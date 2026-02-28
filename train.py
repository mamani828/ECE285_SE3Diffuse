# train.py
import os
import math
import time
import json
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Data: chunking utilities
# -------------------------

def make_chunks(dataset, H=64, stride=16, max_eps=None):
    """
    Returns:
      starts: (N, obs_dim)
      goals:  (N, goal_dim)
      acts:   (N, H, act_dim)
    """
    starts, goals, acts = [], [], []
    for i, ep in enumerate(dataset.iterate_episodes()):
        if max_eps is not None and i >= max_eps:
            break

        obs = ep.observations["observation"]      # (T+1, obs_dim)
        dg  = ep.observations["desired_goal"]     # (T+1, goal_dim)
        act = ep.actions                          # (T, act_dim)

        T = act.shape[0]
        if T < H:
            continue

        for t0 in range(0, T - H + 1, stride):
            starts.append(obs[t0].astype(np.float32))
            goals.append(dg[t0].astype(np.float32))
            acts.append(act[t0:t0+H].astype(np.float32))

    if len(acts) == 0:
        raise RuntimeError("No chunks created. Try smaller H, smaller stride, or more episodes.")
    return np.stack(starts), np.stack(goals), np.stack(acts)


class ChunkDataset(Dataset):
    def __init__(self, starts, goals, actions, stats=None):
        """
        starts:  (N, obs_dim)
        goals:   (N, goal_dim)
        actions: (N, H, act_dim)
        stats: dict with means/stds (optional). If None, computed here.
        """
        self.starts = torch.from_numpy(starts)
        self.goals  = torch.from_numpy(goals)
        self.actions = torch.from_numpy(actions)  # u0 target

        # compute / store normalization
        if stats is None:
            # action stats over all timesteps
            a = self.actions.reshape(-1, self.actions.shape[-1])
            act_mean = a.mean(dim=0)
            act_std  = a.std(dim=0).clamp_min(1e-6)

            c = torch.cat([self.starts, self.goals], dim=-1)
            cond_mean = c.mean(dim=0)
            cond_std  = c.std(dim=0).clamp_min(1e-6)

            stats = {
                "act_mean": act_mean,
                "act_std": act_std,
                "cond_mean": cond_mean,
                "cond_std": cond_std,
            }
        self.stats = stats

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        start = self.starts[idx]
        goal  = self.goals[idx]
        u0    = self.actions[idx]  # (H, act_dim)

        cond = torch.cat([start, goal], dim=-1)

        # normalize
        u0 = (u0 - self.stats["act_mean"]) / self.stats["act_std"]
        cond = (cond - self.stats["cond_mean"]) / self.stats["cond_std"]

        return u0, cond


# -------------------------
# Diffusion utilities
# -------------------------

@dataclass
class DiffusionSchedule:
    betas: torch.Tensor          # (K,)
    alphas: torch.Tensor         # (K,)
    alpha_bars: torch.Tensor     # (K,)

def make_linear_beta_schedule(K, beta_start=1e-4, beta_end=2e-2, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, K, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)

def extract(a, t, x_shape):
    """
    a: (K,)
    t: (B,) int64
    returns (B, 1, 1) broadcastable to x_shape
    """
    out = a.gather(0, t)
    while len(out.shape) < len(x_shape):
        out = out.unsqueeze(-1)
    return out

def q_sample(u0, t, schedule: DiffusionSchedule, noise=None):
    """
    u0: (B, H, act_dim)
    t: (B,)
    """
    if noise is None:
        noise = torch.randn_like(u0)
    ab = extract(schedule.alpha_bars, t, u0.shape)
    return torch.sqrt(ab) * u0 + torch.sqrt(1.0 - ab) * noise, noise


# -------------------------
# Model: temporal Conv ResNet with FiLM conditioning
# -------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) int64 or float
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class ResBlock1D(nn.Module):
    def __init__(self, channels, emb_dim, dilation=1, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)

        self.film = nn.Linear(emb_dim, 2 * channels)  # scale, shift

    def forward(self, x, emb):
        # x: (B, C, H), emb: (B, emb_dim)
        scale_shift = self.film(emb)  # (B, 2C)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)

        h = self.norm1(x)
        h = h * (1 + scale) + shift
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return x + h

class DiffusionPolicy1D(nn.Module):
    def __init__(self, act_dim, cond_dim, hidden=128, emb_dim=256, n_blocks=8):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.in_conv = nn.Conv1d(act_dim, hidden, kernel_size=3, padding=1)
        dilations = [1, 2, 4, 8, 1, 2, 4, 8][:n_blocks]
        self.blocks = nn.ModuleList([ResBlock1D(hidden, emb_dim, dilation=d) for d in dilations])
        self.out_norm = nn.GroupNorm(8, hidden)
        self.out_conv = nn.Conv1d(hidden, act_dim, kernel_size=3, padding=1)

    def forward(self, ut, t, cond):
        """
        ut: (B, H, act_dim) noisy actions
        t:  (B,) timesteps (int64)
        cond: (B, cond_dim)
        returns predicted noise: (B, H, act_dim)
        """
        # build embedding
        te = self.time_mlp(self.time_emb(t))    # (B, emb_dim)
        ce = self.cond_mlp(cond)                # (B, emb_dim)
        emb = te + ce

        x = ut.permute(0, 2, 1)                 # (B, act_dim, H)
        x = self.in_conv(x)                     # (B, hidden, H)
        for blk in self.blocks:
            x = blk(x, emb)
        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)                    # (B, act_dim, H)
        return x.permute(0, 2, 1)               # (B, H, act_dim)


# -------------------------
# Training loop
# -------------------------

def train(args):
    import minari  # keep import here so running --help doesn't require it

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # Load dataset
    ds = minari.load_dataset(args.dataset, download=True)
    print(f"Loaded dataset: {args.dataset} | episodes={ds.total_episodes}")

    # Build chunks
    print("Building chunks...")
    starts, goals, actions = make_chunks(ds, H=args.horizon, stride=args.stride, max_eps=args.max_eps)
    print(f"Chunks: {actions.shape[0]} | horizon={actions.shape[1]} | act_dim={actions.shape[2]}")

    # Dataset + loader
    chunk_ds = ChunkDataset(starts, goals, actions, stats=None)
    stats = {k: v.cpu().numpy() for k, v in chunk_ds.stats.items()}
    with open(os.path.join(args.outdir, "stats.json"), "w") as f:
        json.dump({k: v.tolist() for k, v in stats.items()}, f)

    loader = DataLoader(chunk_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)

    act_dim = actions.shape[-1]
    cond_dim = starts.shape[-1] + goals.shape[-1]

    # Model + diffusion schedule
    model = DiffusionPolicy1D(
        act_dim=act_dim,
        cond_dim=cond_dim,
        hidden=args.hidden,
        emb_dim=args.emb,
        n_blocks=args.blocks
    ).to(device)

    schedule = make_linear_beta_schedule(args.diff_steps, args.beta_start, args.beta_end, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    step = 0
    model.train()
    t0 = time.time()

    for epoch in range(args.epochs):
        for u0, cond in loader:
            u0 = u0.to(device)       # normalized actions (B,H,A)
            cond = cond.to(device)   # normalized condition (B,C)

            B = u0.shape[0]
            t = torch.randint(0, args.diff_steps, (B,), device=device, dtype=torch.long)

            ut, noise = q_sample(u0, t, schedule)

            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                pred = model(ut, t, cond)
                loss = F.mse_loss(pred, noise)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            if step % args.log_every == 0:
                dt = time.time() - t0
                print(f"epoch={epoch} step={step} loss={loss.item():.6f} ({dt:.1f}s)")
            if step % args.ckpt_every == 0 and step > 0:
                save_path = os.path.join(args.outdir, f"ckpt_{step:07d}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args),
                    "step": step,
                }, save_path)
                print(f"Saved: {save_path}")

            step += 1

    # final save
    final_path = os.path.join(args.outdir, "ckpt_final.pt")
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "args": vars(args),
        "step": step,
    }, final_path)
    print(f"Saved final: {final_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="D4RL/pointmaze/umaze-v2")
    p.add_argument("--outdir", type=str, default="runs/umaze")
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--max_eps", type=int, default=None)

    p.add_argument("--diff_steps", type=int, default=100)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)

    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--emb", type=int, default=256)
    p.add_argument("--blocks", type=int, default=8)

    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")

    args = p.parse_args()
    train(args)

if __name__ == "__main__":
    main()