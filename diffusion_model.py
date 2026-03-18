import math
import os
import glob
import json
from dataclasses import dataclass, asdict
#typing helps debugging
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
@dataclass
class TrainConfig:
    data_root: str = "diffdrive_dataset"
    map_mode: str = "sdf"          
    map_size_m: float = 10.0
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-6
    epochs: int = 80
    num_workers: int = 4
    diffusion_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    base_channels: int = 64
    cond_dim: int = 128
    map_emb_dim: int = 128
    pose_emb_dim: int = 64
    time_emb_dim: int = 128
    w_noise: float = 1.0
    w_state: float = 0.25
    w_terminal: float = 1.0
    w_control_smooth: float = 0.02

    grad_clip_norm: float = 1.0
    save_dir: str = "checkpoints"
    seed: int = 7

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def wrap_angle_torch(theta: torch.Tensor) -> torch.Tensor:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi
def angle_diff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return wrap_angle_torch(a - b)
def load_dataset_config(data_root: str) -> Dict:
    cfg_path = os.path.join(data_root, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing dataset config: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
    #diff drive dataset class to load from the other file
class DiffDriveDataset(Dataset):
    def __init__(self,root: str,split: str,map_mode: str, map_size_m: float,v_max: float, w_max: float,horizon: int):
        self.files = sorted(glob.glob(os.path.join(root, split, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {os.path.join(root, split)}")

        self.map_mode = map_mode
        self.map_size_m = map_size_m
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.horizon = int(horizon)
    def __len__(self) -> int:
        return len(self.files)
    # how to condition on pose, the eight dims in the report
    def _pose_condition(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        sx = 2.0 * (start[0] / self.map_size_m) - 1.0
        sy = 2.0 * (start[1] / self.map_size_m) - 1.0
        gx = 2.0 * (goal[0] / self.map_size_m) - 1.0
        gy = 2.0 * (goal[1] / self.map_size_m) - 1.0
        sth = float(start[2])
        gth = float(goal[2])
        return np.asarray(
            [sx, sy, math.cos(sth), math.sin(sth),gx, gy, math.cos(gth), math.sin(gth),],dtype=np.float32,)
    #get controls [-1,1] for both v and w
    def _normalize_controls(self, controls: np.ndarray) -> np.ndarray:
        u = controls.astype(np.float32).copy()
        u[:, 0] /= max(self.v_max, 1e-6)
        u[:, 1] /= max(self.w_max, 1e-6)
        return u
    
    def _build_map_tensor(self, data: np.lib.npyio.NpzFile) -> np.ndarray:
        occ = data["occupancy"].astype(np.float32)
        sdf = data["sdf"].astype(np.float32)
        # diffeent types of occupancy left for future work.
        if self.map_mode == "sdf":
            return sdf[None, ...]
        if self.map_mode == "occupancy":
            return (2.0 * occ - 1.0)[None, ...]
        if self.map_mode == "sdf_occupancy":
            return np.stack([sdf, 2.0 * occ - 1.0], axis=0).astype(np.float32)
        raise ValueError(f"Unsupported map_mode: {self.map_mode}")

    #utils
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.files[idx], allow_pickle=True)

        map_arr = self._build_map_tensor(data)
        start = data["start"].astype(np.float32)
        goal = data["goal"].astype(np.float32)
        controls = data["controls"].astype(np.float32)
        states = data["states"].astype(np.float32)

        valid_horizon = int(data["valid_horizon"]) if "valid_horizon" in data else controls.shape[0]
        valid_horizon = max(1, min(valid_horizon, controls.shape[0], self.horizon))

        controls = controls[: self.horizon]
        states = states[: self.horizon + 1]

        pose_cond = self._pose_condition(start, goal)
        controls_norm = self._normalize_controls(controls)

        ctrl_mask = np.zeros((self.horizon, 1), dtype=np.float32)
        ctrl_mask[:valid_horizon] = 1.0

        state_mask = np.zeros((self.horizon + 1, 1), dtype=np.float32)
        state_mask[: valid_horizon + 1] = 1.0

        return {
            "map": torch.from_numpy(map_arr),                
            "pose_cond": torch.from_numpy(pose_cond),      
            "start": torch.from_numpy(start),               
            "goal": torch.from_numpy(goal),                     
            "controls": torch.from_numpy(controls_norm),         
            "states": torch.from_numpy(states),                  
            "control_mask": torch.from_numpy(ctrl_mask),   
            "state_mask": torch.from_numpy(state_mask),        
            "valid_horizon": torch.tensor(valid_horizon, dtype=torch.long),
        }
#diff schd
class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps: int, beta_start: float, beta_end: float):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.num_steps = int(num_steps)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def predict_x0_from_noise(self, xt: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
        return (xt - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)


#sinousoidal embeddings
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

#CNN encoder
class MapEncoder(nn.Module):
    def __init__(self, in_ch: int, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).flatten(1))

# MLP encoder for pose conditioning
class PoseEncoder(nn.Module):
    def __init__(self, in_dim: int = 8, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

#residual blocks and up/down sampling for the UNet

class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond_proj(cond).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

# main unet
class ConditionalTemporalUNet(nn.Module):
    def __init__(
        self,
        control_dim: int,
        map_in_ch: int,
        base_channels: int,
        cond_dim: int,
        time_emb_dim: int,
        pose_emb_dim: int,
        map_emb_dim: int,
    ):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.pose_encoder = PoseEncoder(in_dim=8, emb_dim=pose_emb_dim)
        self.pose_proj = nn.Linear(pose_emb_dim, cond_dim)

        self.map_encoder = MapEncoder(in_ch=map_in_ch, emb_dim=map_emb_dim)
        self.map_proj = nn.Linear(map_emb_dim, cond_dim)

        self.cond_fuse = nn.Sequential(
            nn.Linear(cond_dim * 3, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        ch = base_channels
        self.in_proj = nn.Conv1d(control_dim, ch, kernel_size=3, padding=1)
        self.down1 = ResBlock1D(ch, ch, cond_dim)
        self.ds1 = Downsample1D(ch)
        self.down2 = ResBlock1D(ch, ch * 2, cond_dim)
        self.ds2 = Downsample1D(ch * 2)
        self.mid1 = ResBlock1D(ch * 2, ch * 4, cond_dim)
        self.mid2 = ResBlock1D(ch * 4, ch * 4, cond_dim)
        self.us1 = Upsample1D(ch * 4)
        self.up1 = ResBlock1D(ch * 4 + ch * 2, ch * 2, cond_dim)
        self.us2 = Upsample1D(ch * 2)
        self.up2 = ResBlock1D(ch * 2 + ch, ch, cond_dim)
        self.out_norm = nn.GroupNorm(num_groups=8, num_channels=ch)
        self.out_proj = nn.Conv1d(ch, control_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, map_tensor: torch.Tensor, pose_cond: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        t_emb = self.time_mlp(self.time_emb(t))
        p_emb = self.pose_proj(self.pose_encoder(pose_cond))
        m_emb = self.map_proj(self.map_encoder(map_tensor))
        cond = self.cond_fuse(torch.cat([t_emb, p_emb, m_emb], dim=1))

        x0 = self.in_proj(x)
        d1 = self.down1(x0, cond)
        x1 = self.ds1(d1)
        d2 = self.down2(x1, cond)
        x2 = self.ds2(d2)
        m = self.mid1(x2, cond)
        m = self.mid2(m, cond)
        u1 = self.us1(m)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up1(u1, cond)
        u2 = self.us2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up2(u2, cond)
        out = self.out_proj(F.silu(self.out_norm(u2)))
        return out.transpose(1, 2)

#denorm

def denormalize_controls(u_norm: torch.Tensor, v_max: float, w_max: float) -> torch.Tensor:
    v = u_norm[..., 0] * v_max
    w = u_norm[..., 1] * w_max
    return torch.stack([v, w], dim=-1)


def rollout_unicycle_batch(start: torch.Tensor, controls: torch.Tensor, dt: float) -> torch.Tensor:
    _, T, _ = controls.shape
    cur = start
    states = [cur]
    for k in range(T):
        x = cur[:, 0]
        y = cur[:, 1]
        th = cur[:, 2]
        v = controls[:, k, 0]
        w = controls[:, k, 1]

        nxt = torch.stack(
            [x + dt * v * torch.cos(th),y + dt * v * torch.sin(th),wrap_angle_torch(th + dt * w),],dim=-1,)
        states.append(nxt)
        cur = nxt

    return torch.stack(states, dim=1)

# all loss funcions

#masked mse for noise
def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    diff = diff * mask
    denom = mask.sum().clamp_min(1.0) * pred.shape[-1]
    return diff.sum() / denom

#state tracking loss, 8 dim state, but only care about x,y and theta, and also mask out invalid states
def state_tracking_loss(pred_states: torch.Tensor, gt_states: torch.Tensor, state_mask: torch.Tensor) -> torch.Tensor:
    pos_loss = ((pred_states[..., :2] - gt_states[..., :2]) ** 2) * state_mask
    th_err = angle_diff_torch(pred_states[..., 2], gt_states[..., 2]).unsqueeze(-1)
    th_loss = (th_err ** 2) * state_mask
    pos_term = pos_loss.sum() / (state_mask.sum().clamp_min(1.0) * 2.0)
    th_term = th_loss.sum() / state_mask.sum().clamp_min(1.0)
    return pos_term + 0.25 * th_term

# terminal loss to encourage reaching the goal, only look at the final valid state and the goal, and also mask out invalid states based on valid horizon
def terminal_loss(pred_states: torch.Tensor, goal: torch.Tensor, valid_horizon: torch.Tensor) -> torch.Tensor:
    B = pred_states.shape[0]
    idx = valid_horizon.clamp(min=1, max=pred_states.shape[1] - 1)
    batch_idx = torch.arange(B, device=pred_states.device)
    final_states = pred_states[batch_idx, idx]
    pos = F.mse_loss(final_states[:, :2], goal[:, :2])
    th = angle_diff_torch(final_states[:, 2], goal[:, 2])
    th = torch.mean(th ** 2)
    return pos + 0.5 * th

# smoothness loss to encourage smoother control sequences, only look at valid control steps based on control mask
def smoothness_loss(u: torch.Tensor, control_mask: torch.Tensor) -> torch.Tensor:
    if u.shape[1] < 2:
        return torch.zeros((), device=u.device, dtype=u.dtype)
    du = u[:, 1:] - u[:, :-1]
    mask = control_mask[:, 1:] * control_mask[:, :-1]
    return masked_mse(du, torch.zeros_like(du), mask)

# Train

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    dt: float,
    v_max: float,
    w_max: float,
    cfg: TrainConfig,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    meters = {
        "loss": 0.0,
        "noise": 0.0,
        "state": 0.0,
        "terminal": 0.0,
        "smooth": 0.0,
        "n": 0,
    }

    for batch in loader:
        maps = batch["map"].to(device, non_blocking=True)
        pose_cond = batch["pose_cond"].to(device, non_blocking=True)
        start = batch["start"].to(device, non_blocking=True)
        goal = batch["goal"].to(device, non_blocking=True)
        u0 = batch["controls"].to(device, non_blocking=True)
        gt_states = batch["states"].to(device, non_blocking=True)
        control_mask = batch["control_mask"].to(device, non_blocking=True)
        state_mask = batch["state_mask"].to(device, non_blocking=True)
        valid_horizon = batch["valid_horizon"].to(device, non_blocking=True)

        B = u0.shape[0]
        t = torch.randint(0, schedule.num_steps, (B,), device=device)
        noise = torch.randn_like(u0)
        uk = schedule.q_sample(u0, t, noise)

        with torch.set_grad_enabled(is_train):
            pred_noise = model(uk, t, maps, pose_cond)
            x0_hat = schedule.predict_x0_from_noise(uk, t, pred_noise)
            x0_hat = x0_hat.clamp(-1.5, 1.5)

            pred_controls = denormalize_controls(x0_hat, v_max=v_max, w_max=w_max)
            pred_states = rollout_unicycle_batch(start, pred_controls, dt=dt)

            noise_loss = masked_mse(pred_noise, noise, control_mask)
            st_loss = state_tracking_loss(pred_states, gt_states, state_mask)
            term_loss = terminal_loss(pred_states, goal, valid_horizon)
            sm_loss = smoothness_loss(pred_controls, control_mask)

            loss = (
                cfg.w_noise * noise_loss
                + cfg.w_state * st_loss
                + cfg.w_terminal * term_loss
                + cfg.w_control_smooth * sm_loss
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

        meters["loss"] += float(loss.item()) * B
        meters["noise"] += float(noise_loss.item()) * B
        meters["state"] += float(st_loss.item()) * B
        meters["terminal"] += float(term_loss.item()) * B
        meters["smooth"] += float(sm_loss.item()) * B
        meters["n"] += B

    n = max(1, meters.pop("n"))
    return {k: v / n for k, v in meters.items()}


#DDPM
@torch.no_grad()
def sample_controls(
    model: nn.Module,
    schedule: DiffusionSchedule,
    map_tensor: torch.Tensor,
    pose_cond: torch.Tensor,
    horizon: int,
    control_dim: int,
    device: str,
    eta_clip: float = 1.5,
) -> torch.Tensor:
    B = map_tensor.shape[0]
    x = torch.randn(B, horizon, control_dim, device=device)

    for step in reversed(range(schedule.num_steps)):
        t = torch.full((B,), step, device=device, dtype=torch.long)
        pred_noise = model(x, t, map_tensor, pose_cond)

        alpha_t = schedule.alphas[t].view(-1, 1, 1)
        alpha_bar_t = schedule.alpha_bars[t].view(-1, 1, 1)
        beta_t = schedule.betas[t].view(-1, 1, 1)
        alpha_bar_prev = schedule.alpha_bars_prev[t].view(-1, 1, 1)

        x0_hat = schedule.predict_x0_from_noise(x, t, pred_noise).clamp(-eta_clip, eta_clip)

        coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        mean = coef1 * x0_hat + coef2 * x

        if step > 0:
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            x = mean + torch.sqrt(posterior_var.clamp_min(1e-8)) * torch.randn_like(x)
        else:
            x = mean

    return x
def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    ds_cfg = load_dataset_config(cfg.data_root)
    horizon = int(ds_cfg["horizon"])
    v_max = float(ds_cfg["v_max"])
    w_max = float(ds_cfg["w_max"])
    dt = float(ds_cfg["dt"])
    if cfg.map_mode == "sdf_occupancy":
        map_in_ch = 2 
    else:
        map_in_ch = 1
    train_ds = DiffDriveDataset(
        root=cfg.data_root,
        split="train",
        map_mode=cfg.map_mode,
        map_size_m=cfg.map_size_m,
        v_max=v_max,
        w_max=w_max,
        horizon=horizon,
    )
    val_ds = DiffDriveDataset(
        root=cfg.data_root,
        split="val",
        map_mode=cfg.map_mode,
        map_size_m=cfg.map_size_m,
        v_max=v_max,
        w_max=w_max,
        horizon=horizon,
    )

    pin = cfg.device.startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=(cfg.num_workers > 0),
    )

    model = ConditionalTemporalUNet(
        control_dim=2,
        map_in_ch=map_in_ch,
        base_channels=cfg.base_channels,
        cond_dim=cfg.cond_dim,
        time_emb_dim=cfg.time_emb_dim,
        pose_emb_dim=cfg.pose_emb_dim,
        map_emb_dim=cfg.map_emb_dim,
    ).to(cfg.device)

    schedule = DiffusionSchedule(
        num_steps=cfg.diffusion_steps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(cfg.save_dir, exist_ok=True)
    best_val = float("inf")

    print(f"train samples: {len(train_ds)} | val samples: {len(val_ds)} | device: {cfg.device}")
    print(f"dataset horizon={horizon} dt={dt} v_max={v_max} w_max={w_max} map_mode={cfg.map_mode}")

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            schedule=schedule,
            optimizer=optimizer,
            device=cfg.device,
            dt=dt,
            v_max=v_max,
            w_max=w_max,
            cfg=cfg,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            schedule=schedule,
            optimizer=None,
            device=cfg.device,
            dt=dt,
            v_max=v_max,
            w_max=w_max,
            cfg=cfg,
        )

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.5f} noise {train_metrics['noise']:.5f} state {train_metrics['state']:.5f} term {train_metrics['terminal']:.5f} | "
            f"val loss {val_metrics['loss']:.5f} noise {val_metrics['noise']:.5f} state {val_metrics['state']:.5f} term {val_metrics['terminal']:.5f}"
        )

        save_payload = {
            "model": model.state_dict(),
            "schedule": schedule.state_dict(),
            "train_cfg": asdict(cfg),
            "dataset_cfg": ds_cfg,
            "best_val": best_val,
        }

        torch.save(save_payload, os.path.join(cfg.save_dir, "diffdrive_last.pt"))

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_payload["best_val"] = best_val
            torch.save(save_payload, os.path.join(cfg.save_dir, "diffdrive_kinodynamic_best.pt"))


if __name__ == "__main__":
    main()
