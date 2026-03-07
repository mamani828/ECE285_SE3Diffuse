import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

@dataclass
class InferConfig:
    checkpoint_path: str = "checkpoints/diffdrive_kinodynamic_best.pt"
    data_root: str = "diffdrive_dataset"
    split: str = "test"
    sample_index: Optional[int] = None
    randomize_test_env: bool = True
    random_seed: Optional[int] = None

    map_mode: str = "sdf"   # must match training: "sdf", "occupancy", or "sdf_occupancy"
    num_samples: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Save / inspect
    out_dir: str = "inference_outputs"
    save_npz: bool = True
    save_plot: bool = True
    show_plot: bool = False
    print_controls_head: int = 10

    # --- test-time projection ---
    use_projection: bool = True
    project_every: int = 10          # apply every p reverse steps
    proj_steps: int = 6              # inner Adam steps
    proj_lr: float = 0.06
    proj_lambda: float = 0.1
    control_clip_norm: float = 1.0   # normalized controls clipped to [-1, 1]
    safety_margin_m: float = 0.05

    # penalty weights for Phi
    w_goal_pos: float = 25.0
    w_goal_theta: float = 2.0
    w_obs: float = 400.0
    w_ctrl: float = 1e-3
    w_smooth: float = 0.02

    # DDPM x0 clipping
    eta_clip: float = 1.5


# ============================================================
# Utilities
# ============================================================

def wrap_angle_torch(theta: torch.Tensor) -> torch.Tensor:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return wrap_angle_torch(a - b)


def choose_sample_index(
    num_files: int,
    sample_index: Optional[int],
    randomize_test_env: bool,
    random_seed: Optional[int]
) -> int:
    if num_files <= 0:
        raise ValueError("num_files must be positive")

    if randomize_test_env:
        rng = random.Random(random_seed)
        return rng.randrange(num_files)

    if sample_index is None:
        return 0

    if sample_index < 0 or sample_index >= num_files:
        raise IndexError(f"sample_index {sample_index} out of range for {num_files} files")

    return int(sample_index)


def pose_condition(start: np.ndarray, goal: np.ndarray, map_size_m: float) -> np.ndarray:
    sx = 2.0 * (start[0] / map_size_m) - 1.0
    sy = 2.0 * (start[1] / map_size_m) - 1.0
    gx = 2.0 * (goal[0] / map_size_m) - 1.0
    gy = 2.0 * (goal[1] / map_size_m) - 1.0
    sth = float(start[2])
    gth = float(goal[2])

    return np.asarray(
        [
            sx, sy, math.cos(sth), math.sin(sth),
            gx, gy, math.cos(gth), math.sin(gth),
        ],
        dtype=np.float32,
    )


def build_map_tensor(data: np.lib.npyio.NpzFile, map_mode: str) -> np.ndarray:
    occ = data["occupancy"].astype(np.float32)
    sdf = data["sdf"].astype(np.float32)

    if map_mode == "sdf":
        return sdf[None, ...]
    if map_mode == "occupancy":
        return (2.0 * occ - 1.0)[None, ...]
    if map_mode == "sdf_occupancy":
        return np.stack([sdf, 2.0 * occ - 1.0], axis=0).astype(np.float32)

    raise ValueError(f"Unsupported map_mode: {map_mode}")


def denormalize_controls(u_norm: torch.Tensor, v_max: float, w_max: float) -> torch.Tensor:
    v = u_norm[..., 0] * v_max
    w = u_norm[..., 1] * w_max
    return torch.stack([v, w], dim=-1)


def rollout_unicycle_batch(start: torch.Tensor, controls: torch.Tensor, dt: float) -> torch.Tensor:
    # start: (B, 3), controls: (B, T, 2)
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
            [
                x + dt * v * torch.cos(th),
                y + dt * v * torch.sin(th),
                wrap_angle_torch(th + dt * w),
            ],
            dim=-1,
        )
        states.append(nxt)
        cur = nxt

    return torch.stack(states, dim=1)


def sdf_query_bilinear_torch(
    sdf: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    map_size_m: float
) -> torch.Tensor:
    # sdf: (H, W), xs/ys: (...)
    H, W = sdf.shape

    gx = torch.clamp((xs / map_size_m) * (W - 1), 0.0, W - 1.0)
    gy = torch.clamp((ys / map_size_m) * (H - 1), 0.0, H - 1.0)

    x0 = torch.floor(gx).long()
    y0 = torch.floor(gy).long()
    x1 = torch.clamp(x0 + 1, max=W - 1)
    y1 = torch.clamp(y0 + 1, max=H - 1)

    tx = gx - x0.float()
    ty = gy - y0.float()

    v00 = sdf[y0, x0]
    v10 = sdf[y0, x1]
    v01 = sdf[y1, x0]
    v11 = sdf[y1, x1]

    v0 = (1.0 - tx) * v00 + tx * v10
    v1 = (1.0 - tx) * v01 + tx * v11
    return (1.0 - ty) * v0 + ty * v1


# ============================================================
# Diffusion schedule
# ============================================================

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

    def predict_x0_from_noise(self, xt: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
        return (xt - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)


# ============================================================
# Embeddings / encoders
# ============================================================

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


# ============================================================
# 1D U-Net
# ============================================================

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


# ============================================================
# Projection objective
# ============================================================

def projection_objective(
    u_norm: torch.Tensor,          # (B, T, 2)
    u_ref: torch.Tensor,           # (B, T, 2)
    start: torch.Tensor,           # (B, 3)
    goal: torch.Tensor,            # (B, 3)
    sdf_map: torch.Tensor,         # (H, W)
    dt: float,
    v_max: float,
    w_max: float,
    map_size_m: float,
    robot_radius: float,
    safety_margin_m: float,
    proj_lambda: float,
    w_goal_pos: float,
    w_goal_theta: float,
    w_obs: float,
    w_ctrl: float,
    w_smooth: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    controls = denormalize_controls(u_norm, v_max=v_max, w_max=w_max)
    states = rollout_unicycle_batch(start, controls, dt=dt)

    # terminal goal error
    goal_pos_term = torch.sum((states[:, -1, :2] - goal[:, :2]) ** 2, dim=1)
    goal_theta_term = angle_diff_torch(states[:, -1, 2], goal[:, 2]) ** 2

    # obstacle / clearance penalty
    clearance = sdf_query_bilinear_torch(
        sdf_map,
        states[..., 0].reshape(-1),
        states[..., 1].reshape(-1),
        map_size_m=map_size_m,
    ).view(states.shape[0], states.shape[1]) - robot_radius

    obs_term = F.relu(safety_margin_m - clearance).pow(2).sum(dim=1)

    # regularization in physical control space
    ctrl_term = controls.pow(2).sum(dim=(1, 2))
    if controls.shape[1] > 1:
        smooth_term = (controls[:, 1:] - controls[:, :-1]).pow(2).sum(dim=(1, 2))
    else:
        smooth_term = torch.zeros_like(ctrl_term)

    phi = (
        w_goal_pos * goal_pos_term
        + w_goal_theta * goal_theta_term
        + w_obs * obs_term
        + w_ctrl * ctrl_term
        + w_smooth * smooth_term
    )

    prox = 0.5 * (u_norm - u_ref).pow(2).sum(dim=(1, 2))
    loss = (prox + proj_lambda * phi).mean()

    aux = {
        "prox": prox.mean().detach(),
        "phi": phi.mean().detach(),
        "goal_pos": goal_pos_term.mean().detach(),
        "goal_theta": goal_theta_term.mean().detach(),
        "obs": obs_term.mean().detach(),
        "ctrl": ctrl_term.mean().detach(),
        "smooth": smooth_term.mean().detach(),
    }
    return loss, aux


# ============================================================
# Sampler
# ============================================================

def sample_controls(
    model: nn.Module,
    schedule: DiffusionSchedule,
    map_tensor: torch.Tensor,
    pose_cond: torch.Tensor,
    horizon: int,
    control_dim: int,
    device: str,
    start_tensor: torch.Tensor,
    goal_tensor: torch.Tensor,
    sdf_map: torch.Tensor,
    dt: float,
    v_max: float,
    w_max: float,
    map_size_m: float,
    robot_radius: float,
    use_projection: bool = True,
    project_every: int = 10,
    proj_steps: int = 6,
    proj_lr: float = 0.06,
    proj_lambda: float = 0.1,
    control_clip_norm: float = 1.0,
    safety_margin_m: float = 0.05,
    w_goal_pos: float = 25.0,
    w_goal_theta: float = 2.0,
    w_obs: float = 400.0,
    w_ctrl: float = 1e-3,
    w_smooth: float = 0.02,
    eta_clip: float = 1.5,
) -> torch.Tensor:
    B = map_tensor.shape[0]
    x = torch.randn(B, horizon, control_dim, device=device)

    for step in reversed(range(schedule.num_steps)):
        # ----------------------------------------------------
        # Standard reverse diffusion step
        # ----------------------------------------------------
        with torch.no_grad():
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
                u_tilde = mean + torch.sqrt(posterior_var.clamp_min(1e-8)) * torch.randn_like(x)
            else:
                u_tilde = mean

        # ----------------------------------------------------
        # Test-time projection every p reverse steps
        # ----------------------------------------------------
        k = step + 1   # 1-based indexing
        do_project = use_projection and (project_every > 0) and (k % project_every == 0)

        if do_project:
            u_proj = u_tilde.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([u_proj], lr=proj_lr)

            for _ in range(proj_steps):
                opt.zero_grad(set_to_none=True)

                loss, _ = projection_objective(
                    u_norm=u_proj,
                    u_ref=u_tilde.detach(),
                    start=start_tensor,
                    goal=goal_tensor,
                    sdf_map=sdf_map,
                    dt=dt,
                    v_max=v_max,
                    w_max=w_max,
                    map_size_m=map_size_m,
                    robot_radius=robot_radius,
                    safety_margin_m=safety_margin_m,
                    proj_lambda=proj_lambda,
                    w_goal_pos=w_goal_pos,
                    w_goal_theta=w_goal_theta,
                    w_obs=w_obs,
                    w_ctrl=w_ctrl,
                    w_smooth=w_smooth,
                )

                loss.backward()
                opt.step()

                with torch.no_grad():
                    u_proj.clamp_(-control_clip_norm, control_clip_norm)

            x = u_proj.detach()
        else:
            x = u_tilde.detach()

    return x


# ============================================================
# Plotting
# ============================================================

@torch.no_grad()
def plot_rollouts(
    out_path: str,
    map_mode: str,
    occupancy_map: np.ndarray,
    sdf_map: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    expert_states: np.ndarray,
    sampled_states: np.ndarray,
    best_idx: int,
    metrics: Dict[str, np.ndarray],
    map_size_m: float,
    scenario_type: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    extent = [0.0, map_size_m, 0.0, map_size_m]
    if map_mode == "occupancy":
        ax.imshow(occupancy_map, origin="lower", extent=extent, alpha=0.9)
    else:
        ax.imshow(sdf_map, origin="lower", extent=extent, alpha=0.9)
        occ_mask = np.ma.masked_where(occupancy_map < 0.5, occupancy_map)
        ax.imshow(occ_mask, origin="lower", extent=extent, alpha=0.35)

    for i in range(sampled_states.shape[0]):
        st = sampled_states[i]
        is_best = i == best_idx
        label = f"best sampled (success={bool(metrics['success'][i])})" if is_best else None
        ax.plot(
            st[:, 0],
            st[:, 1],
            linewidth=2.5 if is_best else 1.2,
            alpha=0.95 if is_best else 0.45,
            zorder=3 if is_best else 2,
            label=label,
        )

    ax.plot(
        expert_states[:, 0],
        expert_states[:, 1],
        linestyle="--",
        linewidth=2.2,
        alpha=0.95,
        label="expert",
        zorder=4,
    )

    ax.scatter([start[0]], [start[1]], s=80, marker="o", label="start", zorder=5)
    ax.scatter([goal[0]], [goal[1]], s=90, marker="*", label="goal", zorder=5)

    start_dx = 0.35 * math.cos(float(start[2]))
    start_dy = 0.35 * math.sin(float(start[2]))
    goal_dx = 0.35 * math.cos(float(goal[2]))
    goal_dy = 0.35 * math.sin(float(goal[2]))

    ax.arrow(float(start[0]), float(start[1]), start_dx, start_dy, width=0.02, length_includes_head=True, zorder=5)
    ax.arrow(float(goal[0]), float(goal[1]), goal_dx, goal_dy, width=0.02, length_includes_head=True, zorder=5)

    ax.set_xlim(0.0, map_size_m)
    ax.set_ylim(0.0, map_size_m)
    ax.set_aspect("equal")
    ax.set_title(
        f"Kinodynamic diffusion inference\n"
        f"scenario={scenario_type} | best pos_err={metrics['final_pos_err'][best_idx]:.3f} | "
        f"best th_err={metrics['final_theta_err_rad'][best_idx]:.3f} rad | "
        f"collision={bool(metrics['collision'][best_idx])}"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig, ax


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_rollouts(
    states: torch.Tensor,
    goal: torch.Tensor,
    sdf_map: torch.Tensor,
    robot_radius: float,
    goal_pos_tol: float,
    goal_theta_tol_deg: float,
    map_size_m: float,
) -> Dict[str, np.ndarray]:
    B = states.shape[0]

    pos_err = torch.linalg.norm(states[:, -1, :2] - goal[:, :2], dim=1)
    th_err = torch.abs(angle_diff_torch(states[:, -1, 2], goal[:, 2]))

    d = sdf_query_bilinear_torch(
        sdf_map,
        states[..., 0].reshape(-1),
        states[..., 1].reshape(-1),
        map_size_m=map_size_m,
    ).view(B, states.shape[1]) - robot_radius

    min_clearance = torch.min(d, dim=1).values
    collision = torch.any(d < 0.0, dim=1)

    success = (
        (pos_err <= goal_pos_tol)
        & (th_err <= math.radians(goal_theta_tol_deg*3))
        & (~collision)
    )

    return {
        "final_pos_err": pos_err.cpu().numpy(),
        "final_theta_err_rad": th_err.cpu().numpy(),
        "min_clearance": min_clearance.cpu().numpy(),
        "collision": collision.cpu().numpy(),
        "success": success.cpu().numpy(),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = InferConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)

    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    train_cfg = ckpt["train_cfg"]
    ds_cfg = ckpt["dataset_cfg"]

    if cfg.map_mode != train_cfg["map_mode"]:
        print(
            f"[warning] infer map_mode={cfg.map_mode} differs from "
            f"train map_mode={train_cfg['map_mode']}. Using infer setting."
        )

    map_in_ch = 2 if cfg.map_mode == "sdf_occupancy" else 1
    horizon = int(ds_cfg["horizon"])
    dt = float(ds_cfg["dt"])
    v_max = float(ds_cfg["v_max"])
    w_max = float(ds_cfg["w_max"])
    map_size_m = float(ds_cfg["map_size_m"])
    robot_radius = float(ds_cfg["robot_radius"])
    goal_pos_tol = float(ds_cfg["goal_pos_tol"])
    goal_theta_tol_deg = float(ds_cfg["goal_theta_tol_deg"])

    model = ConditionalTemporalUNet(
        control_dim=2,
        map_in_ch=map_in_ch,
        base_channels=int(train_cfg["base_channels"]),
        cond_dim=int(train_cfg["cond_dim"]),
        time_emb_dim=int(train_cfg["time_emb_dim"]),
        pose_emb_dim=int(train_cfg["pose_emb_dim"]),
        map_emb_dim=int(train_cfg["map_emb_dim"]),
    ).to(cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    schedule = DiffusionSchedule(
        num_steps=int(train_cfg["diffusion_steps"]),
        beta_start=float(train_cfg["beta_start"]),
        beta_end=float(train_cfg["beta_end"]),
    ).to(cfg.device)
    schedule.load_state_dict(ckpt["schedule"])
    schedule.eval()

    files = sorted(glob.glob(os.path.join(cfg.data_root, cfg.split, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {os.path.join(cfg.data_root, cfg.split)}")

    chosen_index = choose_sample_index(
        num_files=len(files),
        sample_index=cfg.sample_index,
        randomize_test_env=cfg.randomize_test_env,
        random_seed=cfg.random_seed,
    )

    sample_path = files[chosen_index]
    data = np.load(sample_path, allow_pickle=True)

    map_arr = build_map_tensor(data, cfg.map_mode)
    occupancy_map_np = data["occupancy"].astype(np.float32)
    sdf_map_np = data["sdf"].astype(np.float32)
    start_np = data["start"].astype(np.float32)
    goal_np = data["goal"].astype(np.float32)
    expert_controls_np = data["controls"].astype(np.float32)
    expert_states_np = data["states"].astype(np.float32)

    scenario_raw = data["scenario_type"]
    scenario_type = scenario_raw.item() if np.ndim(scenario_raw) == 0 else str(scenario_raw)

    pose_cond_np = pose_condition(start_np, goal_np, map_size_m=map_size_m)

    map_tensor = torch.from_numpy(map_arr).unsqueeze(0).repeat(cfg.num_samples, 1, 1, 1).to(cfg.device)
    pose_tensor = torch.from_numpy(pose_cond_np).unsqueeze(0).repeat(cfg.num_samples, 1).to(cfg.device)
    start_tensor = torch.from_numpy(start_np).unsqueeze(0).repeat(cfg.num_samples, 1).to(cfg.device)
    goal_tensor = torch.from_numpy(goal_np).unsqueeze(0).repeat(cfg.num_samples, 1).to(cfg.device)
    sdf_map = torch.from_numpy(sdf_map_np).to(cfg.device)

    sampled_controls_norm = sample_controls(
        model=model,
        schedule=schedule,
        map_tensor=map_tensor,
        pose_cond=pose_tensor,
        horizon=horizon,
        control_dim=2,
        device=cfg.device,
        start_tensor=start_tensor,
        goal_tensor=goal_tensor,
        sdf_map=sdf_map,
        dt=dt,
        v_max=v_max,
        w_max=w_max,
        map_size_m=map_size_m,
        robot_radius=robot_radius,
        use_projection=cfg.use_projection,
        project_every=cfg.project_every,
        proj_steps=cfg.proj_steps,
        proj_lr=cfg.proj_lr,
        proj_lambda=cfg.proj_lambda,
        control_clip_norm=cfg.control_clip_norm,
        safety_margin_m=cfg.safety_margin_m,
        w_goal_pos=cfg.w_goal_pos,
        w_goal_theta=cfg.w_goal_theta,
        w_obs=cfg.w_obs,
        w_ctrl=cfg.w_ctrl,
        w_smooth=cfg.w_smooth,
        eta_clip=cfg.eta_clip,
    )

    sampled_controls = denormalize_controls(sampled_controls_norm, v_max=v_max, w_max=w_max)
    sampled_states = rollout_unicycle_batch(start_tensor, sampled_controls, dt=dt)

    metrics = evaluate_rollouts(
        states=sampled_states,
        goal=goal_tensor,
        sdf_map=sdf_map,
        robot_radius=robot_radius,
        goal_pos_tol=goal_pos_tol,
        goal_theta_tol_deg=goal_theta_tol_deg,
        map_size_m=map_size_m,
    )

    score = metrics["final_pos_err"] + 0.5 * metrics["final_theta_err_rad"]
    success_mask = metrics["success"].astype(bool)
    if np.any(success_mask):
        best_idx = int(np.argmin(np.where(success_mask, score, np.inf)))
    else:
        best_idx = int(np.argmin(score))

    print(f"checkpoint: {cfg.checkpoint_path}")
    print(f"sample: {sample_path}")
    print(f"sample_index: {chosen_index}")
    print(f"scenario: {scenario_type}")
    print(f"use_projection: {cfg.use_projection}")
    if cfg.use_projection:
        print(
            f"project_every: {cfg.project_every} | proj_steps: {cfg.proj_steps} | "
            f"proj_lr: {cfg.proj_lr} | proj_lambda: {cfg.proj_lambda}"
        )
    print(f"num_samples: {cfg.num_samples} | best_idx: {best_idx}")
    print()

    for i in range(cfg.num_samples):
        print(
            f"traj {i:02d} | "
            f"pos_err={metrics['final_pos_err'][i]:.4f} | "
            f"th_err={metrics['final_theta_err_rad'][i]:.4f} rad | "
            f"min_clear={metrics['min_clearance'][i]:.4f} | "
            f"collision={bool(metrics['collision'][i])} | "
            f"success={bool(metrics['success'][i])}"
        )

    print("\nBest sampled controls (first steps):")
    head = min(cfg.print_controls_head, horizon)
    print(sampled_controls[best_idx, :head].detach().cpu().numpy())

    print("\nExpert controls (first steps):")
    print(expert_controls_np[:head])

    out = {
        "config": asdict(cfg),
        "checkpoint": cfg.checkpoint_path,
        "sample_path": sample_path,
        "sample_index": chosen_index,
        "scenario_type": scenario_type,
        "best_idx": best_idx,
        "metrics": {
            "final_pos_err": metrics["final_pos_err"].tolist(),
            "final_theta_err_rad": metrics["final_theta_err_rad"].tolist(),
            "min_clearance": metrics["min_clearance"].tolist(),
            "collision": metrics["collision"].astype(np.int32).tolist(),
            "success": metrics["success"].astype(np.int32).tolist(),
        },
    }

    with open(os.path.join(cfg.out_dir, "infer_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    sampled_states_np = sampled_states.detach().cpu().numpy()
    sampled_controls_np = sampled_controls.detach().cpu().numpy()

    if cfg.save_npz:
        np.savez_compressed(
            os.path.join(cfg.out_dir, "infer_sample_outputs.npz"),
            map=map_arr,
            occupancy=occupancy_map_np,
            sdf=sdf_map_np,
            start=start_np,
            goal=goal_np,
            expert_controls=expert_controls_np,
            expert_states=expert_states_np,
            sampled_controls=sampled_controls_np,
            sampled_states=sampled_states_np,
            final_pos_err=metrics["final_pos_err"],
            final_theta_err_rad=metrics["final_theta_err_rad"],
            min_clearance=metrics["min_clearance"],
            collision=metrics["collision"].astype(np.uint8),
            success=metrics["success"].astype(np.uint8),
            best_idx=np.array(best_idx, dtype=np.int32),
        )

    if cfg.save_plot or cfg.show_plot:
        fig, _ = plot_rollouts(
            out_path=os.path.join(cfg.out_dir, "infer_plot.png"),
            map_mode=cfg.map_mode,
            occupancy_map=occupancy_map_np,
            sdf_map=sdf_map_np,
            start=start_np,
            goal=goal_np,
            expert_states=expert_states_np,
            sampled_states=sampled_states_np,
            best_idx=best_idx,
            metrics=metrics,
            map_size_m=map_size_m,
            scenario_type=scenario_type,
        )
        if cfg.show_plot:
            plt.show()
        else:
            plt.close(fig)

    print(f"\nSaved outputs to: {cfg.out_dir}")
    if cfg.save_plot:
        print(f"Saved plot to: {os.path.join(cfg.out_dir, 'infer_plot.png')}")


if __name__ == "__main__":
    main()