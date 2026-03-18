import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List

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

    map_mode: str = "sdf"   # must match training: "sdf", "occupancy", or "sdf_occupancy"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Whole-test-set evaluation
    planner_mode: str = "all"   # "all", "diffusion_only", "diffusion_projected", "cem", "mppi"
    max_test_files: Optional[int] = None   # None => evaluate all files
    random_seed: int = 0

    # Diffusion candidate count per test instance
    num_samples: int = 64

    # Output
    out_dir: str = "inference_outputs_full_eval"
    save_example_plots: bool = True
    example_plot_count: int = 20
    print_controls_head: int = 10

    # --- test-time projection ---
    project_every: int = 10
    proj_steps: int = 6
    proj_lr: float = 0.06
    proj_lambda: float = 0.1
    control_clip_norm: float = 1.0
    safety_margin_m: float = 0.05

    # shared planning cost weights
    w_goal_pos: float = 25.0
    w_goal_theta: float = 2.0
    w_obs: float = 400.0
    w_ctrl: float = 1e-3
    w_smooth: float = 0.02

    # DDPM x0 clipping
    eta_clip: float = 1.5

    # --- CEM baseline ---
    cem_population: int = 256
    cem_iters: int = 8
    cem_elite_frac: float = 0.1
    cem_momentum: float = 0.25
    cem_init_std_norm: float = 0.75
    cem_min_std_norm: float = 0.05

    # --- MPPI baseline ---
    mppi_population: int = 256
    mppi_iters: int = 10
    mppi_temperature: float = 1.0
    mppi_sigma_v_norm: float = 0.35
    mppi_sigma_w_norm: float = 0.35


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_angle_torch(theta: torch.Tensor) -> torch.Tensor:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return wrap_angle_torch(a - b)


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


def expand_to_batch(x: torch.Tensor, B: int) -> torch.Tensor:
    if x.shape[0] == B:
        return x
    if x.shape[0] == 1:
        return x.repeat(B, 1)
    raise ValueError(f"Cannot expand tensor with batch {x.shape[0]} to {B}")


# ============================================================
# Shared planning cost
# ============================================================

def planning_cost_from_controls(
    controls: torch.Tensor,         # (B, T, 2) in physical units
    start: torch.Tensor,            # (1,3) or (B,3)
    goal: torch.Tensor,             # (1,3) or (B,3)
    sdf_map: torch.Tensor,          # (H,W)
    dt: float,
    map_size_m: float,
    robot_radius: float,
    safety_margin_m: float,
    w_goal_pos: float,
    w_goal_theta: float,
    w_obs: float,
    w_ctrl: float,
    w_smooth: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    B = controls.shape[0]
    start = expand_to_batch(start, B)
    goal = expand_to_batch(goal, B)

    states = rollout_unicycle_batch(start, controls, dt=dt)

    goal_pos_term = torch.sum((states[:, -1, :2] - goal[:, :2]) ** 2, dim=1)
    goal_theta_term = angle_diff_torch(states[:, -1, 2], goal[:, 2]) ** 2

    clearance = sdf_query_bilinear_torch(
        sdf_map,
        states[..., 0].reshape(-1),
        states[..., 1].reshape(-1),
        map_size_m=map_size_m,
    ).view(B, states.shape[1]) - robot_radius

    obs_term = F.relu(safety_margin_m - clearance).pow(2).sum(dim=1)
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

    aux = {
        "goal_pos": goal_pos_term.detach(),
        "goal_theta": goal_theta_term.detach(),
        "obs": obs_term.detach(),
        "ctrl": ctrl_term.detach(),
        "smooth": smooth_term.detach(),
    }
    return phi, states, aux


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
    u_norm: torch.Tensor,
    u_ref: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    sdf_map: torch.Tensor,
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

    phi, _, aux_terms = planning_cost_from_controls(
        controls=controls,
        start=start,
        goal=goal,
        sdf_map=sdf_map,
        dt=dt,
        map_size_m=map_size_m,
        robot_radius=robot_radius,
        safety_margin_m=safety_margin_m,
        w_goal_pos=w_goal_pos,
        w_goal_theta=w_goal_theta,
        w_obs=w_obs,
        w_ctrl=w_ctrl,
        w_smooth=w_smooth,
    )

    prox = 0.5 * (u_norm - u_ref).pow(2).sum(dim=(1, 2))
    loss = (prox + proj_lambda * phi).mean()

    aux = {
        "prox": prox.mean().detach(),
        "phi": phi.mean().detach(),
        "goal_pos": aux_terms["goal_pos"].mean().detach(),
        "goal_theta": aux_terms["goal_theta"].mean().detach(),
        "obs": aux_terms["obs"].mean().detach(),
        "ctrl": aux_terms["ctrl"].mean().detach(),
        "smooth": aux_terms["smooth"].mean().detach(),
    }
    return loss, aux


# ============================================================
# Diffusion sampler
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

        k = step + 1
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
# CEM / MPPI baselines
# ============================================================

@torch.no_grad()
def cem_plan(
    start_tensor: torch.Tensor,
    goal_tensor: torch.Tensor,
    sdf_map: torch.Tensor,
    horizon: int,
    dt: float,
    v_max: float,
    w_max: float,
    map_size_m: float,
    robot_radius: float,
    safety_margin_m: float,
    w_goal_pos: float,
    w_goal_theta: float,
    w_obs: float,
    w_ctrl: float,
    w_smooth: float,
    population: int,
    iters: int,
    elite_frac: float,
    momentum: float,
    init_std_norm: float,
    min_std_norm: float,
    device: str,
) -> torch.Tensor:
    elite_k = max(1, int(population * elite_frac))

    mean = torch.zeros(1, horizon, 2, device=device)
    std = torch.full((1, horizon, 2), init_std_norm, device=device)

    best_cost = float("inf")
    best_u_norm = mean.clone()

    for _ in range(iters):
        noise = torch.randn(population, horizon, 2, device=device)
        u_norm = (mean + std * noise).clamp(-1.0, 1.0)

        controls = denormalize_controls(u_norm, v_max=v_max, w_max=w_max)
        costs, _, _ = planning_cost_from_controls(
            controls=controls,
            start=start_tensor,
            goal=goal_tensor,
            sdf_map=sdf_map,
            dt=dt,
            map_size_m=map_size_m,
            robot_radius=robot_radius,
            safety_margin_m=safety_margin_m,
            w_goal_pos=w_goal_pos,
            w_goal_theta=w_goal_theta,
            w_obs=w_obs,
            w_ctrl=w_ctrl,
            w_smooth=w_smooth,
        )

        elite_idx = torch.topk(costs, k=elite_k, largest=False).indices
        elite = u_norm[elite_idx]

        elite_mean = elite.mean(dim=0, keepdim=True)
        elite_std = elite.std(dim=0, unbiased=False, keepdim=True).clamp_min(min_std_norm)

        mean = momentum * mean + (1.0 - momentum) * elite_mean
        std = momentum * std + (1.0 - momentum) * elite_std

        iter_best_idx = int(torch.argmin(costs).item())
        iter_best_cost = float(costs[iter_best_idx].item())
        if iter_best_cost < best_cost:
            best_cost = iter_best_cost
            best_u_norm = u_norm[iter_best_idx:iter_best_idx + 1].clone()

    return denormalize_controls(best_u_norm, v_max=v_max, w_max=w_max)


@torch.no_grad()
def mppi_plan(
    start_tensor: torch.Tensor,
    goal_tensor: torch.Tensor,
    sdf_map: torch.Tensor,
    horizon: int,
    dt: float,
    v_max: float,
    w_max: float,
    map_size_m: float,
    robot_radius: float,
    safety_margin_m: float,
    w_goal_pos: float,
    w_goal_theta: float,
    w_obs: float,
    w_ctrl: float,
    w_smooth: float,
    population: int,
    iters: int,
    temperature: float,
    sigma_v_norm: float,
    sigma_w_norm: float,
    device: str,
) -> torch.Tensor:
    u_nom = torch.zeros(1, horizon, 2, device=device)
    sigma = torch.tensor([sigma_v_norm, sigma_w_norm], device=device).view(1, 1, 2)

    best_cost = float("inf")
    best_u_norm = u_nom.clone()

    for _ in range(iters):
        noise = torch.randn(population, horizon, 2, device=device) * sigma
        u_norm = (u_nom + noise).clamp(-1.0, 1.0)
        u_norm[0] = u_nom[0]

        controls = denormalize_controls(u_norm, v_max=v_max, w_max=w_max)
        costs, _, _ = planning_cost_from_controls(
            controls=controls,
            start=start_tensor,
            goal=goal_tensor,
            sdf_map=sdf_map,
            dt=dt,
            map_size_m=map_size_m,
            robot_radius=robot_radius,
            safety_margin_m=safety_margin_m,
            w_goal_pos=w_goal_pos,
            w_goal_theta=w_goal_theta,
            w_obs=w_obs,
            w_ctrl=w_ctrl,
            w_smooth=w_smooth,
        )

        beta = torch.min(costs)
        weights = torch.exp(-(costs - beta) / max(temperature, 1e-6))
        weights = weights / weights.sum().clamp_min(1e-8)

        delta = torch.sum(weights.view(-1, 1, 1) * (u_norm - u_nom), dim=0, keepdim=True)
        u_nom = (u_nom + delta).clamp(-1.0, 1.0)

        iter_best_idx = int(torch.argmin(costs).item())
        iter_best_cost = float(costs[iter_best_idx].item())
        if iter_best_cost < best_cost:
            best_cost = iter_best_cost
            best_u_norm = u_norm[iter_best_idx:iter_best_idx + 1].clone()

        nom_controls = denormalize_controls(u_nom, v_max=v_max, w_max=w_max)
        nom_costs, _, _ = planning_cost_from_controls(
            controls=nom_controls,
            start=start_tensor,
            goal=goal_tensor,
            sdf_map=sdf_map,
            dt=dt,
            map_size_m=map_size_m,
            robot_radius=robot_radius,
            safety_margin_m=safety_margin_m,
            w_goal_pos=w_goal_pos,
            w_goal_theta=w_goal_theta,
            w_obs=w_obs,
            w_ctrl=w_ctrl,
            w_smooth=w_smooth,
        )
        nom_cost = float(nom_costs[0].item())
        if nom_cost < best_cost:
            best_cost = nom_cost
            best_u_norm = u_nom.clone()

    return denormalize_controls(best_u_norm, v_max=v_max, w_max=w_max)


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
    occ_img = 1.0 - (occupancy_map >= 0.5).astype(np.float32)
    ax.imshow(
        occ_img,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        alpha=1.0,
    )

    for i in range(sampled_states.shape[0]):
        st = sampled_states[i]
        is_best = i == best_idx
        label = f"best (success={bool(metrics['success'][i])})" if is_best else None
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
        f"Kinodynamic planning\n"
        f"scenario={scenario_type} | pos_err={metrics['final_pos_err'][best_idx]:.3f} | "
        f"th_err={metrics['final_theta_err_rad'][best_idx]:.3f} rad | "
        f"collision={bool(metrics['collision'][best_idx])}"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_trajectories(
    out_path: str,
    occupancy_map: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    expert_states: np.ndarray,
    method_records: List[Dict],
    map_size_m: float,
    scenario_type: str,
) -> None:
    if not method_records:
        return

    fig, ax = plt.subplots(figsize=(9, 9))
    extent = [0.0, map_size_m, 0.0, map_size_m]
    method_colors = {
        "diffusion_only": "tab:blue",
        "diffusion_projected": "tab:green",
        "cem": "tab:orange",
        "mppi": "tab:red",
    }
    used_labels = set()

    occ_img = 1.0 - (occupancy_map >= 0.5).astype(np.float32)
    ax.imshow(
        occ_img,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        alpha=1.0,
        zorder=0,
    )

    ax.plot(
        expert_states[:, 0],
        expert_states[:, 1],
        linestyle="--",
        linewidth=2.0,
        color="0.35",
        alpha=0.9,
        zorder=1,
        label="expert",
    )
    used_labels.add("expert")

    for record in method_records:
        method = record["method"]
        color = method_colors.get(method, "tab:purple")

        traj_label = method if method not in used_labels else None
        ax.plot(
            record["best_states"][:, 0],
            record["best_states"][:, 1],
            linewidth=2.3,
            color=color,
            alpha=0.9,
            zorder=2,
            label=traj_label,
        )
        if traj_label is not None:
            used_labels.add(method)

        start_label = "start" if "start" not in used_labels else None
        goal_label = "goal" if "goal" not in used_labels else None
        ax.scatter(start[0], start[1], s=60, marker="o", color="black", alpha=0.8, zorder=3, label=start_label)
        ax.scatter(goal[0], goal[1], s=90, marker="*", color="black", alpha=0.85, zorder=3, label=goal_label)
        if start_label is not None:
            used_labels.add("start")
        if goal_label is not None:
            used_labels.add("goal")

    ax.set_xlim(0.0, map_size_m)
    ax.set_ylim(0.0, map_size_m)
    ax.set_aspect("equal")
    ax.set_title(f"Method overlay | scenario={scenario_type}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Evaluation helpers
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
        (pos_err <= goal_pos_tol + 0.1)
        & (th_err <= math.radians(goal_theta_tol_deg)*2)
        & (~collision)
    )

    return {
        "final_pos_err": pos_err.cpu().numpy(),
        "final_theta_err_rad": th_err.cpu().numpy(),
        "min_clearance": min_clearance.cpu().numpy(),
        "collision": collision.cpu().numpy(),
        "success": success.cpu().numpy(),
    }


def best_index_from_metrics(metrics: Dict[str, np.ndarray]) -> int:
    score = metrics["final_pos_err"] + 0.5 * metrics["final_theta_err_rad"]
    success_mask = metrics["success"].astype(bool)
    if np.any(success_mask):
        return int(np.argmin(np.where(success_mask, score, np.inf)))
    return int(np.argmin(score))


def mean_se(arr: List[float]) -> Tuple[float, float]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(arr.size))


def summarize_case(metrics: Dict[str, np.ndarray], idx: int, runtime_ms: float) -> Dict[str, float]:
    return {
        "success": float(bool(metrics["success"][idx])),
        "collision": float(bool(metrics["collision"][idx])),
        "final_pos_err": float(metrics["final_pos_err"][idx]),
        "final_theta_err_rad": float(metrics["final_theta_err_rad"][idx]),
        "min_clearance": float(metrics["min_clearance"][idx]),
        "runtime_ms": float(runtime_ms),
    }


def aggregate_case_records(records: List[Dict[str, float]]) -> Dict[str, float]:
    success = [r["success"] for r in records]
    collision = [r["collision"] for r in records]
    pos = [r["final_pos_err"] for r in records]
    theta = [r["final_theta_err_rad"] for r in records]
    clear = [r["min_clearance"] for r in records]
    runtime = [r["runtime_ms"] for r in records]

    success_mean, success_se = mean_se(success)
    collision_mean, collision_se = mean_se(collision)
    pos_mean, pos_se = mean_se(pos)
    theta_mean, theta_se = mean_se(theta)
    clear_mean, clear_se = mean_se(clear)
    runtime_mean, runtime_se = mean_se(runtime)

    return {
        "num_cases": len(records),
        "success_rate_pct": 100.0 * success_mean,
        "success_rate_se_pct": 100.0 * success_se,
        "collision_rate_pct": 100.0 * collision_mean,
        "collision_rate_se_pct": 100.0 * collision_se,
        "final_pos_err_mean": pos_mean,
        "final_pos_err_se": pos_se,
        "final_theta_err_rad_mean": theta_mean,
        "final_theta_err_rad_se": theta_se,
        "min_clearance_mean": clear_mean,
        "min_clearance_se": clear_se,
        "runtime_ms_mean": runtime_mean,
        "runtime_ms_se": runtime_se,
    }


# ============================================================
# One-case runner
# ============================================================

def run_method_on_case(
    method: str,
    cfg: InferConfig,
    model: nn.Module,
    schedule: DiffusionSchedule,
    case: Dict,
    ds_cfg: Dict,
) -> Tuple[Dict[str, np.ndarray], torch.Tensor, torch.Tensor, int, float]:
    dt = float(ds_cfg["dt"])
    v_max = float(ds_cfg["v_max"])
    w_max = float(ds_cfg["w_max"])
    map_size_m = float(ds_cfg["map_size_m"])
    robot_radius = float(ds_cfg["robot_radius"])
    goal_pos_tol = float(ds_cfg["goal_pos_tol"])
    goal_theta_tol_deg = float(ds_cfg["goal_theta_tol_deg"])
    horizon = int(ds_cfg["horizon"])

    t0 = time.perf_counter()

    if method in ("diffusion_only", "diffusion_projected"):
        if method == "diffusion_only":
           # print(f"Running diffusion sampling without projection on case {case['case_id']}...")
            map_tensor = case["map_tensor"].repeat(cfg.num_samples, 1, 1, 1)
            pose_tensor = case["pose_tensor"].repeat(cfg.num_samples, 1)
            start_tensor = case["start_tensor"].repeat(cfg.num_samples, 1)
            goal_tensor = case["goal_tensor"].repeat(cfg.num_samples, 1)
        if method == "diffusion_projected":
              # print(f"Running diffusion sampling with projection on case {case['case_id']}...")
            map_tensor = case["map_tensor"].repeat(cfg.num_samples, 1, 1, 1)
            pose_tensor = case["pose_tensor"].repeat(cfg.num_samples, 1)
            start_tensor = case["start_tensor"].repeat(cfg.num_samples, 1)
            goal_tensor = case["goal_tensor"].repeat(cfg.num_samples, 1)

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
            sdf_map=case["sdf_map"],
            dt=dt,
            v_max=v_max,
            w_max=w_max,
            map_size_m=map_size_m,
            robot_radius=robot_radius,
            use_projection=(method == "diffusion_projected"),
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

        controls = denormalize_controls(sampled_controls_norm, v_max=v_max, w_max=w_max)
        states = rollout_unicycle_batch(start_tensor, controls, dt=dt)
        metrics = evaluate_rollouts(
            states=states,
            goal=goal_tensor,
            sdf_map=case["sdf_map"],
            robot_radius=robot_radius,
            goal_pos_tol=goal_pos_tol,
            goal_theta_tol_deg=goal_theta_tol_deg,
            map_size_m=map_size_m,
        )
        best_idx = best_index_from_metrics(metrics)

    elif method == "cem":
        controls = cem_plan(
            start_tensor=case["start_tensor"],
            goal_tensor=case["goal_tensor"],
            sdf_map=case["sdf_map"],
            horizon=horizon,
            dt=dt,
            v_max=v_max,
            w_max=w_max,
            map_size_m=map_size_m,
            robot_radius=robot_radius,
            safety_margin_m=cfg.safety_margin_m,
            w_goal_pos=cfg.w_goal_pos,
            w_goal_theta=cfg.w_goal_theta,
            w_obs=cfg.w_obs,
            w_ctrl=cfg.w_ctrl,
            w_smooth=cfg.w_smooth,
            population=cfg.cem_population,
            iters=cfg.cem_iters,
            elite_frac=cfg.cem_elite_frac,
            momentum=cfg.cem_momentum,
            init_std_norm=cfg.cem_init_std_norm,
            min_std_norm=cfg.cem_min_std_norm,
            device=cfg.device,
        )
        states = rollout_unicycle_batch(case["start_tensor"], controls, dt=dt)
        metrics = evaluate_rollouts(
            states=states,
            goal=case["goal_tensor"],
            sdf_map=case["sdf_map"],
            robot_radius=robot_radius,
            goal_pos_tol=goal_pos_tol,
            goal_theta_tol_deg=goal_theta_tol_deg,
            map_size_m=map_size_m,
        )
        best_idx = 0

    elif method == "mppi":
        controls = mppi_plan(
            start_tensor=case["start_tensor"],
            goal_tensor=case["goal_tensor"],
            sdf_map=case["sdf_map"],
            horizon=horizon,
            dt=dt,
            v_max=v_max,
            w_max=w_max,
            map_size_m=map_size_m,
            robot_radius=robot_radius,
            safety_margin_m=cfg.safety_margin_m,
            w_goal_pos=cfg.w_goal_pos,
            w_goal_theta=cfg.w_goal_theta,
            w_obs=cfg.w_obs,
            w_ctrl=cfg.w_ctrl,
            w_smooth=cfg.w_smooth,
            population=cfg.mppi_population,
            iters=cfg.mppi_iters,
            temperature=cfg.mppi_temperature,
            sigma_v_norm=cfg.mppi_sigma_v_norm,
            sigma_w_norm=cfg.mppi_sigma_w_norm,
            device=cfg.device,
        )
        states = rollout_unicycle_batch(case["start_tensor"], controls, dt=dt)
        metrics = evaluate_rollouts(
            states=states,
            goal=case["goal_tensor"],
            sdf_map=case["sdf_map"],
            robot_radius=robot_radius,
            goal_pos_tol=goal_pos_tol,
            goal_theta_tol_deg=goal_theta_tol_deg,
            map_size_m=map_size_m,
        )
        best_idx = 0

    else:
        raise ValueError(f"Unsupported method: {method}")

    runtime_ms = 1000.0 * (time.perf_counter() - t0)
    return metrics, controls, states, best_idx, runtime_ms


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = InferConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.random_seed)

    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    train_cfg = ckpt["train_cfg"]
    ds_cfg = ckpt["dataset_cfg"]

    if cfg.map_mode != train_cfg["map_mode"]:
        print(
            f"[warning] infer map_mode={cfg.map_mode} differs from "
            f"train map_mode={train_cfg['map_mode']}. Using infer setting."
        )

    map_in_ch = 2 if cfg.map_mode == "sdf_occupancy" else 1

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

    if cfg.max_test_files is not None:
        files = files[:cfg.max_test_files]

    example_case_indices = set()
    if cfg.save_example_plots and cfg.example_plot_count > 0:
        rng = np.random.default_rng(cfg.random_seed)
        num_plot_cases = min(cfg.example_plot_count, len(files))
        example_case_indices = set(rng.choice(len(files), size=num_plot_cases, replace=False).tolist())

    methods = [cfg.planner_mode] if cfg.planner_mode != "all" else [
        "diffusion_only",
        "diffusion_projected",
        "cem",
        "mppi",
    ]

    print(f"checkpoint: {cfg.checkpoint_path}")
    print(f"split: {cfg.split}")
    print(f"num_test_files: {len(files)}")
    print(f"device: {cfg.device}")
    print(f"planner_mode: {cfg.planner_mode}")
    print(f"methods: {methods}")
    if example_case_indices:
        print(f"saving one per-case method overlay plot for {len(example_case_indices)} randomly selected cases")

    per_method_case_records: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}
    per_method_detailed: Dict[str, List[Dict]] = {m: [] for m in methods}

    for case_idx, sample_path in enumerate(files):
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

        pose_cond_np = pose_condition(start_np, goal_np, map_size_m=float(ds_cfg["map_size_m"]))

        case = {
            "map_tensor": torch.from_numpy(map_arr).unsqueeze(0).to(cfg.device),
            "pose_tensor": torch.from_numpy(pose_cond_np).unsqueeze(0).to(cfg.device),
            "start_tensor": torch.from_numpy(start_np).unsqueeze(0).to(cfg.device),
            "goal_tensor": torch.from_numpy(goal_np).unsqueeze(0).to(cfg.device),
            "sdf_map": torch.from_numpy(sdf_map_np).to(cfg.device),
            "occupancy_map_np": occupancy_map_np,
            "sdf_map_np": sdf_map_np,
            "start_np": start_np,
            "goal_np": goal_np,
            "expert_controls_np": expert_controls_np,
            "expert_states_np": expert_states_np,
            "scenario_type": scenario_type,
        }
        case_overlay_records: List[Dict] = []

        print(f"\n[{case_idx + 1}/{len(files)}] {os.path.basename(sample_path)} | scenario={scenario_type}")

        for method in methods:
            metrics, controls, states, best_idx, runtime_ms = run_method_on_case(
                method=method,
                cfg=cfg,
                model=model,
                schedule=schedule,
                case=case,
                ds_cfg=ds_cfg,
            )

            case_summary = summarize_case(metrics, best_idx, runtime_ms)
            per_method_case_records[method].append(case_summary)
            per_method_detailed[method].append({
                "sample_path": sample_path,
                "scenario_type": scenario_type,
                "best_idx": int(best_idx),
                **case_summary,
            })

            print(
                f"  {method:20s} | "
                f"success={bool(case_summary['success'])} | "
                f"collision={bool(case_summary['collision'])} | "
                f"pos_err={case_summary['final_pos_err']:.4f} | "
                f"th_err={case_summary['final_theta_err_rad']:.4f} | "
                f"min_clear={case_summary['min_clearance']:.4f} | "
                f"runtime_ms={case_summary['runtime_ms']:.2f}"
            )

            if cfg.save_example_plots and case_idx in example_case_indices:
                states_np = states.detach().cpu().numpy()
                case_overlay_records.append({
                    "method": method,
                    "best_states": states_np[best_idx].copy(),
                })

        if cfg.save_example_plots and case_idx in example_case_indices and case_overlay_records:
            overlay_path = os.path.join(cfg.out_dir, f"methods_overlay_case{case_idx:04d}.png")
            plot_overlay_trajectories(
                out_path=overlay_path,
                occupancy_map=occupancy_map_np,
                start=start_np,
                goal=goal_np,
                expert_states=expert_states_np,
                method_records=case_overlay_records,
                map_size_m=float(ds_cfg["map_size_m"]),
                scenario_type=scenario_type,
            )
            print(f"  saved overlay plot: {overlay_path}")

    aggregate_results = {
        method: aggregate_case_records(records)
        for method, records in per_method_case_records.items()
    }

    print(f"\n{'=' * 24} aggregate summary {'=' * 24}")
    for method, summary in aggregate_results.items():
        print(
            f"{method:20s} | "
            f"success={summary['success_rate_pct']:.2f} ± {summary['success_rate_se_pct']:.2f}% | "
            f"collision={summary['collision_rate_pct']:.2f} ± {summary['collision_rate_se_pct']:.2f}% | "
            f"pos_err={summary['final_pos_err_mean']:.4f} ± {summary['final_pos_err_se']:.4f} | "
            f"th_err={summary['final_theta_err_rad_mean']:.4f} ± {summary['final_theta_err_rad_se']:.4f} | "
            f"min_clear={summary['min_clearance_mean']:.4f} ± {summary['min_clearance_se']:.4f} | "
            f"runtime_ms={summary['runtime_ms_mean']:.2f} ± {summary['runtime_ms_se']:.2f}"
        )

    summary_payload = {
        "config": asdict(cfg),
        "num_test_files": len(files),
        "aggregate_results": aggregate_results,
        "per_method_case_records": per_method_detailed,
    }

    summary_path = os.path.join(cfg.out_dir, "full_testset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\nSaved summary to: {summary_path}")


# ============================================================
# AR-1 correlated noise sampler (used in temporal-correlation ablation)
# ============================================================

def ar1_initial_noise(B: int, horizon: int, control_dim: int, rho: float, device: str) -> torch.Tensor:
    """
    Generate an (B, horizon, control_dim) noise tensor whose time-axis
    follows an AR(1) process with correlation coefficient rho.

    x_0 ~ N(0, I)
    x_t = rho * x_{t-1} + sqrt(1 - rho^2) * eps_t,   eps_t ~ N(0, I)

    When rho=0 this reduces to plain i.i.d. Gaussian noise.
    """
    x = torch.zeros(B, horizon, control_dim, device=device)
    x[:, 0, :] = torch.randn(B, control_dim, device=device)
    innov_scale = math.sqrt(max(1.0 - rho * rho, 0.0))
    for t in range(1, horizon):
        x[:, t, :] = rho * x[:, t - 1, :] + innov_scale * torch.randn(B, control_dim, device=device)
    return x


def sample_controls_ar1(
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
    rho: float = 0.0,
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
    """
    Identical to sample_controls but starts from AR(1)-correlated noise
    (controlled by rho along the horizon/time dimension) rather than i.i.d.
    Gaussian noise.  project_every=0 disables projection entirely.
    """
    B = map_tensor.shape[0]
    x = ar1_initial_noise(B, horizon, control_dim, rho=rho, device=device)

    for step in reversed(range(schedule.num_steps)):
        with torch.no_grad():
            t = torch.full((B,), step, device=device, dtype=torch.long)
            pred_noise = model(x, t, map_tensor, pose_cond)

            alpha_t       = schedule.alphas[t].view(-1, 1, 1)
            alpha_bar_t   = schedule.alpha_bars[t].view(-1, 1, 1)
            beta_t        = schedule.betas[t].view(-1, 1, 1)
            alpha_bar_prev = schedule.alpha_bars_prev[t].view(-1, 1, 1)

            x0_hat = schedule.predict_x0_from_noise(x, t, pred_noise).clamp(-eta_clip, eta_clip)
            coef1  = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
            coef2  = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            mean   = coef1 * x0_hat + coef2 * x

            if step > 0:
                posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                u_tilde = mean + torch.sqrt(posterior_var.clamp_min(1e-8)) * torch.randn_like(x)
            else:
                u_tilde = mean

        k = step + 1
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
# Ablation 1: Effect of projection lambda on success / collision
# ============================================================

def run_ablation_lambda(
    cfg: InferConfig,
    model: nn.Module,
    schedule: DiffusionSchedule,
    ds_cfg: Dict,
    files: List[str],
    lambda_values: Optional[List[float]] = None,
    num_samples: int = 32,
    out_dir: str = "ablation_outputs",
) -> None:
    """
    For each value of proj_lambda, run diffusion_projected on every test file
    and record success / collision rates.  Produces a log-x-axis line plot.
    """
    if lambda_values is None:
        lambda_values = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

    os.makedirs(out_dir, exist_ok=True)

    horizon     = int(ds_cfg["horizon"])
    dt          = float(ds_cfg["dt"])
    v_max       = float(ds_cfg["v_max"])
    w_max       = float(ds_cfg["w_max"])
    map_size_m  = float(ds_cfg["map_size_m"])
    robot_radius  = float(ds_cfg["robot_radius"])
    goal_pos_tol  = float(ds_cfg["goal_pos_tol"])
    goal_theta_tol_deg = float(ds_cfg["goal_theta_tol_deg"])
    map_in_ch   = 2 if cfg.map_mode == "sdf_occupancy" else 1

    success_rates  = []
    collision_rates = []

    for lam in lambda_values:
        print(f"  lambda={lam:.4g} ...", end=" ", flush=True)
        successes  = []
        collisions = []

        for sample_path in files:
            data = np.load(sample_path, allow_pickle=True)
            map_arr   = build_map_tensor(data, cfg.map_mode)
            sdf_np    = data["sdf"].astype(np.float32)
            start_np  = data["start"].astype(np.float32)
            goal_np   = data["goal"].astype(np.float32)
            pcond_np  = pose_condition(start_np, goal_np, map_size_m)

            map_tensor  = torch.from_numpy(map_arr).unsqueeze(0).repeat(num_samples, 1, 1, 1).to(cfg.device)
            pose_tensor = torch.from_numpy(pcond_np).unsqueeze(0).repeat(num_samples, 1).to(cfg.device)
            start_t     = torch.from_numpy(start_np).unsqueeze(0).repeat(num_samples, 1).to(cfg.device)
            goal_t      = torch.from_numpy(goal_np).unsqueeze(0).repeat(num_samples, 1).to(cfg.device)
            sdf_t       = torch.from_numpy(sdf_np).to(cfg.device)

            u_norm = sample_controls(
                model=model, schedule=schedule,
                map_tensor=map_tensor, pose_cond=pose_tensor,
                horizon=horizon, control_dim=2, device=cfg.device,
                start_tensor=start_t, goal_tensor=goal_t, sdf_map=sdf_t,
                dt=dt, v_max=v_max, w_max=w_max,
                map_size_m=map_size_m, robot_radius=robot_radius,
                use_projection=True,
                project_every=cfg.project_every,
                proj_steps=cfg.proj_steps,
                proj_lr=cfg.proj_lr,
                proj_lambda=lam,
                control_clip_norm=cfg.control_clip_norm,
                safety_margin_m=cfg.safety_margin_m,
                w_goal_pos=cfg.w_goal_pos, w_goal_theta=cfg.w_goal_theta,
                w_obs=cfg.w_obs, w_ctrl=cfg.w_ctrl, w_smooth=cfg.w_smooth,
                eta_clip=cfg.eta_clip,
            )
            controls = denormalize_controls(u_norm, v_max=v_max, w_max=w_max)
            states   = rollout_unicycle_batch(start_t, controls, dt=dt)
            metrics  = evaluate_rollouts(
                states=states, goal=goal_t, sdf_map=sdf_t,
                robot_radius=robot_radius,
                goal_pos_tol=goal_pos_tol,
                goal_theta_tol_deg=goal_theta_tol_deg,
                map_size_m=map_size_m,
            )
            best_idx  = best_index_from_metrics(metrics)
            successes.append(float(metrics["success"][best_idx]))
            collisions.append(float(metrics["collision"][best_idx]))

        sr = 100.0 * float(np.mean(successes))
        cr = 100.0 * float(np.mean(collisions))
        success_rates.append(sr)
        collision_rates.append(cr)
        print(f"success={sr:.1f}%  collision={cr:.1f}%")

    # ── Save raw results ──────────────────────────────────────────────────────
    results = {
        "lambda_values":    lambda_values,
        "success_rates":    success_rates,
        "collision_rates":  collision_rates,
    }
    with open(os.path.join(out_dir, "ablation_lambda.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lambda_values, success_rates,  marker="o", linewidth=2,
            color="tab:green", label="Success Rate")
    ax.plot(lambda_values, collision_rates, marker="s", linewidth=2,
            color="tab:red",   label="Collision Rate")
    ax.set_xscale("log")
    ax.set_xlabel(r"Lambda ($\lambda$)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Ablation: Effect of Lambda on Performance")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "ablation_lambda.png")
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Lambda ablation plot saved to: {plot_path}")


# ============================================================
# Ablation 2: AR-1 temporal correlation × projection frequency
# ============================================================

def run_ablation_ar1_projection(
    cfg: InferConfig,
    model: nn.Module,
    schedule: DiffusionSchedule,
    ds_cfg: Dict,
    files: List[str],
    rho_values: Optional[List[float]] = None,
    project_every_values: Optional[List[int]] = None,
    num_samples: int = 32,
    out_dir: str = "ablation_outputs",
) -> None:
    """
    Grid sweep over AR(1) noise correlation (rho) and projection frequency
    (project_every).  project_every=0 means no projection.

    Produces one line per rho value; x-axis is projection frequency
    (displayed in decreasing order: none → 20 → 10 → 5 → 2 → 1).
    """
    if rho_values is None:
        rho_values = [0.0, 0.8, 0.95]
    if project_every_values is None:
        # 0 encodes "no projection"; others are every-N-steps values
        project_every_values = [0, 20, 10, 5, 2, 1]

    os.makedirs(out_dir, exist_ok=True)

    horizon     = int(ds_cfg["horizon"])
    dt          = float(ds_cfg["dt"])
    v_max       = float(ds_cfg["v_max"])
    w_max       = float(ds_cfg["w_max"])
    map_size_m  = float(ds_cfg["map_size_m"])
    robot_radius  = float(ds_cfg["robot_radius"])
    goal_pos_tol  = float(ds_cfg["goal_pos_tol"])
    goal_theta_tol_deg = float(ds_cfg["goal_theta_tol_deg"])

    # results[rho][proj_every] = success_rate_pct
    all_results: Dict[float, Dict[int, float]] = {}

    for rho in rho_values:
        all_results[rho] = {}
        for pe in project_every_values:
            use_proj = (pe > 0)
            label    = f"rho={rho}  proj_every={'none' if pe == 0 else pe}"
            print(f"  {label} ...", end=" ", flush=True)

            successes = []
            for sample_path in files:
                data = np.load(sample_path, allow_pickle=True)
                map_arr   = build_map_tensor(data, cfg.map_mode)
                sdf_np    = data["sdf"].astype(np.float32)
                start_np  = data["start"].astype(np.float32)
                goal_np   = data["goal"].astype(np.float32)
                pcond_np  = pose_condition(start_np, goal_np, map_size_m)

                map_tensor  = torch.from_numpy(map_arr).unsqueeze(0).repeat(num_samples, 1, 1, 1).to(cfg.device)
                pose_tensor = torch.from_numpy(pcond_np).unsqueeze(0).repeat(num_samples, 1).to(cfg.device)
                start_t     = torch.from_numpy(start_np).unsqueeze(0).repeat(num_samples, 1).to(cfg.device)
                goal_t      = torch.from_numpy(goal_np).unsqueeze(0).repeat(num_samples, 1).to(cfg.device)
                sdf_t       = torch.from_numpy(sdf_np).to(cfg.device)

                u_norm = sample_controls_ar1(
                    model=model, schedule=schedule,
                    map_tensor=map_tensor, pose_cond=pose_tensor,
                    horizon=horizon, control_dim=2, device=cfg.device,
                    start_tensor=start_t, goal_tensor=goal_t, sdf_map=sdf_t,
                    dt=dt, v_max=v_max, w_max=w_max,
                    map_size_m=map_size_m, robot_radius=robot_radius,
                    rho=rho,
                    use_projection=use_proj,
                    project_every=pe,
                    proj_steps=cfg.proj_steps,
                    proj_lr=cfg.proj_lr,
                    proj_lambda=cfg.proj_lambda,
                    control_clip_norm=cfg.control_clip_norm,
                    safety_margin_m=cfg.safety_margin_m,
                    w_goal_pos=cfg.w_goal_pos, w_goal_theta=cfg.w_goal_theta,
                    w_obs=cfg.w_obs, w_ctrl=cfg.w_ctrl, w_smooth=cfg.w_smooth,
                    eta_clip=cfg.eta_clip,
                )
                controls = denormalize_controls(u_norm, v_max=v_max, w_max=w_max)
                states   = rollout_unicycle_batch(start_t, controls, dt=dt)
                metrics  = evaluate_rollouts(
                    states=states, goal=goal_t, sdf_map=sdf_t,
                    robot_radius=robot_radius,
                    goal_pos_tol=goal_pos_tol,
                    goal_theta_tol_deg=goal_theta_tol_deg,
                    map_size_m=map_size_m,
                )
                best_idx = best_index_from_metrics(metrics)
                successes.append(float(metrics["success"][best_idx]))

            sr = 100.0 * float(np.mean(successes))
            all_results[rho][pe] = sr
            print(f"success={sr:.1f}%")

    # ── Save raw results ──────────────────────────────────────────────────────
    serialisable = {str(rho): {str(pe): v for pe, v in d.items()}
                    for rho, d in all_results.items()}
    with open(os.path.join(out_dir, "ablation_ar1_projection.json"), "w", encoding="utf-8") as f:
        json.dump({"rho_values": rho_values,
                   "project_every_values": project_every_values,
                   "results": serialisable}, f, indent=2)

    # ── Build x-axis tick labels (none → large-to-small N values) ────────────
    # Display order: no-projection first, then decreasing frequency (1 = most frequent)
    x_labels = ["none" if pe == 0 else str(pe) for pe in project_every_values]
    x_pos    = list(range(len(project_every_values)))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    markers = ["o", "s", "^", "D", "v"]

    for i, rho in enumerate(rho_values):
        y_vals = [all_results[rho][pe] for pe in project_every_values]
        ax.plot(
            x_pos, y_vals,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"rho={rho}",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Projection frequency (every N diffusion steps)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Ablation: Temporal correlation vs projection frequency")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "ablation_ar1_projection.png")
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  AR-1 × projection-frequency ablation plot saved to: {plot_path}")


# ============================================================
# Ablation entry-point
# ============================================================

def run_ablations() -> None:
    """
    Standalone runner for both ablation studies.
    Loads the same checkpoint as main() and evaluates on the full test split.
    Edit the lists below to change the sweep ranges.
    """
    cfg = InferConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)

    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    train_cfg   = ckpt["train_cfg"]
    ds_cfg      = ckpt["dataset_cfg"]

    if cfg.map_mode != train_cfg["map_mode"]:
        print(
            f"[warning] infer map_mode={cfg.map_mode} differs from "
            f"train map_mode={train_cfg['map_mode']}. Using infer setting."
        )

    map_in_ch = 2 if cfg.map_mode == "sdf_occupancy" else 1

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

    files = sorted(glob.glob(os.path.join(cfg.data_root, "test", "*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No .npz files found in {os.path.join(cfg.data_root, 'test')}"
        )

    ablation_out = "ablation_outputs"
    num_samples  = cfg.num_samples   # reuse InferConfig setting

    # ── Ablation 1: lambda sweep ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Ablation 1: Effect of proj_lambda on success / collision rate")
    print("=" * 60)
    run_ablation_lambda(
        cfg=cfg, model=model, schedule=schedule,
        ds_cfg=ds_cfg, files=files,
        lambda_values=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        num_samples=num_samples,
        out_dir=ablation_out,
    )

    # ── Ablation 2: AR-1 rho × projection frequency ───────────────────────────
    print("\n" + "=" * 60)
    print("Ablation 2: AR-1 temporal correlation × projection frequency")
    print("=" * 60)
    run_ablation_ar1_projection(
        cfg=cfg, model=model, schedule=schedule,
        ds_cfg=ds_cfg, files=files,
        rho_values=[0.0, 0.8, 0.95],
        project_every_values=[0, 20, 10, 5, 2, 1],
        num_samples=num_samples,
        out_dir=ablation_out,
    )

    print("\nAll ablations complete.")


if __name__ == "__main__":
        run_ablations()
       # main()