#Function to run the tests for ablations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Configurations
@dataclass
class AblationConfig:
    data_root: str = "diffdrive_dataset"
    split: str = "test"
    map_mode: str = "sdf"  

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 0

    num_samples: int = 64
    eta_clip: float = 1.5

    project_every: int = 10
    proj_steps: int = 6
    proj_lr: float = 0.06
    proj_lambda: float = 0.1
    project_last_n_steps: int = 0
    final_proj_steps: int = 0
    final_proj_lr: float = 0.00
    final_proj_lambda: float = 0.0
    final_proj_goal_pos_scale: float =0
    final_proj_goal_theta_scale: float =0
    control_clip_norm: float = 1.0
    proj_grad_clip_norm: float = 1.0
    safety_margin_m: float = 0.05

    w_goal_pos: float = 25.0
    w_goal_theta: float = 2.0
    w_obs: float = 400.0
    w_ctrl: float = 1e-3
    w_smooth: float = 0.02

    use_relaxed_success_rule: bool = True
    success_pos_extra_m: float = 0.1
    success_theta_scale: float = 2.0

    out_dir: str = "ablation_outputs"

    checkpoint_path_rho00: str = "checkpoints/diffdrive_kinodynamic_best.pt"
    checkpoint_path_rho05: str = "checkpoints_rho0.5/diffdrive_kinodynamic_best.pt"
    checkpoint_path_rho07: str = "checkpoints_rho0.7/diffdrive_kinodynamic_best.pt"
    checkpoint_path_rho08: str = "checkpoints_rho0.8/diffdrive_kinodynamic_best.pt"
    checkpoint_path_rho095: str = "checkpoints_rho0.95/diffdrive_kinodynamic_best.pt"

    lambda_ablation_checkpoint_path: str = "checkpoints/diffdrive_kinodynamic_best.pt"

    lambda_values: Tuple[float, ...] = (0.001, 0.01,0.1)
    rho_values: Tuple[float, ...] = (0.0, 0.5, 0.7, 0.8, 0.95)
    project_every_values: Tuple[int, ...] = (0, 20, 10, 5, 2, 1)


#Utility functions
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

def checkpoint_for_rho(cfg: AblationConfig, rho: float) -> str:
    if abs(rho - 0.0) < 1e-9:
        return cfg.checkpoint_path_rho00
    if abs(rho - 0.5) < 1e-9:
        return cfg.checkpoint_path_rho05
    if abs(rho - 0.7) < 1e-9:
        return cfg.checkpoint_path_rho07
    if abs(rho - 0.8) < 1e-9:
        return cfg.checkpoint_path_rho08
    if abs(rho - 0.95) < 1e-9:
        return cfg.checkpoint_path_rho095
    raise ValueError(f"No checkpoint configured for rho={rho}")


#Shared planner cost
def planning_cost_from_controls(
    controls: torch.Tensor,        
    start: torch.Tensor,            
    goal: torch.Tensor,             
    sdf_map: torch.Tensor,        
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

# Diffusion schedule
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
        return (xt - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t.clamp_min(1e-8))

# Embeddings and encoders
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


#One dimensional UNet
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


# Projection objective
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

def run_projection_steps(
    u_init: torch.Tensor,
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
    proj_steps: int,
    proj_lr: float,
    proj_lambda: float,
    control_clip_norm: float,
    proj_grad_clip_norm: float,
    w_goal_pos: float,
    w_goal_theta: float,
    w_obs: float,
    w_ctrl: float,
    w_smooth: float,
) -> torch.Tensor:
    if proj_steps <= 0:
        return u_init.detach()
    u_proj = u_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([u_proj], lr=proj_lr)

    for proj_step in range(proj_steps):
        opt.zero_grad(set_to_none=True)
        loss, _ = projection_objective(
            u_norm=u_proj,
            u_ref=u_ref,
            start=start,
            goal=goal,
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
        torch.nn.utils.clip_grad_norm_([u_proj], max_norm=proj_grad_clip_norm)
        opt.step()

        decay = 0.5 * (1.0 + math.cos(math.pi * float(proj_step + 1) / float(proj_steps)))
        for group in opt.param_groups:
            group["lr"] = max(1e-4, proj_lr * decay)

        with torch.no_grad():
            u_proj.clamp_(-control_clip_norm, control_clip_norm)
    return u_proj.detach()


# Correlated noise sampling
def sample_ar1_noise(
    batch_size: int,
    horizon: int,
    dim: int,
    rho: float,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"AR(1) rho must satisfy -1 < rho < 1, got {rho}")
    if abs(rho) < 1e-12:
        return torch.randn(batch_size, horizon, dim, device=device, dtype=dtype)
    eps = torch.empty(batch_size, horizon, dim, device=device, dtype=dtype)
    eps[:, 0] = torch.randn(batch_size, dim, device=device, dtype=dtype)
    innov_scale = math.sqrt(max(1.0 - rho * rho, 1e-12))

    for t in range(1, horizon):
        eta_t = torch.randn(batch_size, dim, device=device, dtype=dtype)
        eps[:, t] = rho * eps[:, t - 1] + innov_scale * eta_t
    return eps

def sample_temporal_noise_like(x: torch.Tensor, rho: float) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected x shape (B,T,D), got {tuple(x.shape)}")
    B, T, D = x.shape
    return sample_ar1_noise(
        batch_size=B,
        horizon=T,
        dim=D,
        rho=rho,
        device=x.device,
        dtype=x.dtype,
    )

# Diffusion sampler
def sample_controls_correlated(
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
    rho: float,
    use_projection: bool,
    project_every: int,
    proj_steps: int,
    proj_lr: float,
    proj_lambda: float,
    project_last_n_steps: int,
    final_proj_steps: int,
    final_proj_lr: float,
    final_proj_lambda: float,
    final_proj_goal_pos_scale: float,
    final_proj_goal_theta_scale: float,
    control_clip_norm: float,
    proj_grad_clip_norm: float,
    safety_margin_m: float,
    w_goal_pos: float,
    w_goal_theta: float,
    w_obs: float,
    w_ctrl: float,
    w_smooth: float,
    eta_clip: float,
) -> torch.Tensor:
    B = map_tensor.shape[0]
    x = sample_temporal_noise_like(
        torch.empty(B, horizon, control_dim, device=device),
        rho=rho,
    )
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
                z = sample_temporal_noise_like(x, rho=rho)
                u_tilde = mean + torch.sqrt(posterior_var.clamp_min(1e-8)) * z
            else:
                u_tilde = mean
        k = step + 1
        do_project = use_projection and (
            ((project_every > 0) and (k % project_every == 0))
            or ((project_last_n_steps > 0) and (k <= project_last_n_steps))
        )
        if do_project:
            x = run_projection_steps(
                u_init=u_tilde,
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
                proj_steps=proj_steps,
                proj_lr=proj_lr,
                proj_lambda=proj_lambda,
                control_clip_norm=control_clip_norm,
                proj_grad_clip_norm=proj_grad_clip_norm,
                w_goal_pos=w_goal_pos,
                w_goal_theta=w_goal_theta,
                w_obs=w_obs,
                w_ctrl=w_ctrl,
                w_smooth=w_smooth,
            )
        else:
            x = u_tilde.detach()
    if use_projection and final_proj_steps > 0:
        x = run_projection_steps(
            u_init=x,
            u_ref=x.detach(),
            start=start_tensor,
            goal=goal_tensor,
            sdf_map=sdf_map,
            dt=dt,
            v_max=v_max,
            w_max=w_max,
            map_size_m=map_size_m,
            robot_radius=robot_radius,
            safety_margin_m=safety_margin_m,
            proj_steps=final_proj_steps,
            proj_lr=final_proj_lr,
            proj_lambda=final_proj_lambda,
            control_clip_norm=control_clip_norm,
            proj_grad_clip_norm=proj_grad_clip_norm,
            w_goal_pos=w_goal_pos * final_proj_goal_pos_scale,
            w_goal_theta=w_goal_theta * final_proj_goal_theta_scale,
            w_obs=w_obs,
            w_ctrl=w_ctrl,
            w_smooth=w_smooth,
        )
    return x

#Evaluation
@torch.no_grad()
def evaluate_rollouts(
    states: torch.Tensor,
    goal: torch.Tensor,
    sdf_map: torch.Tensor,
    robot_radius: float,
    goal_pos_tol: float,
    goal_theta_tol_deg: float,
    map_size_m: float,
    use_relaxed_success_rule: bool,
    success_pos_extra_m: float,
    success_theta_scale: float,
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
    pos_thresh = goal_pos_tol + (success_pos_extra_m if use_relaxed_success_rule else 0.0)
    theta_thresh = math.radians(goal_theta_tol_deg) * (success_theta_scale if use_relaxed_success_rule else 1.0)
    success = (
        (pos_err <= pos_thresh)
        & (th_err <= theta_thresh)
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

# Model and checkpoint loading
def load_model_and_schedule_from_ckpt(
    checkpoint_path: str,
    cfg: AblationConfig,
) -> Tuple[nn.Module, DiffusionSchedule, Dict, Dict]:
    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    train_cfg = ckpt["train_cfg"]
    ds_cfg = ckpt["dataset_cfg"]

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

    return model, schedule, train_cfg, ds_cfg


#Lambda ablations
def run_ablation_lambda(
    cfg: AblationConfig,
    model: nn.Module,
    schedule: DiffusionSchedule,
    train_cfg: Dict,
    ds_cfg: Dict,
    files: List[str],
) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    horizon = int(ds_cfg["horizon"])
    dt = float(ds_cfg["dt"])
    v_max = float(ds_cfg["v_max"])
    w_max = float(ds_cfg["w_max"])
    map_size_m = float(ds_cfg["map_size_m"])
    robot_radius = float(ds_cfg["robot_radius"])
    goal_pos_tol = float(ds_cfg["goal_pos_tol"])
    goal_theta_tol_deg = float(ds_cfg["goal_theta_tol_deg"])

    rho = float(train_cfg.get("noise_rho_train", 0.0))
    success_rates = []
    collision_rates = []

    print("\n" + "=" * 64)
    print("Ablation 1: lambda sweep")
    print(f"Checkpoint: {cfg.lambda_ablation_checkpoint_path}")
    print(f"Matched rho used for sampling: {rho}")
    print("=" * 64)
    for lam in cfg.lambda_values:
        print(f"  lambda={lam:.4g} ...", end=" ", flush=True)
        successes = []
        collisions = []
        for sample_path in files:
            data = np.load(sample_path, allow_pickle=True)

            map_arr = build_map_tensor(data, cfg.map_mode)
            sdf_np = data["sdf"].astype(np.float32)
            start_np = data["start"].astype(np.float32)
            goal_np = data["goal"].astype(np.float32)
            pcond_np = pose_condition(start_np, goal_np, map_size_m)
            map_tensor = torch.from_numpy(map_arr).unsqueeze(0).repeat(cfg.num_samples, 1, 1, 1).to(cfg.device)
            pose_tensor = torch.from_numpy(pcond_np).unsqueeze(0).repeat(cfg.num_samples, 1).to(cfg.device)
            start_t = torch.from_numpy(start_np).unsqueeze(0).repeat(cfg.num_samples, 1).to(cfg.device)
            goal_t = torch.from_numpy(goal_np).unsqueeze(0).repeat(cfg.num_samples, 1).to(cfg.device)
            sdf_t = torch.from_numpy(sdf_np).to(cfg.device)

            u_norm = sample_controls_correlated(
                model=model,
                schedule=schedule,
                map_tensor=map_tensor,
                pose_cond=pose_tensor,
                horizon=horizon,
                control_dim=2,
                device=cfg.device,
                start_tensor=start_t,
                goal_tensor=goal_t,
                sdf_map=sdf_t,
                dt=dt,
                v_max=v_max,
                w_max=w_max,
                map_size_m=map_size_m,
                robot_radius=robot_radius,
                rho=rho,
                use_projection=True,
                project_every=cfg.project_every,
                proj_steps=cfg.proj_steps,
                proj_lr=cfg.proj_lr,
                proj_lambda=lam,
                project_last_n_steps=cfg.project_last_n_steps,
                final_proj_steps=cfg.final_proj_steps,
                final_proj_lr=cfg.final_proj_lr,
                final_proj_lambda=cfg.final_proj_lambda,
                final_proj_goal_pos_scale=cfg.final_proj_goal_pos_scale,
                final_proj_goal_theta_scale=cfg.final_proj_goal_theta_scale,
                control_clip_norm=cfg.control_clip_norm,
                proj_grad_clip_norm=cfg.proj_grad_clip_norm,
                safety_margin_m=cfg.safety_margin_m,
                w_goal_pos=cfg.w_goal_pos,
                w_goal_theta=cfg.w_goal_theta,
                w_obs=cfg.w_obs,
                w_ctrl=cfg.w_ctrl,
                w_smooth=cfg.w_smooth,
                eta_clip=cfg.eta_clip,
            )
            controls = denormalize_controls(u_norm, v_max=v_max, w_max=w_max)
            states = rollout_unicycle_batch(start_t, controls, dt=dt)
            metrics = evaluate_rollouts(
                states=states,
                goal=goal_t,
                sdf_map=sdf_t,
                robot_radius=robot_radius,
                goal_pos_tol=goal_pos_tol,
                goal_theta_tol_deg=goal_theta_tol_deg,
                map_size_m=map_size_m,
                use_relaxed_success_rule=cfg.use_relaxed_success_rule,
                success_pos_extra_m=cfg.success_pos_extra_m,
                success_theta_scale=cfg.success_theta_scale,
            )
            best_idx = best_index_from_metrics(metrics)
            successes.append(float(metrics["success"][best_idx]))
            collisions.append(float(metrics["collision"][best_idx]))
        sr = 100.0 * float(np.mean(successes))
        cr = 100.0 * float(np.mean(collisions))
        success_rates.append(sr)
        collision_rates.append(cr)
        print(f"success={sr:.1f}%  collision={cr:.1f}%")
    payload = {
        "config": asdict(cfg),
        "checkpoint": cfg.lambda_ablation_checkpoint_path,
        "matched_rho": rho,
        "lambda_values": list(cfg.lambda_values),
        "success_rates": success_rates,
        "collision_rates": collision_rates,
    }
    with open(os.path.join(cfg.out_dir, "ablation_lambda.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cfg.lambda_values, success_rates, marker="o", linewidth=2, color="tab:green", label="Success rate")
    ax.plot(cfg.lambda_values, collision_rates, marker="s", linewidth=2, color="tab:red", label="Collision rate")
    ax.set_xscale("log")
    ax.set_xlabel(r"Projection strength $\lambda$")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Ablation: projection strength")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(cfg.out_dir, "ablation_lambda.png")
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")
    print(f"Saved: {os.path.join(cfg.out_dir, 'ablation_lambda.json')}")


#Rho ablations
def run_ablation_ar1_projection(
    cfg: AblationConfig,
    files: List[str],
) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    print("\n" + "=" * 64)
    print("Ablation 2: matched rho x projection frequency")
    print("=" * 64)
    all_results: Dict[float, Dict[int, float]] = {}
    ds_cfg_ref = None
    for rho in cfg.rho_values:
        ckpt_path = checkpoint_for_rho(cfg, rho)
        print(f"\nLoading checkpoint for rho={rho}: {ckpt_path}")
        model, schedule, train_cfg, ds_cfg = load_model_and_schedule_from_ckpt(ckpt_path, cfg)
        ds_cfg_ref = ds_cfg

        rho_train = float(train_cfg.get("noise_rho_train", -999.0))
        print(f"  checkpoint train rho={rho_train}")
        horizon = int(ds_cfg["horizon"])
        dt = float(ds_cfg["dt"])
        v_max = float(ds_cfg["v_max"])
        w_max = float(ds_cfg["w_max"])
        map_size_m = float(ds_cfg["map_size_m"])
        robot_radius = float(ds_cfg["robot_radius"])
        goal_pos_tol = float(ds_cfg["goal_pos_tol"])
        goal_theta_tol_deg = float(ds_cfg["goal_theta_tol_deg"])
        all_results[rho] = {}

        for pe in cfg.project_every_values:
            use_proj = (pe > 0)
            label = f"rho={rho}  proj_every={'none' if pe == 0 else pe}"
            print(f"  {label} ...", end=" ", flush=True)

            successes = []
            for sample_path in files:
                data = np.load(sample_path, allow_pickle=True)
                map_arr = build_map_tensor(data, cfg.map_mode)
                sdf_np = data["sdf"].astype(np.float32)
                start_np = data["start"].astype(np.float32)
                goal_np = data["goal"].astype(np.float32)
                pcond_np = pose_condition(start_np, goal_np, map_size_m)

                map_tensor = torch.from_numpy(map_arr).unsqueeze(0).repeat(cfg.num_samples, 1, 1, 1).to(cfg.device)
                pose_tensor = torch.from_numpy(pcond_np).unsqueeze(0).repeat(cfg.num_samples, 1).to(cfg.device)
                start_t = torch.from_numpy(start_np).unsqueeze(0).repeat(cfg.num_samples, 1).to(cfg.device)
                goal_t = torch.from_numpy(goal_np).unsqueeze(0).repeat(cfg.num_samples, 1).to(cfg.device)
                sdf_t = torch.from_numpy(sdf_np).to(cfg.device)
                u_norm = sample_controls_correlated(
                    model=model,
                    schedule=schedule,
                    map_tensor=map_tensor,
                    pose_cond=pose_tensor,
                    horizon=horizon,
                    control_dim=2,
                    device=cfg.device,
                    start_tensor=start_t,
                    goal_tensor=goal_t,
                    sdf_map=sdf_t,
                    dt=dt,
                    v_max=v_max,
                    w_max=w_max,
                    map_size_m=map_size_m,
                    robot_radius=robot_radius,
                    rho=rho,
                    use_projection=use_proj,
                    project_every=pe,
                    proj_steps=cfg.proj_steps,
                    proj_lr=cfg.proj_lr,
                    proj_lambda=cfg.proj_lambda,
                    project_last_n_steps=cfg.project_last_n_steps if use_proj else 0,
                    final_proj_steps=cfg.final_proj_steps if use_proj else 0,
                    final_proj_lr=cfg.final_proj_lr,
                    final_proj_lambda=cfg.final_proj_lambda,
                    final_proj_goal_pos_scale=cfg.final_proj_goal_pos_scale,
                    final_proj_goal_theta_scale=cfg.final_proj_goal_theta_scale,
                    control_clip_norm=cfg.control_clip_norm,
                    proj_grad_clip_norm=cfg.proj_grad_clip_norm,
                    safety_margin_m=cfg.safety_margin_m,
                    w_goal_pos=cfg.w_goal_pos,
                    w_goal_theta=cfg.w_goal_theta,
                    w_obs=cfg.w_obs,
                    w_ctrl=cfg.w_ctrl,
                    w_smooth=cfg.w_smooth,
                    eta_clip=cfg.eta_clip,
                )
                controls = denormalize_controls(u_norm, v_max=v_max, w_max=w_max)
                states = rollout_unicycle_batch(start_t, controls, dt=dt)
                metrics = evaluate_rollouts(
                    states=states,
                    goal=goal_t,
                    sdf_map=sdf_t,
                    robot_radius=robot_radius,
                    goal_pos_tol=goal_pos_tol,
                    goal_theta_tol_deg=goal_theta_tol_deg,
                    map_size_m=map_size_m,
                    use_relaxed_success_rule=cfg.use_relaxed_success_rule,
                    success_pos_extra_m=cfg.success_pos_extra_m,
                    success_theta_scale=cfg.success_theta_scale,
                )
                best_idx = best_index_from_metrics(metrics)
                successes.append(float(metrics["success"][best_idx]))
            sr = 100.0 * float(np.mean(successes))
            all_results[rho][pe] = sr
            print(f"success={sr:.1f}%")
    serialisable = {
        str(rho): {str(pe): v for pe, v in inner.items()}
        for rho, inner in all_results.items()
    }
    payload = {
        "config": asdict(cfg),
        "rho_values": list(cfg.rho_values),
        "project_every_values": list(cfg.project_every_values),
        "results": serialisable,
    }
    with open(os.path.join(cfg.out_dir, "ablation_ar1_projection.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    x_labels = ["none" if pe == 0 else str(pe) for pe in cfg.project_every_values]
    x_pos = list(range(len(cfg.project_every_values)))
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    markers = ["o", "s", "^", "D", "v"]
    for i, rho in enumerate(cfg.rho_values):
        y_vals = [all_results[rho][pe] for pe in cfg.project_every_values]
        ax.plot(
            x_pos,
            y_vals,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"rho={rho}",
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Projection frequency (every N diffusion steps)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Ablation: temporal correlation vs projection frequency")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(cfg.out_dir, "ablation_ar1_projection.png")
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {plot_path}")
    print(f"Saved: {os.path.join(cfg.out_dir, 'ablation_ar1_projection.json')}")


#Main driver
def main() -> None:
    cfg = AblationConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.random_seed)
    files = sorted(glob.glob(os.path.join(cfg.data_root, cfg.split, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {os.path.join(cfg.data_root, cfg.split)}")
    print(f"device: {cfg.device}")
    print(f"num test files: {len(files)}")
    print(f"output dir: {cfg.out_dir}")
    print(f"use_relaxed_success_rule: {cfg.use_relaxed_success_rule}")

    model, schedule, train_cfg, ds_cfg = load_model_and_schedule_from_ckpt(
        cfg.lambda_ablation_checkpoint_path,
        cfg,
    )
    run_ablation_lambda(
        cfg=cfg,
        model=model,
        schedule=schedule,
        train_cfg=train_cfg,
        ds_cfg=ds_cfg,
        files=files,
    )

    run_ablation_ar1_projection(
        cfg=cfg,
        files=files,
    )
    print("\nAll ablations complete.")

if __name__ == "__main__":
    main()