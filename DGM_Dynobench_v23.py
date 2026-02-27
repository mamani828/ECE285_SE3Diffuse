#!/usr/bin/env python3
"""
DGM_Dynobench.py

A single-file Python script version of the provided Jupyter notebook
(DGM_Dynobench.ipynb). It:

1) Downloads/parses Dynobench model + environment YAML (Integrator2_2d_v0 / park)
2) Generates an offline dataset of collision-free trajectories
3) Trains a conditional diffusion model to generate control sequences
4) Evaluates diffusion sampling with and without a proximal projection
5) Runs simple baseline planners (random shooting, MPPI) for sanity checks
6) Produces plots (training curve, example rollouts, batch metrics, lambda ablation)

Notes:
- Default settings can be compute-heavy. Use CLI flags to reduce runtime (e.g. --epochs 5).
- Requires: torch, numpy, matplotlib, tqdm, requests
"""

from __future__ import annotations

import argparse
import os
import re
import time
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import requests

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm


# ----------------------------
# Globals (kept to match notebook structure)
# ----------------------------
env_cfg = None
model_cfg = None
norm = None
POS_CENTER = None
POS_SCALE = None

box_obstacles = None
start_n = None
goal_n = None

max_vel_n = None
max_acc_n = None
robot_radius_n = None


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # best-effort determinism (can reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------
# Dynamics + rollout
# ----------------------------
class BaseDynamics(nn.Module):
    """Abstract base class for dynamics models."""

    def __init__(self, state_dim: int, control_dim: int, dt: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dt = dt

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PointMass2D(BaseDynamics):
    """2D point mass with velocity: state=[x,y,vx,vy], control=[ax,ay]."""

    def __init__(self, dt: float = 0.1, max_velocity: float = 2.0, damping: float = 0.9):
        super().__init__(state_dim=4, control_dim=2, dt=dt)
        self.max_velocity = float(max_velocity)
        self.damping = float(damping)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pos = x[..., :2]
        vel = x[..., 2:]

        vel_new = self.damping * vel + self.dt * u
        vel_new = torch.clamp(vel_new, -self.max_velocity, self.max_velocity)
        pos_new = pos + self.dt * vel_new

        return torch.cat([pos_new, vel_new], dim=-1)


class SingleIntegrator2D(BaseDynamics):
    """Simple velocity control: state=[x,y], control=[vx,vy]."""

    def __init__(self, dt: float = 0.1):
        super().__init__(state_dim=2, control_dim=2, dt=dt)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + self.dt * u


class DifferentiableRollout(nn.Module):
    """Roll out controls through dynamics (enables backprop)."""

    def __init__(self, dynamics: BaseDynamics):
        super().__init__()
        self.dynamics = dynamics

    def forward(self, x0: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        batch_size, T = controls.shape[:2]
        state_dim = self.dynamics.state_dim

        states = torch.zeros(batch_size, T + 1, state_dim, device=x0.device, dtype=x0.dtype)
        states[:, 0] = x0

        x = x0
        for t in range(T):
            x = self.dynamics(x, controls[:, t])
            states[:, t + 1] = x

        return states


# ----------------------------
# SDFs + constraints + projection
# ----------------------------
class SignedDistanceField2D(nn.Module):
    def __init__(self, obstacles, grid_size: int = 100, bounds: Tuple[float, float] = (-1, 1), precompute_grid: bool = True):
        super().__init__()
        self.obstacles = obstacles
        self.bounds = bounds

        if obstacles:
            centers = torch.tensor([c for (c, r) in obstacles], dtype=torch.float32)
            radii = torch.tensor([r for (c, r) in obstacles], dtype=torch.float32)
        else:
            centers = torch.zeros(0, 2)
            radii = torch.zeros(0)

        self.register_buffer("centers", centers)
        self.register_buffer("radii", radii)
        self.sdf_grid = self._compute_sdf_grid(grid_size) if precompute_grid and obstacles else None

    def _compute_sdf_grid(self, grid_size: int) -> torch.Tensor:
        x = torch.linspace(self.bounds[0], self.bounds[1], grid_size)
        y = torch.linspace(self.bounds[0], self.bounds[1], grid_size)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        return self(torch.stack([xx, yy], dim=-1))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if self.centers.numel() == 0:
            return torch.full(positions.shape[:-1], 1e6, device=positions.device, dtype=positions.dtype)
        centers = self.centers.to(positions.device, dtype=positions.dtype)
        radii = self.radii.to(positions.device, dtype=positions.dtype)
        diff = positions[..., None, :] - centers
        dist = torch.sqrt((diff * diff).sum(-1) + 1e-9)
        return torch.min(dist - radii, dim=-1).values


class BoxSDF2D(nn.Module):
    """Signed distance field for axis-aligned rectangular obstacles."""

    def __init__(self, boxes: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        super().__init__()
        self.boxes = boxes

        centers = torch.tensor([[cx, cy] for (cx, cy), _ in boxes], dtype=torch.float32)
        halfs = torch.tensor([[hx, hy] for _, (hx, hy) in boxes], dtype=torch.float32)
        self.register_buffer("centers", centers)
        self.register_buffer("halfs", halfs)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        centers = self.centers.to(positions.device, dtype=positions.dtype)
        halfs = self.halfs.to(positions.device, dtype=positions.dtype)
        d = torch.abs(positions[..., None, :] - centers) - halfs  # [..., N, 2]
        d_out = torch.sqrt((d.clamp(min=0) ** 2).sum(-1) + 1e-9)  # [..., N]
        d_in = d.max(dim=-1).values.clamp(max=0)                  # [..., N]
        return torch.min(d_out + d_in, dim=-1).values             # [...]


class MazeSDF2D(nn.Module):
    """Union of circular and rectangular obstacles. min(sdf_circles, sdf_boxes)."""

    def __init__(self, circle_obstacles=None, box_obstacles=None):
        super().__init__()
        self.has_circles = bool(circle_obstacles)
        self.has_boxes = bool(box_obstacles)
        if self.has_circles:
            self.circle_sdf = SignedDistanceField2D(circle_obstacles, precompute_grid=False)
        if self.has_boxes:
            self.box_sdf = BoxSDF2D(box_obstacles)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        sdfs = []
        if self.has_circles:
            sdfs.append(self.circle_sdf(positions))
        if self.has_boxes:
            sdfs.append(self.box_sdf(positions))
        if not sdfs:
            return torch.full(positions.shape[:-1], 1e6, device=positions.device, dtype=positions.dtype)
        return torch.stack(sdfs, dim=-1).min(dim=-1).values


class ConstraintFunction(nn.Module):
    def __init__(
        self,
        dynamics: BaseDynamics,
        sdf: nn.Module,
        delta: float = 0.05,
        goal_weight: float = 10.0,
        obstacle_weight: float = 100.0,
        control_weight: float = 0.1,
        smooth_weight: float = 0.0,
    ):
        super().__init__()
        self.dynamics = dynamics
        self.sdf = sdf
        self.delta = float(delta)
        self.goal_weight = float(goal_weight)
        self.obstacle_weight = float(obstacle_weight)
        self.control_weight = float(control_weight)
        self.smooth_weight = float(smooth_weight)

    def forward(self, controls: torch.Tensor, x0: torch.Tensor, x_goal: torch.Tensor, return_components: bool = False):
        B, T = controls.shape[:2]
        states = torch.zeros(B, T + 1, self.dynamics.state_dim, device=controls.device, dtype=x0.dtype)
        states[:, 0] = x0

        x = x0
        for t in range(T):
            x = self.dynamics(x, controls[:, t])
            states[:, t + 1] = x

        positions = states[..., :2]
        goal_cost = ((positions[:, -1] - x_goal[..., :2]) ** 2).sum(-1)
        sdf_vals = self.sdf(positions)
        obs_cost = F.relu(self.delta - sdf_vals).pow(2).sum(-1)
        ctrl_cost = (controls ** 2).sum((1, 2))

        smth_cost = torch.zeros_like(goal_cost)
        if self.smooth_weight > 0 and T >= 2:
            smth_cost = ((controls[:, 1:] - controls[:, :-1]) ** 2).sum((1, 2))

        total = (
            self.goal_weight * goal_cost
            + self.obstacle_weight * obs_cost
            + self.control_weight * ctrl_cost
            + self.smooth_weight * smth_cost
        )

        if return_components:
            return total, {
                "goal": goal_cost.mean().item(),
                "obstacle": obs_cost.mean().item(),
                "control": ctrl_cost.mean().item(),
                "smooth": smth_cost.mean().item(),
                "total": total.mean().item(),
            }
        return total


class ProximalProjection(nn.Module):
    def __init__(
        self,
        constraint_fn: ConstraintFunction,
        lambda_penalty: float = 0.1,
        num_steps: int = 10,
        lr: float = 0.01,
        u_clip: Optional[float] = None,
    ):
        super().__init__()
        self.constraint_fn = constraint_fn
        self.lambda_penalty = float(lambda_penalty)
        self.num_steps = int(num_steps)
        self.lr = float(lr)
        self.u_clip = u_clip

    def project(self, u_tilde: torch.Tensor, x0: torch.Tensor, x_goal: torch.Tensor, verbose: bool = False):
        u = u_tilde.clone().detach().requires_grad_(True)
        opt = Adam([u], lr=self.lr)
        losses = []
        last_comp = None
        for _ in range(self.num_steps):
            opt.zero_grad(set_to_none=True)
            prox = ((u - u_tilde) ** 2).mean()
            c, comp = self.constraint_fn(u, x0, x_goal, return_components=True)
            last_comp = comp
            loss = prox + self.lambda_penalty * c.mean()
            loss.backward()
            opt.step()
            if self.u_clip is not None:
                with torch.no_grad():
                    u.clamp_(-float(self.u_clip), float(self.u_clip))
            losses.append(float(loss.item()))
        return u.detach(), {"losses": losses, "final_constraint": last_comp}


class FeasibleDiffusionSampler(nn.Module):
    def __init__(self, diffusion_model: "ControlDiffusion", proximal_projection: ProximalProjection, project_every: int = 1):
        super().__init__()
        self.diffusion = diffusion_model
        self.projection = proximal_projection
        self.project_every = int(project_every)

    @torch.no_grad()
    def sample(self, batch_size: int, T: int, x0: torch.Tensor, x_goal: torch.Tensor, method: str = "ddim", eta: float = 0.0, verbose: bool = False):
        device = next(self.diffusion.parameters()).device
        C = self.diffusion.denoiser.control_dim
        condition = create_condition_encoding(x0, x_goal, state_dim=x0.shape[-1])
        u = torch.randn(batch_size, T, C, device=device)
        for i in reversed(range(self.diffusion.T_diffusion)):
            k = torch.full((batch_size,), i, device=device, dtype=torch.long)
            u = self.diffusion.p_sample(u, k, condition) if method.lower() == "ddpm" else self.diffusion.p_sample_ddim(u, k, condition, eta=eta)
            if self.project_every and (i % self.project_every == 0):
                with torch.enable_grad():
                    u, _ = self.projection.project(u, x0, x_goal)
        return u, {}


# ----------------------------
# Diffusion model (Temporal U-Net + DDPM/DDIM)
# ----------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class TemporalUNet(nn.Module):
    """Temporal U-Net denoiser for control sequences with (x0,x_goal) conditioning."""

    def __init__(self, control_dim: int, hidden_dim: int = 128, time_emb_dim: int = 64, cond_dim: int = 32):
        super().__init__()
        self.control_dim = int(control_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Encoder
        self.enc1 = TemporalConvBlock(control_dim, hidden_dim // 2)
        self.enc2 = TemporalConvBlock(hidden_dim // 2, hidden_dim)
        self.enc3 = TemporalConvBlock(hidden_dim, hidden_dim * 2)

        # Bottleneck
        self.bottleneck = TemporalConvBlock(hidden_dim * 2, hidden_dim * 2)

        # Decoder
        self.dec3 = TemporalConvBlock(hidden_dim * 4, hidden_dim)
        self.dec2 = TemporalConvBlock(hidden_dim * 2, hidden_dim // 2)
        self.dec1 = TemporalConvBlock(hidden_dim, hidden_dim // 2)

        self.out = nn.Conv1d(hidden_dim // 2, control_dim, 1)

        # Time modulation (additive)
        self.time_mod = nn.ModuleList([
            nn.Linear(time_emb_dim, hidden_dim // 2),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.Linear(time_emb_dim, hidden_dim * 2),
            nn.Linear(time_emb_dim, hidden_dim * 2),
        ])

        # Condition modulation (additive)
        self.cond_mod = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Linear(hidden_dim, hidden_dim * 2),
        ])

    def forward(self, u_noisy: torch.Tensor, k: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = u_noisy.transpose(1, 2)  # [B, C, T]
        t_emb = self.time_mlp(k.float())     # [B, time_emb_dim]
        c_emb = self.cond_encoder(condition) # [B, hidden_dim]

        e1 = self.enc1(x) + self.time_mod[0](t_emb)[:, :, None] + self.cond_mod[0](c_emb)[:, :, None]
        e2_in = F.avg_pool1d(e1, 2)
        e2 = self.enc2(e2_in) + self.time_mod[1](t_emb)[:, :, None] + self.cond_mod[1](c_emb)[:, :, None]

        e3_in = F.avg_pool1d(e2, 2)
        e3 = self.enc3(e3_in) + self.time_mod[2](t_emb)[:, :, None] + self.cond_mod[2](c_emb)[:, :, None]

        b_in = F.avg_pool1d(e3, 2)
        b = self.bottleneck(b_in) + self.time_mod[3](t_emb)[:, :, None] + self.cond_mod[3](c_emb)[:, :, None]

        d3 = F.interpolate(b, size=e3.shape[-1], mode="linear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[-1], mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out(d1)  # [B, C, T]
        return out.transpose(1, 2)  # [B, T, C]


class ControlDiffusion(nn.Module):
    """DDPM/DDIM diffusion model for control sequences."""

    def __init__(self, denoiser: TemporalUNet, T_diffusion: int = 100, beta_schedule: str = "linear"):
        super().__init__()
        self.denoiser = denoiser
        self.T_diffusion = int(T_diffusion)

        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.T_diffusion)
        else:
            s = 0.008
            steps = self.T_diffusion + 1
            x = torch.linspace(0, self.T_diffusion, steps)
            alphas_cumprod = torch.cos(((x / self.T_diffusion) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0, 0.999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, u0: torch.Tensor, k: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(u0)
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[k][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[k][:, None, None]
        u_k = sqrt_alpha_cumprod * u0 + sqrt_one_minus * noise
        return u_k, noise

    def p_sample(self, u_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        eps = self.denoiser(u_t, t, condition)
        abar_t = self.alphas_cumprod[t][:, None, None]
        alpha_t = self.alphas[t][:, None, None]
        beta_t = self.betas[t][:, None, None]

        u0_hat = (u_t - torch.sqrt(1.0 - abar_t) * eps) / torch.sqrt(abar_t)

        abar_prev = torch.where(
            (t > 0)[:, None, None],
            self.alphas_cumprod[(t - 1).clamp(min=0)][:, None, None],
            torch.ones_like(abar_t),
        )

        beta_tilde = beta_t * (1.0 - abar_prev) / (1.0 - abar_t)
        sigma = torch.sqrt(beta_tilde)

        coef1 = torch.sqrt(abar_prev) * beta_t / (1.0 - abar_t)
        coef2 = torch.sqrt(alpha_t) * (1.0 - abar_prev) / (1.0 - abar_t)
        mu = coef1 * u0_hat + coef2 * u_t

        noise = torch.randn_like(u_t)
        nonzero = (t > 0)[:, None, None].float()
        return mu + nonzero * sigma * noise

    def p_sample_ddim(self, u_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        eps = self.denoiser(u_t, t, condition)
        abar_t = self.alphas_cumprod[t][:, None, None]
        alpha_t = self.alphas[t][:, None, None]

        u0_hat = (u_t - torch.sqrt(1.0 - abar_t) * eps) / torch.sqrt(abar_t)

        abar_prev = torch.where(
            (t > 0)[:, None, None],
            self.alphas_cumprod[(t - 1).clamp(min=0)][:, None, None],
            torch.ones_like(abar_t),
        )

        sigma = eta * torch.sqrt((1.0 - abar_prev) / (1.0 - abar_t) * (1.0 - alpha_t))
        eps_coeff = torch.sqrt(torch.clamp(1.0 - abar_prev - sigma * sigma, min=0.0))

        noise = torch.randn_like(u_t)
        nonzero = (t > 0)[:, None, None].float()
        u_prev = torch.sqrt(abar_prev) * u0_hat + eps_coeff * eps + nonzero * sigma * noise
        return u_prev

    @torch.no_grad()
    def sample(self, shape: Tuple[int, int, int], condition: torch.Tensor, method: str = "ddim", eta: float = 0.0) -> torch.Tensor:
        device = self.betas.device
        u = torch.randn(shape, device=device)
        for i in reversed(range(self.T_diffusion)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            u = self.p_sample(u, t, condition) if method.lower() == "ddpm" else self.p_sample_ddim(u, t, condition, eta=eta)
        return u


# ----------------------------
# Dynobench fetch/parse + offline dataset generation
# ----------------------------
DYNOBENCH_RAW_BASE = "https://raw.githubusercontent.com/quimortiz/dynobench/main/"
MODEL_URL = DYNOBENCH_RAW_BASE + "models/integrator2_2d_v0.yaml"
ENV_URL = DYNOBENCH_RAW_BASE + "envs/integrator2_2d_v0/park.yaml"


def download_text(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        with open(dest, "r", encoding="utf-8") as f:
            return f.read()
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    txt = r.text
    with open(dest, "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


def _parse_float_list(inner: str) -> np.ndarray:
    return np.array([float(x.strip()) for x in inner.split(",")], dtype=np.float32)


def _extract_array(text: str, key: str) -> np.ndarray:
    m = re.search(rf"{re.escape(key)}\s*:\s*\[([^\]]+)\]", text)
    if m is None:
        raise ValueError(f"Could not parse array for key='{key}'")
    return _parse_float_list(m.group(1))


def _extract_float(text: str, key: str, default: Optional[float] = None) -> float:
    m = re.search(rf"{re.escape(key)}\s*:\s*([0-9eE+\-.]+)", text)
    if m is None:
        if default is None:
            raise ValueError(f"Could not parse float for key='{key}'")
        return float(default)
    return float(m.group(1))


def parse_dynobench_model(text: str) -> Dict[str, float]:
    # Prefer YAML parsing (more robust), fall back to regex parsing if PyYAML isn't available.
    if yaml is not None:
        data = yaml.safe_load(text)

        def dig(keys, default=None):
            cur = data
            for k in keys:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    return default
            return cur

        max_vel = dig(["max_vel"])
        max_acc = dig(["max_acc"])
        radius = dig(["radius"], 0.0)

        # Common alternative nesting
        if max_vel is None:
            max_vel = dig(["limits", "max_vel"])
        if max_acc is None:
            max_acc = dig(["limits", "max_acc"])
        if radius in (None, "None"):
            radius = dig(["robot", "radius"], 0.0)

        if max_vel is None or max_acc is None:
            raise ValueError("Could not find max_vel/max_acc in model YAML.")
        return {"max_vel": float(max_vel), "max_acc": float(max_acc), "radius": float(radius or 0.0)}

    # Fallback: regex extraction
    max_vel = _extract_float(text, "max_vel")
    max_acc = _extract_float(text, "max_acc")
    radius = _extract_float(text, "radius", default=0.0)
    return dict(max_vel=max_vel, max_acc=max_acc, radius=radius)



def parse_dynobench_env(text: str) -> Dict[str, np.ndarray]:
    """
    Parse a Dynobench environment YAML.

    Dynobench env files are not fully standardized across versions; common layouts include:
      - start/goal at the top level
      - start/goal under `problem` / `task`
      - start/goal inside the first entry of `robots: [...]` (e.g. park.yaml)

    This parser tries several known key paths before failing.
    """
    # Prefer YAML parsing; fall back to regex parsing if PyYAML isn't available.
    if yaml is not None:
        data = yaml.safe_load(text)

        def as_arr(v):
            return np.asarray(v, dtype=np.float32)

        def dig(d, path):
            cur = d
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    return None
                cur = cur[k]
            return cur

        def first_in_robots(d):
            if not isinstance(d, dict):
                return None
            robots = d.get("robots")
            if isinstance(robots, list) and robots:
                for r in robots:
                    if isinstance(r, dict):
                        return r
            return None

        # bounds may be at top-level or under `environment` / `env` / `bounds`
        env_min = data.get("min") if isinstance(data, dict) else None
        env_max = data.get("max") if isinstance(data, dict) else None

        for container_key in ["environment", "env", "bounds"]:
            if (env_min is None or env_max is None) and isinstance(data, dict) and isinstance(data.get(container_key), dict):
                d = data[container_key]
                env_min = env_min if env_min is not None else d.get("min")
                env_max = env_max if env_max is not None else d.get("max")

        if env_min is None or env_max is None:
            raise ValueError("Could not find env min/max in environment YAML.")
        env_min = as_arr(env_min)
        env_max = as_arr(env_max)

        # start/goal may live in several places
        start = None
        goal = None

        if isinstance(data, dict):
            # top-level
            start = data.get("start") or data.get("start_state") or data.get("x0")
            goal = data.get("goal") or data.get("goal_state") or data.get("xf")

        # common alternative nesting: problem/task
        for container_key in ["problem", "task"]:
            if (start is None or goal is None) and isinstance(data, dict) and isinstance(data.get(container_key), dict):
                d = data[container_key]
                start = start if start is not None else (d.get("start") or d.get("start_state") or d.get("x0"))
                goal = goal if goal is not None else (d.get("goal") or d.get("goal_state") or d.get("xf"))

        # dynobench often stores per-robot start/goal under `robots: - ...`
        if (start is None or goal is None) and isinstance(data, dict):
            r0 = first_in_robots(data)
            if isinstance(r0, dict):
                start = start if start is not None else (r0.get("start") or r0.get("start_state") or r0.get("x0"))
                goal = goal if goal is not None else (r0.get("goal") or r0.get("goal_state") or r0.get("xf"))

        if start is None or goal is None:
            # Provide a more actionable error message (include top-level keys if possible)
            keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
            raise ValueError(f"Could not find start/goal in environment YAML. Top-level keys: {keys}")

        start = as_arr(start)
        goal = as_arr(goal)

        # Obstacles
        obstacles = []
        obs_list = None
        for k in ["obstacles", "objects"]:
            if isinstance(data, dict) and isinstance(data.get(k), list):
                obs_list = data[k]
                break
        if obs_list is None:
            for container_key in ["environment", "env"]:
                if isinstance(data, dict) and isinstance(data.get(container_key), dict):
                    d = data[container_key]
                    for k in ["obstacles", "objects"]:
                        if isinstance(d.get(k), list):
                            obs_list = d[k]
                            break

        if obs_list:
            for o in obs_list:
                if not isinstance(o, dict):
                    continue
                # Dynobench uses `type: box`; other types exist but are ignored here
                if o.get("type") != "box":
                    continue
                c = as_arr(o.get("center", [0, 0]))
                size = as_arr(o.get("size", [0, 0]))
                half = 0.5 * size
                obstacles.append((c, half))

        return dict(env_min=env_min, env_max=env_max, obstacles=obstacles, start=start, goal=goal)

    # Fallback: regex extraction (works for inline single-line YAML too)
    env_min = _extract_array(text, "min")
    env_max = _extract_array(text, "max")
    start = _extract_array(text, "start")
    goal = _extract_array(text, "goal")

    obstacles = []
    for m in re.finditer(r"-\s*type:\s*box\s*center:\s*\[([^\]]+)\]\s*size:\s*\[([^\]]+)\]", text):
        c = _parse_float_list(m.group(1))
        size = _parse_float_list(m.group(2))
        half = 0.5 * size
        obstacles.append((c, half))
    return dict(env_min=env_min, env_max=env_max, obstacles=obstacles, start=start, goal=goal)




def make_normalizer(env_min, env_max):
    env_min = np.asarray(env_min, dtype=np.float32)
    env_max = np.asarray(env_max, dtype=np.float32)
    center = 0.5 * (env_min + env_max)
    scale = 0.5 * float(np.max(env_max - env_min))

    def npos(p):  # [...,2]
        return (p - center) / scale

    def dpos(pn):
        return pn * scale + center

    def nvel(v):
        return v / scale

    def dvel(vn):
        return vn * scale

    return dict(center=center, scale=scale, npos=npos, dpos=dpos, nvel=nvel, dvel=dvel)


def setup_dynobench(cache_dir: str) -> None:
    """Download and parse Dynobench configs; populate module-level globals."""
    global env_cfg, model_cfg, norm, POS_CENTER, POS_SCALE
    global box_obstacles, start_n, goal_n, max_vel_n, max_acc_n, robot_radius_n

    model_cache = os.path.join(cache_dir, "dynobench_models_integrator2_2d_v0.yaml")
    env_cache = os.path.join(cache_dir, "dynobench_env_integrator2_2d_v0_park.yaml")

    model_txt = download_text(MODEL_URL, model_cache)
    env_txt = download_text(ENV_URL, env_cache)

    model_cfg = parse_dynobench_model(model_txt)
    env_cfg = parse_dynobench_env(env_txt)

    norm = make_normalizer(env_cfg["env_min"], env_cfg["env_max"])
    POS_CENTER = norm["center"]
    POS_SCALE = norm["scale"]

    box_obstacles = []
    for c, half in env_cfg["obstacles"]:
        c_n = norm["npos"](c)
        half_n = half / POS_SCALE
        box_obstacles.append(((float(c_n[0]), float(c_n[1])), (float(half_n[0]), float(half_n[1]))))

    start_raw = env_cfg["start"].astype(np.float32)
    goal_raw = env_cfg["goal"].astype(np.float32)
    start_n = np.concatenate([norm["npos"](start_raw[:2]), norm["nvel"](start_raw[2:])]).astype(np.float32)
    goal_n = np.concatenate([norm["npos"](goal_raw[:2]), norm["nvel"](goal_raw[2:])]).astype(np.float32)

    max_vel_n = float(model_cfg["max_vel"] / POS_SCALE)
    max_acc_n = float(model_cfg["max_acc"] / POS_SCALE)
    robot_radius_n = float(model_cfg["radius"] / POS_SCALE)

    print("✓ Dynobench files loaded")
    print("  Model:", model_cfg)
    print("  Bounds:", env_cfg["env_min"], env_cfg["env_max"])
    print("  Obstacles:", len(env_cfg["obstacles"]))
    print(f"  Normalisation: center={POS_CENTER}, scale={POS_SCALE:.3f}")
    print(f"  max_vel_n={max_vel_n:.3f}, max_acc_n={max_acc_n:.3f}, radius_n={robot_radius_n:.3f}")


def sample_polyline(points: List[np.ndarray], N: int) -> np.ndarray:
    """Sample N points along a polyline with arclength parametrisation."""
    pts = np.asarray(points, dtype=np.float32)  # [K,2]
    seg = pts[1:] - pts[:-1]
    seglen = np.linalg.norm(seg, axis=1) + 1e-9
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = s[-1]
    ts = np.linspace(0, total, N, dtype=np.float32)
    out = np.zeros((N, 2), dtype=np.float32)
    j = 0
    for i, t in enumerate(ts):
        while j < len(seglen) - 1 and t > s[j + 1]:
            j += 1
        alpha = (t - s[j]) / (seglen[j] + 1e-9)
        out[i] = pts[j] + alpha * seg[j]
    return out


def plan_via_waypoint(
    s0: np.ndarray,
    sg: np.ndarray,
    horizon: int,
    dt: float,
    sdf: nn.Module,
    max_vel: float,
    max_acc: float,
    margin: float = 0.03,
    kp: float = 18.0,
    kd: float = 6.0,
    max_tries: int = 50,
):
    """Heuristic planner for double-integrator: pick waypoint above obstacles, PD-track it."""
    assert env_cfg is not None and norm is not None and box_obstacles is not None and robot_radius_n is not None, "Call setup_dynobench() first."

    s0 = np.asarray(s0, dtype=np.float32)
    sg = np.asarray(sg, dtype=np.float32)

    env_min_n = norm["npos"](env_cfg["env_min"])
    env_max_n = norm["npos"](env_cfg["env_max"])

    obs_tops = [cy + hy for (cx, cy), (hx, hy) in box_obstacles]
    top = max(obs_tops) if obs_tops else -1.0

    for _ in range(max_tries):
        y_wp = min(
            float(env_max_n[1] - margin),
            float(max(top + (robot_radius_n + margin), s0[1], sg[1]) + np.random.uniform(0.10, 0.60)),
        )
        x_wp = float(np.random.uniform(min(s0[0], sg[0]), max(s0[0], sg[0])))
        wp = np.array([x_wp, y_wp], dtype=np.float32)

        ref = sample_polyline([s0[:2], wp, sg[:2]], horizon + 1)  # [T+1,2]

        x = s0.copy()
        states = [x.copy()]
        actions = []
        ok = True

        for _t in range(horizon):
            p = x[:2]
            v = x[2:]
            # one-step lookahead
            idx = len(actions)
            p_ref_next = ref[idx + 1]
            v_ref = (ref[idx + 1] - ref[idx]) / dt

            u = kp * (p_ref_next - p) + kd * (v_ref - v)
            u = np.clip(u, -max_acc, max_acc).astype(np.float32)

            v_new = np.clip(v + dt * u, -max_vel, max_vel)
            p_new = p + dt * v_new
            x = np.concatenate([p_new, v_new]).astype(np.float32)

            if sdf(torch.tensor(p_new[None], dtype=torch.float32)).item() < (robot_radius_n + margin):
                ok = False
                break

            actions.append(u)
            states.append(x.copy())

        if not ok:
            continue

        goal_err = np.linalg.norm(states[-1][:2] - sg[:2])
        if goal_err < 0.12:  # normalised units
            return np.stack(states, 0), np.stack(actions, 0)

    return None, None


class DynobenchIntegrator2DDataset(Dataset):
    """Generate collision-free (state, action) windows for integrator2_2d."""

    def __init__(self, horizon: int = 50, num_trajectories: int = 2000, seed: int = 0):
        super().__init__()
        assert box_obstacles is not None, "Call setup_dynobench() first."
        self.horizon = int(horizon)
        self.dt = None  # set later
        self.box_obstacles = box_obstacles
        self.sdf = MazeSDF2D(circle_obstacles=[], box_obstacles=self.box_obstacles)
        self.max_vel_n = float(max_vel_n)
        self.max_acc_n = float(max_acc_n)
        self.windows = []
        self._rng = np.random.RandomState(seed)
        self.num_trajectories = int(num_trajectories)

    def build(self, dt: float):
        assert env_cfg is not None and norm is not None and robot_radius_n is not None, "Call setup_dynobench() first."
        self.dt = float(dt)
        rng = self._rng

        env_min_n = norm["npos"](env_cfg["env_min"])
        env_max_n = norm["npos"](env_cfg["env_max"])

        def sample_state() -> np.ndarray:
            for _ in range(2000):
                p = rng.uniform(env_min_n, env_max_n).astype(np.float32)
                if self.sdf(torch.tensor(p[None], dtype=torch.float32)).item() > (robot_radius_n + 0.05):
                    return np.array([p[0], p[1], 0.0, 0.0], dtype=np.float32)
            return start_n.copy()

        kept = 0
        attempts = 0

        while kept < self.num_trajectories and attempts < self.num_trajectories * 30:
            attempts += 1
            s0 = sample_state()
            sg = sample_state()
            if np.linalg.norm(s0[:2] - sg[:2]) < 0.35:
                continue

            states, actions = plan_via_waypoint(
                s0, sg, horizon=self.horizon, dt=self.dt, sdf=self.sdf, max_vel=self.max_vel_n, max_acc=self.max_acc_n
            )
            if states is None:
                continue

            self.windows.append(
                {
                    "states": states.astype(np.float32),
                    "controls": actions.astype(np.float32),
                    "x0": states[0].astype(np.float32),
                    "x_goal": sg.astype(np.float32),
                }
            )
            kept += 1

        if kept == 0:
            raise RuntimeError("Failed to generate any trajectories. Try increasing max_tries or relaxing collision margins.")

        print(f"✓ Generated {kept} trajectories (attempts={attempts})")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        return {k: torch.from_numpy(w[k]) for k in w}


def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def create_condition_encoding(x0: torch.Tensor, x_goal: torch.Tensor, state_dim: int = 4) -> torch.Tensor:
    """Concatenate x0 and goal (padded/clipped to state_dim) into a condition vector."""
    B = x0.shape[0]
    gp = torch.zeros(B, state_dim, device=x0.device, dtype=x0.dtype)
    gg = x_goal[..., :state_dim]
    gp[:, : gg.shape[-1]] = gg
    return torch.cat([x0[..., :state_dim], gp], dim=-1)


# ----------------------------
# Evaluation helpers + baselines
# ----------------------------
@torch.no_grad()
def evaluate_controls(controls, x0, x_goal, rollout, sdf, goal_tol: float = 0.12):
    states = rollout(x0, controls)
    positions = states[..., :2]
    goal_err = torch.norm(positions[:, -1] - x_goal[..., :2], dim=-1)
    sdf_vals = sdf(positions)
    collision = (sdf_vals.min(dim=1).values < float(robot_radius_n))
    success = (goal_err < goal_tol) & (~collision)
    return {
        "success_rate": success.float().mean().item(),
        "collision_rate": collision.float().mean().item(),
        "avg_goal_error": goal_err.mean().item(),
        "success": success.cpu().numpy(),
        "collision": collision.cpu().numpy(),
        "goal_error": goal_err.cpu().numpy(),
    }


@torch.no_grad()
def diffusion_best_of_k(diffusion_model, constraint_fn, x0, x_goal, T, C, K: int = 32, method: str = "ddim", eta: float = 0.0, u_clip=None):
    device = next(diffusion_model.parameters()).device
    B = x0.shape[0]
    state_dim = x0.shape[-1]
    cond = create_condition_encoding(x0, x_goal, state_dim=state_dim)
    cond_rep = cond.repeat_interleave(K, dim=0)

    u = diffusion_model.sample((B * K, T, C), cond_rep, method=method, eta=eta)
    if u_clip is not None:
        u = u.clamp(-float(u_clip), float(u_clip))

    x0_rep = x0.repeat_interleave(K, dim=0)
    xgoal_rep = x_goal.repeat_interleave(K, dim=0)
    costs = constraint_fn(u, x0_rep, xgoal_rep).view(B, K)
    best_i = torch.argmin(costs, dim=1)

    u = u.view(B, K, T, C)
    u_best = u[torch.arange(B, device=device), best_i]
    best_cost = costs[torch.arange(B, device=device), best_i]
    info = {
        "best_cost_mean": best_cost.mean().item(),
        "best_cost_min": best_cost.min().item(),
        "best_cost_max": best_cost.max().item(),
    }
    return u_best, info


@torch.no_grad()
def random_shooting(constraint_fn, x0, x_goal, T, C, num_samples: int = 512, u_clip=None, dt=None):
    device = x0.device
    B = x0.shape[0]

    u = torch.randn(B, num_samples, T, C, device=device, dtype=x0.dtype)
    if u_clip is not None:
        u = u.clamp(-float(u_clip), float(u_clip))

    u_flat = u.view(B * num_samples, T, C)
    x0_rep = x0.repeat_interleave(num_samples, dim=0)
    xgoal_rep = x_goal.repeat_interleave(num_samples, dim=0)

    costs = constraint_fn(u_flat, x0_rep, xgoal_rep).view(B, num_samples)
    best_i = torch.argmin(costs, dim=1)
    u_best = u[torch.arange(B, device=device), best_i]
    best_cost = costs[torch.arange(B, device=device), best_i]
    info = {
        "best_cost_mean": best_cost.mean().item(),
        "best_cost_min": best_cost.min().item(),
        "best_cost_max": best_cost.max().item(),
    }
    return u_best, info


@torch.no_grad()
def mppi(constraint_fn, x0, x_goal, T, C, dt=None, num_samples: int = 256, num_iters: int = 6, lam: float = 1.0, sigma: float = 0.8, u_clip=None):
    device = x0.device
    B = x0.shape[0]
    u_nom = torch.zeros(B, T, C, device=device, dtype=x0.dtype)

    for _ in range(num_iters):
        noise = torch.randn(B, num_samples, T, C, device=device, dtype=x0.dtype)
        u_cand = u_nom[:, None, :, :] + sigma * noise
        if u_clip is not None:
            u_cand = u_cand.clamp(-float(u_clip), float(u_clip))

        u_flat = u_cand.view(B * num_samples, T, C)
        x0_rep = x0.repeat_interleave(num_samples, dim=0)
        xgoal_rep = x_goal.repeat_interleave(num_samples, dim=0)
        costs = constraint_fn(u_flat, x0_rep, xgoal_rep).view(B, num_samples)

        Jmin = costs.min(dim=1, keepdim=True).values
        w = torch.exp(-(costs - Jmin) / max(lam, 1e-6))
        w = w / (w.sum(dim=1, keepdim=True) + 1e-9)
        u_nom = (w[:, :, None, None] * u_cand).sum(dim=1)

    info = {"final_cost_mean": constraint_fn(u_nom, x0, x_goal).mean().item()}
    return u_nom, info


def draw_env(ax, boxes, env_min=None, env_max=None):
    for (cx, cy), (hx, hy) in boxes:
        ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, color="dimgray", alpha=0.7, zorder=1))
    if env_min is not None and env_max is not None:
        ax.set_xlim(env_min[0], env_max[0])
        ax.set_ylim(env_min[1], env_max[1])
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.2)


def compute_stats(results: List[Dict], name: str) -> Dict:
    success = np.array([r["success"] for r in results], dtype=np.float64)
    collision = np.array([r["collision"] for r in results], dtype=np.float64)
    goal_error = np.array([r["goal_error"] for r in results], dtype=np.float64)

    has_time = ("time_s" in results[0])
    if has_time:
        times = np.array([r["time_s"] for r in results], dtype=np.float64)

    success_rate = success.mean() * 100.0
    collision_rate = collision.mean() * 100.0
    avg_goal_error = goal_error.mean()

    n = len(results)
    se_success = np.sqrt(max(success.mean() * (1 - success.mean()), 1e-9) / n) * 100.0
    se_collision = np.sqrt(max(collision.mean() * (1 - collision.mean()), 1e-9) / n) * 100.0
    se_goal = goal_error.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0

    out = {
        "name": name,
        "n": n,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "avg_goal_error": avg_goal_error,
        "se_success": se_success,
        "se_collision": se_collision,
        "se_goal": se_goal,
    }

    if has_time:
        out.update(
            {
                "avg_time_s": times.mean(),
                "se_time_s": (times.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                "median_time_s": np.median(times),
                "p90_time_s": np.percentile(times, 90),
            }
        )
    return out


# ----------------------------
# Main experiment
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Dynobench diffusion + projection experiment (single-file).")
    p.add_argument("--cache-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynobench_cache"),
                   help="Directory for caching downloaded Dynobench YAMLs.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num-trajectories", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--T-diffusion", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--no-plots", action="store_true", help="Disable matplotlib plots (still computes metrics).")
    p.add_argument("--num-test", type=int, default=200, help="Number of random start/goal pairs for evaluation.")
    p.add_argument(
        "--eval-k",
        type=int,
        default=1,
        help="Best-of-K sampling for random-pairs eval. K=1 uses a single diffusion sample; K>1 uses diffusion_best_of_k.",
    )
    p.add_argument(
        "--eval-reachable",
        action="store_true",
        help="If set, only evaluate start/goal pairs that a simple waypoint heuristic can solve within the horizon (reduces unreachable cases that force 0%% success).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device(prefer_cuda=(not args.cpu))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Dynobench setup (downloads + normalisation)
    setup_dynobench(args.cache_dir)

    # Config (very close to notebook defaults)
    config = {
        "dynamics_type": "integrator2_2d_v0",
        "dt": 0.10,
        "horizon": int(args.horizon),
        "state_dim": 4,
        "control_dim": 2,
        "hidden_dim": int(args.hidden_dim),
        "cond_dim": 8,
        "T_diffusion": int(args.T_diffusion),
        "num_trajectories": int(args.num_trajectories),
        "batch_size": int(args.batch_size),
        "num_epochs": int(args.epochs),
        "learning_rate": float(args.lr),
        "u_clip": float(max_acc_n),
    }
    print("\nConfig:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Dataset
    print("\nPreparing Dynobench dataset …")
    train_dataset = DynobenchIntegrator2DDataset(horizon=config["horizon"], num_trajectories=config["num_trajectories"], seed=args.seed)
    train_dataset.build(dt=config["dt"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=0)
    print(f"✓ {len(train_dataset):,} trajectories, {len(train_loader)} batches/epoch")

    # Model
    print("\nCreating model …")
    denoiser = TemporalUNet(control_dim=config["control_dim"], hidden_dim=config["hidden_dim"], cond_dim=config["cond_dim"]).to(device)
    diffusion_model = ControlDiffusion(denoiser, T_diffusion=config["T_diffusion"]).to(device)
    optimizer = Adam(diffusion_model.parameters(), lr=config["learning_rate"])
    print(f"✓ {sum(p.numel() for p in diffusion_model.parameters()):,} parameters")

    # Training
    print("\nTraining …\n")
    diffusion_model.train()
    loss_history: List[float] = []

    for epoch in range(config["num_epochs"]):
        ep_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in pbar:
            controls = batch["controls"].to(device)
            x0 = batch["x0"].to(device)
            x_goal = batch["x_goal"].to(device)
            B = controls.shape[0]

            condition = create_condition_encoding(x0, x_goal, state_dim=config["state_dim"])
            k = torch.randint(0, config["T_diffusion"], (B,), device=device)
            u_noisy, noise_true = diffusion_model.q_sample(controls, k)
            noise_pred = denoiser(u_noisy, k, condition)

            loss = F.mse_loss(noise_pred, noise_true)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
            optimizer.step()

            ep_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg = ep_loss / max(len(train_loader), 1)
        loss_history.append(avg)
        print(f"Epoch {epoch+1:3d}  loss: {avg:.4f}")

    print("\n✓ Training complete!")
    if loss_history:
        print(f"Final loss: {loss_history[-1]:.4f}")

    if not args.no_plots:
        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training Loss", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Evaluation stack
    dynamics = PointMass2D(dt=config["dt"], max_velocity=max_vel_n, damping=1.0)
    rollout = DifferentiableRollout(dynamics)
    sdf = MazeSDF2D(circle_obstacles=[], box_obstacles=train_dataset.box_obstacles)

    constraint_fn = ConstraintFunction(
        dynamics,
        sdf,
        delta=float(robot_radius_n) + 0.03,
        goal_weight=20.0,
        obstacle_weight=200.0,
        control_weight=0.05,
        smooth_weight=0.01,
    )
    projection = ProximalProjection(
        constraint_fn,
        lambda_penalty=0.5,
        num_steps=6,
        lr=0.06,
        u_clip=config["u_clip"],
    )
    sampler = FeasibleDiffusionSampler(diffusion_model, projection, project_every=1)
    print("\n✓ Evaluation stack ready (Dynobench Integrator2_2d)")

    # Baseline comparison on one batch
    print("\nBaseline comparison on one training batch:")
    batch = next(iter(train_loader))
    x0 = batch["x0"].to(device)
    x_goal = batch["x_goal"].to(device)
    B, T, C = x0.shape[0], config["horizon"], config["control_dim"]

    u_diff, _ = diffusion_best_of_k(diffusion_model, constraint_fn, x0, x_goal, T, C, K=32, u_clip=config["u_clip"])
    u_proj, _ = sampler.sample(B, T, x0, x_goal, method="ddim", eta=0.0)
    u_shoot, _ = random_shooting(constraint_fn, x0, x_goal, T, C, u_clip=config["u_clip"])
    u_mppi, _ = mppi(constraint_fn, x0, x_goal, T, C, u_clip=config["u_clip"])

    for name, u in [
        ("Diffusion (no proj)", u_diff),
        ("Diffusion+projection", u_proj),
        ("Random shooting", u_shoot),
        ("MPPI", u_mppi),
    ]:
        m = evaluate_controls(u, x0, x_goal, rollout, sdf)
        print(
            f"  {name:28s}  success={100*m['success_rate']:.1f}%  "
            f"collision={100*m['collision_rate']:.1f}%  goal_err={m['avg_goal_error']:.3f}"
        )

    # Single canonical start/goal example plot
    x0_state = torch.tensor(start_n[None], dtype=torch.float32, device=device)
    x_goal_state = torch.tensor(goal_n[None], dtype=torch.float32, device=device)
    diffusion_model.eval()
    cond = create_condition_encoding(x0_state, x_goal_state, state_dim=config["state_dim"])

    u_no = diffusion_model.sample((1, config["horizon"], config["control_dim"]), cond).clamp(-config["u_clip"], config["u_clip"])
    u_yes, _ = projection.project(u_no.clone(), x0_state, x_goal_state)

    sno = rollout(x0_state, u_no)
    syes = rollout(x0_state, u_yes)

    ok_no = (sdf(sno[..., :2]).min().item() > float(robot_radius_n))
    ok_yes = (sdf(syes[..., :2]).min().item() > float(robot_radius_n))

    if not args.no_plots:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        gx, gy = x_goal_state[0, 0].item(), x_goal_state[0, 1].item()

        for i, (st, title, color, ok) in enumerate(
            [
                (sno, "Without projection", "darkorange", ok_no),
                (syes, "With projection", "royalblue", ok_yes),
            ]
        ):
            ax = axes[i]
            draw_env(ax, train_dataset.box_obstacles)
            ax.plot(st[0, :, 0].cpu(), st[0, :, 1].cpu(), "-", lw=3, color=color, zorder=3)
            ax.scatter([x0_state[0, 0].item()], [x0_state[0, 1].item()], c="lime", s=120, edgecolor="k", zorder=4, label="Start")
            ax.scatter([gx], [gy], c="red", s=120, edgecolor="k", zorder=4, label="Goal")
            ax.set_title(f"{title}  |  collision-free={ok}", fontsize=13, fontweight="bold")
            ax.legend()

        plt.tight_layout()
        plt.show()

    # Batch eval: diffusion with and without projection
    num_test = int(args.num_test)
    goal_tol = 0.12
    min_dist = 0.35

    def _sync_if_cuda():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def sample_dynobench_pair(sdf_model, n_tries: int = 5000):
        env_min_n = norm["npos"](env_cfg["env_min"])
        env_max_n = norm["npos"](env_cfg["env_max"])
        for _ in range(n_tries):
            s = torch.tensor(np.random.uniform(env_min_n, env_max_n, (1, 2)), dtype=torch.float32, device=device)
            g = torch.tensor(np.random.uniform(env_min_n, env_max_n, (1, 2)), dtype=torch.float32, device=device)
            if (
                sdf_model(s[0]).item() > (float(robot_radius_n) + 0.05)
                and sdf_model(g[0]).item() > (float(robot_radius_n) + 0.05)
                and torch.norm(s - g).item() > min_dist
            ):
                s_state = torch.cat([s, torch.zeros(1, 2, device=device)], -1)
                g_state = torch.cat([g, torch.zeros(1, 2, device=device)], -1)

                if args.eval_reachable:
                    # Filter to pairs that are plausibly solvable within the horizon.
                    # This avoids evaluating on many effectively-unreachable random pairs.
                    states_np, actions_np = plan_via_waypoint(
                        s0=s_state[0].detach().cpu().numpy(),
                        sg=g_state[0].detach().cpu().numpy(),
                        horizon=int(config["horizon"]),
                        dt=float(config["dt"]),
                        sdf=sdf_model,
                        max_vel=float(max_vel_n),
                        max_acc=float(max_acc_n),
                    )
                    if states_np is None or actions_np is None:
                        continue

                return s_state, g_state
        raise RuntimeError("Could not sample a valid (start, goal) pair")

    test_pairs = [sample_dynobench_pair(sdf) for _ in range(num_test)]
    print(f"\n✓ Sampled {len(test_pairs)} Dynobench start/goal pairs")

    T, C = config["horizon"], config["control_dim"]
    results_no_proj = []
    results_with_proj = []

    K_eval = max(int(args.eval_k), 1)
    if K_eval > 1:
        print(f"\nUsing best-of-K for eval: K={K_eval}")
    if args.eval_reachable:
        print("Using reachable-pair filter for eval")

    for (x0p, x_goalp) in tqdm(test_pairs, desc="Evaluating"):
        x0p = x0p.to(device)
        x_goalp = x_goalp.to(device)
        condp = create_condition_encoding(x0p, x_goalp, state_dim=config["state_dim"])

        _sync_if_cuda()
        t0 = time.time()
        if K_eval == 1:
            u_no = diffusion_model.sample((1, T, C), condp).clamp(-config["u_clip"], config["u_clip"])
        else:
            u_no, _ = diffusion_best_of_k(
                diffusion_model,
                constraint_fn,
                x0p,
                x_goalp,
                T,
                C,
                K=K_eval,
                u_clip=config["u_clip"],
            )
        _sync_if_cuda()
        t1 = time.time()
        sample_time_s = float(t1 - t0)
        m_no = evaluate_controls(u_no, x0p, x_goalp, rollout, sdf, goal_tol=goal_tol)
        results_no_proj.append(
            {
                "success": bool(m_no["success"][0]),
                "collision": bool(m_no["collision"][0]),
                "goal_error": float(m_no["goal_error"][0]),
                "time_s": sample_time_s,
            }
        )

        _sync_if_cuda()
        t0 = time.time()
        u_yes, _ = projection.project(u_no.clone(), x0p, x_goalp)
        _sync_if_cuda()
        t1 = time.time()
        proj_time_s = float(t1 - t0)
        m_yes = evaluate_controls(u_yes, x0p, x_goalp, rollout, sdf, goal_tol=goal_tol)
        results_with_proj.append(
            {
                "success": bool(m_yes["success"][0]),
                "collision": bool(m_yes["collision"][0]),
                "goal_error": float(m_yes["goal_error"][0]),
                # Total time for "diffusion + projection" (previously this incorrectly timed only projection)
                "time_s": sample_time_s + proj_time_s,
                "time_sample_s": sample_time_s,
                "time_project_s": proj_time_s,
            }
        )

    stats_no = compute_stats(results_no_proj, "Without projection")
    stats_yes = compute_stats(results_with_proj, "With projection")

    print("\nBatch results:")
    for s in [stats_no, stats_yes]:
        print(f"\n{s['name']} (n={s['n']}):")
        print(f"  Success rate:   {s['success_rate']:.1f}% ± {s['se_success']:.1f}% (SE)")
        print(f"  Collision rate: {s['collision_rate']:.1f}% ± {s['se_collision']:.1f}% (SE)")
        print(f"  Avg goal error: {s['avg_goal_error']:.3f} ± {s['se_goal']:.3f} (SE)")
        print(f"  Avg time:       {s['avg_time_s']*1000:.1f} ms ± {s['se_time_s']*1000:.1f} ms (SE)")
        print(f"  Median / P90:   {s['median_time_s']*1000:.1f} ms / {s['p90_time_s']*1000:.1f} ms")

    if not args.no_plots:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        metrics = ["success_rate", "collision_rate", "avg_goal_error"]
        titles = ["Success rate (%)", "Collision rate (%)", "Avg goal error"]
        colors = ["green", "red", "blue"]

        for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            ax = axes[i]
            labels = ["Without\nprojection", "With\nprojection"]
            values = [stats_no[metric], stats_yes[metric]]
            bars = ax.bar(labels, values, color=[color, color], edgecolor="black", linewidth=2)
            bars[0].set_alpha(0.5)
            bars[1].set_alpha(1.0)

            if metric == "success_rate":
                err = [stats_no["se_success"], stats_yes["se_success"]]
            elif metric == "collision_rate":
                err = [stats_no["se_collision"], stats_yes["se_collision"]]
            else:
                err = [stats_no["se_goal"], stats_yes["se_goal"]]

            ax.errorbar(labels, values, yerr=err, fmt="none", ecolor="black", capsize=5, linewidth=2)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.2, axis="y")

        plt.tight_layout()
        plt.show()

    # Ablation on lambda
    lambda_values = [0.01, 0.05, 0.1, 0.5]
    ablation_results = []
    test_cases_per_lambda = 30

    print("\nRunning lambda ablation...\n")
    for lam in lambda_values:
        print(f"Testing λ = {lam}...")

        proj_test = ProximalProjection(constraint_fn, lambda_penalty=lam, num_steps=6, lr=0.06, u_clip=config["u_clip"])
        sampler_test = FeasibleDiffusionSampler(diffusion_model, proj_test, project_every=1)

        pairs = [sample_dynobench_pair(sdf) for _ in range(test_cases_per_lambda)]
        results = []
        for (x0p, x_goalp) in pairs:
            x0p = x0p.to(device)
            x_goalp = x_goalp.to(device)
            t0 = time.time()
            u, _ = sampler_test.sample(1, T, x0p, x_goalp, method="ddim", eta=0.0)
            t1 = time.time()
            m = evaluate_controls(u, x0p, x_goalp, rollout, sdf, goal_tol=goal_tol)
            results.append(
                {
                    "success": bool(m["success"][0]),
                    "collision": bool(m["collision"][0]),
                    "goal_error": float(m["goal_error"][0]),
                    "time_s": float(t1 - t0),
                }
            )

        s = compute_stats(results, f"lambda={lam}")
        ablation_results.append({"lambda": lam, "success_rate": s["success_rate"], "collision_rate": s["collision_rate"]})
        print(f"  Success: {s['success_rate']:.1f}% | Collision: {s['collision_rate']:.1f}%")

    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(10, 5))
        lambdas = [r["lambda"] for r in ablation_results]
        success_rates = [r["success_rate"] for r in ablation_results]
        collision_rates = [r["collision_rate"] for r in ablation_results]

        ax.plot(lambdas, success_rates, "o-", linewidth=2, markersize=10, label="Success Rate", color="green")
        ax.plot(lambdas, collision_rates, "s-", linewidth=2, markersize=10, label="Collision Rate", color="red")

        ax.set_xlabel("Lambda (λ)", fontsize=13)
        ax.set_ylabel("Rate (%)", fontsize=13)
        ax.set_title("Ablation: Effect of Lambda on Performance", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    print("\n✓ All done.")


if __name__ == "__main__":
    main()
