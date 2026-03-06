import math
import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    # Workspace / map
    map_size_m: float = 10.0
    grid_size: int = 64
    robot_radius: float = 0.20
    safety_margin: float = 0.05

    # Dynamics
    dt: float = 0.1
    horizon: int = 40
    v_min: float = -0.2
    v_max: float = 2.8
    w_min: float = -1.5
    w_max: float = 1.5

    # Goal tolerances
    goal_pos_tol: float = 0.20
    goal_theta_tol_deg: float = 15.0

    # Sampling
    min_start_goal_dist: float = 4.0
    min_clearance_start_goal: float = 0.35
    max_start_goal_dist: float = 8.0

    # CEM expert
    cem_iterations: int = 8
    cem_population: int = 256
    cem_elites: int = 32
    init_v_mean: float = 0.25
    init_w_mean: float = 0.0
    init_v_std: float = 0.35
    init_w_std: float = 0.80
    smooth_alpha: float = 0.65  # higher = smoother sampled controls

    # Cost weights
    w_goal_pos: float = 80.0
    w_goal_theta: float = 20.0
    w_obs: float = 120.0
    w_ctrl: float = 0.2
    w_smooth: float = 1.0
    collision_penalty: float = 500.0

    # Dataset
    out_dir: str = "diffdrive_dataset"
    n_train: int = 2000
    n_val: int = 200
    n_test: int = 200
    max_attempts_per_sample: int = 30
    seed: int = 7
    n_train = 20
    n_val = 5
    n_test = 5

    cem_population = 64
    cem_iterations = 4
    horizon = 100
    max_attempts_per_sample = 10
    goal_pos_tol = 0.30
    goal_theta_tol_deg = 25.0

    cem_population = 96
    cem_iterations = 5
    cem_elites = 16

    max_attempts_per_sample = 15
# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed()
    np.random.seed()


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a: float, b: float) -> float:
    return float(wrap_angle(np.array([a - b]))[0])


def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def world_to_grid(x: float, y: float, cfg: Config) -> Tuple[int, int]:
    scale = cfg.grid_size / cfg.map_size_m
    gx = int(np.clip(x * scale, 0, cfg.grid_size - 1))
    gy = int(np.clip(y * scale, 0, cfg.grid_size - 1))
    return gx, gy


def grid_to_world_centers(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(cfg.grid_size) + 0.5) * (cfg.map_size_m / cfg.grid_size)
    ys = (np.arange(cfg.grid_size) + 0.5) * (cfg.map_size_m / cfg.grid_size)
    return np.meshgrid(xs, ys, indexing="xy")


# ============================================================
# Map generation
# ============================================================
class OccupancyMap:
    """
    occupancy[y, x] = 1 for obstacle, 0 for free.
    sdf[y, x] > 0 outside obstacles, < 0 inside obstacles, units in meters.
    meta stores scenario-specific geometry for smarter goal sampling.
    """
    def __init__(
        self,
        occupancy: np.ndarray,
        sdf: np.ndarray,
        scenario_type: str,
        meta: Optional[Dict] = None,
    ):
        assert occupancy.shape == sdf.shape
        self.occupancy = occupancy.astype(np.uint8)
        self.sdf = sdf.astype(np.float32)
        self.scenario_type = scenario_type
        self.meta = meta or {}


def add_rect_obstacle(occ: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    x0, x1 = sorted((max(0, x0), min(occ.shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(occ.shape[0], y1)))
    occ[y0:y1, x0:x1] = 1


def compute_sdf_from_rects(rects_m: List[Tuple[float, float, float, float]], cfg: Config) -> np.ndarray:
    """
    Approximate exact rectangle SDF on the grid.
    Positive outside, negative inside.
    """
    X, Y = grid_to_world_centers(cfg)
    sdf = np.full((cfg.grid_size, cfg.grid_size), np.inf, dtype=np.float32)

    for (xmin, ymin, xmax, ymax) in rects_m:
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        hx = 0.5 * (xmax - xmin)
        hy = 0.5 * (ymax - ymin)

        dx = np.abs(X - cx) - hx
        dy = np.abs(Y - cy) - hy
        dx_clip = np.maximum(dx, 0.0)
        dy_clip = np.maximum(dy, 0.0)
        outside = np.sqrt(dx_clip ** 2 + dy_clip ** 2)
        inside = np.minimum(np.maximum(dx, dy), 0.0)
        rect_sdf = outside + inside
        sdf = np.minimum(sdf, rect_sdf.astype(np.float32))

    # Outer walls as obstacles
    wall_dist = np.minimum.reduce([X, Y, cfg.map_size_m - X, cfg.map_size_m - Y]).astype(np.float32)
    sdf = np.minimum(sdf, wall_dist)
    return sdf


def build_occ_from_rects(rects_m: List[Tuple[float, float, float, float]], cfg: Config) -> np.ndarray:
    occ = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.uint8)

    occ[0:1, :] = 1
    occ[-1:, :] = 1
    occ[:, 0:1] = 1
    occ[:, -1:] = 1

    scale = cfg.grid_size / cfg.map_size_m
    for xmin, ymin, xmax, ymax in rects_m:
        add_rect_obstacle(
            occ,
            int(xmin * scale),
            int(ymin * scale),
            int(math.ceil(xmax * scale)),
            int(math.ceil(ymax * scale)),
        )
    return occ


def generate_open_map(cfg: Config) -> OccupancyMap:
    rects: List[Tuple[float, float, float, float]] = []
    occ = build_occ_from_rects(rects, cfg)
    sdf = compute_sdf_from_rects(rects, cfg)
    return OccupancyMap(occ, sdf, "open", {"task_hint": "free_space"})


def generate_clutter_map(cfg: Config) -> OccupancyMap:
    rects: List[Tuple[float, float, float, float]] = []
    n_obs = random.randint(6, 14)

    for _ in range(n_obs):
        w = random.uniform(0.5, 1.4)
        h = random.uniform(0.5, 1.4)
        x = random.uniform(0.5, cfg.map_size_m - 0.5 - w)
        y = random.uniform(0.5, cfg.map_size_m - 0.5 - h)
        rects.append((x, y, x + w, y + h))

    occ = build_occ_from_rects(rects, cfg)
    sdf = compute_sdf_from_rects(rects, cfg)
    return OccupancyMap(occ, sdf, "clutter", {"task_hint": "obstacle_avoidance"})


def generate_corridor_map(cfg: Config) -> OccupancyMap:
    rects: List[Tuple[float, float, float, float]] = []

    orientation = random.choice(["horizontal", "vertical"])
    corridor_width = random.uniform(1.0, 1.6)

    if orientation == "horizontal":
        cy = random.uniform(3.0, cfg.map_size_m - 3.0)
        rects.append((0.0, 0.0, cfg.map_size_m, cy - corridor_width / 2))
        rects.append((0.0, cy + corridor_width / 2, cfg.map_size_m, cfg.map_size_m))
        meta = {
            "orientation": "horizontal",
            "center": cy,
            "width": corridor_width,
        }
    else:
        cx = random.uniform(3.0, cfg.map_size_m - 3.0)
        rects.append((0.0, 0.0, cx - corridor_width / 2, cfg.map_size_m))
        rects.append((cx + corridor_width / 2, 0.0, cfg.map_size_m, cfg.map_size_m))
        meta = {
            "orientation": "vertical",
            "center": cx,
            "width": corridor_width,
        }

    occ = build_occ_from_rects(rects, cfg)
    sdf = compute_sdf_from_rects(rects, cfg)
    return OccupancyMap(occ, sdf, "corridor", meta)


def generate_doorway_map(cfg: Config) -> OccupancyMap:
    rects: List[Tuple[float, float, float, float]] = []

    orientation = random.choice(["horizontal", "vertical"])
    door_width = random.uniform(0.9, 1.4)
    wall_thickness = random.uniform(0.3, 0.6)

    if orientation == "vertical":
        x0 = random.uniform(4.0, 6.0)
        door_y = random.uniform(3.0, 7.0)
        rects.append((x0, 0.0, x0 + wall_thickness, door_y - door_width / 2))
        rects.append((x0, door_y + door_width / 2, x0 + wall_thickness, cfg.map_size_m))
        meta = {
            "orientation": "vertical",
            "wall_x": x0,
            "wall_thickness": wall_thickness,
            "door_center": (x0 + 0.5 * wall_thickness, door_y),
            "door_width": door_width,
        }
    else:
        y0 = random.uniform(4.0, 6.0)
        door_x = random.uniform(3.0, 7.0)
        rects.append((0.0, y0, door_x - door_width / 2, y0 + wall_thickness))
        rects.append((door_x + door_width / 2, y0, cfg.map_size_m, y0 + wall_thickness))
        meta = {
            "orientation": "horizontal",
            "wall_y": y0,
            "wall_thickness": wall_thickness,
            "door_center": (door_x, y0 + 0.5 * wall_thickness),
            "door_width": door_width,
        }

    # light clutter
    for _ in range(random.randint(2, 5)):
        w = random.uniform(0.4, 1.0)
        h = random.uniform(0.4, 1.0)
        x = random.uniform(0.5, cfg.map_size_m - 0.5 - w)
        y = random.uniform(0.5, cfg.map_size_m - 0.5 - h)
        rects.append((x, y, x + w, y + h))

    occ = build_occ_from_rects(rects, cfg)
    sdf = compute_sdf_from_rects(rects, cfg)
    return OccupancyMap(occ, sdf, "doorway", meta)


def generate_parking_map(cfg: Config) -> OccupancyMap:
    rects: List[Tuple[float, float, float, float]] = []

    bay_x = random.uniform(6.5, 8.0)
    bay_y = random.uniform(3.0, 6.0)
    bay_w = 1.6
    bay_h = 2.2
    wall = 0.25

    rects.append((bay_x, bay_y, bay_x + wall, bay_y + bay_h))
    rects.append((bay_x + bay_w - wall, bay_y, bay_x + bay_w, bay_y + bay_h))
    rects.append((bay_x, bay_y + bay_h - wall, bay_x + bay_w, bay_y + bay_h))

    for _ in range(random.randint(3, 6)):
        w = random.uniform(0.5, 1.3)
        h = random.uniform(0.5, 1.3)
        x = random.uniform(0.5, cfg.map_size_m - 0.5 - w)
        y = random.uniform(0.5, cfg.map_size_m - 0.5 - h)
        if x > bay_x - 1.5 and y > bay_y - 1.0 and y < bay_y + bay_h + 1.0:
            continue
        rects.append((x, y, x + w, y + h))

    meta = {
        "bay_x": bay_x,
        "bay_y": bay_y,
        "bay_w": bay_w,
        "bay_h": bay_h,
        "opening_side": "bottom",
    }

    occ = build_occ_from_rects(rects, cfg)
    sdf = compute_sdf_from_rects(rects, cfg)
    return OccupancyMap(occ, sdf, "parking", meta)


def line_min_clearance(x0, xg, omap, cfg, n=100):
    vals = []
    for i in range(n):
        t = i / max(1, n - 1)
        x = (1 - t) * x0[0] + t * xg[0]
        y = (1 - t) * x0[1] + t * xg[1]
        vals.append(sdf_query(omap.sdf, float(x), float(y), cfg) - cfg.robot_radius)
    return float(np.min(vals))


def generate_map(cfg: Config) -> OccupancyMap:
    generator = random.choices(
        population=[
            generate_open_map,
            generate_clutter_map,
            generate_corridor_map,
            generate_doorway_map,
            generate_parking_map,
        ],
        weights=[0.1, 0.50, 0.5, 0.5, 0.5],
        k=1,
    )[0]
    return generator(cfg)


# ============================================================
# Sampling start / goal
# ============================================================

def sdf_query(sdf_grid: np.ndarray, x: float, y: float, cfg: Config) -> float:
    gx = np.clip((x / cfg.map_size_m) * (cfg.grid_size - 1), 0.0, cfg.grid_size - 1.0)
    gy = np.clip((y / cfg.map_size_m) * (cfg.grid_size - 1), 0.0, cfg.grid_size - 1.0)

    x0 = int(np.floor(gx))
    x1 = min(x0 + 1, cfg.grid_size - 1)
    y0 = int(np.floor(gy))
    y1 = min(y0 + 1, cfg.grid_size - 1)

    tx = gx - x0
    ty = gy - y0

    v00 = sdf_grid[y0, x0]
    v10 = sdf_grid[y0, x1]
    v01 = sdf_grid[y1, x0]
    v11 = sdf_grid[y1, x1]

    v0 = (1.0 - tx) * v00 + tx * v10
    v1 = (1.0 - tx) * v01 + tx * v11
    return float((1.0 - ty) * v0 + ty * v1)


def sample_pose_free(omap: OccupancyMap, cfg: Config) -> Tuple[float, float, float]:
    for _ in range(1000):
        x = random.uniform(0.5, cfg.map_size_m - 0.5)
        y = random.uniform(0.5, cfg.map_size_m - 0.5)
        th = random.uniform(-np.pi, np.pi)
        if sdf_query(omap.sdf, x, y, cfg) > cfg.robot_radius + cfg.min_clearance_start_goal:
            return x, y, th
    raise RuntimeError("Failed to sample a collision-free pose.")


def sample_start_goal(omap: OccupancyMap, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(1000):
        x0 = np.array(sample_pose_free(omap, cfg), dtype=np.float32)
        xg = np.array(sample_pose_free(omap, cfg), dtype=np.float32)

        dist = np.linalg.norm(x0[:2] - xg[:2])
        if dist < cfg.min_start_goal_dist:
            continue
        if dist > cfg.max_start_goal_dist:
            continue
        if line_min_clearance(x0, xg, omap, cfg) > 0.8:
            continue
        return x0, xg

    raise RuntimeError("Failed to sample a valid start/goal pair.")


# ============================================================
# Dynamics and rollout
# ============================================================

def rollout_unicycle(
    x0: np.ndarray,
    controls: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """
    x = [x, y, theta]
    u = [v, w]
    Returns states of shape (T+1, 3)
    """
    T = controls.shape[0]
    states = np.zeros((T + 1, 3), dtype=np.float32)
    states[0] = x0.astype(np.float32)

    for t in range(T):
        x, y, th = states[t]
        v, w = controls[t]

        x_next = x + cfg.dt * v * math.cos(th)
        y_next = y + cfg.dt * v * math.sin(th)
        th_next = wrap_angle(np.array([th + cfg.dt * w], dtype=np.float32))[0]

        states[t + 1] = np.array([x_next, y_next, th_next], dtype=np.float32)

    return states


def path_min_clearance(states: np.ndarray, omap: OccupancyMap, cfg: Config) -> float:
    clearances = [sdf_query(omap.sdf, float(s[0]), float(s[1]), cfg) - cfg.robot_radius for s in states]
    return float(np.min(clearances))


def collision_flags(states: np.ndarray, omap: OccupancyMap, cfg: Config) -> np.ndarray:
    vals = [sdf_query(omap.sdf, float(s[0]), float(s[1]), cfg) - cfg.robot_radius for s in states]
    return np.array(vals, dtype=np.float32) < 0.0


# ============================================================
# Expert cost and optimizer
# ============================================================

def obstacle_penalty(states: np.ndarray, omap: OccupancyMap, cfg: Config) -> float:
    margin = cfg.robot_radius + cfg.safety_margin
    penalties = []
    for s in states:
        d = sdf_query(omap.sdf, float(s[0]), float(s[1]), cfg) - margin
        penalties.append(max(0.0, -d) ** 2)
    return float(np.sum(penalties))


def trajectory_cost(
    controls: np.ndarray,
    x0: np.ndarray,
    xg: np.ndarray,
    omap: OccupancyMap,
    cfg: Config,
) -> Tuple[float, np.ndarray]:
    states = rollout_unicycle(x0, controls, cfg)

    pos_err = np.linalg.norm(states[-1, :2] - xg[:2])
    th_err = abs(angle_diff(float(states[-1, 2]), float(xg[2])))
    obs_pen = obstacle_penalty(states, omap, cfg)

    ctrl_cost = float(np.sum(controls[:, 0] ** 2 + 0.2 * controls[:, 1] ** 2))
    smooth_cost = float(np.sum((controls[1:] - controls[:-1]) ** 2))

    collision = np.any(collision_flags(states, omap, cfg))
    total = (
        cfg.w_goal_pos * (pos_err ** 2)
        + cfg.w_goal_theta * (th_err ** 2)
        + cfg.w_obs * obs_pen
        + cfg.w_ctrl * ctrl_cost
        + cfg.w_smooth * smooth_cost
        + (cfg.collision_penalty if collision else 0.0)
    )
    return total, states


def sample_smoothed_controls(
    mean: np.ndarray,
    std: np.ndarray,
    n: int,
    cfg: Config,
) -> np.ndarray:
    """
    Samples controls with simple temporal smoothing.
    mean/std shape: (T, 2)
    Returns shape: (n, T, 2)
    """
    T = mean.shape[0]
    samples = np.random.randn(n, T, 2).astype(np.float32)

    # Smooth over time so commands are not too jittery
    alpha = cfg.smooth_alpha
    for i in range(n):
        for t in range(1, T):
            samples[i, t] = alpha * samples[i, t - 1] + math.sqrt(max(1e-6, 1.0 - alpha ** 2)) * samples[i, t]

    samples = mean[None, :, :] + std[None, :, :] * samples

    samples[:, :, 0] = clamp(samples[:, :, 0], cfg.v_min, cfg.v_max)
    samples[:, :, 1] = clamp(samples[:, :, 1], cfg.w_min, cfg.w_max)
    return samples.astype(np.float32)


def make_initial_guess(x0: np.ndarray, xg: np.ndarray, cfg: Config) -> np.ndarray:
    T = cfg.horizon
    controls = np.zeros((T, 2), dtype=np.float32)

    dx = float(xg[0] - x0[0])
    dy = float(xg[1] - x0[1])
    target_heading = math.atan2(dy, dx)
    heading_err = angle_diff(target_heading, float(x0[2]))
    dist = math.sqrt(dx * dx + dy * dy)

    v_nom = np.clip(dist / (T * cfg.dt), 0.0, cfg.v_max * 0.8)
    w_nom = np.clip(heading_err / (max(1, T // 3) * cfg.dt), cfg.w_min * 0.6, cfg.w_max * 0.6)

    # First turn, then move, then terminal heading adjustment
    for t in range(T):
        if t < T // 4:
            controls[t] = np.array([0.15, w_nom], dtype=np.float32)
        elif t < 3 * T // 4:
            controls[t] = np.array([v_nom, 0.0], dtype=np.float32)
        else:
            final_heading_err = angle_diff(float(xg[2]), target_heading)
            w2 = np.clip(final_heading_err / (max(1, T // 4) * cfg.dt), cfg.w_min * 0.6, cfg.w_max * 0.6)
            controls[t] = np.array([0.10, w2], dtype=np.float32)

    return controls


def cem_plan(
    x0: np.ndarray,
    xg: np.ndarray,
    omap: OccupancyMap,
    cfg: Config,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    T = cfg.horizon
    mean = make_initial_guess(x0, xg, cfg)

    std = np.zeros((T, 2), dtype=np.float32)
    std[:, 0] = cfg.init_v_std
    std[:, 1] = cfg.init_w_std

    best_controls = None
    best_states = None
    best_cost = float("inf")

    for _ in range(cfg.cem_iterations):
        batch = sample_smoothed_controls(mean, std, cfg.cem_population, cfg)
        costs = np.zeros(cfg.cem_population, dtype=np.float32)

        rollout_cache: List[np.ndarray] = []
        for i in range(cfg.cem_population):
            c, states = trajectory_cost(batch[i], x0, xg, omap, cfg)
            costs[i] = c
            rollout_cache.append(states)

        elite_idx = np.argsort(costs)[:cfg.cem_elites]
        elites = batch[elite_idx]

        mean = np.mean(elites, axis=0)
        std = np.std(elites, axis=0) + 1e-3

        cur_best = int(np.argmin(costs))
        if float(costs[cur_best]) < best_cost:
            best_cost = float(costs[cur_best])
            best_controls = batch[cur_best].copy()
            best_states = rollout_cache[cur_best].copy()

    if best_controls is None or best_states is None:
        return None

    pos_err = float(np.linalg.norm(best_states[-1, :2] - xg[:2]))
    theta_err = abs(angle_diff(float(best_states[-1, 2]), float(xg[2])))
    min_clear = path_min_clearance(best_states, omap, cfg)
    success = (
        pos_err <= cfg.goal_pos_tol
        and theta_err <= math.radians(cfg.goal_theta_tol_deg)
        and min_clear > 0.0
    )

    info = {
        "cost": best_cost,
        "pos_err": pos_err,
        "theta_err_rad": theta_err,
        "min_clearance": min_clear,
        "success": success,
    }
    return best_controls, best_states, info


# ============================================================
# Dataset generation
# ============================================================

def make_sample_dict(
    omap: OccupancyMap,
    x0: np.ndarray,
    xg: np.ndarray,
    controls: np.ndarray,
    states: np.ndarray,
    info: Dict,
) -> Dict:
    return {
        "occupancy": omap.occupancy.astype(np.uint8),
        "sdf": omap.sdf.astype(np.float32),
        "start": x0.astype(np.float32),
        "goal": xg.astype(np.float32),
        "controls": controls.astype(np.float32),
        "states": states.astype(np.float32),
        "scenario_type": omap.scenario_type,
        "info": info,
    }


def save_sample_npz(sample: Dict, path: str) -> None:
    np.savez_compressed(
        path,
        occupancy=sample["occupancy"],
        sdf=sample["sdf"],
        start=sample["start"],
        goal=sample["goal"],
        controls=sample["controls"],
        states=sample["states"],
        scenario_type=np.array(sample["scenario_type"]),
        info_json=np.array(json.dumps(sample["info"])),
    )


def generate_one_sample(cfg: Config) -> Optional[Dict]:
    for _ in range(cfg.max_attempts_per_sample):
        omap = generate_map(cfg)
        try:
            x0, xg = sample_start_goal(omap, cfg)
        except RuntimeError:
            continue

        result = cem_plan(x0, xg, omap, cfg)
        if result is None:
            continue

        controls, states, info = result
        if not info["success"]:
            continue

        return make_sample_dict(omap, x0, xg, controls, states, info)

    return None


def generate_split(n_samples: int, split_name: str, cfg: Config) -> None:
    split_dir = os.path.join(cfg.out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    count = 0
    attempts = 0
    scenario_hist: Dict[str, int] = {}

    pbar = tqdm(total=n_samples, desc=f"{split_name}", dynamic_ncols=True)

    while count < n_samples:
        sample = generate_one_sample(cfg)
        attempts += 1

        if sample is None:
            if attempts % 10 == 0:
                pbar.set_postfix({
                    "attempts": attempts,
                    "saved": count,
                    "yield": f"{count / max(attempts, 1):.2f}"
                })
            continue

        fname = os.path.join(split_dir, f"{split_name}_{count:06d}.npz")
        save_sample_npz(sample, fname)

        sc = sample["scenario_type"]
        scenario_hist[sc] = scenario_hist.get(sc, 0) + 1
        count += 1
        pbar.update(1)
        pbar.set_postfix({
            "attempts": attempts,
            "yield": f"{count / max(attempts, 1):.2f}",
            "last": sc
        })

    pbar.close()

    metadata = {
        "split": split_name,
        "n_samples": n_samples,
        "config": asdict(cfg),
        "scenario_hist": scenario_hist,
        "attempts": attempts,
        "acceptance_rate": count / max(attempts, 1),
    }
    with open(os.path.join(split_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("Generating train split...")
    generate_split(cfg.n_train, "train", cfg)

    print("Generating val split...")
    generate_split(cfg.n_val, "val", cfg)

    print("Generating test split...")
    generate_split(cfg.n_test, "test", cfg)

    print(f"Done. Dataset saved to: {cfg.out_dir}")


if __name__ == "__main__":
    main()