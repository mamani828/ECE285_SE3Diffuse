# Fast standalone differential-drive dataset generator
# - Self-contained (stdlib + numpy)
# - Much faster than CEM-based generation
# - Uses grid A* + pure-pursuit-like tracker to create expert labels
# - Saves train/val/test as compressed NPZ files

import os
import json
import math
import heapq
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    # World
    map_size_m: float = 10.0
    grid_size: int = 64
    robot_radius: float = 0.10
    safety_margin: float = 0.03

    # Rollout
    dt: float = 0.10
    horizon: int = 64
    v_max: float = 1.5
    w_max: float = 1.5
    a_v: float = 2.0          # soft accel limit used by tracker
    a_w: float = 4.0          # soft angular accel limit

    # Goal
    goal_pos_tol: float = 0.20
    goal_theta_tol_deg: float = 18.0

    # Sampling
    min_start_goal_dist: float = 3.0
    max_start_goal_dist: float = 7.0
    min_clearance_start_goal: float = 0.25
    reject_easy_if_line_clearer_than: float = 0.90
    max_attempts_per_sample: int = 20

    # Dataset size
    out_dir: str = "diffdrive_fast_dataset"
    n_train: int = 1000
    n_val: int = 200
    n_test: int = 100
    seed: int = 7

    # Scenario mix
    p_open: float = 0.08
    p_clutter: float = 0.42
    p_corridor: float = 0.18
    p_doorway: float = 0.20
    p_parking: float = 0.12


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def angle_diff(a: float, b: float) -> float:
    return float(wrap_angle(np.array([a - b], dtype=np.float32))[0])


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def world_to_grid(x: float, y: float, cfg: Config) -> Tuple[int, int]:
    s = cfg.grid_size / cfg.map_size_m
    gx = int(np.clip(x * s, 0, cfg.grid_size - 1))
    gy = int(np.clip(y * s, 0, cfg.grid_size - 1))
    return gx, gy


def grid_to_world(path: List[Tuple[int, int]], cfg: Config) -> np.ndarray:
    s = cfg.map_size_m / cfg.grid_size
    out = np.empty((len(path), 2), dtype=np.float32)
    for i, (gx, gy) in enumerate(path):
        out[i, 0] = (gx + 0.5) * s
        out[i, 1] = (gy + 0.5) * s
    return out


def simplify_polyline(pts: np.ndarray, min_spacing: float = 0.20) -> np.ndarray:
    if len(pts) <= 1:
        return pts.astype(np.float32)
    kept = [pts[0]]
    last = pts[0]
    for p in pts[1:]:
        if np.linalg.norm(p - last) >= min_spacing:
            kept.append(p)
            last = p
    if np.linalg.norm(kept[-1] - pts[-1]) > 1e-6:
        kept.append(pts[-1])
    return np.asarray(kept, dtype=np.float32)


def rasterize_waypoints_to_grid(waypoints: np.ndarray, cfg: Config) -> np.ndarray:
    grid = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float32)
    if len(waypoints) == 0:
        return grid
    meters_per_cell = cfg.map_size_m / cfg.grid_size
    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        seg_len = float(np.linalg.norm(p1 - p0))
        n = max(2, int(seg_len / meters_per_cell * 2.0))
        for j in range(n):
            t = j / max(1, n - 1)
            p = (1.0 - t) * p0 + t * p1
            gx, gy = world_to_grid(float(p[0]), float(p[1]), cfg)
            grid[gy, gx] = 1.0
    return grid


# ============================================================
# Map representation
# ============================================================

class OccupancyMap:
    def __init__(self, occupancy: np.ndarray, sdf: np.ndarray, scenario_type: str, meta: Optional[Dict] = None):
        self.occupancy = occupancy.astype(np.uint8)
        self.sdf = sdf.astype(np.float32)
        self.scenario_type = scenario_type
        self.meta = meta or {}


def add_rect_obstacle(occ: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    x0 = max(0, min(occ.shape[1], x0))
    x1 = max(0, min(occ.shape[1], x1))
    y0 = max(0, min(occ.shape[0], y0))
    y1 = max(0, min(occ.shape[0], y1))
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    occ[y0:y1, x0:x1] = 1


def build_occ_from_rects(rects_m: List[Tuple[float, float, float, float]], cfg: Config) -> np.ndarray:
    occ = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.uint8)
    occ[0:1, :] = 1
    occ[-1:, :] = 1
    occ[:, 0:1] = 1
    occ[:, -1:] = 1

    s = cfg.grid_size / cfg.map_size_m
    for xmin, ymin, xmax, ymax in rects_m:
        add_rect_obstacle(
            occ,
            int(xmin * s),
            int(ymin * s),
            int(math.ceil(xmax * s)),
            int(math.ceil(ymax * s)),
        )
    return occ


def grid_world_centers(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(cfg.grid_size, dtype=np.float32) + 0.5) * (cfg.map_size_m / cfg.grid_size)
    ys = (np.arange(cfg.grid_size, dtype=np.float32) + 0.5) * (cfg.map_size_m / cfg.grid_size)
    return np.meshgrid(xs, ys, indexing="xy")


def compute_sdf_from_rects(rects_m: List[Tuple[float, float, float, float]], cfg: Config) -> np.ndarray:
    # Analytical SDF to rectangles + box walls. Fast and standalone.
    X, Y = grid_world_centers(cfg)
    sdf = np.full((cfg.grid_size, cfg.grid_size), np.inf, dtype=np.float32)

    for xmin, ymin, xmax, ymax in rects_m:
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        hx = 0.5 * (xmax - xmin)
        hy = 0.5 * (ymax - ymin)

        dx = np.abs(X - cx) - hx
        dy = np.abs(Y - cy) - hy
        dxp = np.maximum(dx, 0.0)
        dyp = np.maximum(dy, 0.0)
        outside = np.sqrt(dxp * dxp + dyp * dyp)
        inside = np.minimum(np.maximum(dx, dy), 0.0)
        rect_sdf = outside + inside
        sdf = np.minimum(sdf, rect_sdf.astype(np.float32))

    wall_dist = np.minimum.reduce([X, Y, cfg.map_size_m - X, cfg.map_size_m - Y]).astype(np.float32)
    sdf = np.minimum(sdf, wall_dist)
    return sdf


def sdf_query_bilinear(sdf: np.ndarray, xs: np.ndarray, ys: np.ndarray, cfg: Config) -> np.ndarray:
    gx = np.clip((xs / cfg.map_size_m) * (cfg.grid_size - 1), 0.0, cfg.grid_size - 1.0)
    gy = np.clip((ys / cfg.map_size_m) * (cfg.grid_size - 1), 0.0, cfg.grid_size - 1.0)

    x0 = np.floor(gx).astype(np.int32)
    y0 = np.floor(gy).astype(np.int32)
    x1 = np.minimum(x0 + 1, cfg.grid_size - 1)
    y1 = np.minimum(y0 + 1, cfg.grid_size - 1)

    tx = gx - x0
    ty = gy - y0

    v00 = sdf[y0, x0]
    v10 = sdf[y0, x1]
    v01 = sdf[y1, x0]
    v11 = sdf[y1, x1]

    v0 = (1.0 - tx) * v00 + tx * v10
    v1 = (1.0 - tx) * v01 + tx * v11
    return ((1.0 - ty) * v0 + ty * v1).astype(np.float32)


def sdf_query_single(sdf: np.ndarray, x: float, y: float, cfg: Config) -> float:
    return float(sdf_query_bilinear(
        sdf,
        np.array([x], dtype=np.float32),
        np.array([y], dtype=np.float32),
        cfg,
    )[0])


# ============================================================
# Map generators
# ============================================================

def generate_open_map(cfg: Config) -> OccupancyMap:
    rects: List[Tuple[float, float, float, float]] = []
    return OccupancyMap(build_occ_from_rects(rects, cfg), compute_sdf_from_rects(rects, cfg), "open", {"task_hint": "free_space"})


def generate_clutter_map(cfg: Config) -> OccupancyMap:
    rects: List[Tuple[float, float, float, float]] = []
    for _ in range(random.randint(8, 16)):
        w = random.uniform(0.4, 1.0)
        h = random.uniform(0.4, 1.0)
        x = random.uniform(0.5, cfg.map_size_m - 0.5 - w)
        y = random.uniform(0.5, cfg.map_size_m - 0.5 - h)
        rects.append((x, y, x + w, y + h))
    return OccupancyMap(build_occ_from_rects(rects, cfg), compute_sdf_from_rects(rects, cfg), "clutter", {"task_hint": "obstacle_avoidance"})


def generate_corridor_map(cfg: Config) -> OccupancyMap:
    rects: List[Tuple[float, float, float, float]] = []
    corridor_width = random.uniform(1.0, 1.6)
    orientation = random.choice(["horizontal", "vertical"])
    if orientation == "horizontal":
        cy = random.uniform(3.0, cfg.map_size_m - 3.0)
        rects.append((0.0, 0.0, cfg.map_size_m, cy - corridor_width / 2.0))
        rects.append((0.0, cy + corridor_width / 2.0, cfg.map_size_m, cfg.map_size_m))
        meta = {"orientation": orientation, "center": cy, "width": corridor_width}
    else:
        cx = random.uniform(3.0, cfg.map_size_m - 3.0)
        rects.append((0.0, 0.0, cx - corridor_width / 2.0, cfg.map_size_m))
        rects.append((cx + corridor_width / 2.0, 0.0, cfg.map_size_m, cfg.map_size_m))
        meta = {"orientation": orientation, "center": cx, "width": corridor_width}
    return OccupancyMap(build_occ_from_rects(rects, cfg), compute_sdf_from_rects(rects, cfg), "corridor", meta)


def generate_doorway_map(cfg: Config) -> OccupancyMap:
    rects: List[Tuple[float, float, float, float]] = []
    door_width = random.uniform(0.9, 1.4)
    wall_thickness = random.uniform(0.3, 0.6)
    orientation = random.choice(["horizontal", "vertical"])

    if orientation == "vertical":
        x0 = random.uniform(4.0, 6.0)
        door_y = random.uniform(3.0, 7.0)
        rects.append((x0, 0.0, x0 + wall_thickness, door_y - door_width / 2.0))
        rects.append((x0, door_y + door_width / 2.0, x0 + wall_thickness, cfg.map_size_m))
        meta = {"orientation": orientation, "door_center": (x0 + 0.5 * wall_thickness, door_y), "door_width": door_width}
        door_box = (x0 - 0.6, door_y - door_width * 0.8, x0 + wall_thickness + 0.6, door_y + door_width * 0.8)
    else:
        y0 = random.uniform(4.0, 6.0)
        door_x = random.uniform(3.0, 7.0)
        rects.append((0.0, y0, door_x - door_width / 2.0, y0 + wall_thickness))
        rects.append((door_x + door_width / 2.0, y0, cfg.map_size_m, y0 + wall_thickness))
        meta = {"orientation": orientation, "door_center": (door_x, y0 + 0.5 * wall_thickness), "door_width": door_width}
        door_box = (door_x - door_width * 0.8, y0 - 0.6, door_x + door_width * 0.8, y0 + wall_thickness + 0.6)

    for _ in range(random.randint(2, 5)):
        w = random.uniform(0.4, 1.0)
        h = random.uniform(0.4, 1.0)
        x = random.uniform(0.5, cfg.map_size_m - 0.5 - w)
        y = random.uniform(0.5, cfg.map_size_m - 0.5 - h)
        # Avoid blocking the doorway region itself.
        if not (x + w < door_box[0] or x > door_box[2] or y + h < door_box[1] or y > door_box[3]):
            continue
        rects.append((x, y, x + w, y + h))

    return OccupancyMap(build_occ_from_rects(rects, cfg), compute_sdf_from_rects(rects, cfg), "doorway", meta)


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
        w = random.uniform(0.5, 1.2)
        h = random.uniform(0.5, 1.2)
        x = random.uniform(0.5, cfg.map_size_m - 0.5 - w)
        y = random.uniform(0.5, cfg.map_size_m - 0.5 - h)
        if x > bay_x - 1.5 and y > bay_y - 1.0 and y < bay_y + bay_h + 1.0:
            continue
        rects.append((x, y, x + w, y + h))

    meta = {"bay_x": bay_x, "bay_y": bay_y, "bay_w": bay_w, "bay_h": bay_h}
    return OccupancyMap(build_occ_from_rects(rects, cfg), compute_sdf_from_rects(rects, cfg), "parking", meta)


def generate_map(cfg: Config) -> OccupancyMap:
    generator = random.choices(
        population=[generate_open_map, generate_clutter_map, generate_corridor_map, generate_doorway_map, generate_parking_map],
        weights=[cfg.p_open, cfg.p_clutter, cfg.p_corridor, cfg.p_doorway, cfg.p_parking],
        k=1,
    )[0]
    return generator(cfg)


# ============================================================
# Start/goal sampling
# ============================================================

def sample_pose_free(omap: OccupancyMap, cfg: Config) -> Tuple[float, float, float]:
    min_clear = cfg.robot_radius + cfg.min_clearance_start_goal
    for _ in range(1000):
        x = random.uniform(0.5, cfg.map_size_m - 0.5)
        y = random.uniform(0.5, cfg.map_size_m - 0.5)
        th = random.uniform(-math.pi, math.pi)
        if sdf_query_single(omap.sdf, x, y, cfg) > min_clear:
            return x, y, th
    raise RuntimeError("Could not sample collision-free pose")


def line_min_clearance(x0: np.ndarray, xg: np.ndarray, omap: OccupancyMap, cfg: Config, n: int = 72) -> float:
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    xs = (1.0 - ts) * x0[0] + ts * xg[0]
    ys = (1.0 - ts) * x0[1] + ts * xg[1]
    d = sdf_query_bilinear(omap.sdf, xs, ys, cfg) - cfg.robot_radius
    return float(np.min(d))


def sample_start_goal(omap: OccupancyMap, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(1000):
        x0 = np.asarray(sample_pose_free(omap, cfg), dtype=np.float32)
        xg = np.asarray(sample_pose_free(omap, cfg), dtype=np.float32)
        dist = float(np.linalg.norm(x0[:2] - xg[:2]))
        if dist < cfg.min_start_goal_dist or dist > cfg.max_start_goal_dist:
            continue
        if line_min_clearance(x0, xg, omap, cfg) > cfg.reject_easy_if_line_clearer_than:
            continue
        return x0, xg
    raise RuntimeError("Could not sample valid start-goal pair")


# ============================================================
# A* planner
# ============================================================

def build_astar_free_mask(omap: OccupancyMap, cfg: Config) -> np.ndarray:
    clearance = cfg.robot_radius + cfg.safety_margin
    return omap.sdf > clearance


def astar_grid(free_mask: np.ndarray, start_xy: Tuple[int, int], goal_xy: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    h, w = free_mask.shape
    sx, sy = start_xy
    gx, gy = goal_xy
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if not free_mask[sy, sx] or not free_mask[gy, gx]:
        return None

    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.41421356), (-1, 1, 1.41421356), (1, -1, 1.41421356), (1, 1, 1.41421356),
    ]

    def heuristic(ax: int, ay: int, bx: int, by: int) -> float:
        return math.hypot(ax - bx, ay - by)

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
    heap: List[Tuple[float, int, int]] = [(heuristic(sx, sy, gx, gy), sx, sy)]

    while heap:
        _, cx, cy = heapq.heappop(heap)
        if cx == gx and cy == gy:
            path = [(cx, cy)]
            cur = (cx, cy)
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        cur_g = g_score[(cx, cy)]
        for dx, dy, step_cost in neighbors:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h or not free_mask[ny, nx]:
                continue
            ng = cur_g + step_cost
            key = (nx, ny)
            if ng < g_score.get(key, float("inf")):
                g_score[key] = ng
                came_from[key] = (cx, cy)
                f = ng + heuristic(nx, ny, gx, gy)
                heapq.heappush(heap, (f, nx, ny))
    return None


# ============================================================
# Path follower / rollout
# ============================================================

def choose_lookahead_target(state_xy: np.ndarray, waypoints: np.ndarray, current_idx: int, lookahead_m: float) -> Tuple[np.ndarray, int]:
    idx = current_idx
    while idx < len(waypoints) - 1 and np.linalg.norm(waypoints[idx] - state_xy) < lookahead_m:
        idx += 1
    return waypoints[min(idx, len(waypoints) - 1)], idx


def rollout_tracker(x0: np.ndarray, xg: np.ndarray, waypoints: np.ndarray, omap: OccupancyMap, cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict]:
    T = cfg.horizon
    controls = np.zeros((T, 2), dtype=np.float32)
    states = np.zeros((T + 1, 3), dtype=np.float32)
    states[0] = x0.astype(np.float32)

    wp_idx = 0
    v_prev = 0.0
    w_prev = 0.0
    done_step = T

    for t in range(T):
        x, y, th = [float(v) for v in states[t]]
        pos = np.array([x, y], dtype=np.float32)

        # Near goal: stop and align.
        goal_dx = float(xg[0] - x)
        goal_dy = float(xg[1] - y)
        goal_dist = math.hypot(goal_dx, goal_dy)

        if goal_dist < 0.55:
            target_heading = math.atan2(goal_dy, goal_dx) if goal_dist > 1e-6 else float(xg[2])
            heading_err = angle_diff(target_heading, th)
            final_heading_err = angle_diff(float(xg[2]), th)

            v_cmd = min(cfg.v_max, 1.6 * goal_dist)
            if abs(heading_err) > 0.7:
                v_cmd *= 0.3
            if goal_dist < cfg.goal_pos_tol:
                v_cmd *= 0.2
                w_cmd = 2.5 * final_heading_err
            else:
                w_cmd = 2.2 * heading_err
        else:
            target, wp_idx = choose_lookahead_target(pos, waypoints, wp_idx, lookahead_m=0.55)
            dx = float(target[0] - x)
            dy = float(target[1] - y)
            target_heading = math.atan2(dy, dx)
            heading_err = angle_diff(target_heading, th)

            # Curvature-aware speed reduction.
            v_cmd = min(cfg.v_max, 1.0 * math.hypot(dx, dy))
            v_cmd *= max(0.25, 1.0 - 0.55 * min(abs(heading_err), 1.2))
            w_cmd = 2.4 * heading_err

        # Soft accel limits.
        dv = np.clip(v_cmd - v_prev, -cfg.a_v * cfg.dt, cfg.a_v * cfg.dt)
        dw = np.clip(w_cmd - w_prev, -cfg.a_w * cfg.dt, cfg.a_w * cfg.dt)
        v = clamp(v_prev + float(dv), 0.0, cfg.v_max)
        w = clamp(w_prev + float(dw), -cfg.w_max, cfg.w_max)
        controls[t] = (v, w)

        xn = x + cfg.dt * v * math.cos(th)
        yn = y + cfg.dt * v * math.sin(th)
        thn = wrap_angle(np.array([th + cfg.dt * w], dtype=np.float32))[0]
        states[t + 1] = np.array([xn, yn, thn], dtype=np.float32)

        v_prev = v
        w_prev = w

        pos_err = float(np.linalg.norm(states[t + 1, :2] - xg[:2]))
        th_err = abs(angle_diff(float(xg[2]), float(states[t + 1, 2])))
        if pos_err <= cfg.goal_pos_tol and th_err <= math.radians(cfg.goal_theta_tol_deg):
            done_step = t + 1
            controls[t + 1:] = 0.0
            states[t + 2:] = states[t + 1]
            break

    d = sdf_query_bilinear(omap.sdf, states[:, 0], states[:, 1], cfg) - cfg.robot_radius
    min_clearance = float(np.min(d))
    collision = bool(np.any(d < 0.0))
    final_pos_err = float(np.linalg.norm(states[min(done_step, T), :2] - xg[:2]))
    final_th_err = abs(angle_diff(float(xg[2]), float(states[min(done_step, T), 2])))

    info = {
        "goal_reached_step": int(done_step),
        "final_pos_err": final_pos_err,
        "final_theta_err_rad": float(final_th_err),
        "min_clearance": min_clearance,
        "collision": collision,
    }
    return controls, states, info


# ============================================================
# Planner wrapper
# ============================================================

def make_plan(x0: np.ndarray, xg: np.ndarray, omap: OccupancyMap, cfg: Config) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]]:
    free_mask = build_astar_free_mask(omap, cfg)
    sx, sy = world_to_grid(float(x0[0]), float(x0[1]), cfg)
    gx, gy = world_to_grid(float(xg[0]), float(xg[1]), cfg)

    grid_path = astar_grid(free_mask, (sx, sy), (gx, gy))
    if grid_path is None or len(grid_path) < 2:
        return None

    waypoints = grid_to_world(grid_path, cfg)
    waypoints = simplify_polyline(waypoints, min_spacing=0.20)

    controls, states, info = rollout_tracker(x0, xg, waypoints, omap, cfg)
    success = (
        (not info["collision"]) and
        info["final_pos_err"] <= cfg.goal_pos_tol and
        info["final_theta_err_rad"] <= math.radians(cfg.goal_theta_tol_deg)
    )
    info["success"] = bool(success)
    info["astar_num_waypoints"] = int(len(waypoints))
    return controls, states, waypoints, info


# ============================================================
# Serialization
# ============================================================

def make_sample_dict(
    omap: OccupancyMap,
    x0: np.ndarray,
    xg: np.ndarray,
    controls: np.ndarray,
    states: np.ndarray,
    astar_waypoints: np.ndarray,
    cfg: Config,
    info: Dict,
) -> Dict:
    valid_horizon = int(min(cfg.horizon, max(1, info.get("goal_reached_step", cfg.horizon))))
    return {
        "occupancy": omap.occupancy.astype(np.uint8),
        "sdf": omap.sdf.astype(np.float32),
        "start": x0.astype(np.float32),
        "goal": xg.astype(np.float32),
        "controls": controls.astype(np.float32),
        "states": states.astype(np.float32),
        "astar_waypoints": astar_waypoints.astype(np.float32),
        "path_mask": rasterize_waypoints_to_grid(astar_waypoints, cfg).astype(np.float32),
        "valid_horizon": np.array(valid_horizon, dtype=np.int32),
        "scenario_type": omap.scenario_type,
        "info": info,
        "map_meta": omap.meta,
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
        astar_waypoints=sample["astar_waypoints"],
        path_mask=sample["path_mask"],
        valid_horizon=sample["valid_horizon"],
        scenario_type=np.array(sample["scenario_type"]),
        info_json=np.array(json.dumps(sample["info"])),
        map_meta_json=np.array(json.dumps(sample["map_meta"])),
    )


# ============================================================
# Generation loop
# ============================================================

def generate_one_sample(cfg: Config) -> Optional[Dict]:
    for _ in range(cfg.max_attempts_per_sample):
        omap = generate_map(cfg)
        try:
            x0, xg = sample_start_goal(omap, cfg)
        except RuntimeError:
            continue

        plan = make_plan(x0, xg, omap, cfg)
        if plan is None:
            continue

        controls, states, waypoints, info = plan
        if not info["success"]:
            continue

        return make_sample_dict(omap, x0, xg, controls, states, waypoints, cfg, info)
    return None


def progress_line(split: str, count: int, n_samples: int, attempts: int, last_scenario: str) -> str:
    acc = count / max(1, attempts)
    return f"\r[{split}] saved={count}/{n_samples} attempts={attempts} yield={acc:.2f} last={last_scenario:<10}"


def generate_split(n_samples: int, split_name: str, cfg: Config) -> None:
    split_dir = os.path.join(cfg.out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    count = 0
    attempts = 0
    scenario_hist: Dict[str, int] = {}

    while count < n_samples:
        sample = generate_one_sample(cfg)
        attempts += 1
        if sample is None:
            if attempts % 10 == 0:
                print(progress_line(split_name, count, n_samples, attempts, "none"), end="", flush=True)
            continue

        filename = os.path.join(split_dir, f"{split_name}_{count:06d}.npz")
        save_sample_npz(sample, filename)

        sc = sample["scenario_type"]
        scenario_hist[sc] = scenario_hist.get(sc, 0) + 1
        count += 1
        print(progress_line(split_name, count, n_samples, attempts, sc), end="", flush=True)

    print()

    metadata = {
        "split": split_name,
        "n_samples": n_samples,
        "config": asdict(cfg),
        "scenario_hist": scenario_hist,
        "attempts": attempts,
        "acceptance_rate": count / max(1, attempts),
    }
    with open(os.path.join(split_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ============================================================
# Main
# ============================================================

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
