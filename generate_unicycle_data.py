import math
import json
import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

# -----------------------------
# Map + SDF
# -----------------------------
def make_random_map(H=64, W=64, obstacle_prob=0.25, border=2, seed=None):
    rng = np.random.default_rng(seed)
    occ = rng.random((H, W)) < obstacle_prob

    # Add solid border to keep robot inside map
    occ[:border, :] = True
    occ[-border:, :] = True
    occ[:, :border] = True
    occ[:, -border:] = True

    return occ

def compute_sdf(occ: np.ndarray, resolution=1.0):
    # Positive outside obstacles, negative inside obstacles
    occ = occ.astype(bool)
    outside = ~occ
    dist_out = distance_transform_edt(outside) * resolution
    dist_in = distance_transform_edt(occ) * resolution
    sdf = dist_out - dist_in
    return sdf.astype(np.float32)

def in_bounds(p, H, W):
    x, y = p
    return (0 <= x < W) and (0 <= y < H)

def bilinear_sample(grid, x, y):
    H, W = grid.shape

    # Ensure x0 in [0, W-2], y0 in [0, H-2]
    x = float(x); y = float(y)
    x = np.clip(x, 0.0, W - 2.000001)
    y = np.clip(y, 0.0, H - 2.000001)

    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    x1 = x0 + 1; y1 = y0 + 1

    wx = x - x0; wy = y - y0

    v00 = grid[y0, x0]
    v10 = grid[y0, x1]
    v01 = grid[y1, x0]
    v11 = grid[y1, x1]

    return (1-wx)*(1-wy)*v00 + wx*(1-wy)*v10 + (1-wx)*wy*v01 + wx*wy*v11

def sample_free_pose(rng, sdf, delta):
    H, W = sdf.shape
    for _ in range(20000):
        x = rng.uniform(0, W)
        y = rng.uniform(0, H)
        if bilinear_sample(sdf, x, y) >= delta:
            theta = rng.uniform(-math.pi, math.pi)
            return np.array([x, y, theta], dtype=np.float32)
    return None

# -----------------------------
# Unicycle dynamics
# -----------------------------
def rollout_unicycle(x0, U, dt, W=64, H=64):
    # x: [x, y, theta], U: [T, 2] with [v, w]
    T = U.shape[0]
    X = np.zeros((T + 1, 3), dtype=np.float32)
    X[0] = x0
    for t in range(T):
        x, y, th = X[t]
        v, w = U[t]
        x_next = x + v * math.cos(th) * dt

        y_next = y + v * math.sin(th) * dt
        x_next = np.clip(x_next, 0.0, W - 1e-3)
        y_next = np.clip(y_next, 0.0, H - 1e-3)
        th_next = th + w * dt
        # wrap angle
        th_next = (th_next + math.pi) % (2 * math.pi) - math.pi
        X[t + 1] = np.array([x_next, y_next, th_next], dtype=np.float32)
    return X

# -----------------------------
# MPPI planner for unicycle
# -----------------------------
def mppi_plan(
    x0, goal_xy, sdf,
    T=64, dt=0.1,
    K=256, iters=15,
    v_max=1.0, w_max=2.0,
    sigma_v=0.4, sigma_w=0.8,
    lam=1.0,
    delta=1.5,
    w_goal=10.0,
    w_ctrl=0.1,
    w_col=1000.0,
    seed=None,
):
    rng = np.random.default_rng(seed)

    # nominal controls
    U = np.zeros((T, 2), dtype=np.float32)
    U[:, 0] = 0.5 * v_max  # start moving forward

    def clamp(Ucand):
        Uc = Ucand.copy()
        Uc[:, 0] = np.clip(Uc[:, 0], 0.0, v_max)
        Uc[:, 1] = np.clip(Uc[:, 1], -w_max, w_max)
        return Uc

    best_X = None
    best_U = None
    best_J = np.inf

    for _ in range(iters):
        eps = np.zeros((K, T, 2), dtype=np.float32)
        eps[:, :, 0] = rng.normal(0.0, sigma_v, size=(K, T))
        eps[:, :, 1] = rng.normal(0.0, sigma_w, size=(K, T))

        Js = np.zeros((K,), dtype=np.float32)
        for k in range(K):
            Uk = clamp(U + eps[k])
            Xk = rollout_unicycle(x0, Uk, dt)

            # cost
            gx, gy = goal_xy
            px, py = Xk[-1, 0], Xk[-1, 1]
            J = w_goal * ((px - gx) ** 2 + (py - gy) ** 2)
            J += w_ctrl * float(np.sum(Uk[:, 0] ** 2 + Uk[:, 1] ** 2))

            # collision penalty via sdf hinge
            col = 0.0
            for t in range(T + 1):
                sx = bilinear_sample(sdf, Xk[t, 0], Xk[t, 1])
                if sx < delta:
                    col += (delta - sx) ** 2
            J += w_col * col

            Js[k] = J

            if J < best_J:
                best_J = J
                best_X = Xk
                best_U = Uk

        # MPPI weights (stabilize by subtracting min)
        Jmin = float(np.min(Js))
        w = np.exp(-(Js - Jmin) / max(lam, 1e-6))
        w = w / (np.sum(w) + 1e-9)

        # update nominal: U <- U + sum w_k * eps_k
        dU = np.tensordot(w, eps, axes=(0, 0))  # [T,2]
        U = clamp(U + dU)

    return best_U, best_X, best_J

def trajectory_feasible(X, sdf, delta):
    for t in range(X.shape[0]):
        if bilinear_sample(sdf, X[t, 0], X[t, 1]) < delta:
            return False
    return True

# -----------------------------
# Dataset generation
# -----------------------------
def generate_dataset(
    out_npz="data/unicycle_mppi.npz",
    n_maps=10,
    traj_per_map=200,
    H=64, W=64,
    obstacle_prob=0.25,
    resolution=1.0,
    delta=1.5,
    T=64, dt=0.1,
    seed=0,
):
    rng = np.random.default_rng(seed)

    maps_occ = []
    maps_sdf = []
    samples = []

    for mid in range(n_maps):
        occ = make_random_map(H, W, obstacle_prob=obstacle_prob, seed=int(rng.integers(1e9)))
        sdf = compute_sdf(occ, resolution=resolution)

        maps_occ.append(occ.astype(np.uint8))
        maps_sdf.append(sdf)

        made = 0
        attempts = 0
        pbar = tqdm(total=traj_per_map, desc=f"map {mid}", leave=False)
        while made < traj_per_map and attempts < traj_per_map * 50:
            attempts += 1
            x0 = sample_free_pose(rng, sdf, delta=delta)
            xg = sample_free_pose(rng, sdf, delta=delta)
            if x0 is None or xg is None:
                continue

            # enforce minimum distance between start/goal
            if np.linalg.norm(x0[:2] - xg[:2]) < 15.0:
                continue

            U, X, J = mppi_plan(
                x0=x0,
                goal_xy=xg[:2],
                sdf=sdf,
                T=T, dt=dt,
                K=256, iters=20,
                v_max=1.2, w_max=2.5,
                sigma_v=0.5, sigma_w=1.0,
                lam=1.0,
                delta=delta,
                w_goal=20.0, w_ctrl=0.05, w_col=2000.0,
                seed=int(rng.integers(1e9)),
            )

            # accept criteria
            if not trajectory_feasible(X, sdf, delta=delta):
                continue
            if np.linalg.norm(X[-1, :2] - xg[:2]) > 3.0:
                continue

            samples.append((mid, x0, xg, U, X))
            made += 1
            pbar.update(1)

        pbar.close()
        if made < traj_per_map:
            print(f"Warning: map {mid} only produced {made}/{traj_per_map} feasible trajectories.")

    # pack arrays
    map_id = np.array([s[0] for s in samples], dtype=np.int32)
    X0 = np.stack([s[1] for s in samples]).astype(np.float32)
    XG = np.stack([s[2] for s in samples]).astype(np.float32)
    U  = np.stack([s[3] for s in samples]).astype(np.float32)   # [N,T,2]
    X  = np.stack([s[4] for s in samples]).astype(np.float32)   # [N,T+1,3]

    # normalization stats for controls
    u_flat = U.reshape(-1, U.shape[-1])
    u_mean = u_flat.mean(axis=0).astype(np.float32)
    u_std = (u_flat.std(axis=0) + 1e-6).astype(np.float32)

    np.savez_compressed(
        out_npz,
        map_id=map_id,
        X0=X0,
        XG=XG,
        U=U,
        X=X,
        maps_occ=np.stack(maps_occ),
        maps_sdf=np.stack(maps_sdf),
        meta=json.dumps({
            "H": H, "W": W, "resolution": resolution,
            "delta": float(delta), "T": T, "dt": float(dt),
            "control": "[v, w]",
            "state": "[x, y, theta]"
        })
    )

    print("Saved:", out_npz)
    print("N:", len(samples), "maps:", n_maps)
    print("U:", U.shape, "X:", X.shape)
    print("u_mean:", u_mean, "u_std:", u_std)

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    generate_dataset()
