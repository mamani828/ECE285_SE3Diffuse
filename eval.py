# eval_vis.py
import os, json, time, math, argparse
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Import your model + diffusion schedule builder from train.py
from train import DiffusionPolicy1D, make_linear_beta_schedule


# ---- Maze maps from Gymnasium-Robotics docs (discrete cell grids) ----
# 1=wall, 0=free
OPEN = [
    [1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1],
]
U_MAZE = [
    [1,1,1,1,1],
    [1,0,0,0,1],
    [1,1,1,0,1],
    [1,0,0,0,1],
    [1,1,1,1,1],
]
MEDIUM = [
    [1,1,1,1,1,1,1,1],
    [1,0,0,1,1,0,0,1],
    [1,0,0,1,0,0,0,1],
    [1,1,0,0,0,1,1,1],
    [1,0,0,1,0,0,0,1],
    [1,0,1,0,0,1,0,1],
    [1,0,0,0,1,0,0,1],
    [1,1,1,1,1,1,1,1],
]
LARGE = [
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,1,1,0,1,0,1,0,1,0,1],
    [1,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,1,1,1,1,0,1,1,1,0,1],
    [1,0,0,1,0,1,0,0,0,0,0,1],
    [1,1,0,1,0,1,0,1,0,1,1,1],
    [1,0,0,1,0,0,0,1,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1],
]

MAZE_MAPS = {
    "PointMaze_Open-v3": OPEN,
    "PointMaze_UMaze-v3": U_MAZE,
    "PointMaze_Medium-v3": MEDIUM,
    "PointMaze_Large-v3": LARGE,
}


def free_cells(maze_map):
    cells = []
    for i, row in enumerate(maze_map):
        for j, v in enumerate(row):
            if v == 0:
                cells.append((i, j))
    return cells


def load_stats(path, device):
    with open(path, "r") as f:
        s = json.load(f)

    def t(x): return torch.tensor(x, dtype=torch.float32, device=device)
    return {
        "act_mean": t(s["act_mean"]),
        "act_std": t(s["act_std"]),
        "cond_mean": t(s["cond_mean"]),
        "cond_std": t(s["cond_std"]),
    }


def extract(a, t, x_shape):
    out = a.gather(0, t)
    while len(out.shape) < len(x_shape):
        out = out.unsqueeze(-1)
    return out


@torch.no_grad()
def ddpm_sample_actions(model, schedule, cond, H, act_dim, n_samples, stats, action_low, action_high):
    """
    cond: (cond_dim,) float32 (unnormalized)
    returns: (n_samples, H, act_dim) in ENV ACTION units (unnormalized + clipped)
    """
    device = next(model.parameters()).device
    K = schedule.betas.shape[0]

    cond = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0).repeat(n_samples, 1)
    cond = (cond - stats["cond_mean"]) / stats["cond_std"]

    x = torch.randn((n_samples, H, act_dim), device=device)  # normalized action space

    betas = schedule.betas
    alphas = schedule.alphas
    abar = schedule.alpha_bars

    for t in reversed(range(K)):
        tt = torch.full((n_samples,), t, device=device, dtype=torch.long)
        eps = model(x, tt, cond)

        a_t = alphas[t]
        ab_t = abar[t]
        b_t = betas[t]

        coef1 = 1.0 / torch.sqrt(a_t)
        coef2 = (1.0 - a_t) / torch.sqrt(1.0 - ab_t)
        mean = coef1 * (x - coef2 * eps)

        if t > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(b_t) * z
        else:
            x = mean

    u = x * stats["act_std"] + stats["act_mean"]
    u = torch.clamp(
        u,
        torch.tensor(action_low, device=device),
        torch.tensor(action_high, device=device),
    )
    return u.cpu().numpy()


def rollout(env, reset_opts, reset_seed, u_seq):
    obs, info = env.reset(seed=reset_seed, options=reset_opts)

    traj_ag = [obs["achieved_goal"].copy()]     # (2,)
    traj_obs = [obs["observation"].copy()]      # (obs_dim,)
    terminated = truncated = False
    final_info = info

    for a in u_seq:
        obs, reward, terminated, truncated, info = env.step(a)
        traj_ag.append(obs["achieved_goal"].copy())
        traj_obs.append(obs["observation"].copy())
        final_info = info
        if terminated or truncated:
            break

    ag = obs["achieved_goal"]
    dg = obs["desired_goal"]
    dist = float(np.linalg.norm(ag - dg))
    success = bool(final_info.get("success", False)) or bool(terminated)

    return {
        "traj_ag": np.asarray(traj_ag, dtype=np.float32),
        "traj_obs": np.asarray(traj_obs, dtype=np.float32),
        "success": success,
        "goal_dist": dist,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_info": final_info,
    }


def rollout_cost(u_seq, roll, goal_weight=1.0, act_weight=0.01):
    dist = roll["goal_dist"]
    success = roll["success"]
    total_u2 = float(np.sum(u_seq * u_seq))
    fail_pen = 0.0 if success else 100.0
    return goal_weight * dist + act_weight * total_u2 + fail_pen


def mppi_refine(env, reset_opts, reset_seed, H, act_dim,
               u_init, n_rollouts=64, iters=4, sigma=0.5, temp=1.0,
               action_low=None, action_high=None,
               prox_anchor=None, prox_lam=0.0):
    """
    MPPI update. If prox_anchor provided, adds prox_lam * mean||u - prox_anchor||^2
    to encourage staying near the diffusion plan (your "projection-ish" operator).
    """
    u = u_init.copy()
    action_low = np.array(action_low, dtype=np.float32)
    action_high = np.array(action_high, dtype=np.float32)

    for _ in range(iters):
        eps = np.random.randn(n_rollouts, H, act_dim).astype(np.float32) * sigma
        U = u[None, :, :] + eps
        U = np.clip(U, action_low, action_high)

        costs = np.zeros((n_rollouts,), dtype=np.float32)
        for i in range(n_rollouts):
            roll = rollout(env, reset_opts, reset_seed, U[i])
            c = rollout_cost(U[i], roll)
            if prox_anchor is not None and prox_lam > 0:
                diff = (U[i] - prox_anchor)
                c = c + prox_lam * float(np.mean(diff * diff))
            costs[i] = c

        cmin = float(costs.min())
        w = np.exp(-(costs - cmin) / max(1e-6, temp))
        w = w / (w.sum() + 1e-8)

        u = u + np.tensordot(w, eps, axes=(0, 0))
        u = np.clip(u, action_low, action_high)

    return u


# --------- Visualization: fit cell->world affine map and draw walls ---------

def fit_cell_to_world(env, maze_map, seed=123, n_cells=25):
    """
    Fits an affine map from cell coords (col, row, 1) -> world xy, using env.reset at many cells.
    This lets us draw wall polygons aligned with the actual achieved_goal coordinates.
    """
    empties = free_cells(maze_map)
    if len(empties) < 2:
        raise RuntimeError("Not enough free cells to calibrate.")

    rng = np.random.RandomState(seed)
    picks = [empties[rng.randint(len(empties))] for _ in range(min(n_cells, len(empties)))]

    # Use a fixed goal cell (any free cell)
    goal_cell = np.array(empties[0], dtype=int)

    rows, cols = [], []
    Xw, Yw = [], []

    for k, cell in enumerate(picks):
        reset_cell = np.array(cell, dtype=int)
        reset_opts = {"reset_cell": reset_cell, "goal_cell": goal_cell}
        obs, info = env.reset(seed=seed + k, options=reset_opts)
        ag = obs["achieved_goal"].astype(np.float64)  # (2,)
        r, c = cell[0], cell[1]
        rows.append(r); cols.append(c)
        Xw.append(ag[0]); Yw.append(ag[1])

    cols = np.asarray(cols, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    Xw = np.asarray(Xw, dtype=np.float64)
    Yw = np.asarray(Yw, dtype=np.float64)

    # Design matrix: [col, row, 1]
    A = np.stack([cols, rows, np.ones_like(cols)], axis=1)  # (N,3)

    # Solve least squares for x and y separately
    px, *_ = np.linalg.lstsq(A, Xw, rcond=None)  # (3,)
    py, *_ = np.linalg.lstsq(A, Yw, rcond=None)  # (3,)

    # Mapping:
    # world_x = px0*col + px1*row + px2
    # world_y = py0*col + py1*row + py2
    return px, py


def cell_center_world(px, py, row, col):
    x = px[0]*col + px[1]*row + px[2]
    y = py[0]*col + py[1]*row + py[2]
    return np.array([x, y], dtype=np.float64)


def cell_corners_world(px, py, row, col):
    """
    Build a parallelogram cell using the fitted step vectors.
    """
    origin = np.array([px[2], py[2]], dtype=np.float64)
    step_c = np.array([px[0], py[0]], dtype=np.float64)  # +1 col
    step_r = np.array([px[1], py[1]], dtype=np.float64)  # +1 row

    center = origin + step_c*col + step_r*row

    # Corners: center +/- 0.5*step_c +/- 0.5*step_r
    corners = []
    for sc in (-0.5, 0.5):
        for sr in (-0.5, 0.5):
            corners.append(center + sc*step_c + sr*step_r)
    # Order them roughly (make polygon non-self-intersecting):
    # sort by angle around center
    corners = np.asarray(corners)
    angles = np.arctan2(corners[:,1]-center[1], corners[:,0]-center[0])
    corners = corners[np.argsort(angles)]
    return corners


def plot_maze_and_paths(maze_map, px, py, paths_dict, start_xy=None, goal_xy=None,
                        title="", save_path=None, show=False):
    """
    paths_dict: {name: traj_ag (T,2)}
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw walls
    for r, row in enumerate(maze_map):
        for c, v in enumerate(row):
            if v == 1:
                corners = cell_corners_world(px, py, r, c)
                poly = Polygon(corners, closed=True, alpha=0.25)
                ax.add_patch(poly)

    # Draw paths
    for name, traj in paths_dict.items():
        if traj is None or len(traj) < 2:
            continue
        ax.plot(traj[:, 0], traj[:, 1], label=name)

    # Mark start/goal
    if start_xy is not None:
        ax.scatter([start_xy[0]], [start_xy[1]], marker="o", s=60, label="start")
    if goal_xy is not None:
        ax.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=90, label="goal")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)

    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PointMaze_UMaze-v3")
    parser.add_argument("--ckpt", type=str, default="runs/umaze/ckpt_final.pt")
    parser.add_argument("--stats", type=str, default="runs/umaze/stats.json")

    parser.add_argument("--diff_steps", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=64)

    parser.add_argument("--n_trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)

    # compute budget fairness
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--diff_candidates", type=int, default=128)

    # MPPI params
    parser.add_argument("--mppi_sigma", type=float, default=0.5)
    parser.add_argument("--mppi_temp", type=float, default=1.0)

    # "projection" refinement around diffusion best
    parser.add_argument("--proj_iters", type=int, default=2)
    parser.add_argument("--proj_pop", type=int, default=64)
    parser.add_argument("--prox_lam", type=float, default=1.0)

    # output/visualization
    parser.add_argument("--outdir", type=str, default="eval_out")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--plot_every", type=int, default=1, help="Plot every k trials (if save_png/show).")
    parser.add_argument("--render", action="store_true", help="Use MuJoCo render window (slow).")

    # model arch (must match train.py settings you used)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--emb", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=8)

    args = parser.parse_args()

    assert args.env in MAZE_MAPS, f"Unknown env {args.env}. Choose from: {list(MAZE_MAPS.keys())}"
    os.makedirs(args.outdir, exist_ok=True)
    if args.save_npz:
        os.makedirs(os.path.join(args.outdir, "npz"), exist_ok=True)
    if args.save_png:
        os.makedirs(os.path.join(args.outdir, "png"), exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gym.register_envs(gymnasium_robotics)
    render_mode = "human" if args.render else None
    env = gym.make(args.env, continuing_task=False, render_mode=render_mode, max_episode_steps=args.horizon)

    maze_map = MAZE_MAPS[args.env]
    empties = free_cells(maze_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = load_stats(args.stats, device=device)

    act_dim = int(stats["act_mean"].shape[0])
    cond_dim = int(stats["cond_mean"].shape[0])

    model = DiffusionPolicy1D(
        act_dim=act_dim,
        cond_dim=cond_dim,
        hidden=args.hidden,
        emb_dim=args.emb,
        n_blocks=args.blocks
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    schedule = make_linear_beta_schedule(args.diff_steps, 1e-4, 2e-2, device=device)

    action_low = env.action_space.low
    action_high = env.action_space.high

    # Calibrate for plotting walls aligned to world coords
    px, py = fit_cell_to_world(env, maze_map, seed=args.seed + 999, n_cells=25)

    # Budget splitting
    mppi_pop = 64
    mppi_iters = max(1, args.budget // mppi_pop)

    diffN = min(args.diff_candidates, args.budget)

    proj_rollouts = args.proj_pop * args.proj_iters
    diffN_proj = max(1, args.budget - proj_rollouts)
    diffN_proj = min(diffN_proj, args.diff_candidates)

    results = {
        "mppi": {"succ": 0, "cost": [], "t": []},
        "diff": {"succ": 0, "cost": [], "t": []},
        "diff_proj": {"succ": 0, "cost": [], "t": []},
    }

    for trial in range(args.n_trials):
        # pick distinct start/goal cells
        s = empties[np.random.randint(len(empties))]
        g = empties[np.random.randint(len(empties))]
        while g == s:
            g = empties[np.random.randint(len(empties))]

        reset_opts = {
            "reset_cell": np.array(s, dtype=int),
            "goal_cell": np.array(g, dtype=int),
        }
        reset_seed = args.seed + 10000 + trial

        # get condition from reset
        obs, info = env.reset(seed=reset_seed, options=reset_opts)
        start = obs["observation"].astype(np.float32)
        goal = obs["desired_goal"].astype(np.float32)
        cond = np.concatenate([start, goal], axis=0)

        start_xy = obs["achieved_goal"].astype(np.float32)
        goal_xy = goal.astype(np.float32)

        # ----- MPPI -----
        t0 = time.time()
        u0 = np.zeros((args.horizon, act_dim), dtype=np.float32)
        u_mppi = mppi_refine(
            env, reset_opts, reset_seed,
            H=args.horizon, act_dim=act_dim,
            u_init=u0,
            n_rollouts=mppi_pop, iters=mppi_iters,
            sigma=args.mppi_sigma, temp=args.mppi_temp,
            action_low=action_low, action_high=action_high,
        )
        roll_mppi = rollout(env, reset_opts, reset_seed, u_mppi)
        c_mppi = rollout_cost(u_mppi, roll_mppi)
        results["mppi"]["succ"] += int(roll_mppi["success"])
        results["mppi"]["cost"].append(c_mppi)
        results["mppi"]["t"].append(time.time() - t0)

        # ----- Diffusion-only (best of N) -----
        t0 = time.time()
        U = ddpm_sample_actions(model, schedule, cond, args.horizon, act_dim, diffN, stats, action_low, action_high)
        best_c, best_u, best_roll = float("inf"), None, None
        for i in range(U.shape[0]):
            r = rollout(env, reset_opts, reset_seed, U[i])
            c = rollout_cost(U[i], r)
            if c < best_c:
                best_c, best_u, best_roll = c, U[i], r
        results["diff"]["succ"] += int(best_roll["success"])
        results["diff"]["cost"].append(best_c)
        results["diff"]["t"].append(time.time() - t0)

        # ----- Diffusion + "projection" (prox-MPPI refine) -----
        t0 = time.time()
        U = ddpm_sample_actions(model, schedule, cond, args.horizon, act_dim, diffN_proj, stats, action_low, action_high)
        best_c2, best_u2, best_roll2 = float("inf"), None, None
        for i in range(U.shape[0]):
            r = rollout(env, reset_opts, reset_seed, U[i])
            c = rollout_cost(U[i], r)
            if c < best_c2:
                best_c2, best_u2, best_roll2 = c, U[i], r

        u_ref = mppi_refine(
            env, reset_opts, reset_seed,
            H=args.horizon, act_dim=act_dim,
            u_init=best_u2,
            n_rollouts=args.proj_pop, iters=args.proj_iters,
            sigma=args.mppi_sigma * 0.5, temp=args.mppi_temp,
            action_low=action_low, action_high=action_high,
            prox_anchor=best_u2, prox_lam=args.prox_lam
        )
        roll_ref = rollout(env, reset_opts, reset_seed, u_ref)
        c_ref = rollout_cost(u_ref, roll_ref)
        results["diff_proj"]["succ"] += int(roll_ref["success"])
        results["diff_proj"]["cost"].append(c_ref)
        results["diff_proj"]["t"].append(time.time() - t0)

        print(f"[{trial+1:03d}/{args.n_trials}] "
              f"MPPI succ={roll_mppi['success']} cost={c_mppi:.2f} | "
              f"DIFF succ={best_roll['success']} cost={best_c:.2f} | "
              f"DIFF+PROJ succ={roll_ref['success']} cost={c_ref:.2f}")

        # Save per-trial artifacts
        if args.save_npz:
            np.savez(
                os.path.join(args.outdir, "npz", f"{args.env}_trial{trial:04d}.npz"),
                env=args.env,
                start_cell=np.array(s, dtype=np.int32),
                goal_cell=np.array(g, dtype=np.int32),

                u_mppi=u_mppi, traj_mppi=roll_mppi["traj_ag"], succ_mppi=roll_mppi["success"], cost_mppi=c_mppi,
                u_diff=best_u, traj_diff=best_roll["traj_ag"], succ_diff=best_roll["success"], cost_diff=best_c,
                u_proj=u_ref,  traj_proj=roll_ref["traj_ag"],  succ_proj=roll_ref["success"], cost_proj=c_ref,

                start_xy=start_xy, goal_xy=goal_xy,
            )

        # Plot paths
        do_plot = (args.save_png or args.show) and ((trial % max(1, args.plot_every)) == 0)
        if do_plot:
            title = (f"{args.env} trial {trial} | "
                     f"MPPI:{roll_mppi['success']} DIFF:{best_roll['success']} PROJ:{roll_ref['success']}")
            save_path = None
            if args.save_png:
                save_path = os.path.join(args.outdir, "png", f"{args.env}_trial{trial:04d}.png")

            plot_maze_and_paths(
                maze_map=maze_map,
                px=px, py=py,
                paths_dict={
                    "MPPI": roll_mppi["traj_ag"],
                    "Diffusion": best_roll["traj_ag"],
                    "Diff+Proj": roll_ref["traj_ag"],
                },
                start_xy=start_xy,
                goal_xy=goal_xy,
                title=title,
                save_path=save_path,
                show=args.show
            )

    # Summaries
    def summarize(tag):
        succ = results[tag]["succ"]
        costs = np.array(results[tag]["cost"], dtype=np.float32)
        times = np.array(results[tag]["t"], dtype=np.float32)
        return succ / args.n_trials, float(costs.mean()), float(times.mean())

    print("\n=== SUMMARY ===")
    for tag in ["mppi", "diff", "diff_proj"]:
        sr, mc, mt = summarize(tag)
        print(f"{tag:10s} | success={sr*100:.1f}% | mean_cost={mc:.3f} | mean_time={mt:.3f}s")

    env.close()


if __name__ == "__main__":
    main()