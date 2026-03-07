import os
import glob
import json
import math
import random
import argparse
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_samples(root, split):
    files = sorted(glob.glob(os.path.join(root, split, "*.npz")))
    samples = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        info = json.loads(str(d["info_json"])) if "info_json" in d else {}
        samples.append({
            "file": f,
            "occupancy": d["occupancy"],
            "sdf": d["sdf"],
            "start": d["start"],
            "goal": d["goal"],
            "controls": d["controls"],
            "states": d["states"],
            "scenario_type": str(d["scenario_type"]),
            "info": info,
        })
    return samples


def draw_pose(ax, pose, color, label=None, scale=0.25):
    x, y, th = pose
    ax.plot(x, y, "o", color=color, markersize=6, label=label)
    ax.arrow(
        x, y,
        scale * np.cos(th),
        scale * np.sin(th),
        head_width=0.10,
        head_length=0.12,
        fc=color,
        ec=color,
        length_includes_head=True,
    )


def summarize(samples):
    counts = Counter(s["scenario_type"] for s in samples)

    dists = []
    pos_errs = []
    theta_errs = []
    clears = []
    traj_lengths = []

    for s in samples:
        start = s["start"]
        goal = s["goal"]
        states = s["states"]
        info = s["info"]

        dists.append(np.linalg.norm(start[:2] - goal[:2]))
        pos_errs.append(info.get("pos_err", np.nan))
        theta_errs.append(np.degrees(info.get("theta_err_rad", np.nan)))
        clears.append(info.get("min_clearance", np.nan))
        traj_lengths.append(np.sum(np.linalg.norm(states[1:, :2] - states[:-1, :2], axis=1)))

    print("\n=== Dataset summary ===")
    print(f"num samples: {len(samples)}")
    print("scenario counts:", dict(counts))
    print(f"start-goal dist: mean={np.nanmean(dists):.3f}, std={np.nanstd(dists):.3f}")
    print(f"pos err:         mean={np.nanmean(pos_errs):.3f}, std={np.nanstd(pos_errs):.3f}")
    print(f"theta err deg:   mean={np.nanmean(theta_errs):.3f}, std={np.nanstd(theta_errs):.3f}")
    print(f"min clearance:   mean={np.nanmean(clears):.3f}, std={np.nanstd(clears):.3f}")
    print(f"traj length:     mean={np.nanmean(traj_lengths):.3f}, std={np.nanstd(traj_lengths):.3f}")


def plot_random_trajectories(samples, map_size_m=10.0, n=9, scenario=None):
    if scenario is not None:
        samples = [s for s in samples if s["scenario_type"] == scenario]
    if len(samples) == 0:
        print(f"No samples found for scenario={scenario}")
        return

    chosen = random.sample(samples, min(n, len(samples)))
    rows = int(math.ceil(len(chosen) / 3))
    cols = min(3, len(chosen))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, s in zip(axes, chosen):
        occ = s["occupancy"]
        states = s["states"]
        start = s["start"]
        goal = s["goal"]
        info = s["info"]

        ax.imshow(
            occ,
            cmap="gray_r",
            origin="lower",
            extent=[0, map_size_m, 0, map_size_m],
        )
        ax.plot(states[:, 0], states[:, 1], linewidth=2)
        draw_pose(ax, start, "green", "start")
        draw_pose(ax, goal, "red", "goal")

        ax.set_title(
            f"{s['scenario_type']} | "
            f"pos_err={info.get('pos_err', np.nan):.2f} | "
            f"clear={info.get('min_clearance', np.nan):.2f}"
        )
        ax.set_xlim(0, map_size_m)
        ax.set_ylim(0, map_size_m)
        ax.set_aspect("equal")

    for ax in axes[len(chosen):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_histograms(samples):
    dists = []
    pos_errs = []
    theta_errs = []
    clears = []
    traj_lengths = []
    max_vs = []
    max_ws = []

    for s in samples:
        start = s["start"]
        goal = s["goal"]
        states = s["states"]
        controls = s["controls"]
        info = s["info"]

        dists.append(np.linalg.norm(start[:2] - goal[:2]))
        pos_errs.append(info.get("pos_err", np.nan))
        theta_errs.append(np.degrees(info.get("theta_err_rad", np.nan)))
        clears.append(info.get("min_clearance", np.nan))
        traj_lengths.append(np.sum(np.linalg.norm(states[1:, :2] - states[:-1, :2], axis=1)))
        max_vs.append(np.max(np.abs(controls[:, 0])))
        max_ws.append(np.max(np.abs(controls[:, 1])))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    axes[0].hist(dists, bins=30)
    axes[0].set_title("Start-goal distance")

    axes[1].hist(pos_errs, bins=30)
    axes[1].set_title("Final position error")

    axes[2].hist(theta_errs, bins=30)
    axes[2].set_title("Final heading error (deg)")

    axes[3].hist(clears, bins=30)
    axes[3].set_title("Min clearance")

    axes[4].hist(traj_lengths, bins=30)
    axes[4].set_title("Trajectory length")

    axes[5].hist(max_vs, bins=30, alpha=0.7, label="max |v|")
    axes[5].hist(max_ws, bins=30, alpha=0.7, label="max |w|")
    axes[5].set_title("Control saturation")
    axes[5].legend()

    plt.tight_layout()
    plt.show()


def plot_controls(samples, n=6, scenario=None):
    if scenario is not None:
        samples = [s for s in samples if s["scenario_type"] == scenario]
    if len(samples) == 0:
        print(f"No samples found for scenario={scenario}")
        return

    chosen = random.sample(samples, min(n, len(samples)))

    fig, axes = plt.subplots(len(chosen), 2, figsize=(10, 3 * len(chosen)))
    if len(chosen) == 1:
        axes = np.array([axes])

    for i, s in enumerate(chosen):
        u = s["controls"]
        axes[i, 0].plot(u[:, 0])
        axes[i, 0].set_title(f"{s['scenario_type']} | v")
        axes[i, 1].plot(u[:, 1])
        axes[i, 1].set_title(f"{s['scenario_type']} | omega")

    plt.tight_layout()
    plt.show()


def plot_per_scenario_montage(samples, map_size_m=10.0):
    grouped = defaultdict(list)
    for s in samples:
        grouped[s["scenario_type"]].append(s)

    scenarios = sorted(grouped.keys())
    if len(scenarios) == 0:
        return

    fig, axes = plt.subplots(len(scenarios), 3, figsize=(12, 4 * len(scenarios)))
    if len(scenarios) == 1:
        axes = np.array([axes])

    for r, sc in enumerate(scenarios):
        chosen = random.sample(grouped[sc], min(3, len(grouped[sc])))
        for c in range(3):
            ax = axes[r, c]
            if c >= len(chosen):
                ax.axis("off")
                continue

            s = chosen[c]
            ax.imshow(
                s["occupancy"],
                cmap="gray_r",
                origin="lower",
                extent=[0, map_size_m, 0, map_size_m],
            )
            ax.plot(s["states"][:, 0], s["states"][:, 1], linewidth=2)
            draw_pose(ax, s["start"], "green")
            draw_pose(ax, s["goal"], "red")
            ax.set_xlim(0, map_size_m)
            ax.set_ylim(0, map_size_m)
            ax.set_aspect("equal")
            ax.set_title(f"{sc}")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="diffdrive_dataset")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--map_size_m", type=float, default=10.0)
    parser.add_argument("--scenario", type=str, default=None)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    samples = load_samples(args.root, args.split)
    summarize(samples)
    plot_histograms(samples)
    plot_per_scenario_montage(samples, map_size_m=args.map_size_m)
    plot_random_trajectories(samples, map_size_m=args.map_size_m, n=9, scenario=args.scenario)
    plot_controls(samples, n=6, scenario=args.scenario)


if __name__ == "__main__":
    main()