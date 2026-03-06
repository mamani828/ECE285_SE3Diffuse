import json
import sys
import numpy as np
import matplotlib.pyplot as plt


def draw_pose(ax, pose, color, label, scale=0.35):
    x, y, th = pose
    ax.plot(x, y, "o", color=color, markersize=8, label=label)
    ax.arrow(
        x, y,
        scale * np.cos(th),
        scale * np.sin(th),
        head_width=0.12,
        head_length=0.16,
        fc=color,
        ec=color,
        length_includes_head=True,
    )


def main(path):
    data = np.load(path, allow_pickle=True)

    occupancy = data["occupancy"]      # (H, W)
    start = data["start"]              # (3,)
    goal = data["goal"]                # (3,)
    states = data["states"]            # (T+1, 3)
    controls = data["controls"]        # (T, 2)
    scenario_type = str(data["scenario_type"])
    info = json.loads(str(data["info_json"]))

    H, W = occupancy.shape
    map_size_m = 10.0  # must match dataset config

    fig, ax = plt.subplots(figsize=(7, 7))

    # show occupancy map
    ax.imshow(
        occupancy,
        cmap="gray_r",
        origin="lower",
        extent=[0, map_size_m, 0, map_size_m],
    )

    # trajectory
    ax.plot(states[:, 0], states[:, 1], linewidth=2, label="trajectory")

    # sparse heading arrows along path
    step = max(1, len(states) // 12)
    for i in range(0, len(states), step):
        x, y, th = states[i]
        ax.arrow(
            x, y,
            0.18 * np.cos(th),
            0.18 * np.sin(th),
            head_width=0.08,
            head_length=0.10,
            fc="tab:blue",
            ec="tab:blue",
            alpha=0.7,
            length_includes_head=True,
        )

    draw_pose(ax, start, "green", "start")
    draw_pose(ax, goal, "red", "goal")

    ax.set_title(
        f"{scenario_type} | "
        f"pos_err={info['pos_err']:.2f}, "
        f"clear={info['min_clearance']:.2f}"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(0, map_size_m)
    ax.set_ylim(0, map_size_m)
    ax.set_aspect("equal")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # optional: print controls summary
    print("scenario_type:", scenario_type)
    print("start:", start)
    print("goal:", goal)
    print("controls shape:", controls.shape)
    print("states shape:", states.shape)
    print("info:", info)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python view_sample.py path/to/sample.npz")
        sys.exit(1)
    main(sys.argv[1])