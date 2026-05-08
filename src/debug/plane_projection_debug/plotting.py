from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_uv_projection_vs_laser_intersections(
    output_path: str | Path,
    uv_points_robot: list[dict],
    laser_intersections: list[dict],
    xlim: tuple[float, float] = (-0.09, -0.05),
    ylim: tuple[float, float] = (0.94, 1.0),
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    laser_by_frame = {r["frame_idx"]: r for r in laser_intersections}

    fig, ax = plt.subplots(figsize=(8, 8))

    uv_x = []
    uv_y = []
    laser_x = []
    laser_y = []

    for uv in uv_points_robot:
        frame_idx = uv["frame_idx"]
        laser = laser_by_frame.get(frame_idx)

        if laser is None or not laser["valid"]:
            continue

        uv_x.append(uv["x"])
        uv_y.append(uv["y"])
        laser_x.append(laser["x"])
        laser_y.append(laser["y"])

        ax.plot(
            [uv["x"], laser["x"]],
            [uv["y"], laser["y"]],
            linewidth=0.8,
            alpha=0.6,
        )

        ax.text(
            uv["x"],
            uv["y"],
            str(frame_idx),
            fontsize=7,
        )

    ax.scatter(uv_x, uv_y, marker="o", s=35, label="UV → Debug-Ebene")
    ax.scatter(laser_x, laser_y, marker="x", s=45, label="Laser-Ray ∩ Debug-Ebene")

    ax.set_title("UV-Projektion vs. Laserray-Schnittpunkte auf Debug-Ebene")
    ax.set_xlabel("x_R [m]")
    ax.set_ylabel("y_R [m]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path