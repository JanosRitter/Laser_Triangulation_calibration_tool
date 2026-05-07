from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.calibration.laser_kinematics.robot_base_offsets import (
    build_absolute_transforms_from_robot_base_offsets,
)


def _ray_from_transform(T: np.ndarray, local_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin = T[:3, 3].copy()
    R = T[:3, :3]

    direction = R @ np.asarray(local_direction, dtype=float).reshape(3)
    direction = direction / np.linalg.norm(direction)

    return origin, direction


def plot_laser_rays_in_robot_base(
    run_data: dict,
    output_path: str | Path,
    local_ray_direction: np.ndarray | None = None,
    ray_length: float = 0.5,
) -> Path:
    if local_ray_direction is None:
        local_ray_direction = np.array([0.0, 1.0, 0.0], dtype=float)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory_config = run_data["run_metadata"]["scan"]["trajectory_config"]
    transforms = build_absolute_transforms_from_robot_base_offsets(trajectory_config)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    points_for_limits = []

    rows = []

    for i, T in enumerate(transforms):
        origin, direction = _ray_from_transform(T, local_ray_direction)

        p0 = origin
        p1 = origin + ray_length * direction

        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            linewidth=1,
        )

        ax.scatter([p0[0]], [p0[1]], [p0[2]], s=15)
        ax.text(p0[0], p0[1], p0[2], str(i), fontsize=7)

        points_for_limits.append(p0)
        points_for_limits.append(p1)

        rows.append({
            "frame_idx": i,
            "origin_x": float(origin[0]),
            "origin_y": float(origin[1]),
            "origin_z": float(origin[2]),
            "dir_x": float(direction[0]),
            "dir_y": float(direction[1]),
            "dir_z": float(direction[2]),
            "end_x": float(p1[0]),
            "end_y": float(p1[1]),
            "end_z": float(p1[2]),
        })

    ax.set_title("Laser-Rays im Roboter-Basis-KS")
    ax.set_xlabel("x_R [m]")
    ax.set_ylabel("y_R [m]")
    ax.set_zlabel("z_R [m]")

    points = np.asarray(points_for_limits)
    center = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1))

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path