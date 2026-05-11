from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.calibration.laser_rays import (
    Ray3D,
    build_laser_rays_robot_base_from_run_data,
)


def plot_laser_rays(
    rays: list[Ray3D],
    output_path: str | Path,
    title: str,
    axis_labels: tuple[str, str, str],
    ray_length: float = 0.5,
) -> Path:
    """
    Plottet eine Liste von 3D-Rays und speichert zusätzlich eine CSV-Datei.

    Dieses Modul rekonstruiert keine Rays selbst.
    Es visualisiert nur Rays, die aus src.calibration.* kommen.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(rays) == 0:
        raise ValueError("Keine Rays zum Plotten übergeben.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    points_for_limits: list[np.ndarray] = []
    rows: list[dict] = []

    for ray in rays:
        p0 = ray.origin
        p1 = ray.origin + ray_length * ray.direction

        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            linewidth=1,
        )

        ax.scatter([p0[0]], [p0[1]], [p0[2]], s=15)

        if ray.frame_idx is not None:
            ax.text(
                p0[0],
                p0[1],
                p0[2],
                str(ray.frame_idx),
                fontsize=7,
            )

        points_for_limits.append(p0)
        points_for_limits.append(p1)

        rows.append(
            {
                "frame_idx": ray.frame_idx,
                "origin_x": float(ray.origin[0]),
                "origin_y": float(ray.origin[1]),
                "origin_z": float(ray.origin[2]),
                "dir_x": float(ray.direction[0]),
                "dir_y": float(ray.direction[1]),
                "dir_z": float(ray.direction[2]),
                "end_x": float(p1[0]),
                "end_y": float(p1[1]),
                "end_z": float(p1[2]),
            }
        )

    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])

    points = np.asarray(points_for_limits, dtype=float)
    center = np.mean(points, axis=0)
    radius = float(np.max(np.linalg.norm(points - center, axis=1)))

    if radius <= 1e-12:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")
    save_laser_rays_csv(rays=rays, output_path=csv_path, ray_length=ray_length)

    return output_path


def save_laser_rays_csv(
    rays: list[Ray3D],
    output_path: str | Path,
    ray_length: float = 0.5,
) -> Path:
    """
    Speichert Ray-Ursprung, Richtung und Endpunkt als CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for ray in rays:
        p1 = ray.origin + ray_length * ray.direction

        rows.append(
            {
                "frame_idx": ray.frame_idx,
                "origin_x": float(ray.origin[0]),
                "origin_y": float(ray.origin[1]),
                "origin_z": float(ray.origin[2]),
                "dir_x": float(ray.direction[0]),
                "dir_y": float(ray.direction[1]),
                "dir_z": float(ray.direction[2]),
                "end_x": float(p1[0]),
                "end_y": float(p1[1]),
                "end_z": float(p1[2]),
            }
        )

    if len(rows) == 0:
        raise ValueError("Keine Rays zum Speichern übergeben.")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def plot_laser_rays_in_robot_base(
    run_data: dict,
    output_path: str | Path,
    local_ray_direction: np.ndarray | None = None,
    ray_length: float = 0.5,
) -> Path:
    """
    Debug-Wrapper für absolute Laserrays im Roboter-Basis-KS.

    Die Rekonstruktion erfolgt in src.calibration.laser_rays.
    """
    rays = build_laser_rays_robot_base_from_run_data(
        run_data=run_data,
        local_direction=local_ray_direction,
    )

    return plot_laser_rays(
        rays=rays,
        output_path=output_path,
        title="Laser-Rays im Roboter-Basis-KS",
        axis_labels=("x_R [m]", "y_R [m]", "z_R [m]"),
        ray_length=ray_length,
    )