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

    for i, ray in enumerate(rays):
        p0 = ray.origin
        p1 = ray.origin + ray_length * ray.direction
    
        color = "tab:blue" if i < 36 else "tab:red"
    
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            linewidth=1,
            color=color,
        )
    
        ax.scatter(
            [p0[0]],
            [p0[1]],
            [p0[2]],
            s=15,
            color=color,
        )

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

    #ax.set_xlim(center[0] - radius, center[0] + radius)
    #ax.set_ylim(center[1] - radius, center[1] + radius)
    #ax.set_zlim(center[2] - radius, center[2] + radius)
    
    ax.set_xlim(-0.1, 0.04)
    ax.set_ylim(0.6, 0.85)
    ax.set_zlim(0.65, 0.8)

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

def plot_laser_ray_offset_debug_in_robot_base(
    run_data: dict,
    output_path: str | Path,
    local_ray_direction: np.ndarray | None = None,
    ray_length: float = 0.5,
    offset_arrow_scale: float = 1.0,
) -> Path:
    """
    Erstellt einen Debugplot für die Tool-Offset-Wirkung.

    Gezeigt wird:
        - Ray ohne Tool-Offset
        - Ray mit Tool-Offset
        - Verschiebungsvektor vom alten zum neuen Ray-Ursprung

    Der normale plot_laser_rays_in_robot_base bleibt unverändert.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory_config = run_data["run_metadata"]["scan"]["trajectory_config"]

    original_tool_offset = trajectory_config.get("tool_offset", None)

    trajectory_without_offset = dict(trajectory_config)
    trajectory_without_offset["tool_offset"] = {
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_deg": [0.0, 0.0, 0.0],
    }

    trajectory_with_offset = dict(trajectory_config)
    trajectory_with_offset["tool_offset"] = original_tool_offset

    run_data_without_offset = dict(run_data)
    run_data_with_offset = dict(run_data)

    run_data_without_offset["run_metadata"] = dict(run_data["run_metadata"])
    run_data_with_offset["run_metadata"] = dict(run_data["run_metadata"])

    run_data_without_offset["run_metadata"]["scan"] = dict(
        run_data["run_metadata"]["scan"]
    )
    run_data_with_offset["run_metadata"]["scan"] = dict(
        run_data["run_metadata"]["scan"]
    )

    run_data_without_offset["run_metadata"]["scan"]["trajectory_config"] = (
        trajectory_without_offset
    )
    run_data_with_offset["run_metadata"]["scan"]["trajectory_config"] = (
        trajectory_with_offset
    )

    rays_without_offset = build_laser_rays_robot_base_from_run_data(
        run_data=run_data_without_offset,
        local_direction=local_ray_direction,
    )

    rays_with_offset = build_laser_rays_robot_base_from_run_data(
        run_data=run_data_with_offset,
        local_direction=local_ray_direction,
    )

    if len(rays_without_offset) != len(rays_with_offset):
        raise ValueError(
            "Ray-Anzahl mit und ohne Offset ist unterschiedlich."
        )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    points_for_limits: list[np.ndarray] = []

    for ray_no, ray_off in zip(rays_without_offset, rays_with_offset):
        p_no = ray_no.origin
        p_off = ray_off.origin

        p_no_end = p_no + ray_length * ray_no.direction
        p_off_end = p_off + ray_length * ray_off.direction

        # Ray ohne Offset
        ax.plot(
            [p_no[0], p_no_end[0]],
            [p_no[1], p_no_end[1]],
            [p_no[2], p_no_end[2]],
            linewidth=1,
            linestyle="--",
            label="ohne Offset" if ray_no is rays_without_offset[0] else None,
        )

        # Ray mit Offset
        ax.plot(
            [p_off[0], p_off_end[0]],
            [p_off[1], p_off_end[1]],
            [p_off[2], p_off_end[2]],
            linewidth=1,
            label="mit Offset" if ray_off is rays_with_offset[0] else None,
        )

        # Ursprungspunkte
        ax.scatter([p_no[0]], [p_no[1]], [p_no[2]], s=12)
        ax.scatter([p_off[0]], [p_off[1]], [p_off[2]], s=18)

        # Offset-Vektor
        delta = p_off - p_no

        ax.quiver(
            p_no[0],
            p_no[1],
            p_no[2],
            delta[0] * offset_arrow_scale,
            delta[1] * offset_arrow_scale,
            delta[2] * offset_arrow_scale,
            arrow_length_ratio=0.25,
            linewidth=1.5,
        )

        if ray_off.frame_idx is not None:
            ax.text(
                p_off[0],
                p_off[1],
                p_off[2],
                str(ray_off.frame_idx),
                fontsize=7,
            )

        points_for_limits.extend([p_no, p_no_end, p_off, p_off_end])

    ax.set_title("Laser-Rays im Roboter-Basis-KS: Tool-Offset Debug")
    ax.set_xlabel("x_R [m]")
    ax.set_ylabel("y_R [m]")
    ax.set_zlabel("z_R [m]")
    ax.legend()

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

    return output_path

def plot_laser_offset_top_view_in_robot_base(
    run_data: dict,
    output_path: str | Path,
    local_ray_direction: np.ndarray | None = None,
) -> Path:
    """
    Plottet die Wirkung des Tool-Offsets als 2D-Draufsicht in z-Richtung.

    Gezeigt wird nur:
        - Startpunkt ohne Offset: (x, y)
        - Startpunkt mit Offset:  (x', y')
        - Verschiebungspfeil von ohne Offset zu mit Offset

    Die Laserstrahlen selbst werden nicht geplottet.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory_config = run_data["run_metadata"]["scan"]["trajectory_config"]
    original_tool_offset = trajectory_config.get("tool_offset", None)

    trajectory_without_offset = dict(trajectory_config)
    trajectory_without_offset["tool_offset"] = {
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_deg": [0.0, 0.0, 0.0],
    }

    trajectory_with_offset = dict(trajectory_config)
    trajectory_with_offset["tool_offset"] = original_tool_offset

    run_data_without_offset = dict(run_data)
    run_data_with_offset = dict(run_data)

    run_data_without_offset["run_metadata"] = dict(run_data["run_metadata"])
    run_data_with_offset["run_metadata"] = dict(run_data["run_metadata"])

    run_data_without_offset["run_metadata"]["scan"] = dict(
        run_data["run_metadata"]["scan"]
    )
    run_data_with_offset["run_metadata"]["scan"] = dict(
        run_data["run_metadata"]["scan"]
    )

    run_data_without_offset["run_metadata"]["scan"]["trajectory_config"] = (
        trajectory_without_offset
    )
    run_data_with_offset["run_metadata"]["scan"]["trajectory_config"] = (
        trajectory_with_offset
    )

    rays_without_offset = build_laser_rays_robot_base_from_run_data(
        run_data=run_data_without_offset,
        local_direction=local_ray_direction,
    )

    rays_with_offset = build_laser_rays_robot_base_from_run_data(
        run_data=run_data_with_offset,
        local_direction=local_ray_direction,
    )

    if len(rays_without_offset) != len(rays_with_offset):
        raise ValueError(
            "Ray-Anzahl mit und ohne Offset ist unterschiedlich."
        )

    fig, ax = plt.subplots(figsize=(9, 9))

    all_xy_points: list[np.ndarray] = []

    for ray_no, ray_off in zip(rays_without_offset, rays_with_offset):
        p_no = ray_no.origin
        p_off = ray_off.origin

        dx = p_off[0] - p_no[0]
        dy = p_off[1] - p_no[1]

        ax.scatter(
            p_no[0],
            p_no[1],
            marker="o",
            s=35,
            label="ohne Offset" if ray_no is rays_without_offset[0] else None,
        )

        ax.scatter(
            p_off[0],
            p_off[1],
            marker="x",
            s=45,
            label="mit Offset" if ray_off is rays_with_offset[0] else None,
        )

        ax.arrow(
            p_no[0],
            p_no[1],
            dx,
            dy,
            length_includes_head=True,
            head_width=0.01,
            head_length=0.015,
            linewidth=1.2,
        )

        if ray_off.frame_idx is not None:
            ax.text(
                p_off[0],
                p_off[1],
                str(ray_off.frame_idx),
                fontsize=7,
                ha="left",
                va="bottom",
            )

        all_xy_points.append(np.array([p_no[0], p_no[1]], dtype=float))
        all_xy_points.append(np.array([p_off[0], p_off[1]], dtype=float))

    points = np.asarray(all_xy_points, dtype=float)
    center = np.mean(points, axis=0)
    radius = float(np.max(np.linalg.norm(points - center, axis=1)))

    if radius <= 1e-12:
        radius = 1.0

    margin = 0.1 * radius
    radius = radius + margin

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    ax.set_title("Tool-Offset Debug: Draufsicht im Roboter-KS")
    ax.set_xlabel("x_R [m]")
    ax.set_ylabel("y_R [m]")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path

def plot_laser_ray_intersections_with_z_plane_in_robot_base(
    run_data: dict,
    output_path: str | Path,
    z_plane: float | list[float] | tuple[float, ...] | np.ndarray,
    local_ray_direction: np.ndarray | None = None,
    max_abs_t: float | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z_values = np.atleast_1d(np.asarray(z_plane, dtype=float))

    rays = build_laser_rays_robot_base_from_run_data(
        run_data=run_data,
        local_direction=local_ray_direction,
    )

    if len(rays) == 0:
        raise ValueError("Keine Rays zum Plotten vorhanden.")

    n = len(z_values)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.5 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    all_points_by_z: list[np.ndarray] = []
    all_frame_indices_by_z: list[list[int | None]] = []

    eps = 1e-12

    for z in z_values:
        points_xy: list[np.ndarray] = []
        frame_indices: list[int | None] = []

        for ray in rays:
            origin = np.asarray(ray.origin, dtype=float).reshape(3)
            direction = np.asarray(ray.direction, dtype=float).reshape(3)

            dz = direction[2]

            if abs(dz) <= eps:
                continue

            t = (float(z) - origin[2]) / dz

            if max_abs_t is not None and abs(t) > max_abs_t:
                continue

            p = origin + t * direction
            points_xy.append(np.array([p[0], p[1]], dtype=float))
            frame_indices.append(ray.frame_idx)

        if len(points_xy) == 0:
            all_points_by_z.append(np.empty((0, 2), dtype=float))
        else:
            all_points_by_z.append(np.asarray(points_xy, dtype=float))

        all_frame_indices_by_z.append(frame_indices)

    non_empty = [p for p in all_points_by_z if len(p) > 0]

    if len(non_empty) == 0:
        raise ValueError("Keine gültigen Schnittpunkte für die angegebenen z-Ebenen gefunden.")

    all_points = np.vstack(non_empty)
    center = np.mean(all_points, axis=0)
    radius = float(np.max(np.linalg.norm(all_points - center, axis=1)))

    if radius <= 1e-12:
        radius = 1.0

    radius *= 1.1

    for idx, z in enumerate(z_values):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        points = all_points_by_z[idx]
        frame_indices = all_frame_indices_by_z[idx]

        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], marker="o", s=25)

            for point, frame_idx in zip(points, frame_indices):
                if frame_idx is not None:
                    ax.text(
                        point[0],
                        point[1],
                        str(frame_idx),
                        fontsize=6,
                        ha="left",
                        va="bottom",
                    )

        ax.set_title(f"z = {float(z):.4f} m")
        ax.set_xlabel("x_R [m]")
        ax.set_ylabel("y_R [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)

    for idx in range(n, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis("off")

    fig.suptitle("Laserray-Schnittpunkte mit z-Ebenen im Roboter-KS")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path