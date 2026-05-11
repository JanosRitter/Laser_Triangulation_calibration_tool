from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.calibration.laser_rays import Ray3D
from src.calibration.ray_geometry import closest_points_between_rays


def _ray_end(ray: Ray3D, ray_length: float) -> np.ndarray:
    return ray.origin + ray_length * ray.direction


def save_ray_pair_debug_csv(
    laser_rays_R: list[Ray3D],
    camera_rays_R: list[Ray3D],
    output_path: str | Path,
    laser_ray_length: float = 0.3,
    camera_ray_length: float = 1.0,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(laser_rays_R) != len(camera_rays_R):
        raise ValueError(
            "laser_rays_R und camera_rays_R müssen dieselbe Länge haben."
        )

    rows: list[dict] = []

    for i, (laser_ray, camera_ray) in enumerate(zip(laser_rays_R, camera_rays_R)):
        laser_end = _ray_end(laser_ray, laser_ray_length)
        camera_end = _ray_end(camera_ray, camera_ray_length)

        closest = closest_points_between_rays(
            origin_a=laser_ray.origin,
            direction_a=laser_ray.direction,
            origin_b=camera_ray.origin,
            direction_b=camera_ray.direction,
        )

        rows.append(
            {
                "idx": i,
                "frame_idx_laser": laser_ray.frame_idx,
                "frame_idx_camera": camera_ray.frame_idx,
                "laser_origin_x": float(laser_ray.origin[0]),
                "laser_origin_y": float(laser_ray.origin[1]),
                "laser_origin_z": float(laser_ray.origin[2]),
                "laser_dir_x": float(laser_ray.direction[0]),
                "laser_dir_y": float(laser_ray.direction[1]),
                "laser_dir_z": float(laser_ray.direction[2]),
                "laser_end_x": float(laser_end[0]),
                "laser_end_y": float(laser_end[1]),
                "laser_end_z": float(laser_end[2]),
                "camera_origin_x": float(camera_ray.origin[0]),
                "camera_origin_y": float(camera_ray.origin[1]),
                "camera_origin_z": float(camera_ray.origin[2]),
                "camera_dir_x": float(camera_ray.direction[0]),
                "camera_dir_y": float(camera_ray.direction[1]),
                "camera_dir_z": float(camera_ray.direction[2]),
                "camera_end_x": float(camera_end[0]),
                "camera_end_y": float(camera_end[1]),
                "camera_end_z": float(camera_end[2]),
                "lambda_laser": float(closest["lambda_a"]),
                "lambda_camera": float(closest["lambda_b"]),
                "closest_laser_x": float(closest["point_a"][0]),
                "closest_laser_y": float(closest["point_a"][1]),
                "closest_laser_z": float(closest["point_a"][2]),
                "closest_camera_x": float(closest["point_b"][0]),
                "closest_camera_y": float(closest["point_b"][1]),
                "closest_camera_z": float(closest["point_b"][2]),
                "midpoint_x": float(closest["midpoint"][0]),
                "midpoint_y": float(closest["midpoint"][1]),
                "midpoint_z": float(closest["midpoint"][2]),
                "distance_m": float(closest["distance"]),
            }
        )

    if len(rows) == 0:
        raise ValueError("Keine Ray-Paare zum Speichern übergeben.")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def plot_ray_pairs_in_robot_frame(
    laser_rays_R: list[Ray3D],
    camera_rays_R: list[Ray3D],
    output_path: str | Path,
    laser_ray_length: float = 0.3,
    camera_ray_length: float = 0.4,
    max_pairs: int | None = 100,
    draw_closest_segments: bool = True,
    annotate_indices: bool = True,
) -> Path:
    """
    Plottet Laserrays und transformierte Kamerarays gemeinsam im Roboter-KS.

    Erwartung:
        laser_rays_R:
            Laserrays im Roboterbasis-KS

        camera_rays_R:
            Kamerarays nach Anwendung der groben Kamerapose im Roboterbasis-KS
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(laser_rays_R) != len(camera_rays_R):
        raise ValueError(
            "laser_rays_R und camera_rays_R müssen dieselbe Länge haben."
        )

    if len(laser_rays_R) == 0:
        raise ValueError("Keine Ray-Paare zum Plotten übergeben.")

    if max_pairs is not None:
        laser_rays_plot = laser_rays_R[:max_pairs]
        camera_rays_plot = camera_rays_R[:max_pairs]
    else:
        laser_rays_plot = laser_rays_R
        camera_rays_plot = camera_rays_R

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    points_for_limits: list[np.ndarray] = []

    for i, (laser_ray, camera_ray) in enumerate(zip(laser_rays_plot, camera_rays_plot)):
        laser_p0 = laser_ray.origin
        laser_p1 = _ray_end(laser_ray, laser_ray_length)

        camera_p0 = camera_ray.origin
        camera_p1 = _ray_end(camera_ray, camera_ray_length)

        # Laser-Ray
        ax.plot(
            [laser_p0[0], laser_p1[0]],
            [laser_p0[1], laser_p1[1]],
            [laser_p0[2], laser_p1[2]],
            linewidth=1,
        )
        ax.scatter([laser_p0[0]], [laser_p0[1]], [laser_p0[2]], s=12)

        # Kamera-Ray
        ax.plot(
            [camera_p0[0], camera_p1[0]],
            [camera_p0[1], camera_p1[1]],
            [camera_p0[2], camera_p1[2]],
            linewidth=1,
            linestyle="--",
        )

        if i == 0:
            ax.scatter(
                [camera_p0[0]],
                [camera_p0[1]],
                [camera_p0[2]],
                s=50,
            )
            ax.text(
                camera_p0[0],
                camera_p0[1],
                camera_p0[2],
                "camera",
                fontsize=9,
            )

        if draw_closest_segments:
            closest = closest_points_between_rays(
                origin_a=laser_ray.origin,
                direction_a=laser_ray.direction,
                origin_b=camera_ray.origin,
                direction_b=camera_ray.direction,
            )

            pa = closest["point_a"]
            pb = closest["point_b"]

            ax.plot(
                [pa[0], pb[0]],
                [pa[1], pb[1]],
                [pa[2], pb[2]],
                linewidth=0.8,
                linestyle=":",
            )

            points_for_limits.append(pa)
            points_for_limits.append(pb)

        if annotate_indices:
            label = (
                str(laser_ray.frame_idx)
                if laser_ray.frame_idx is not None
                else str(i)
            )
            ax.text(laser_p1[0], laser_p1[1], laser_p1[2], label, fontsize=7)

        points_for_limits.extend([laser_p0, laser_p1, camera_p0, camera_p1])

    ax.set_title("Laser- und Kamera-Ray-Paare im Roboter-KS")
    ax.set_xlabel("x_R [m]")
    ax.set_ylabel("y_R [m]")
    ax.set_zlabel("z_R [m]")

    points = np.asarray(points_for_limits, dtype=float)
    center = np.mean(points, axis=0)
    radius = float(np.max(np.linalg.norm(points - center, axis=1)))

    if radius <= 1e-12:
        radius = 1.0

    # Achsenlimits bewusst primär aus Laserrays ableiten.
    # Die Kamerarays können sonst wegen der langen Blickstrahlen
    # den Plot unnötig groß ziehen.
    laser_points = []

    for laser_ray in laser_rays_plot:
        laser_p0 = laser_ray.origin
        laser_p1 = _ray_end(laser_ray, laser_ray_length)
        laser_points.extend([laser_p0, laser_p1])

    laser_points = np.asarray(laser_points, dtype=float)

    laser_min = np.min(laser_points, axis=0)
    laser_max = np.max(laser_points, axis=0)

    center = 0.5 * (laser_min + laser_max)
    span = laser_max - laser_min

    # etwas Luft um die Laser-Geometrie
    margin = 0.5
    span = span + 1.0 * margin

    # Mindestgröße, damit flache Geometrien nicht kollabieren
    span = np.maximum(span, 0.30)

    ax.set_xlim(center[0] - span[0] / 2.0, center[0] + span[0] / 2.0)
    ax.set_ylim(center[1] - span[1] / 2.0, center[1] + span[1] / 2.0)
    ax.set_zlim(center[2] - span[2] / 2.0, center[2] + span[2] / 2.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")
    save_ray_pair_debug_csv(
        laser_rays_R=laser_rays_R,
        camera_rays_R=camera_rays_R,
        output_path=csv_path,
        laser_ray_length=laser_ray_length,
        camera_ray_length=camera_ray_length,
    )

    return output_path