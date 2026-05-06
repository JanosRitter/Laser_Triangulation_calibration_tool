from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import csv

from src.calibration.camera_rays import pixel_to_camera_ray
from src.calibration.laser_kinematics import laser_ray_in_camera
from src.calibration.parameterization import extrinsic_pose_from_vector


def _plot_ray(ax, origin, direction, length, label=None, linestyle="-"):
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = np.asarray(direction, dtype=float).reshape(3)
    direction = direction / np.linalg.norm(direction)

    p0 = origin
    p1 = origin + length * direction

    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        [p0[2], p1[2]],
        linestyle=linestyle,
        label=label,
    )


def plot_camera_and_laser_rays(
    calib_observations: list,
    intrinsics,
    params: np.ndarray,
    output_path: str | Path,
    max_rays: int = 100,
    ray_length: float = 1.0,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extrinsic_pose = extrinsic_pose_from_vector(params)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cam_origin = np.zeros(3, dtype=float)

    used_observations = calib_observations[:max_rays]

    for i, obs in enumerate(used_observations):
        cam_dir = pixel_to_camera_ray(obs.uv, intrinsics)

        laser_origin, laser_dir = laser_ray_in_camera(
            extrinsic_pose=extrinsic_pose,
            relative_pose=obs.relative_pose,
        )

        _plot_ray(
            ax,
            cam_origin,
            cam_dir,
            ray_length,
            label="camera ray" if i == 0 else None,
            linestyle="-",
        )

        _plot_ray(
            ax,
            laser_origin,
            laser_dir,
            ray_length,
            label="laser ray" if i == 0 else None,
            linestyle="--",
        )

        ax.scatter(
            [laser_origin[0]],
            [laser_origin[1]],
            [laser_origin[2]],
            s=20,
        )

        ax.text(
            laser_origin[0],
            laser_origin[1],
            laser_origin[2],
            str(obs.frame_idx),
            fontsize=8,
        )

    ax.scatter([0], [0], [0], s=80, marker="o", label="camera origin")

    ax.set_title("Kamera-Rays und Laser-Rays im Kamera-KS")
    ax.set_xlabel("x_C [m]")
    ax.set_ylabel("y_C [m]")
    ax.set_zlabel("z_C [m]")
    ax.legend()

    # halbwegs gleiche Skalierung
    all_points = []

    all_points.append(np.zeros(3))
    for obs in used_observations:
        cam_dir = pixel_to_camera_ray(obs.uv, intrinsics)
        all_points.append(cam_origin + ray_length * cam_dir)

        laser_origin, laser_dir = laser_ray_in_camera(
            extrinsic_pose=extrinsic_pose,
            relative_pose=obs.relative_pose,
        )
        all_points.append(laser_origin)
        all_points.append(laser_origin + ray_length * laser_dir)

    all_points = np.asarray(all_points)
    center = np.mean(all_points, axis=0)
    radius = np.max(np.linalg.norm(all_points - center, axis=1))

    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(-0.1, center[2] + radius)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path

def save_ray_debug_csv(
    calib_observations: list,
    intrinsics,
    params: np.ndarray,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extrinsic_pose = extrinsic_pose_from_vector(params)
    cam_origin = np.zeros(3, dtype=float)

    rows = []

    for obs in calib_observations:
        cam_dir = pixel_to_camera_ray(obs.uv, intrinsics)

        laser_origin, laser_dir = laser_ray_in_camera(
            extrinsic_pose=extrinsic_pose,
            relative_pose=obs.relative_pose,
        )

        rows.append({
            "frame_idx": int(obs.frame_idx),

            "uv_u": float(obs.uv[0]),
            "uv_v": float(obs.uv[1]),

            "cam_origin_x": float(cam_origin[0]),
            "cam_origin_y": float(cam_origin[1]),
            "cam_origin_z": float(cam_origin[2]),
            "cam_dir_x": float(cam_dir[0]),
            "cam_dir_y": float(cam_dir[1]),
            "cam_dir_z": float(cam_dir[2]),

            "laser_origin_x": float(laser_origin[0]),
            "laser_origin_y": float(laser_origin[1]),
            "laser_origin_z": float(laser_origin[2]),
            "laser_dir_x": float(laser_dir[0]),
            "laser_dir_y": float(laser_dir[1]),
            "laser_dir_z": float(laser_dir[2]),

            "rel_tx": float(obs.relative_pose.translation[0]),
            "rel_ty": float(obs.relative_pose.translation[1]),
            "rel_tz": float(obs.relative_pose.translation[2]),
        })

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path