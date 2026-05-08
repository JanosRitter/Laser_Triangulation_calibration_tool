from __future__ import annotations

import numpy as np

from src.calibration.camera_rays import pixel_to_normalized_camera_coordinates
from src.calibration.laser_kinematics.robot_base_offsets import (
    build_absolute_transforms_from_robot_base_offsets,
)
from src.calibration.laser_kinematics.transforms import euler_xyz_deg_to_matrix


def ray_from_transform(
    T: np.ndarray,
    local_ray_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origin = T[:3, 3].copy()
    rotation = T[:3, :3]

    direction = rotation @ np.asarray(local_ray_direction, dtype=float).reshape(3)
    direction = direction / np.linalg.norm(direction)

    return origin, direction


def intersect_ray_with_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = np.asarray(direction, dtype=float).reshape(3)
    plane_point = np.asarray(plane_point, dtype=float).reshape(3)
    plane_normal = np.asarray(plane_normal, dtype=float).reshape(3)

    denom = float(np.dot(plane_normal, direction))

    if abs(denom) < 1e-12:
        return None, None

    t = float(np.dot(plane_normal, plane_point - origin) / denom)
    point = origin + t * direction

    return point, t


def project_uv_rows_to_debug_plane(
    uv_rows: list[dict],
    intrinsics,
    center: np.ndarray,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    projection_distance_m: float,
    base_x_axis_robot: np.ndarray = np.array([-1.0, 0.0, 0.0]),
    base_y_axis_robot: np.ndarray = np.array([0.0, -1.0, 0.0]),
) -> list[dict]:
    """
    Vereinfachte Debug-Projektion.

    Wichtig:
    - Das ist bewusst KEINE vollständige Kamera-Roboter-Extrinsik.
    - Die UV-Punkte werden zuerst zentral über camera_rays.py in normierte
      Kamerakoordinaten umgerechnet.
    - Danach werden diese normierten Koordinaten mit projection_distance_m
      auf eine Debug-Ebene im Roboter-KS skaliert.
    """
    center = np.asarray(center, dtype=float).reshape(3)

    rotation = euler_xyz_deg_to_matrix(rx_deg, ry_deg, rz_deg)

    base_x = np.asarray(base_x_axis_robot, dtype=float).reshape(3)
    base_y = np.asarray(base_y_axis_robot, dtype=float).reshape(3)

    base_x = base_x / np.linalg.norm(base_x)
    base_y = base_y / np.linalg.norm(base_y)

    x_axis = rotation @ base_x
    y_axis = rotation @ base_y

    points = []

    for row in uv_rows:
        uv = np.array([float(row["u"]), float(row["v"])], dtype=float)

        x_norm, y_norm = pixel_to_normalized_camera_coordinates(
            uv=uv,
            intrinsics=intrinsics,
        )

        x_local = x_norm * projection_distance_m
        y_local = y_norm * projection_distance_m

        point_robot = center + x_local * x_axis + y_local * y_axis

        points.append({
            "frame_idx": int(row["frame_idx"]),
            "u": float(row["u"]),
            "v": float(row["v"]),
            "x": float(point_robot[0]),
            "y": float(point_robot[1]),
            "z": float(point_robot[2]),
            "x_local": float(x_local),
            "y_local": float(y_local),
        })

    return points


def build_laser_intersections_with_debug_plane(
    run_data: dict,
    frame_indices: list[int],
    center: np.ndarray,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    local_ray_direction: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> list[dict]:
    trajectory_config = run_data["run_metadata"]["scan"]["trajectory_config"]
    transforms = build_absolute_transforms_from_robot_base_offsets(trajectory_config)

    rotation = euler_xyz_deg_to_matrix(rx_deg, ry_deg, rz_deg)
    plane_normal = rotation @ np.array([0.0, 0.0, 1.0], dtype=float)

    intersections = []

    for frame_idx in frame_indices:
        T = transforms[frame_idx]
        origin, direction = ray_from_transform(T, local_ray_direction)

        point, t = intersect_ray_with_plane(
            origin=origin,
            direction=direction,
            plane_point=center,
            plane_normal=plane_normal,
        )

        if point is None:
            intersections.append({
                "frame_idx": int(frame_idx),
                "valid": False,
                "ray_t": np.nan,
                "x": np.nan,
                "y": np.nan,
                "z": np.nan,
                "origin_x": float(origin[0]),
                "origin_y": float(origin[1]),
                "origin_z": float(origin[2]),
                "dir_x": float(direction[0]),
                "dir_y": float(direction[1]),
                "dir_z": float(direction[2]),
            })
            continue

        intersections.append({
            "frame_idx": int(frame_idx),
            "valid": True,
            "ray_t": float(t),
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
            "origin_x": float(origin[0]),
            "origin_y": float(origin[1]),
            "origin_z": float(origin[2]),
            "dir_x": float(direction[0]),
            "dir_y": float(direction[1]),
            "dir_z": float(direction[2]),
        })

    return intersections