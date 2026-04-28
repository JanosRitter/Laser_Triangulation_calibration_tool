from __future__ import annotations

import numpy as np

from src.calibration.calibration_types import ExtrinsicPose, RelativePose


def rotation_matrix_x(rx_rad: float) -> np.ndarray:
    c = np.cos(rx_rad)
    s = np.sin(rx_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c,   -s],
        [0.0, s,    c],
    ], dtype=float)


def rotation_matrix_y(ry_rad: float) -> np.ndarray:
    c = np.cos(ry_rad)
    s = np.sin(ry_rad)
    return np.array([
        [c,   0.0, s],
        [0.0, 1.0, 0.0],
        [-s,  0.0, c],
    ], dtype=float)


def rotation_matrix_z(rz_rad: float) -> np.ndarray:
    c = np.cos(rz_rad)
    s = np.sin(rz_rad)
    return np.array([
        [c,  -s,  0.0],
        [s,   c,  0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def euler_xyz_deg_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    Rx = rotation_matrix_x(rx)
    Ry = rotation_matrix_y(ry)
    Rz = rotation_matrix_z(rz)

    return Rz @ Ry @ Rx


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    translation = np.asarray(translation, dtype=float).reshape(3)

    T = np.eye(4, dtype=float)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(4, 4)

    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def compose_transforms(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    T_a = np.asarray(T_a, dtype=float).reshape(4, 4)
    T_b = np.asarray(T_b, dtype=float).reshape(4, 4)
    return T_a @ T_b


def pose_xyzrpy_deg_to_transform(
    x: float,
    y: float,
    z: float,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> np.ndarray:
    rotation = euler_xyz_deg_to_matrix(rx_deg, ry_deg, rz_deg)
    translation = np.array([x, y, z], dtype=float)
    return make_transform(rotation, translation)


def build_extrinsic_pose_from_xyzrpy_deg(
    x: float,
    y: float,
    z: float,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> ExtrinsicPose:
    rotation = euler_xyz_deg_to_matrix(rx_deg, ry_deg, rz_deg)
    translation = np.array([x, y, z], dtype=float)
    return ExtrinsicPose(
        translation=translation,
        rotation=rotation,
    )


def laser_pose_in_camera(
    extrinsic_pose: ExtrinsicPose,
    relative_pose: RelativePose,
) -> tuple[np.ndarray, np.ndarray]:
    T_c_l0 = make_transform(extrinsic_pose.rotation, extrinsic_pose.translation)
    T_l0_li = make_transform(relative_pose.rotation, relative_pose.translation)

    T_c_li = compose_transforms(T_c_l0, T_l0_li)

    R_c_li = T_c_li[:3, :3].copy()
    t_c_li = T_c_li[:3, 3].copy()

    return R_c_li, t_c_li


def laser_ray_in_camera(
    extrinsic_pose: ExtrinsicPose,
    relative_pose: RelativePose,
    local_ray_origin: np.ndarray | None = None,
    local_ray_direction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if local_ray_origin is None:
        local_ray_origin = np.array([0.0, 0.0, 0.0], dtype=float)
    if local_ray_direction is None:
        local_ray_direction = np.array([0.0, 0.0, 1.0], dtype=float)

    local_ray_origin = np.asarray(local_ray_origin, dtype=float).reshape(3)
    local_ray_direction = np.asarray(local_ray_direction, dtype=float).reshape(3)

    local_ray_direction_norm = np.linalg.norm(local_ray_direction)
    if local_ray_direction_norm < 1e-12:
        raise ValueError("local_ray_direction hat Norm 0.")

    local_ray_direction = local_ray_direction / local_ray_direction_norm

    R_c_li, t_c_li = laser_pose_in_camera(
        extrinsic_pose=extrinsic_pose,
        relative_pose=relative_pose,
    )

    ray_origin_c = R_c_li @ local_ray_origin + t_c_li
    ray_direction_c = R_c_li @ local_ray_direction

    ray_direction_norm = np.linalg.norm(ray_direction_c)
    if ray_direction_norm < 1e-12:
        raise ValueError("Transformierte Laserrichtung hat Norm 0.")

    ray_direction_c = ray_direction_c / ray_direction_norm

    return ray_origin_c, ray_direction_c