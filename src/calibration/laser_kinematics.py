from __future__ import annotations

import numpy as np

from src.calibration.calibration_types import ExtrinsicPose, RelativePose


# ============================================================
# ROTATIONSMATRIZEN
# ============================================================
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


def euler_xyz_deg_to_matrix(
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> np.ndarray:
    """
    Gleiche Konvention wie im Simulations-Tool:

    - erst Rotation um X
    - dann um Y
    - dann um Z

    Gesamtrotation:
        R = Rz @ Ry @ Rx
    """
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    Rx = rotation_matrix_x(rx)
    Ry = rotation_matrix_y(ry)
    Rz = rotation_matrix_z(rz)

    return Rz @ Ry @ Rx


# ============================================================
# TRANSFORMATIONS-HILFSFUNKTIONEN
# ============================================================
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
    """
    Gleiches Verhalten wie im Simulations-Tool:
        T_result = T_a @ T_b
    """
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


def local_increment_to_transform(
    dx: float,
    dy: float,
    dz: float,
    drx_deg: float,
    dry_deg: float,
    drz_deg: float,
) -> np.ndarray:
    """
    Lokales Inkrement identisch zur Simulation.
    """
    return pose_xyzrpy_deg_to_transform(
        dx, dy, dz,
        drx_deg, dry_deg, drz_deg,
    )


# ============================================================
# ABSOLUTE UND RELATIVE POSEN
# ============================================================
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


def build_absolute_transforms_from_trajectory_config(trajectory_config: dict) -> list[np.ndarray]:
    """
    Rekonstruiert die absolute Posefolge aus trajectory_config
    exakt analog zur Simulation für den Typ 'local_increments'.

    Returns
    -------
    list[np.ndarray]
        Liste von 4x4-Transformationen T_i
    """
    trajectory_type = trajectory_config["type"]
    if trajectory_type != "local_increments":
        raise NotImplementedError(
            f"Trajektorientyp aktuell nicht unterstützt: {trajectory_type}"
        )

    start_pose = trajectory_config["start_pose"]
    include_start_pose = bool(trajectory_config.get("include_start_pose", True))
    increments = trajectory_config["increments"]

    T_current = pose_xyzrpy_deg_to_transform(
        x=float(start_pose[0]),
        y=float(start_pose[1]),
        z=float(start_pose[2]),
        rx_deg=float(start_pose[3]),
        ry_deg=float(start_pose[4]),
        rz_deg=float(start_pose[5]),
    )

    transforms: list[np.ndarray] = []

    if include_start_pose:
        transforms.append(T_current.copy())

    for step in increments:
        dx = float(step["dx_m"])
        dy = float(step["dy_m"])
        dz = float(step["dz_m"])
        drx = float(step["drx_deg"])
        dry = float(step["dry_deg"])
        drz = float(step["drz_deg"])
        repeat = int(step["repeat"])

        T_increment = local_increment_to_transform(
            dx=dx,
            dy=dy,
            dz=dz,
            drx_deg=drx,
            dry_deg=dry,
            drz_deg=drz,
        )

        for _ in range(repeat):
            T_current = compose_transforms(T_current, T_increment)
            transforms.append(T_current.copy())

    return transforms


def build_relative_poses_from_trajectory_config(trajectory_config: dict) -> list[RelativePose]:
    """
    Baut relative Posen ^L0 T_Li aus trajectory_config.

    Dazu wird zuerst die absolute Posefolge rekonstruiert und dann
    relativ zur Startpose normiert:

        ^L0 T_Li = inv(T_0) @ T_i
    """
    absolute_transforms = build_absolute_transforms_from_trajectory_config(trajectory_config)

    if len(absolute_transforms) == 0:
        return []

    T_0 = absolute_transforms[0]
    T_0_inv = invert_transform(T_0)

    relative_poses: list[RelativePose] = []

    for T_i in absolute_transforms:
        T_rel = compose_transforms(T_0_inv, T_i)

        relative_poses.append(
            RelativePose(
                translation=T_rel[:3, 3].copy(),
                rotation=T_rel[:3, :3].copy(),
            )
        )

    return relative_poses


# ============================================================
# LASERPOSE / LASERSTRAHL IM KAMERA-KS
# ============================================================
def laser_pose_in_camera(
    extrinsic_pose: ExtrinsicPose,
    relative_pose: RelativePose,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gegeben:
        ^C T_L0
        ^L0 T_Li

    Gesucht:
        ^C T_Li = ^C T_L0 @ ^L0 T_Li
    """
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
    """
    Berechnet Laserursprung und Laserrichtung im Kamera-KS.

    Standardannahme:
    - lokaler Ursprung = (0, 0, 0)
    - lokale Strahlrichtung = +z
    """
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