# src/calibration/initialization.py
from __future__ import annotations

import numpy as np

from src.calibration.laser_kinematics.transforms import (
    make_transform,
    invert_transform,
)


def _normalize_vector(v: np.ndarray, label: str) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    norm = float(np.linalg.norm(v))

    if norm < 1e-12:
        raise ValueError(f"{label} darf kein Nullvektor sein.")

    return v / norm


def _matrix_to_euler_xyz_deg(R: np.ndarray) -> np.ndarray:
    """
    Inverse zu euler_xyz_deg_to_matrix aus transforms.py.

    Konvention:
        R = Rz @ Ry @ Rx

    Returns:
        [rx_deg, ry_deg, rz_deg]
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)

    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    ry = np.arcsin(sy)

    cy = np.cos(ry)

    if abs(cy) > 1e-12:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal-Lock-Fallback
        rx = 0.0
        rz = np.arctan2(-R[0, 1], R[1, 1])

    return np.rad2deg(np.array([rx, ry, rz], dtype=float))


def build_camera_pose_in_laser_frame(
    camera_position_L0: np.ndarray,
    camera_z_axis_L0: np.ndarray,
    camera_x_axis_L0: np.ndarray,
) -> np.ndarray:
    """
    Baut ^L0 T_C aus einer groben Kamera-Pose im Laserstart-KS.

    Bedeutung:
        p_L0 = R_L0_C @ p_C + t_L0_C

    camera_position_L0:
        Position des Kameraursprungs im Laserstart-/Roboter-KS.

    camera_z_axis_L0:
        Blickrichtung der Kamera, ausgedrückt im Laserstart-/Roboter-KS.
        Da im Kamera-KS die Blickrichtung +z_C ist, ist dies die z_C-Achse in L0.

    camera_x_axis_L0:
        x_C-Achse der Kamera, ausgedrückt im Laserstart-/Roboter-KS.

    Die y_C-Achse wird daraus rechtshändig konstruiert.
    """
    t_L0_C = np.asarray(camera_position_L0, dtype=float).reshape(3)

    z_axis = _normalize_vector(camera_z_axis_L0, "camera_z_axis_L0")
    x_axis_raw = _normalize_vector(camera_x_axis_L0, "camera_x_axis_L0")

    # x-Achse orthogonal zu z machen
    x_axis = x_axis_raw - np.dot(x_axis_raw, z_axis) * z_axis
    x_axis = _normalize_vector(x_axis, "orthogonalisierte camera_x_axis_L0")

    # Rechtshändiges Kamera-KS:
    # x_C × y_C = z_C  =>  y_C = z_C × x_C
    y_axis = np.cross(z_axis, x_axis)
    y_axis = _normalize_vector(y_axis, "camera_y_axis_L0")

    # Spalten sind die Kameraachsen ausgedrückt in L0:
    # p_L0 = R_L0_C @ p_C + t_L0_C
    R_L0_C = np.column_stack([x_axis, y_axis, z_axis])

    return make_transform(
        rotation=R_L0_C,
        translation=t_L0_C,
    )


def initial_guess_from_camera_pose_in_laser_frame(
    camera_position_L0: np.ndarray,
    camera_z_axis_L0: np.ndarray,
    camera_x_axis_L0: np.ndarray,
) -> np.ndarray:
    """
    Erzeugt einen Solver-Startvektor aus einer groben Kamera-Pose im Laser-KS.

    Input:
        ^L0 T_C, beschrieben durch Kameraursprung und Kameraachsen in L0.

    Solver erwartet:
        ^C T_L0

    Daher:
        ^C T_L0 = inverse(^L0 T_C)

    Returns:
        np.ndarray shape (6,)
        [x, y, z, rx_deg, ry_deg, rz_deg]
    """
    T_L0_C = build_camera_pose_in_laser_frame(
        camera_position_L0=camera_position_L0,
        camera_z_axis_L0=camera_z_axis_L0,
        camera_x_axis_L0=camera_x_axis_L0,
    )

    T_C_L0 = invert_transform(T_L0_C)

    R_C_L0 = T_C_L0[:3, :3]
    t_C_L0 = T_C_L0[:3, 3]

    rpy_deg = _matrix_to_euler_xyz_deg(R_C_L0)

    params = np.array([
        float(t_C_L0[0]),
        float(t_C_L0[1]),
        float(t_C_L0[2]),
        float(rpy_deg[0]),
        float(rpy_deg[1]),
        float(rpy_deg[2]),
    ], dtype=float)

    return params


def print_initial_guess(params: np.ndarray, label: str = "Initial Guess") -> None:
    params = np.asarray(params, dtype=float).reshape(6)

    print(f"\n🎯 {label}:")
    print(f"  x  = {params[0]:+.10f} m")
    print(f"  y  = {params[1]:+.10f} m")
    print(f"  z  = {params[2]:+.10f} m")
    print(f"  rx = {params[3]:+.10f} deg")
    print(f"  ry = {params[4]:+.10f} deg")
    print(f"  rz = {params[5]:+.10f} deg")