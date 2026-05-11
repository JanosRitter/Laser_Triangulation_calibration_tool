from __future__ import annotations

import numpy as np

from src.calibration.calibration_types import ExtrinsicPose
from src.calibration.laser_kinematics import (
    build_extrinsic_pose_from_xyzrpy_deg,
)


def extrinsic_pose_from_vector(params: np.ndarray) -> ExtrinsicPose:
    """
    Wandelt einen 6D-Parametervektor in eine ExtrinsicPose um.

    Parameterkonvention:
        params = [x, y, z, rx_deg, ry_deg, rz_deg]

    Einheiten:
    - Translation in Meter
    - Rotation in Grad
    """
    params = np.asarray(params, dtype=float).reshape(6)

    return build_extrinsic_pose_from_xyzrpy_deg(
        x=float(params[0]),
        y=float(params[1]),
        z=float(params[2]),
        rx_deg=float(params[3]),
        ry_deg=float(params[4]),
        rz_deg=float(params[5]),
    )


def vector_from_extrinsic_pose(
    translation: np.ndarray,
    rotation_deg_xyz: np.ndarray,
) -> np.ndarray:
    """
    Baut einen 6D-Parametervektor aus Translation + Eulerwinkeln.

    Parameters
    ----------
    translation : np.ndarray, shape (3,)
        [x, y, z] in Meter
    rotation_deg_xyz : np.ndarray, shape (3,)
        [rx_deg, ry_deg, rz_deg]

    Returns
    -------
    np.ndarray, shape (6,)
        [x, y, z, rx_deg, ry_deg, rz_deg]
    """
    translation = np.asarray(translation, dtype=float).reshape(3)
    rotation_deg_xyz = np.asarray(rotation_deg_xyz, dtype=float).reshape(3)

    return np.array([
        translation[0],
        translation[1],
        translation[2],
        rotation_deg_xyz[0],
        rotation_deg_xyz[1],
        rotation_deg_xyz[2],
    ], dtype=float)


def initial_guess_from_ground_truth_start_pose(run_data: dict) -> np.ndarray:
    """
    Baut einen Startvektor aus der GT-Startpose.

    Nur für Debug / Tests gedacht.
    Nicht für die echte Kalibrierung verwenden.
    """
    start_pose_gt = run_data["ground_truth"]["start_pose"]
    if start_pose_gt is None:
        raise ValueError("Keine Ground-Truth-Startpose vorhanden.")

    return np.array([
        float(start_pose_gt["laser_x"]),
        float(start_pose_gt["laser_y"]),
        float(start_pose_gt["laser_z"]),
        float(start_pose_gt["laser_rx"]),
        float(start_pose_gt["laser_ry"]),
        float(start_pose_gt["laser_rz"]),
    ], dtype=float)


def zero_initial_guess() -> np.ndarray:
    """
    Einfache Nullinitialisierung:
        [0, 0, 0, 0, 0, 0]
    """
    return np.zeros(6, dtype=float)


def perturb_parameter_vector(
    params: np.ndarray,
    translation_offset: np.ndarray | None = None,
    rotation_offset_deg: np.ndarray | None = None,
) -> np.ndarray:
    """
    Erzeugt aus einem Parametervektor einen gestörten Vektor.

    Parameters
    ----------
    params : np.ndarray, shape (6,)
    translation_offset : np.ndarray, shape (3,), optional
        Offset in Meter
    rotation_offset_deg : np.ndarray, shape (3,), optional
        Offset in Grad
    """
    params = np.asarray(params, dtype=float).reshape(6).copy()

    if translation_offset is not None:
        translation_offset = np.asarray(translation_offset, dtype=float).reshape(3)
        params[:3] += translation_offset

    if rotation_offset_deg is not None:
        rotation_offset_deg = np.asarray(rotation_offset_deg, dtype=float).reshape(3)
        params[3:] += rotation_offset_deg

    return params