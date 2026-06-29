from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.calibration.laser_rays import Ray3D
from src.calibration.camera_pose_in_robot_frame import (
    CameraPoseInRobotFrame,
    camera_pose_to_matrix,
)


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(4, 4)

    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def robot_to_camera_transform_from_camera_pose(
    camera_pose_R: CameraPoseInRobotFrame,
) -> np.ndarray:
    """
    Baut ^C T_R aus der optimierten Kamerapose ^R T_C.

    camera_pose_R beschreibt:
        p_R = R_R_C @ p_C + t_R_C

    Rückgabe:
        p_C = R_C_R @ p_R + t_C_R
    """
    T_R_C = camera_pose_to_matrix(camera_pose_R)
    return invert_transform(T_R_C)


def transform_point_R_to_C(
    point_R: np.ndarray,
    T_C_R: np.ndarray,
) -> np.ndarray:
    point_R = np.asarray(point_R, dtype=float).reshape(3)
    T_C_R = np.asarray(T_C_R, dtype=float).reshape(4, 4)

    point_R_h = np.array([point_R[0], point_R[1], point_R[2], 1.0], dtype=float)
    point_C_h = T_C_R @ point_R_h

    return point_C_h[:3]


def transform_direction_R_to_C(
    direction_R: np.ndarray,
    T_C_R: np.ndarray,
) -> np.ndarray:
    direction_R = np.asarray(direction_R, dtype=float).reshape(3)
    T_C_R = np.asarray(T_C_R, dtype=float).reshape(4, 4)

    R_C_R = T_C_R[:3, :3]
    direction_C = R_C_R @ direction_R
    direction_C /= np.linalg.norm(direction_C)

    return direction_C


def transform_ray_R_to_C(
    ray_R: Ray3D,
    T_C_R: np.ndarray,
) -> Ray3D:
    origin_C = transform_point_R_to_C(ray_R.origin, T_C_R)
    direction_C = transform_direction_R_to_C(ray_R.direction, T_C_R)

    return Ray3D(
        origin=origin_C,
        direction=direction_C,
        frame_idx=ray_R.frame_idx,
    )


def transform_rays_R_to_C(
    rays_R: list[Ray3D],
    T_C_R: np.ndarray,
) -> list[Ray3D]:
    return [
        transform_ray_R_to_C(ray_R=ray, T_C_R=T_C_R)
        for ray in rays_R
    ]


def save_robot_to_camera_transform_json(
    T_C_R: np.ndarray,
    output_path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """
    Speichert die finale Kalibrierung für die spätere Verwendung
    im Triangulationssystem.

    Enthalten:
        - vollständige Transformationsmatrix ^C T_R
        - Rotationsmatrix R_C_R
        - Translation t_C_R
        - explizite Kamera-KS-Konvention
        - optionale Kalibrier-/Debug-Metadaten
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    T_C_R = np.asarray(T_C_R, dtype=float).reshape(4, 4)

    R_C_R = T_C_R[:3, :3]
    t_C_R = T_C_R[:3, 3]

    det_R = float(np.linalg.det(R_C_R))

    data = {
        "description": (
            "Transformation from robot frame R to camera frame C"
        ),

        # -----------------------------------------------------
        # Transformationskonvention
        # -----------------------------------------------------
        "transformation_convention": {
            "point_transform": (
                "p_C = R_C_R @ p_R + t_C_R"
            ),
            "direction_transform": (
                "d_C = R_C_R @ d_R"
            ),
            "homogeneous_transform": (
                "p_C_h = T_C_R @ p_R_h"
            ),
        },

        # -----------------------------------------------------
        # Kamera-KS-Konvention
        # -----------------------------------------------------
        "camera_frame_convention": {
            "x_C": "image right",
            "y_C": "image up",
            "z_C": (
                "backward, opposite to viewing direction"
            ),
            "viewing_direction": "-z_C",

            "camera_ray_model": (
                "ray_C = normalize([(u-cx)/fx, (cy-v)/fy, -1])"
            ),

            "image_coordinates": {
                "u_direction": "right",
                "v_direction": "down",
                "origin": "top_left",
            },

            "handedness": "right-handed",
        },

        # -----------------------------------------------------
        # Finale Transformation
        # -----------------------------------------------------
        "matrix_T_C_R": T_C_R.tolist(),

        "R_C_R": {
            "matrix": R_C_R.tolist(),
            "determinant": det_R,
        },

        "t_C_R": {
            "translation_m": t_C_R.tolist(),
        },

        # -----------------------------------------------------
        # Zusatzinformationen
        # -----------------------------------------------------
        "notes": [
            (
                "Laser rays transformed into camera frame "
                "should have negative z-values for objects "
                "in front of the camera."
            ),
            (
                "Camera viewing direction corresponds to -z_C."
            ),
            (
                "This calibration assumes a right-handed "
                "camera coordinate system."
            ),
        ],
    }

    if metadata is not None:
        data["metadata"] = metadata

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return output_path

def build_calibration_export_metadata(
    output_dir,
    calib_observations,
    intrinsics,
    camera_pose_optimization_result,
    ray_pair_distance_analysis_optimized,
) -> dict:
    dist_coeffs = getattr(intrinsics, "dist_coeffs", None)

    return {
        "run_folder": str(output_dir),
        "num_observations": len(calib_observations),
        "intrinsics": {
            "source": getattr(intrinsics, "source", "unknown"),
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.cx),
            "cy": float(intrinsics.cy),
            "img_width": int(intrinsics.img_width),
            "img_height": int(intrinsics.img_height),
            "dist_coeffs": (
                None
                if dist_coeffs is None
                else np.asarray(dist_coeffs, dtype=float).reshape(-1).tolist()
            ),
        },
        "optimization": {
            "initial_params": camera_pose_optimization_result.initial_params.tolist(),
            "optimized_params": camera_pose_optimization_result.optimized_params.tolist(),
            "success": bool(camera_pose_optimization_result.solver_result.success),
            "status": int(camera_pose_optimization_result.solver_result.status),
            "message": str(camera_pose_optimization_result.solver_result.message),
            "nfev": int(camera_pose_optimization_result.solver_result.nfev),
            "cost": float(camera_pose_optimization_result.solver_result.cost),
        },
        "ray_pair_distance_summary": {
            "mean_distance_m": float(ray_pair_distance_analysis_optimized.mean_distance_m),
            "median_distance_m": float(ray_pair_distance_analysis_optimized.median_distance_m),
            "rmse_distance_m": float(ray_pair_distance_analysis_optimized.rmse_distance_m),
            "max_distance_m": float(ray_pair_distance_analysis_optimized.max_distance_m),
            "fitted_z_plane_m": float(ray_pair_distance_analysis_optimized.fitted_z_m),
            "mean_abs_z_residual_m": float(ray_pair_distance_analysis_optimized.mean_abs_z_residual_m),
            "max_abs_z_residual_m": float(ray_pair_distance_analysis_optimized.max_abs_z_residual_m),
        },
    }