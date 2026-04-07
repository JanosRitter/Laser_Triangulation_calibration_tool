from __future__ import annotations

import numpy as np

from src.calibration.camera_rays import pixel_to_camera_ray
from src.calibration.laser_kinematics import laser_ray_in_camera
from src.calibration.parameterization import extrinsic_pose_from_vector
from src.calibration.ray_geometry import closest_points_between_rays


def compute_frame_residual(
    params: np.ndarray,
    observation,
    intrinsics,
    use_weight: bool = False,
) -> dict:
    """
    Berechnet das Residuum für genau eine Beobachtung.

    Vorgehen:
    - Kamerastrahl aus UV bestimmen
    - Laserstrahl aus Kandidat für Startpose + relativer Pose bestimmen
    - nächstgelegene Punkte auf beiden Strahlen berechnen
    - Residuum = point_cam - point_laser

    Parameters
    ----------
    params : np.ndarray, shape (6,)
        [x, y, z, rx_deg, ry_deg, rz_deg]
    observation : CalibrationObservation
    intrinsics : CameraIntrinsics
    use_weight : bool
        Wenn True, wird das Residuum mit sqrt(weight) skaliert.

    Returns
    -------
    dict mit u. a.:
    - frame_idx
    - residual_vector
    - residual_norm
    - point_cam
    - point_laser
    - lambda_cam
    - lambda_laser
    - camera_ray_direction
    - laser_ray_origin
    - laser_ray_direction
    """
    extrinsic_pose = extrinsic_pose_from_vector(params)

    cam_origin = np.zeros(3, dtype=float)
    cam_direction = pixel_to_camera_ray(observation.uv, intrinsics)

    laser_origin, laser_direction = laser_ray_in_camera(
        extrinsic_pose=extrinsic_pose,
        relative_pose=observation.relative_pose,
    )

    closest = closest_points_between_rays(
        origin_a=cam_origin,
        direction_a=cam_direction,
        origin_b=laser_origin,
        direction_b=laser_direction,
    )

    point_cam = closest["point_a"]
    point_laser = closest["point_b"]

    residual_vector = point_cam - point_laser

    if use_weight:
        weight = float(getattr(observation, "weight", 1.0))
        weight = max(weight, 1e-12)
        residual_vector = np.sqrt(weight) * residual_vector

    residual_norm = float(np.linalg.norm(residual_vector))

    return {
        "frame_idx": int(observation.frame_idx),
        "residual_vector": residual_vector,
        "residual_norm": residual_norm,
        "point_cam": point_cam,
        "point_laser": point_laser,
        "lambda_cam": float(closest["lambda_a"]),
        "lambda_laser": float(closest["lambda_b"]),
        "midpoint": closest["midpoint"],
        "camera_ray_direction": cam_direction,
        "laser_ray_origin": laser_origin,
        "laser_ray_direction": laser_direction,
    }


def compute_residual_vector(
    params: np.ndarray,
    observations: list,
    intrinsics,
    use_weight: bool = False,
) -> np.ndarray:
    """
    Stapelt die 3D-Residuen aller Beobachtungen zu einem langen Vektor.

    Für N Beobachtungen entsteht ein Vektor der Länge 3N.
    """
    residual_blocks = []

    for obs in observations:
        frame_result = compute_frame_residual(
            params=params,
            observation=obs,
            intrinsics=intrinsics,
            use_weight=use_weight,
        )
        residual_blocks.append(frame_result["residual_vector"])

    if len(residual_blocks) == 0:
        return np.zeros(0, dtype=float)

    return np.concatenate(residual_blocks, axis=0)


def compute_frame_residuals_detailed(
    params: np.ndarray,
    observations: list,
    intrinsics,
    use_weight: bool = False,
) -> list[dict]:
    """
    Berechnet detaillierte Residueninformationen für alle Beobachtungen.
    """
    results = []

    for obs in observations:
        result = compute_frame_residual(
            params=params,
            observation=obs,
            intrinsics=intrinsics,
            use_weight=use_weight,
        )
        results.append(result)

    return results


def summarize_residuals(frame_results: list[dict]) -> dict:
    """
    Erzeugt eine kompakte Zusammenfassung über alle Frame-Residuen.
    """
    if len(frame_results) == 0:
        return {
            "num_frames": 0,
            "mean_residual_norm_m": np.nan,
            "max_residual_norm_m": np.nan,
            "rmse_residual_m": np.nan,
            "mean_lambda_cam_m": np.nan,
            "mean_lambda_laser_m": np.nan,
            "worst_frame_idx": None,
        }

    residual_norms = np.array([r["residual_norm"] for r in frame_results], dtype=float)
    lambda_cam = np.array([r["lambda_cam"] for r in frame_results], dtype=float)
    lambda_laser = np.array([r["lambda_laser"] for r in frame_results], dtype=float)

    worst_idx = int(np.argmax(residual_norms))
    worst_frame_idx = int(frame_results[worst_idx]["frame_idx"])

    return {
        "num_frames": len(frame_results),
        "mean_residual_norm_m": float(np.mean(residual_norms)),
        "max_residual_norm_m": float(np.max(residual_norms)),
        "rmse_residual_m": float(np.sqrt(np.mean(residual_norms ** 2))),
        "mean_lambda_cam_m": float(np.mean(lambda_cam)),
        "mean_lambda_laser_m": float(np.mean(lambda_laser)),
        "worst_frame_idx": worst_frame_idx,
    }


def print_residual_summary(summary: dict, label: str = "Residuen") -> None:
    """
    Konsolenausgabe einer Residuenzusammenfassung.
    """
    print(f"\n📐 {label}:")
    print(f"  Anzahl Frames: {summary['num_frames']}")
    print(f"  mittlerer Residualbetrag: {summary['mean_residual_norm_m']:.10f} m")
    print(f"  maximaler Residualbetrag: {summary['max_residual_norm_m']:.10f} m")
    print(f"  RMSE Residuum: {summary['rmse_residual_m']:.10f} m")
    print(f"  mittlere Kamera-Tiefe λ: {summary['mean_lambda_cam_m']:.10f} m")
    print(f"  mittlere Laser-Tiefe μ:  {summary['mean_lambda_laser_m']:.10f} m")
    print(f"  schlechtester Frame: {summary['worst_frame_idx']}")


def print_first_frame_residuals(
    frame_results: list[dict],
    first_n: int = 10,
    label: str = "Erste Residuen",
) -> None:
    """
    Druckt die ersten Residuen frameweise aus.
    """
    print(f"\n🔎 {label}:")
    for result in frame_results[:first_n]:
        rv = result["residual_vector"]
        print(
            f"  frame {result['frame_idx']:3d} | "
            f"|r|={result['residual_norm']:.10f} m | "
            f"rx={rv[0]:+.10f}, "
            f"ry={rv[1]:+.10f}, "
            f"rz={rv[2]:+.10f} | "
            f"λ={result['lambda_cam']:.10f}, "
            f"μ={result['lambda_laser']:.10f}"
        )