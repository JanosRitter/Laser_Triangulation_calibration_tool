from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.calibration.laser_rays import Ray3D
from src.calibration.ray_geometry import closest_points_between_rays


@dataclass
class RayPairClosestApproach:
    idx: int
    frame_idx: int | None

    laser_point: np.ndarray
    camera_point: np.ndarray
    midpoint: np.ndarray

    distance_m: float
    lambda_laser: float
    lambda_camera: float

    def __post_init__(self):
        self.laser_point = np.asarray(self.laser_point, dtype=float).reshape(3)
        self.camera_point = np.asarray(self.camera_point, dtype=float).reshape(3)
        self.midpoint = np.asarray(self.midpoint, dtype=float).reshape(3)


@dataclass
class RayPairDistanceAnalysis:
    closest_approaches: list[RayPairClosestApproach]
    fitted_z_m: float

    mean_distance_m: float
    median_distance_m: float
    max_distance_m: float
    rmse_distance_m: float

    mean_abs_z_residual_m: float
    max_abs_z_residual_m: float


def analyze_ray_pair_distances(
    laser_rays_R: list[Ray3D],
    camera_rays_R: list[Ray3D],
) -> RayPairDistanceAnalysis:
    """
    Analysiert die kürzesten Abstände paarweiser Laser-/Kamerarays.

    Beide Ray-Listen müssen bereits im selben KS liegen, hier Roboter-KS R.

    Für jedes Paar:
        - nächster Punkt auf Laser-Ray
        - nächster Punkt auf Kamera-Ray
        - Mittelpunkt der Verbindungsstrecke
        - Abstand

    Zusätzlich wird eine Ebene parallel zur x-y-Ebene gefittet:
        z = mean(midpoint_z)
    """
    if len(laser_rays_R) != len(camera_rays_R):
        raise ValueError(
            "laser_rays_R und camera_rays_R müssen dieselbe Länge haben."
        )

    if len(laser_rays_R) == 0:
        raise ValueError("Keine Ray-Paare übergeben.")

    closest_approaches: list[RayPairClosestApproach] = []

    for i, (laser_ray, camera_ray) in enumerate(zip(laser_rays_R, camera_rays_R)):
        closest = closest_points_between_rays(
            origin_a=laser_ray.origin,
            direction_a=laser_ray.direction,
            origin_b=camera_ray.origin,
            direction_b=camera_ray.direction,
        )

        frame_idx = laser_ray.frame_idx
        if camera_ray.frame_idx is not None and frame_idx != camera_ray.frame_idx:
            frame_idx = None

        closest_approaches.append(
            RayPairClosestApproach(
                idx=i,
                frame_idx=frame_idx,
                laser_point=closest["point_a"],
                camera_point=closest["point_b"],
                midpoint=closest["midpoint"],
                distance_m=float(closest["distance"]),
                lambda_laser=float(closest["lambda_a"]),
                lambda_camera=float(closest["lambda_b"]),
            )
        )

    distances = np.array(
        [c.distance_m for c in closest_approaches],
        dtype=float,
    )

    midpoints = np.array(
        [c.midpoint for c in closest_approaches],
        dtype=float,
    )

    fitted_z_m = float(np.mean(midpoints[:, 2]))
    z_residuals = midpoints[:, 2] - fitted_z_m

    return RayPairDistanceAnalysis(
        closest_approaches=closest_approaches,
        fitted_z_m=fitted_z_m,
        mean_distance_m=float(np.mean(distances)),
        median_distance_m=float(np.median(distances)),
        max_distance_m=float(np.max(distances)),
        rmse_distance_m=float(np.sqrt(np.mean(distances**2))),
        mean_abs_z_residual_m=float(np.mean(np.abs(z_residuals))),
        max_abs_z_residual_m=float(np.max(np.abs(z_residuals))),
    )


def print_ray_pair_distance_summary(
    analysis: RayPairDistanceAnalysis,
    label: str = "Ray-Pair-Abstandsanalyse",
) -> None:
    print(f"\n📏 {label}:")
    print(f"  Anzahl Paare:       {len(analysis.closest_approaches)}")
    print(f"  mean distance:      {analysis.mean_distance_m:.10f} m")
    print(f"  median distance:    {analysis.median_distance_m:.10f} m")
    print(f"  max distance:       {analysis.max_distance_m:.10f} m")
    print(f"  RMSE distance:      {analysis.rmse_distance_m:.10f} m")
    print(f"  fitted z-plane:     z = {analysis.fitted_z_m:.10f} m")
    print(f"  mean |z residual|:  {analysis.mean_abs_z_residual_m:.10f} m")
    print(f"  max |z residual|:   {analysis.max_abs_z_residual_m:.10f} m")