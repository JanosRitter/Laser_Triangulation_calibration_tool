from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.calibration.laser_rays import Ray3D
from src.calibration.ray_geometry import closest_points_between_rays
from src.calibration.camera_pose_in_robot_frame import (
    CameraPoseInRobotFrame,
    transform_camera_rays_to_robot_frame,
)


@dataclass
class CameraPoseOptimizationResult:
    initial_params: np.ndarray
    optimized_params: np.ndarray

    initial_pose_R: CameraPoseInRobotFrame
    optimized_pose_R: CameraPoseInRobotFrame

    solver_result: object

    mean_distance_initial_m: float
    mean_distance_optimized_m: float
    rmse_distance_initial_m: float
    rmse_distance_optimized_m: float
    max_distance_initial_m: float
    max_distance_optimized_m: float


def camera_pose_to_params(
    camera_pose_R: CameraPoseInRobotFrame,
) -> np.ndarray:
    """
    Parametrisierung:
        [x, y, z, rx_deg, ry_deg, rz_deg]

    Rotation als Euler xyz in Grad.
    """
    t = camera_pose_R.translation
    rpy_deg = Rotation.from_matrix(camera_pose_R.rotation).as_euler(
        "xyz",
        degrees=True,
    )

    return np.array(
        [t[0], t[1], t[2], rpy_deg[0], rpy_deg[1], rpy_deg[2]],
        dtype=float,
    )


def params_to_camera_pose(
    params: np.ndarray,
) -> CameraPoseInRobotFrame:
    """
    Baut Kamerapose im Roboter-KS aus:
        [x, y, z, rx_deg, ry_deg, rz_deg]
    """
    params = np.asarray(params, dtype=float).reshape(6)

    t = params[:3]
    rpy_deg = params[3:6]

    R_R_C = Rotation.from_euler(
        "xyz",
        rpy_deg,
        degrees=True,
    ).as_matrix()

    return CameraPoseInRobotFrame(
        translation=t,
        rotation=R_R_C,
    )


def _ray_pair_residuals_for_pose(
    params: np.ndarray,
    laser_rays_R: list[Ray3D],
    camera_rays_C: list[np.ndarray],
    frame_indices: list[int] | None = None,
    use_sqrt_weight: bool = False,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Residual-Vektor für least_squares.

    Für jedes Ray-Paar:
        residual_i = closest_laser_point - closest_camera_point

    Rückgabe ist flach:
        [rx0, ry0, rz0, rx1, ry1, rz1, ...]
    """
    camera_pose_R = params_to_camera_pose(params)

    camera_rays_R = transform_camera_rays_to_robot_frame(
        camera_rays_C=camera_rays_C,
        camera_pose_R=camera_pose_R,
        frame_indices=frame_indices,
    )

    residuals: list[np.ndarray] = []

    for i, (laser_ray, camera_ray) in enumerate(zip(laser_rays_R, camera_rays_R)):
        closest = closest_points_between_rays(
            origin_a=laser_ray.origin,
            direction_a=laser_ray.direction,
            origin_b=camera_ray.origin,
            direction_b=camera_ray.direction,
        )

        r = closest["point_a"] - closest["point_b"]

        if weights is not None:
            w = float(weights[i])
            if use_sqrt_weight:
                r = np.sqrt(w) * r
            else:
                r = w * r

        residuals.append(r)

    return np.concatenate(residuals)


def _distance_stats(
    laser_rays_R: list[Ray3D],
    camera_rays_C: list[np.ndarray],
    camera_pose_R: CameraPoseInRobotFrame,
    frame_indices: list[int] | None = None,
) -> dict:
    camera_rays_R = transform_camera_rays_to_robot_frame(
        camera_rays_C=camera_rays_C,
        camera_pose_R=camera_pose_R,
        frame_indices=frame_indices,
    )

    distances = []

    for laser_ray, camera_ray in zip(laser_rays_R, camera_rays_R):
        closest = closest_points_between_rays(
            origin_a=laser_ray.origin,
            direction_a=laser_ray.direction,
            origin_b=camera_ray.origin,
            direction_b=camera_ray.direction,
        )
        distances.append(float(closest["distance"]))

    distances = np.asarray(distances, dtype=float)

    return {
        "mean": float(np.mean(distances)),
        "rmse": float(np.sqrt(np.mean(distances**2))),
        "max": float(np.max(distances)),
    }


def solve_camera_pose_from_ray_pairs(
    laser_rays_R: list[Ray3D],
    camera_rays_C: list[np.ndarray],
    initial_pose_R: CameraPoseInRobotFrame,
    frame_indices: list[int] | None = None,
    weights: np.ndarray | None = None,
    position_bounds_m: tuple[np.ndarray, np.ndarray] | None = None,
    rotation_bounds_deg: tuple[np.ndarray, np.ndarray] | None = None,
    verbose: int = 1,
) -> CameraPoseOptimizationResult:
    """
    Optimiert die Kamerapose im Roboter-KS so, dass die Ray-Pair-Abstände minimal werden.

    Optimierte Parameter:
        x, y, z, rx_deg, ry_deg, rz_deg

    Bounds:
        position_bounds_m:
            (lower_xyz, upper_xyz)

        rotation_bounds_deg:
            (lower_rpy_deg, upper_rpy_deg)
    """
    if len(laser_rays_R) != len(camera_rays_C):
        raise ValueError(
            "laser_rays_R und camera_rays_C müssen dieselbe Länge haben."
        )

    if len(laser_rays_R) == 0:
        raise ValueError("Keine Ray-Paare für Optimierung übergeben.")

    if frame_indices is not None and len(frame_indices) != len(camera_rays_C):
        raise ValueError(
            "frame_indices muss dieselbe Länge wie camera_rays_C haben."
        )

    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(camera_rays_C):
            raise ValueError("weights muss dieselbe Länge wie camera_rays_C haben.")

    x0 = camera_pose_to_params(initial_pose_R)

    if position_bounds_m is None:
        lower_xyz = x0[:3] - np.array([0.25, 0.25, 0.25], dtype=float)
        upper_xyz = x0[:3] + np.array([0.25, 0.25, 0.25], dtype=float)
    else:
        lower_xyz = np.asarray(position_bounds_m[0], dtype=float).reshape(3)
        upper_xyz = np.asarray(position_bounds_m[1], dtype=float).reshape(3)

    if rotation_bounds_deg is None:
        lower_rpy = x0[3:6] - np.array([30.0, 30.0, 30.0], dtype=float)
        upper_rpy = x0[3:6] + np.array([30.0, 30.0, 30.0], dtype=float)
    else:
        lower_rpy = np.asarray(rotation_bounds_deg[0], dtype=float).reshape(3)
        upper_rpy = np.asarray(rotation_bounds_deg[1], dtype=float).reshape(3)

    lower = np.concatenate([lower_xyz, lower_rpy])
    upper = np.concatenate([upper_xyz, upper_rpy])

    initial_stats = _distance_stats(
        laser_rays_R=laser_rays_R,
        camera_rays_C=camera_rays_C,
        camera_pose_R=initial_pose_R,
        frame_indices=frame_indices,
    )

    result = least_squares(
        fun=_ray_pair_residuals_for_pose,
        x0=x0,
        bounds=(lower, upper),
        args=(laser_rays_R, camera_rays_C, frame_indices, True, weights),
        method="trf",
        loss="soft_l1",
        f_scale=0.005,
        verbose=verbose,
        x_scale=np.array([0.05, 0.05, 0.05, 5.0, 5.0, 5.0], dtype=float),
        max_nfev=500,
    )

    optimized_pose_R = params_to_camera_pose(result.x)

    optimized_stats = _distance_stats(
        laser_rays_R=laser_rays_R,
        camera_rays_C=camera_rays_C,
        camera_pose_R=optimized_pose_R,
        frame_indices=frame_indices,
    )

    return CameraPoseOptimizationResult(
        initial_params=x0,
        optimized_params=result.x,
        initial_pose_R=initial_pose_R,
        optimized_pose_R=optimized_pose_R,
        solver_result=result,
        mean_distance_initial_m=initial_stats["mean"],
        mean_distance_optimized_m=optimized_stats["mean"],
        rmse_distance_initial_m=initial_stats["rmse"],
        rmse_distance_optimized_m=optimized_stats["rmse"],
        max_distance_initial_m=initial_stats["max"],
        max_distance_optimized_m=optimized_stats["max"],
    )


def print_camera_pose_optimization_result(
    opt: CameraPoseOptimizationResult,
) -> None:
    print("\n🎯 Kamerapose-Optimierung:")

    print("\n  Initial:")
    print(
        f"    x,y,z = "
        f"({opt.initial_params[0]:+.6f}, "
        f"{opt.initial_params[1]:+.6f}, "
        f"{opt.initial_params[2]:+.6f}) m"
    )
    print(
        f"    rx,ry,rz = "
        f"({opt.initial_params[3]:+.6f}, "
        f"{opt.initial_params[4]:+.6f}, "
        f"{opt.initial_params[5]:+.6f}) deg"
    )

    print("\n  Optimiert:")
    print(
        f"    x,y,z = "
        f"({opt.optimized_params[0]:+.6f}, "
        f"{opt.optimized_params[1]:+.6f}, "
        f"{opt.optimized_params[2]:+.6f}) m"
    )
    print(
        f"    rx,ry,rz = "
        f"({opt.optimized_params[3]:+.6f}, "
        f"{opt.optimized_params[4]:+.6f}, "
        f"{opt.optimized_params[5]:+.6f}) deg"
    )

    print("\n  Abstände:")
    print(
        f"    mean: "
        f"{opt.mean_distance_initial_m * 1000.0:.3f} mm -> "
        f"{opt.mean_distance_optimized_m * 1000.0:.3f} mm"
    )
    print(
        f"    RMSE: "
        f"{opt.rmse_distance_initial_m * 1000.0:.3f} mm -> "
        f"{opt.rmse_distance_optimized_m * 1000.0:.3f} mm"
    )
    print(
        f"    max:  "
        f"{opt.max_distance_initial_m * 1000.0:.3f} mm -> "
        f"{opt.max_distance_optimized_m * 1000.0:.3f} mm"
    )

    print("\n  Solver:")
    print(f"    success: {opt.solver_result.success}")
    print(f"    status:  {opt.solver_result.status}")
    print(f"    message: {opt.solver_result.message}")
    print(f"    nfev:    {opt.solver_result.nfev}")
    print(f"    cost:    {opt.solver_result.cost:.12e}")