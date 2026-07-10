from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.calibration.laser_rays import Ray3D
from src.calibration.camera_pose_in_robot_frame import (
    CameraPoseInRobotFrame,
)


DEFAULT_CAMERA_POSITION_BOUND_DELTA_M = np.array(
    [0.10, 0.10, 0.10],
    dtype=float,
)
DEFAULT_CAMERA_ROTATION_BOUND_DELTA_DEG = np.array(
    [5.0, 5.0, 5.0],
    dtype=float,
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
    origins_a = np.stack([ray.origin for ray in laser_rays_R])
    directions_a = np.stack([ray.direction for ray in laser_rays_R])
    directions_b_C = np.asarray(camera_rays_C, dtype=float).reshape(-1, 3)
    origins_b = np.broadcast_to(camera_pose_R.translation, origins_a.shape)
    directions_b = directions_b_C @ camera_pose_R.rotation.T

    directions_a = directions_a / np.linalg.norm(
        directions_a, axis=1, keepdims=True
    )
    directions_b = directions_b / np.linalg.norm(
        directions_b, axis=1, keepdims=True
    )
    w0 = origins_a - origins_b
    a = np.einsum("ij,ij->i", directions_a, directions_a)
    b = np.einsum("ij,ij->i", directions_a, directions_b)
    c = np.einsum("ij,ij->i", directions_b, directions_b)
    d = np.einsum("ij,ij->i", directions_a, w0)
    e = np.einsum("ij,ij->i", directions_b, w0)
    denominator = a * c - b * b

    parallel = np.abs(denominator) < 1e-12
    lambda_a = np.zeros_like(denominator)
    lambda_b = np.zeros_like(denominator)
    regular = ~parallel
    lambda_a[regular] = (
        b[regular] * e[regular] - c[regular] * d[regular]
    ) / denominator[regular]
    lambda_b[regular] = (
        a[regular] * e[regular] - b[regular] * d[regular]
    ) / denominator[regular]
    valid_c = parallel & (np.abs(c) > 1e-12)
    lambda_b[valid_c] = e[valid_c] / c[valid_c]

    point_a = origins_a + lambda_a[:, None] * directions_a
    point_b = origins_b + lambda_b[:, None] * directions_b
    residuals = point_a - point_b

    if weights is not None:
        factors = np.sqrt(weights) if use_sqrt_weight else weights
        residuals = residuals * factors[:, None]

    return residuals.reshape(-1)


def _distance_stats(
    laser_rays_R: list[Ray3D],
    camera_rays_C: list[np.ndarray],
    camera_pose_R: CameraPoseInRobotFrame,
    frame_indices: list[int] | None = None,
) -> dict:
    residuals = _ray_pair_residuals_for_pose(
        params=camera_pose_to_params(camera_pose_R),
        laser_rays_R=laser_rays_R,
        camera_rays_C=camera_rays_C,
        frame_indices=frame_indices,
    ).reshape(-1, 3)
    distances = np.linalg.norm(residuals, axis=1)

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
        lower_xyz = x0[:3] - DEFAULT_CAMERA_POSITION_BOUND_DELTA_M
        upper_xyz = x0[:3] + DEFAULT_CAMERA_POSITION_BOUND_DELTA_M
    else:
        lower_xyz = np.asarray(position_bounds_m[0], dtype=float).reshape(3)
        upper_xyz = np.asarray(position_bounds_m[1], dtype=float).reshape(3)

    if rotation_bounds_deg is None:
        lower_rpy = x0[3:6] - DEFAULT_CAMERA_ROTATION_BOUND_DELTA_DEG
        upper_rpy = x0[3:6] + DEFAULT_CAMERA_ROTATION_BOUND_DELTA_DEG
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
