from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from src.calibration.residuals import compute_residual_vector


def solve_extrinsic_pose(
    initial_params: np.ndarray,
    observations: list,
    intrinsics,
    use_weight: bool = False,
    method: str = "trf",
    verbose: int = 0,
    laser_origin_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    laser_origin_bound_weight: float = 100.0,
):
    """
    Optimiert die 6D-Startpose des Lasers im Kamera-KS.

    Parameterkonvention:
        [x, y, z, rx_deg, ry_deg, rz_deg]

    Parameter-Bounds:
    - beschränken die Startpose des Laser-Bündels.

    laser_origin_bounds:
    - optionale weiche Bounds für alle transformierten Laser-Ray-Startpunkte.
    """
    initial_params = np.asarray(initial_params, dtype=float).reshape(6)

    lower_bounds = np.array([
        -0.30,   # x [m]
        -0.10,   # y [m]
         0.10,   # z [m]
        -90.0,   # rx [deg]
        -90.0,   # ry [deg]
        -90.0,   # rz [deg]
    ], dtype=float)

    upper_bounds = np.array([
         0.30,   # x [m]
         0.40,   # y [m]
         0.30,   # z [m]
         90.0,   # rx [deg]
         90.0,   # ry [deg]
         90.0,   # rz [deg]
    ], dtype=float)

    eps = 1e-9
    initial_params = np.clip(
        initial_params,
        lower_bounds + eps,
        upper_bounds - eps,
    )

    def objective(params: np.ndarray) -> np.ndarray:
        return compute_residual_vector(
            params=params,
            observations=observations,
            intrinsics=intrinsics,
            use_weight=use_weight,
            laser_origin_bounds=laser_origin_bounds,
            laser_origin_bound_weight=laser_origin_bound_weight,
        )

    result = least_squares(
        fun=objective,
        x0=initial_params,
        bounds=(lower_bounds, upper_bounds),
        method=method,
        verbose=verbose,
    )

    return result


def parameter_difference(params_est: np.ndarray, params_ref: np.ndarray) -> dict:
    """
    Vergleicht zwei 6D-Parametervektoren.
    """
    params_est = np.asarray(params_est, dtype=float).reshape(6)
    params_ref = np.asarray(params_ref, dtype=float).reshape(6)

    diff = params_est - params_ref

    return {
        "dx_m": float(diff[0]),
        "dy_m": float(diff[1]),
        "dz_m": float(diff[2]),
        "dtx_norm_m": float(np.linalg.norm(diff[:3])),
        "drx_deg": float(diff[3]),
        "dry_deg": float(diff[4]),
        "drz_deg": float(diff[5]),
        "drot_norm_deg": float(np.linalg.norm(diff[3:])),
    }


def print_parameter_vector(params: np.ndarray, label: str = "Parameter") -> None:
    params = np.asarray(params, dtype=float).reshape(6)

    print(f"\n📦 {label}:")
    print(f"  x  = {params[0]:+.10f} m")
    print(f"  y  = {params[1]:+.10f} m")
    print(f"  z  = {params[2]:+.10f} m")
    print(f"  rx = {params[3]:+.10f} deg")
    print(f"  ry = {params[4]:+.10f} deg")
    print(f"  rz = {params[5]:+.10f} deg")


def print_parameter_difference(diff: dict, label: str = "Abweichung") -> None:
    print(f"\n📏 {label}:")
    print(f"  dx = {diff['dx_m']:+.10f} m")
    print(f"  dy = {diff['dy_m']:+.10f} m")
    print(f"  dz = {diff['dz_m']:+.10f} m")
    print(f"  |dt| = {diff['dtx_norm_m']:.10f} m")
    print(f"  drx = {diff['drx_deg']:+.10f} deg")
    print(f"  dry = {diff['dry_deg']:+.10f} deg")
    print(f"  drz = {diff['drz_deg']:+.10f} deg")
    print(f"  |drot| = {diff['drot_norm_deg']:.10f} deg")


def print_solver_report(result) -> None:
    """
    Kompakte Ausgabe des least_squares-Ergebnisses.
    """
    print("\n🧠 Solver-Report:")
    print(f"  success: {result.success}")
    print(f"  status: {result.status}")
    print(f"  message: {result.message}")
    print(f"  nfev: {result.nfev}")
    print(f"  njev: {result.njev}")
    print(f"  cost: {result.cost:.12e}")
    print(f"  optimality: {result.optimality:.12e}")