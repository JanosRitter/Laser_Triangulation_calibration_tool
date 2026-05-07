from __future__ import annotations

import numpy as np

from src.io.calibration_io import load_crop_array
from src.io.io_utils import save_fit_table_csv
from src.fitting.crop_fitting import fit_multiple_observations
from src.evaluation.fit_summary import (
    build_fit_results_table,
    apply_quality_flags,
)
from src.calibration.observation_builder import (
    build_calibration_observations_from_fit_table,
)
from src.calibration.parameterization import zero_initial_guess
from src.calibration.residuals import (
    compute_frame_residuals_detailed,
    summarize_residuals,
)
from src.calibration.solver import solve_extrinsic_pose


def prepare_calibration_observations(
    run_data: dict,
    method: str = "gaussian",
    threshold_factor: float = 2.5,
    subtract_background: bool = False,
) -> dict:
    observations_raw = run_data["observations"]
    if len(observations_raw) == 0:
        raise ValueError("Keine gültigen Beobachtungen gefunden.")

    def crop_loader(obs):
        return load_crop_array(run_data["input_folder"], obs["crop_npy_file"])

    fit_results = fit_multiple_observations(
        observations=observations_raw,
        crop_loader=crop_loader,
        method=method,
        threshold_factor=threshold_factor,
        subtract_background=subtract_background,
    )

    fit_table = build_fit_results_table(
        observations=observations_raw,
        fit_results=fit_results,
        crop_loader=crop_loader,
    )

    fit_table = apply_quality_flags(
        fit_table,
        min_snr=4.0,
        min_amplitude=10.0,
        sigma_min=0.8,
        sigma_max=40.0,
        reject_near_border=False,
    )

    fit_table_path = run_data["input_folder"] / "laser_point_fit_table.csv"
    save_fit_table_csv(fit_table, fit_table_path)
    print(f"\n💾 Fit-Tabelle gespeichert: {fit_table_path}")

    print("\n🔎 Fit-Reject-Gründe:")
    for row in fit_table:
        print(
            f"frame {row['frame_idx']:3d}: "
            f"use={row['use_for_calibration']} | "
            f"reason={row['reject_reason']} | "
            f"snr={row['snr_estimate']:.3f} | "
            f"amp={row['amplitude']:.3f} | "
            f"sigma={row['sigma_mean']:.3f} | "
            f"border={row['is_near_crop_border']}"
        )

    calib_observations = build_calibration_observations_from_fit_table(
        run_data=run_data,
        fit_table=fit_table,
        only_usable=True,
    )

    stats = {
        "num_fit_ok": sum(1 for r in fit_results if r["fit_ok"]),
        "num_fit_total": len(fit_results),
        "num_good": sum(1 for row in fit_table if row["use_for_calibration"]),
        "num_fit_table": len(fit_table),
        "num_calib_observations": len(calib_observations),
    }

    return {
        "fit_results": fit_results,
        "fit_table": fit_table,
        "calib_observations": calib_observations,
        "stats": stats,
    }


def build_default_initial_guess(run_data: dict) -> np.ndarray:
    _ = run_data
    return zero_initial_guess()


def run_calibration_without_gt(
    run_data: dict,
    calib_observations: list,
    intrinsics,
    initial_params: np.ndarray | None = None,
    use_weight: bool = False,
) -> dict:
    if initial_params is None:
        initial_params = build_default_initial_guess(run_data)

    initial_params = np.asarray(initial_params, dtype=float).reshape(6)

    frame_results_initial = compute_frame_residuals_detailed(
        params=initial_params,
        observations=calib_observations,
        intrinsics=intrinsics,
        use_weight=use_weight,
    )
    residual_summary_initial = summarize_residuals(frame_results_initial)

    laser_origin_bounds = (
        np.array([-0.30, -0.10, 0.10], dtype=float),
        np.array([0.30, 0.40, 0.30], dtype=float),
    )

    result = solve_extrinsic_pose(
        initial_params=initial_params,
        observations=calib_observations,
        intrinsics=intrinsics,
        use_weight=use_weight,
        method="trf",
        verbose=0,
        laser_origin_bounds=laser_origin_bounds,
        laser_origin_bound_weight=100.0,
    )

    params_optimized = result.x

    frame_results_optimized = compute_frame_residuals_detailed(
        params=params_optimized,
        observations=calib_observations,
        intrinsics=intrinsics,
        use_weight=use_weight,
    )
    residual_summary_optimized = summarize_residuals(frame_results_optimized)

    return {
        "params_initial": initial_params,
        "params_optimized": params_optimized,
        "frame_results_initial": frame_results_initial,
        "frame_results_optimized": frame_results_optimized,
        "residual_summary_initial": residual_summary_initial,
        "residual_summary_optimized": residual_summary_optimized,
        "solver_result": result,
    }