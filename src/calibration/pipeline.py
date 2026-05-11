from __future__ import annotations

from pathlib import Path

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


def prepare_calibration_observations(
    run_data: dict,
    method: str = "gaussian",
    threshold_factor: float = 2.5,
    subtract_background: bool = False,
    save_fit_overlays: bool = False,
    fit_overlay_output_dir: str | Path | None = None,
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
        save_fit_overlays=save_fit_overlays,
        fit_overlay_output_dir=fit_overlay_output_dir,
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


