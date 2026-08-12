from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from src.app.calibration_app import RunOptions
from src.calibration.laser_rays import resolve_tool_offset_from_run_metadata
from src.calibration.statistical_pipeline import (
    prepare_statistical_calibration,
    prepared_statistical_calibration_with_tool_offset,
)
from src.evaluation.intrarun_evaluation import (
    _can_use_prepared_fast_path,
    _run_single_observation_count,
)
from src.io.calibration_io import load_calibration_run, resolve_input_folder


OffsetObjectiveMetric = Literal[
    "translation_mean_variance_mm2",
    "translation_rms_3d_deviation_mm",
]


def _make_unique_output_dir(base_dir: Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / f"{prefix}_{timestamp}"
    if not candidate.exists():
        return candidate

    for counter in range(2, 1000):
        numbered = base_dir / f"{prefix}_{timestamp}_{counter:02d}"
        if not numbered.exists():
            return numbered

    return base_dir / f"{prefix}_{timestamp}_{datetime.now().microsecond:06d}"


def _offset_with_dz(base_tool_offset: dict, dz_m: float) -> dict:
    tool_offset = deepcopy(base_tool_offset)
    translation = list(tool_offset.get("translation_m", [0.0, 0.0, 0.0]))
    if len(translation) != 3:
        raise ValueError("tool_offset['translation_m'] muss drei Werte enthalten.")
    translation[2] = float(dz_m)
    tool_offset["translation_m"] = translation
    return tool_offset


def _offset_record(
    dz_m: float,
    result: dict,
    objective_metric: OffsetObjectiveMetric,
    evaluation_dir: Path,
) -> dict:
    statistics = result["statistics"]
    translation_std = np.asarray(
        statistics["translation"]["sample_std_mm"],
        dtype=float,
    )
    rotation_std = np.asarray(
        statistics["rotation"]["sample_std_rotation_vector_deg"],
        dtype=float,
    )
    objective_value = (
        float(np.mean(translation_std**2))
        if objective_metric == "translation_mean_variance_mm2"
        else float(statistics["translation"]["rms_3d_deviation_mm"])
    )

    return {
        "offset_dz_m": float(dz_m),
        "offset_dz_mm": float(dz_m * 1000.0),
        "objective_metric": objective_metric,
        "objective_value": objective_value,
        "num_successful_subruns": len(result["records"]),
        "num_failed_subruns": len(result["failures"]),
        "translation_sample_std_x_mm": float(translation_std[0]),
        "translation_sample_std_y_mm": float(translation_std[1]),
        "translation_sample_std_z_mm": float(translation_std[2]),
        "translation_mean_variance_mm2": float(np.mean(translation_std**2)),
        "translation_rms_3d_deviation_mm": float(
            statistics["translation"]["rms_3d_deviation_mm"]
        ),
        "translation_max_3d_deviation_mm": float(
            statistics["translation"]["max_3d_deviation_mm"]
        ),
        "rotation_sample_std_x_deg": float(rotation_std[0]),
        "rotation_sample_std_y_deg": float(rotation_std[1]),
        "rotation_sample_std_z_deg": float(rotation_std[2]),
        "rotation_rms_angular_deviation_deg": float(
            statistics["rotation"]["rms_angular_deviation_deg"]
        ),
        "mean_ray_distance_mm": float(
            statistics["fit_quality"]["mean_ray_distance_mm"]
        ),
        "mean_rmse_ray_distance_mm": float(
            statistics["fit_quality"]["mean_rmse_ray_distance_mm"]
        ),
        "max_ray_distance_across_runs_mm": float(
            statistics["fit_quality"]["max_ray_distance_across_runs_mm"]
        ),
        "evaluation_dir": str(evaluation_dir),
    }


def _write_offset_records(records: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _save_offset_plots(
    records: list[dict],
    output_dir: Path,
    objective_metric: OffsetObjectiveMetric,
) -> None:
    sorted_records = sorted(records, key=lambda record: record["offset_dz_m"])
    dz_mm = np.asarray([record["offset_dz_mm"] for record in sorted_records])
    objective = np.asarray([
        record["objective_value"] for record in sorted_records
    ])
    best_index = int(np.argmin(objective))

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(dz_mm, objective, "o-", linewidth=1.5)
    axis.scatter(
        [dz_mm[best_index]],
        [objective[best_index]],
        color="red",
        zorder=5,
        label="Best grid result",
    )
    axis.set_xlabel("Laser offset dz in mm")
    ylabel = (
        "Mean translation variance in mm²"
        if objective_metric == "translation_mean_variance_mm2"
        else "RMS 3D deviation in mm"
    )
    axis.set_ylabel(ylabel)
    axis.set_title("Offset dz scan: camera-position stability")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "objective_vs_offset_dz.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for component in "xyz":
        axis.plot(
            dz_mm,
            [
                record[f"translation_sample_std_{component}_mm"]
                for record in sorted_records
            ],
            "o-",
            label=component,
        )
    axis.set_xlabel("Laser offset dz in mm")
    axis.set_ylabel("Camera-position sample std in mm")
    axis.set_title("Camera-position spread by offset dz")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "translation_std_components_vs_offset_dz.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        dz_mm,
        [record["mean_rmse_ray_distance_mm"] for record in sorted_records],
        "o-",
        label="Mean ray-pair RMSE",
    )
    axis.plot(
        dz_mm,
        [record["mean_ray_distance_mm"] for record in sorted_records],
        "o-",
        label="Mean ray distance",
    )
    axis.set_xlabel("Laser offset dz in mm")
    axis.set_ylabel("Ray distance in mm")
    axis.set_title("Fit quality by offset dz")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fit_quality_vs_offset_dz.png", dpi=180)
    plt.close(fig)


def run_offset_dz_intrarun_optimization(
    folder_name: str,
    dz_values_m: list[float] | np.ndarray,
    observations_per_subrun: int,
    num_subruns: int,
    random_seed: int = 42,
    run_options: RunOptions | None = None,
    max_workers: int = 1,
    objective_metric: OffsetObjectiveMetric = "translation_mean_variance_mm2",
) -> dict:
    """
    Scans laser tool-offset dz and evaluates each value with the same Intra-Run
    random subsets. The objective is the camera-pose scatter across subruns.
    """
    dz_values = np.asarray(dz_values_m, dtype=float).reshape(-1)
    if len(dz_values) == 0:
        raise ValueError("dz_values_m darf nicht leer sein.")

    input_folder = resolve_input_folder(folder_name)
    run_data = load_calibration_run(input_folder)
    base_tool_offset = resolve_tool_offset_from_run_metadata(
        run_data["run_metadata"]
    )

    base_output_dir = input_folder / "statistical_evaluation"
    evaluation_dir = _make_unique_output_dir(
        base_output_dir,
        "offset_dz_optimization",
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    if run_options is None:
        run_options = RunOptions(
            save_fit_crop_overlays=False,
            run_trajectory_debug=False,
            run_robot_ray_debug=False,
            run_initial_ray_pair_debug=False,
            run_camera_pose_optimization=True,
            run_optimized_ray_pair_debug=False,
        )

    if not _can_use_prepared_fast_path(run_options):
        raise ValueError(
            "Offset-Optimierung benötigt den vorbereiteten Fast-Path. "
            "Bitte Debug-Einzelplots für diese Auswertung deaktivieren."
        )

    preparation_dir = evaluation_dir / "preparation"
    prepared_base = prepare_statistical_calibration(
        folder_name=str(input_folder),
        output_dir=preparation_dir,
    )

    records: list[dict] = []
    results_by_dz: list[dict] = []
    for index, dz_m in enumerate(dz_values, start=1):
        print(
            f"\nOffset-dz-Auswertung {index}/{len(dz_values)}: "
            f"dz={dz_m * 1000.0:.3f} mm"
        )
        tool_offset = _offset_with_dz(base_tool_offset, float(dz_m))
        prepared = prepared_statistical_calibration_with_tool_offset(
            prepared_base,
            tool_offset=tool_offset,
        )
        dz_dir = evaluation_dir / f"dz_{dz_m * 1000.0:+08.3f}mm"
        result = _run_single_observation_count(
            folder_name=str(input_folder),
            observations_per_subrun=observations_per_subrun,
            num_subruns=num_subruns,
            random_seed=random_seed,
            run_options=run_options,
            run_data=prepared.run_data,
            evaluation_dir=dz_dir,
            prepared_data=prepared,
            max_workers=max_workers,
        )
        record = _offset_record(
            dz_m=float(dz_m),
            result=result,
            objective_metric=objective_metric,
            evaluation_dir=dz_dir,
        )
        records.append(record)
        results_by_dz.append({
            "offset_dz_m": float(dz_m),
            "tool_offset": tool_offset,
            "result": result,
        })

    best_record = min(records, key=lambda record: record["objective_value"])
    csv_path = evaluation_dir / "offset_dz_optimization_results.csv"
    _write_offset_records(records, csv_path)
    _save_offset_plots(records, evaluation_dir, objective_metric)

    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_run": folder_name,
        "source_run_path": str(input_folder),
        "base_tool_offset": base_tool_offset,
        "parameter": "tool_offset.translation_m[2]",
        "dz_values_m": dz_values.tolist(),
        "observations_per_subrun": observations_per_subrun,
        "num_subruns": num_subruns,
        "random_seed": random_seed,
        "objective_metric": objective_metric,
        "best": best_record,
        "records": records,
        "paths": {
            "evaluation_dir": str(evaluation_dir),
            "results_csv": str(csv_path),
            "objective_plot": str(evaluation_dir / "objective_vs_offset_dz.png"),
            "translation_std_plot": str(
                evaluation_dir / "translation_std_components_vs_offset_dz.png"
            ),
            "fit_quality_plot": str(
                evaluation_dir / "fit_quality_vs_offset_dz.png"
            ),
        },
    }
    summary_path = evaluation_dir / "offset_dz_optimization_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\nOffset-dz-Optimierung gespeichert unter: {evaluation_dir}")
    print(
        "Bestes Grid-Ergebnis: "
        f"dz={best_record['offset_dz_mm']:.3f} mm, "
        f"{objective_metric}={best_record['objective_value']:.6g}"
    )

    return {
        "records": records,
        "best_record": best_record,
        "results_by_dz": results_by_dz,
        "evaluation_dir": evaluation_dir,
        "summary_path": summary_path,
    }
