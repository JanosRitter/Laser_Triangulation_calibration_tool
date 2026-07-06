from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.app.calibration_app import RunOptions, run_calibration_app
from src.calibration.statistical_pipeline import (
    PreparedStatisticalCalibration,
    prepare_statistical_calibration,
    run_prepared_statistical_calibration,
)
from src.evaluation.multirun_evaluation import (
    MultiRunRecord,
    _build_statistics,
    _record_from_result,
    _save_plots,
    _write_results_csv,
)
from src.io.calibration_io import load_calibration_run


def _can_use_prepared_fast_path(run_options: RunOptions | None) -> bool:
    if run_options is None:
        return True
    return not any((
        run_options.save_fit_crop_overlays,
        run_options.run_trajectory_debug,
        run_options.run_robot_ray_debug,
        run_options.run_initial_ray_pair_debug,
        run_options.run_optimized_ray_pair_debug,
    ))


def _write_selection_table(
    selections: list[dict],
    output_path: Path,
) -> None:
    fieldnames = [
        "subrun_index",
        "subrun_name",
        "frame_idx",
        "sample_position",
        "crop_center_u_px",
        "crop_center_v_px",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selections)


def _save_sampling_plots(
    available_observations: list[dict],
    selections: list[dict],
    output_dir: Path,
) -> None:
    counts = Counter(row["frame_idx"] for row in selections)
    frame_indices = np.array(
        [int(observation["frame_idx"]) for observation in available_observations]
    )
    frequencies = np.array([counts[int(index)] for index in frame_indices])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(frame_indices, frequencies, width=1.0)
    ax.axhline(
        np.mean(frequencies),
        color="black",
        linestyle="--",
        linewidth=1,
        label="mittlere Auswahlhäufigkeit",
    )
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("Anzahl Auswahlen")
    ax.set_title("Auswahlhäufigkeit der Beobachtungen")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sampling_frequency_by_frame.png", dpi=180)
    plt.close(fig)

    centers = np.array([
        [
            float(observation["crop_x0"]) + float(observation["crop_width"]) / 2.0,
            float(observation["crop_y0"]) + float(observation["crop_height"]) / 2.0,
        ]
        for observation in available_observations
    ])
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c=frequencies,
        cmap="viridis",
        s=35,
    )
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u [px]")
    ax.set_ylabel("v [px]")
    ax.set_title("Räumliche Stichprobenabdeckung im Kamerabild")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Anzahl Auswahlen")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "sampling_coverage_uv.png", dpi=180)
    plt.close(fig)


def _run_single_observation_count(
    folder_name: str,
    observations_per_subrun: int,
    num_subruns: int,
    random_seed: int,
    run_options: RunOptions | None,
    run_data: dict,
    evaluation_dir: Path,
    prepared_data: PreparedStatisticalCalibration | None = None,
    max_workers: int = 1,
) -> dict:
    if observations_per_subrun <= 0:
        raise ValueError("observations_per_subrun muss positiv sein.")
    if num_subruns <= 0:
        raise ValueError("num_subruns muss positiv sein.")

    available_observations = run_data["observations"]
    num_available = len(available_observations)
    if observations_per_subrun > num_available:
        raise ValueError(
            f"Es wurden {observations_per_subrun} Beobachtungen pro Sub-Run "
            f"angefordert, aber nur {num_available} sind verfügbar."
        )

    if run_options is None:
        run_options = RunOptions(
            save_fit_crop_overlays=False,
            run_trajectory_debug=False,
            run_robot_ray_debug=False,
            run_initial_ray_pair_debug=False,
            run_camera_pose_optimization=True,
            run_optimized_ray_pair_debug=False,
        )

    evaluation_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)
    available_indices = np.array(
        [int(observation["frame_idx"]) for observation in available_observations],
        dtype=int,
    )
    observation_by_index = {
        int(observation["frame_idx"]): observation
        for observation in available_observations
    }

    records: list[MultiRunRecord] = []
    selections: list[dict] = []
    selection_lists: list[dict] = []
    failures: list[dict] = []
    subrun_tasks: list[tuple[int, str, np.ndarray, Path]] = []

    for subrun_index in range(1, num_subruns + 1):
        subrun_name = f"subrun_{subrun_index:03d}"
        selected_indices = np.sort(
            rng.choice(
                available_indices,
                size=observations_per_subrun,
                replace=False,
            )
        )
        subrun_dir = evaluation_dir / subrun_name
        subrun_tasks.append(
            (subrun_index, subrun_name, selected_indices, subrun_dir)
        )
        selection_lists.append({
            "subrun_index": subrun_index,
            "subrun_name": subrun_name,
            "selected_frame_indices": selected_indices.tolist(),
        })

        for sample_position, frame_idx in enumerate(selected_indices, start=1):
            observation = observation_by_index[int(frame_idx)]
            selections.append({
                "subrun_index": subrun_index,
                "subrun_name": subrun_name,
                "frame_idx": int(frame_idx),
                "sample_position": sample_position,
                "crop_center_u_px": (
                    float(observation["crop_x0"])
                    + float(observation["crop_width"]) / 2.0
                ),
                "crop_center_v_px": (
                    float(observation["crop_y0"])
                    + float(observation["crop_height"]) / 2.0
                ),
            })

    def execute_subrun(task):
        subrun_index, subrun_name, selected_indices, subrun_dir = task
        if prepared_data is not None:
            result = run_prepared_statistical_calibration(
                prepared=prepared_data,
                selected_frame_indices=selected_indices.tolist(),
                result_output_dir=subrun_dir,
                verbose=0,
            )
        else:
            result = run_calibration_app(
                folder_name=folder_name,
                options=run_options,
                observation_frame_indices=selected_indices.tolist(),
                result_output_dir=subrun_dir,
            )
        if result is None:
            raise RuntimeError("Kalibrierung lieferte kein Ergebnis.")
        return subrun_index, subrun_name, result

    if max_workers < 1:
        raise ValueError("max_workers muss mindestens 1 sein.")
    if max_workers > 1 and prepared_data is None:
        raise ValueError(
            "Parallele Sub-Runs benötigen vorbereitete Kalibrierdaten."
        )

    if max_workers == 1:
        completed_tasks = []
        for task in subrun_tasks:
            print(
                f"Intra-Run {task[0]}/{num_subruns}: "
                f"{len(task[2])} Beobachtungen"
            )
            try:
                completed_tasks.append(execute_subrun(task))
            except Exception as exc:
                failures.append({
                    "subrun_index": task[0],
                    "subrun_name": task[1],
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })
                print(
                    f"FEHLER in {task[1]}: "
                    f"{type(exc).__name__}: {exc}"
                )
    else:
        print(
            f"Starte {num_subruns} Sub-Runs mit "
            f"{max_workers} parallelen Workern."
        )
        completed_tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_by_task = {
                executor.submit(execute_subrun, task): task
                for task in subrun_tasks
            }
            for future, task in future_by_task.items():
                try:
                    completed_tasks.append(future.result())
                except Exception as exc:
                    failures.append({
                        "subrun_index": task[0],
                        "subrun_name": task[1],
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    })
                    print(
                        f"FEHLER in {task[1]}: "
                        f"{type(exc).__name__}: {exc}"
                    )

    for _, subrun_name, result in sorted(completed_tasks):
        try:
            records.append(_record_from_result(subrun_name, result))
        except Exception as exc:
            failures.append({
                "subrun_index": int(subrun_name.rsplit("_", 1)[-1]),
                "subrun_name": subrun_name,
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
            print(f"FEHLER in {subrun_name}: {type(exc).__name__}: {exc}")

    _write_selection_table(
        selections,
        evaluation_dir / "subrun_observation_selections.csv",
    )
    _save_sampling_plots(available_observations, selections, evaluation_dir)

    if not records:
        raise RuntimeError("Keiner der Sub-Runs konnte ausgewertet werden.")

    statistics = _build_statistics(records)
    _write_results_csv(records, evaluation_dir / "subrun_results.csv")
    _save_plots(records, evaluation_dir)

    selection_counts = Counter(row["frame_idx"] for row in selections)
    all_selection_frequencies = [
        selection_counts[int(frame_idx)] for frame_idx in available_indices
    ]
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_run": folder_name,
        "source_run_path": str(run_data["input_folder"]),
        "sampling": {
            "method": "uniform_random_without_replacement_per_subrun",
            "random_seed": random_seed,
            "prepared_fast_path": prepared_data is not None,
            "max_workers": max_workers,
            "num_available_observations": num_available,
            "num_requested_subruns": num_subruns,
            "observations_per_subrun": observations_per_subrun,
            "num_successful_subruns": len(records),
            "num_failed_subruns": len(failures),
            "total_observation_uses": len(selections),
            "selection_frequency_min": min(all_selection_frequencies),
            "selection_frequency_mean": float(
                np.mean(all_selection_frequencies)
            ),
            "selection_frequency_max": max(all_selection_frequencies),
        },
        "subruns": selection_lists,
        "failures": failures,
        "statistics": statistics,
    }
    summary_path = evaluation_dir / "intrarun_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\nIntra-Run-Auswertung gespeichert unter: {evaluation_dir}")
    print(f"  Erfolgreiche Sub-Runs: {len(records)}/{num_subruns}")
    print(
        "  Positionsstreuung (1 s, x/y/z): "
        f"{statistics['translation']['sample_std_mm']} mm"
    )
    print(
        "  RMS-Orientierungsabweichung: "
        f"{statistics['rotation']['rms_angular_deviation_deg']:.6f} deg"
    )

    return {
        "records": records,
        "failures": failures,
        "statistics": statistics,
        "evaluation_dir": evaluation_dir,
        "summary_path": summary_path,
    }


def _image_count_record(result: dict, observations_per_subrun: int) -> dict:
    statistics = result["statistics"]
    records = result["records"]
    translation_std = np.asarray(
        statistics["translation"]["sample_std_mm"],
        dtype=float,
    )
    translation_range = np.asarray(
        statistics["translation"]["range_mm"],
        dtype=float,
    )
    rotation_std = np.asarray(
        statistics["rotation"]["sample_std_rotation_vector_deg"],
        dtype=float,
    )
    mean_ray_distances_mm = np.asarray(
        [record.mean_ray_distance_m * 1000.0 for record in records],
        dtype=float,
    )
    rmse_ray_distances_mm = np.asarray(
        [record.rmse_ray_distance_m * 1000.0 for record in records],
        dtype=float,
    )
    solver_costs = np.asarray(
        [record.solver_cost for record in records],
        dtype=float,
    )
    solver_nfev = np.asarray(
        [record.solver_nfev for record in records],
        dtype=float,
    )

    def sample_std(values: np.ndarray) -> float:
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    return {
        "observations_per_subrun": observations_per_subrun,
        "num_successful_subruns": len(records),
        "num_failed_subruns": len(result["failures"]),
        "translation_sample_std_x_mm": float(translation_std[0]),
        "translation_sample_std_y_mm": float(translation_std[1]),
        "translation_sample_std_z_mm": float(translation_std[2]),
        "translation_variance_x_mm2": float(translation_std[0] ** 2),
        "translation_variance_y_mm2": float(translation_std[1] ** 2),
        "translation_variance_z_mm2": float(translation_std[2] ** 2),
        "translation_mean_variance_mm2": float(np.mean(translation_std**2)),
        "translation_range_x_mm": float(translation_range[0]),
        "translation_range_y_mm": float(translation_range[1]),
        "translation_range_z_mm": float(translation_range[2]),
        "translation_rms_3d_deviation_mm": float(
            statistics["translation"]["rms_3d_deviation_mm"]
        ),
        "translation_max_3d_deviation_mm": float(
            statistics["translation"]["max_3d_deviation_mm"]
        ),
        "rotation_sample_std_x_deg": float(rotation_std[0]),
        "rotation_sample_std_y_deg": float(rotation_std[1]),
        "rotation_sample_std_z_deg": float(rotation_std[2]),
        "rotation_variance_x_deg2": float(rotation_std[0] ** 2),
        "rotation_variance_y_deg2": float(rotation_std[1] ** 2),
        "rotation_variance_z_deg2": float(rotation_std[2] ** 2),
        "rotation_mean_variance_deg2": float(np.mean(rotation_std**2)),
        "rotation_rms_angular_deviation_deg": float(
            statistics["rotation"]["rms_angular_deviation_deg"]
        ),
        "rotation_max_angular_deviation_deg": float(
            statistics["rotation"]["max_angular_deviation_deg"]
        ),
        "mean_ray_distance_mm": float(
            statistics["fit_quality"]["mean_ray_distance_mm"]
        ),
        "sample_std_mean_ray_distance_mm": sample_std(mean_ray_distances_mm),
        "mean_rmse_ray_distance_mm": float(
            statistics["fit_quality"]["mean_rmse_ray_distance_mm"]
        ),
        "sample_std_rmse_ray_distance_mm": sample_std(rmse_ray_distances_mm),
        "max_ray_distance_across_runs_mm": float(
            statistics["fit_quality"]["max_ray_distance_across_runs_mm"]
        ),
        "solver_success_rate": float(
            np.mean([record.solver_success for record in records])
        ),
        "mean_solver_cost": float(np.mean(solver_costs)),
        "sample_std_solver_cost": sample_std(solver_costs),
        "mean_solver_nfev": float(np.mean(solver_nfev)),
        "sample_std_solver_nfev": sample_std(solver_nfev),
        "evaluation_dir": str(result["evaluation_dir"]),
    }


def _write_image_count_results(
    records: list[dict],
    output_path: Path,
) -> None:
    fieldnames = list(records[0])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _metric_summary(records: list[dict], metric: str) -> dict:
    values = np.asarray([record[metric] for record in records], dtype=float)
    return {
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


PLOT_DATA_FIELDS = [
    "observations_per_subrun",
    "translation_variance_x_mm2",
    "translation_variance_y_mm2",
    "translation_variance_z_mm2",
    "translation_mean_variance_mm2",
    "translation_rms_3d_deviation_mm",
    "translation_max_3d_deviation_mm",
    "rotation_variance_x_deg2",
    "rotation_variance_y_deg2",
    "rotation_variance_z_deg2",
    "rotation_mean_variance_deg2",
    "rotation_rms_angular_deviation_deg",
    "rotation_max_angular_deviation_deg",
    "mean_ray_distance_mm",
    "sample_std_mean_ray_distance_mm",
    "mean_rmse_ray_distance_mm",
    "sample_std_rmse_ray_distance_mm",
    "max_ray_distance_across_runs_mm",
    "solver_success_rate",
    "mean_solver_cost",
    "sample_std_solver_cost",
    "mean_solver_nfev",
    "sample_std_solver_nfev",
]


def _write_plot_data(records: list[dict], output_path: Path) -> None:
    sorted_records = sorted(
        records,
        key=lambda record: record["observations_per_subrun"],
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLOT_DATA_FIELDS)
        writer.writeheader()
        for record in sorted_records:
            writer.writerow({
                field: record[field]
                for field in PLOT_DATA_FIELDS
            })


def _save_image_count_plots(records: list[dict], output_dir: Path) -> None:
    sorted_records = sorted(
        records,
        key=lambda record: record["observations_per_subrun"],
    )
    image_counts = np.asarray(
        [record["observations_per_subrun"] for record in sorted_records],
        dtype=int,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes[0, 0].plot(
        image_counts,
        [record["translation_mean_variance_mm2"] for record in sorted_records],
        "o-",
    )
    axes[0, 0].set_ylabel("Mittlere Varianz [mm²]")
    axes[0, 0].set_title("Translation")

    for component in "xyz":
        axes[0, 1].plot(
            image_counts,
            [
                record[f"translation_variance_{component}_mm2"]
                for record in sorted_records
            ],
            "o-",
            label=component,
        )
    axes[0, 1].set_ylabel("Varianz [mm²]")
    axes[0, 1].set_title("Translationskomponenten")
    axes[0, 1].legend()

    axes[1, 0].plot(
        image_counts,
        [record["rotation_mean_variance_deg2"] for record in sorted_records],
        "o-",
    )
    axes[1, 0].set_ylabel("Mittlere Varianz [deg²]")
    axes[1, 0].set_title("Rotation")

    for component in "xyz":
        axes[1, 1].plot(
            image_counts,
            [
                record[f"rotation_variance_{component}_deg2"]
                for record in sorted_records
            ],
            "o-",
            label=component,
        )
    axes[1, 1].set_ylabel("Varianz [deg²]")
    axes[1, 1].set_title("Rotationsvektorkomponenten")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Verwendete Bilder pro Subrun")
        axis.set_xticks(image_counts)
        axis.grid(True, alpha=0.3)

    fig.suptitle("Kalibrierungsstabilität über der Anzahl verwendeter Bilder")
    fig.tight_layout()
    fig.savefig(output_dir / "stability_vs_image_count.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes[0, 0].errorbar(
        image_counts,
        [record["mean_ray_distance_mm"] for record in sorted_records],
        yerr=[
            record["sample_std_mean_ray_distance_mm"]
            for record in sorted_records
        ],
        fmt="o-",
        capsize=4,
    )
    axes[0, 0].set_ylabel("Mittlere Ray-Distanz [mm]")
    axes[0, 0].set_title("Mittlere Fitqualität")

    axes[0, 1].errorbar(
        image_counts,
        [record["mean_rmse_ray_distance_mm"] for record in sorted_records],
        yerr=[
            record["sample_std_rmse_ray_distance_mm"]
            for record in sorted_records
        ],
        fmt="o-",
        capsize=4,
    )
    axes[0, 1].set_ylabel("Ray-Pair-RMSE [mm]")
    axes[0, 1].set_title("Mittlerer RMSE")

    axes[1, 0].plot(
        image_counts,
        [
            record["max_ray_distance_across_runs_mm"]
            for record in sorted_records
        ],
        "o-",
    )
    axes[1, 0].set_ylabel("Maximale Ray-Distanz [mm]")
    axes[1, 0].set_title("Schlechtester Wert aller Subruns")

    success_axis = axes[1, 1]
    success_axis.plot(
        image_counts,
        [record["solver_success_rate"] * 100.0 for record in sorted_records],
        "o-",
        color="tab:blue",
        label="Erfolgsquote",
    )
    success_axis.set_ylabel("Solver-Erfolgsquote [%]", color="tab:blue")
    success_axis.tick_params(axis="y", labelcolor="tab:blue")
    nfev_axis = success_axis.twinx()
    nfev_axis.plot(
        image_counts,
        [record["mean_solver_nfev"] for record in sorted_records],
        "s--",
        color="tab:orange",
        label="Funktionsauswertungen",
    )
    nfev_axis.set_ylabel(
        "Mittlere Funktionsauswertungen",
        color="tab:orange",
    )
    nfev_axis.tick_params(axis="y", labelcolor="tab:orange")
    success_axis.set_title("Solver-Verhalten")

    for axis in axes.flat:
        axis.set_xlabel("Verwendete Bilder pro Subrun")
        axis.set_xticks(image_counts)
        axis.grid(True, alpha=0.3)

    fig.suptitle("Fitqualität über der Anzahl verwendeter Bilder")
    fig.tight_layout()
    fig.savefig(output_dir / "fit_quality_vs_image_count.png", dpi=180)
    plt.close(fig)


def _normalize_observation_counts(
    observations_per_subrun: int | Sequence[int],
) -> tuple[list[int], bool]:
    if isinstance(observations_per_subrun, bool):
        raise TypeError("observations_per_subrun darf kein boolescher Wert sein.")

    if isinstance(observations_per_subrun, int):
        counts = [observations_per_subrun]
        is_sweep = False
    elif isinstance(observations_per_subrun, Sequence) and not isinstance(
        observations_per_subrun, (str, bytes)
    ):
        counts = list(observations_per_subrun)
        is_sweep = True
    else:
        raise TypeError(
            "observations_per_subrun muss ein Integer oder eine Liste "
            "von Integern sein."
        )

    if not counts:
        raise ValueError("Die Liste der Bildanzahlen darf nicht leer sein.")
    if any(isinstance(count, bool) or not isinstance(count, int) for count in counts):
        raise TypeError("Alle Bildanzahlen müssen Integer sein.")
    if any(count <= 0 for count in counts):
        raise ValueError("Alle Bildanzahlen müssen positiv sein.")
    if len(set(counts)) != len(counts):
        raise ValueError("Bildanzahlen dürfen nicht doppelt angegeben werden.")

    return counts, is_sweep


def run_intrarun_evaluation(
    folder_name: str,
    observations_per_subrun: int | Sequence[int],
    num_subruns: int,
    random_seed: int = 42,
    run_options: RunOptions | None = None,
    max_workers: int = 1,
) -> dict:
    counts, is_sweep = _normalize_observation_counts(observations_per_subrun)
    if num_subruns <= 0:
        raise ValueError("num_subruns muss positiv sein.")

    run_data = load_calibration_run(folder_name)
    num_available = len(run_data["observations"])
    oversized_counts = [count for count in counts if count > num_available]
    if oversized_counts:
        raise ValueError(
            f"Angeforderte Bildanzahlen {oversized_counts} überschreiten die "
            f"{num_available} verfügbaren Beobachtungen."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_dir = (
        Path(run_data["input_folder"])
        / "statistical_evaluation"
        / f"evaluation_{timestamp}"
    )
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    use_fast_path = _can_use_prepared_fast_path(run_options)
    prepared_data = None
    effective_max_workers = max_workers
    if use_fast_path:
        prepared_data = prepare_statistical_calibration(
            folder_name=folder_name,
            output_dir=evaluation_dir / "_prepared_full_run",
        )
    elif max_workers > 1:
        print(
            "Debug-Ausgaben sind aktiviert; verwende die vollständige "
            "Pipeline sequenziell."
        )
        effective_max_workers = 1

    if not is_sweep:
        return _run_single_observation_count(
            folder_name=folder_name,
            observations_per_subrun=counts[0],
            num_subruns=num_subruns,
            random_seed=random_seed,
            run_options=run_options,
            run_data=run_data,
            evaluation_dir=evaluation_dir,
            prepared_data=prepared_data,
            max_workers=effective_max_workers,
        )

    folder_width = max(3, len(str(max(counts))))
    image_count_results: list[dict] = []
    failed_image_counts: list[dict] = []

    for index, count in enumerate(counts, start=1):
        count_dir = evaluation_dir / f"{count:0{folder_width}d}_images_used"
        print(f"\n{'#' * 72}")
        print(f"Bildanzahl {index}/{len(counts)}: {count}")
        print(f"{'#' * 72}")
        try:
            result = _run_single_observation_count(
                folder_name=folder_name,
                observations_per_subrun=count,
                num_subruns=num_subruns,
                random_seed=random_seed,
                run_options=run_options,
                run_data=run_data,
                evaluation_dir=count_dir,
                prepared_data=prepared_data,
                max_workers=effective_max_workers,
            )
            image_count_results.append(_image_count_record(result, count))
        except Exception as exc:
            failed_image_counts.append({
                "observations_per_subrun": count,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "evaluation_dir": str(count_dir),
            })
            print(
                f"FEHLER für {count} Bilder: "
                f"{type(exc).__name__}: {exc}"
            )

    if not image_count_results:
        raise RuntimeError("Keine Bildanzahl konnte erfolgreich ausgewertet werden.")

    results_csv = evaluation_dir / "statistics_by_image_count.csv"
    plot_data_csv = evaluation_dir / "plot_data_by_image_count.csv"
    summary_path = evaluation_dir / "image_count_evaluation_summary.json"
    _write_image_count_results(image_count_results, results_csv)
    _write_plot_data(image_count_results, plot_data_csv)
    _save_image_count_plots(image_count_results, evaluation_dir)

    aggregate_metrics = {
        metric: _metric_summary(image_count_results, metric)
        for metric in (
            "translation_mean_variance_mm2",
            "translation_rms_3d_deviation_mm",
            "translation_max_3d_deviation_mm",
            "rotation_mean_variance_deg2",
            "rotation_rms_angular_deviation_deg",
            "rotation_max_angular_deviation_deg",
            "mean_ray_distance_mm",
            "sample_std_mean_ray_distance_mm",
            "mean_rmse_ray_distance_mm",
            "sample_std_rmse_ray_distance_mm",
            "max_ray_distance_across_runs_mm",
            "solver_success_rate",
            "mean_solver_cost",
            "mean_solver_nfev",
        )
    }
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_run": folder_name,
        "source_run_path": str(run_data["input_folder"]),
        "random_seed_per_image_count": random_seed,
        "num_subruns_per_image_count": num_subruns,
        "prepared_fast_path": prepared_data is not None,
        "max_workers": effective_max_workers,
        "requested_image_counts": counts,
        "successful_image_counts": [
            record["observations_per_subrun"]
            for record in image_count_results
        ],
        "failed_image_counts": failed_image_counts,
        "statistics_by_image_count": image_count_results,
        "statistics_across_image_counts": aggregate_metrics,
        "artifacts": {
            "statistics_csv": str(results_csv),
            "plot_data_csv": str(plot_data_csv),
            "stability_plot": str(
                evaluation_dir / "stability_vs_image_count.png"
            ),
            "fit_quality_plot": str(
                evaluation_dir / "fit_quality_vs_image_count.png"
            ),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\nBildanzahl-Auswertung gespeichert unter: {evaluation_dir}")
    print(
        "  Erfolgreiche Bildanzahlen: "
        f"{len(image_count_results)}/{len(counts)}"
    )

    return {
        "image_count_results": image_count_results,
        "failed_image_counts": failed_image_counts,
        "evaluation_dir": evaluation_dir,
        "results_csv": results_csv,
        "plot_data_csv": plot_data_csv,
        "summary_path": summary_path,
    }
