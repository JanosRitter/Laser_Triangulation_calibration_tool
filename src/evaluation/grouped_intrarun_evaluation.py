from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.app.calibration_app import RunOptions
from src.calibration.statistical_pipeline import prepare_statistical_calibration
from src.evaluation.intrarun_evaluation import (
    _can_use_prepared_fast_path,
    _image_count_record,
    _run_single_observation_count,
)
from src.io.calibration_io import load_calibration_run


@dataclass(frozen=True)
class ObservationGroup:
    """Benannte Auswahl aus inklusiven Frame-Index-Bereichen."""

    name: str
    frame_ranges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Der Gruppenname darf nicht leer sein.")
        if not self.frame_ranges:
            raise ValueError(f"Gruppe {self.name!r} enthält keine Frame-Bereiche.")

        covered_indices: set[int] = set()
        for frame_range in self.frame_ranges:
            if len(frame_range) != 2:
                raise ValueError(
                    f"Ungültiger Frame-Bereich in Gruppe {self.name!r}: "
                    f"{frame_range!r}"
                )
            start, end = frame_range
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
            ):
                raise TypeError("Frame-Bereichsgrenzen müssen Integer sein.")
            if start < 0 or end < start:
                raise ValueError(
                    f"Ungültiger inklusiver Bereich ({start}, {end}) "
                    f"in Gruppe {self.name!r}."
                )
            indices = set(range(start, end + 1))
            overlap = covered_indices & indices
            if overlap:
                raise ValueError(
                    f"Überlappende Bereiche in Gruppe {self.name!r}; "
                    f"erste doppelte frame_idx: {min(overlap)}"
                )
            covered_indices.update(indices)

    @property
    def requested_frame_indices(self) -> list[int]:
        return [
            frame_idx
            for start, end in self.frame_ranges
            for frame_idx in range(start, end + 1)
        ]

    @property
    def ranges_label(self) -> str:
        return " + ".join(
            f"{start}-{end}" for start, end in self.frame_ranges
        )


def _safe_folder_name(index: int, name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    return f"group_{index:02d}_{slug or 'unnamed'}"


def _build_group_run_data(run_data: dict, group: ObservationGroup) -> tuple[dict, dict]:
    requested = set(group.requested_frame_indices)
    selected_observations = [
        observation
        for observation in run_data["observations"]
        if int(observation["frame_idx"]) in requested
    ]
    available_indices = sorted(
        int(observation["frame_idx"])
        for observation in selected_observations
    )
    missing_indices = sorted(requested - set(available_indices))

    filtered_run_data = dict(run_data)
    filtered_run_data["observations"] = selected_observations
    group_info = {
        "name": group.name,
        "frame_ranges_inclusive": [list(value) for value in group.frame_ranges],
        "frame_ranges_label": group.ranges_label,
        "num_nominal_frames": len(requested),
        "num_available_observations": len(available_indices),
        "available_frame_indices": available_indices,
        "filtered_or_missing_frame_indices": missing_indices,
    }
    return filtered_run_data, group_info


def _group_result_record(
    result: dict,
    group_info: dict,
    observations_per_subrun: int,
) -> dict:
    record = _image_count_record(result, observations_per_subrun)
    mean_position_m = np.asarray(
        result["statistics"]["translation"]["mean_m"],
        dtype=float,
    )
    record = {
        "group_name": group_info["name"],
        "frame_ranges": group_info["frame_ranges_label"],
        "num_nominal_frames": group_info["num_nominal_frames"],
        "num_available_observations": group_info["num_available_observations"],
        "mean_camera_x_m": float(mean_position_m[0]),
        "mean_camera_y_m": float(mean_position_m[1]),
        "mean_camera_z_m": float(mean_position_m[2]),
        **record,
    }
    return record


def _write_group_results(records: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _save_group_comparison_plots(
    records: list[dict],
    output_dir: Path,
) -> None:
    labels = [record["group_name"] for record in records]
    x = np.arange(len(records))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes[0, 0].plot(
        x,
        [record["translation_mean_variance_mm2"] for record in records],
        "o-",
    )
    axes[0, 0].set_ylabel("Mittlere Varianz [mm²]")
    axes[0, 0].set_title("Translation")

    for component in "xyz":
        axes[0, 1].plot(
            x,
            [
                record[f"translation_variance_{component}_mm2"]
                for record in records
            ],
            "o-",
            label=component,
        )
    axes[0, 1].set_ylabel("Varianz [mm²]")
    axes[0, 1].set_title("Translationskomponenten")
    axes[0, 1].legend()

    axes[1, 0].plot(
        x,
        [record["rotation_mean_variance_deg2"] for record in records],
        "o-",
    )
    axes[1, 0].set_ylabel("Mittlere Varianz [deg²]")
    axes[1, 0].set_title("Rotation")

    for component in "xyz":
        axes[1, 1].plot(
            x,
            [
                record[f"rotation_variance_{component}_deg2"]
                for record in records
            ],
            "o-",
            label=component,
        )
    axes[1, 1].set_ylabel("Varianz [deg²]")
    axes[1, 1].set_title("Rotationsvektorkomponenten")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(True, alpha=0.3)

    fig.suptitle("Kalibrierungsstabilität nach Beobachtungsgruppe")
    fig.tight_layout()
    fig.savefig(output_dir / "stability_by_observation_group.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes[0, 0].bar(
        x,
        [record["translation_rms_3d_deviation_mm"] for record in records],
    )
    axes[0, 0].set_ylabel("RMS-Positionsabweichung [mm]")
    axes[0, 0].set_title("Positionsstreuung")

    axes[0, 1].bar(
        x,
        [record["rotation_rms_angular_deviation_deg"] for record in records],
    )
    axes[0, 1].set_ylabel("RMS-Winkelabweichung [deg]")
    axes[0, 1].set_title("Orientierungsstreuung")

    axes[1, 0].errorbar(
        x,
        [record["mean_rmse_ray_distance_mm"] for record in records],
        yerr=[
            record["sample_std_rmse_ray_distance_mm"]
            for record in records
        ],
        fmt="o",
        capsize=4,
    )
    axes[1, 0].set_ylabel("Ray-Pair-RMSE [mm]")
    axes[1, 0].set_title("Fitqualität: Mittelwert ± 1σ")

    axes[1, 1].bar(
        x,
        [record["solver_success_rate"] * 100.0 for record in records],
    )
    axes[1, 1].set_ylabel("Solver-Erfolgsquote [%]")
    axes[1, 1].set_ylim(0.0, 105.0)
    axes[1, 1].set_title("Solver-Verhalten")

    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Qualitätskennzahlen nach Beobachtungsgruppe")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_by_observation_group.png", dpi=180)
    plt.close(fig)

    mean_positions_mm = np.asarray([
        [
            record["mean_camera_x_m"],
            record["mean_camera_y_m"],
            record["mean_camera_z_m"],
        ]
        for record in records
    ]) * 1000.0
    overall_mean_mm = np.mean(mean_positions_mm, axis=0)

    fig = plt.figure(figsize=(12, 8))
    axis = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(records)))
    for index, label in enumerate(labels):
        axis.scatter(
            mean_positions_mm[index, 0],
            mean_positions_mm[index, 1],
            mean_positions_mm[index, 2],
            color=colors[index],
            s=75,
            depthshade=False,
            label=label,
        )
    axis.scatter(
        overall_mean_mm[0],
        overall_mean_mm[1],
        overall_mean_mm[2],
        marker="x",
        s=140,
        linewidths=2.5,
        color="black",
        label="Mittelwert aller Gruppen",
    )
    spans = np.ptp(mean_positions_mm, axis=0)
    largest_span = max(float(np.max(spans)), 1e-9)
    axis.set_box_aspect(np.maximum(spans, largest_span * 0.05))
    axis.set_xlabel("x_R [mm]")
    axis.set_ylabel("y_R [mm]")
    axis.set_zlabel("z_R [mm]")
    axis.set_title("Mittlere Kameraposition je Beobachtungsgruppe")
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(
        output_dir / "camera_mean_position_by_observation_group_3d.png",
        dpi=180,
    )
    plt.close(fig)


def run_grouped_intrarun_evaluation(
    folder_name: str,
    observation_groups: list[ObservationGroup],
    observations_per_subrun: int,
    num_subruns: int,
    random_seed: int = 42,
    run_options: RunOptions | None = None,
    max_workers: int = 1,
) -> dict:
    if not observation_groups:
        raise ValueError("observation_groups darf nicht leer sein.")
    if len({group.name for group in observation_groups}) != len(observation_groups):
        raise ValueError("Gruppennamen müssen eindeutig sein.")
    if observations_per_subrun <= 0:
        raise ValueError("observations_per_subrun muss positiv sein.")
    if num_subruns <= 0:
        raise ValueError("num_subruns muss positiv sein.")

    run_data = load_calibration_run(folder_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_dir = (
        Path(run_data["input_folder"])
        / "statistical_evaluation"
        / f"group_evaluation_{timestamp}"
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

    group_results: list[dict] = []
    group_details: list[dict] = []
    failed_groups: list[dict] = []

    for index, group in enumerate(observation_groups, start=1):
        group_run_data, group_info = _build_group_run_data(run_data, group)
        group_details.append(group_info)
        if observations_per_subrun > group_info["num_available_observations"]:
            raise ValueError(
                f"Gruppe {group.name!r} enthält nur "
                f"{group_info['num_available_observations']} gültige "
                f"Beobachtungen, angefordert sind {observations_per_subrun}."
            )

        group_dir = evaluation_dir / _safe_folder_name(index, group.name)
        print(f"\n{'#' * 72}")
        print(
            f"Beobachtungsgruppe {index}/{len(observation_groups)}: "
            f"{group.name} ({group.ranges_label})"
        )
        print(
            f"Gültig: {group_info['num_available_observations']}/"
            f"{group_info['num_nominal_frames']}"
        )
        print(f"{'#' * 72}")

        try:
            result = _run_single_observation_count(
                folder_name=folder_name,
                observations_per_subrun=observations_per_subrun,
                num_subruns=num_subruns,
                random_seed=random_seed,
                run_options=run_options,
                run_data=group_run_data,
                evaluation_dir=group_dir,
                prepared_data=prepared_data,
                max_workers=effective_max_workers,
            )
            group_results.append(
                _group_result_record(
                    result,
                    group_info,
                    observations_per_subrun,
                )
            )
        except Exception as exc:
            failed_groups.append({
                "group_name": group.name,
                "frame_ranges": group.ranges_label,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "evaluation_dir": str(group_dir),
            })
            print(
                f"FEHLER in Gruppe {group.name!r}: "
                f"{type(exc).__name__}: {exc}"
            )

    if not group_results:
        raise RuntimeError("Keine Beobachtungsgruppe konnte ausgewertet werden.")

    results_csv = evaluation_dir / "statistics_by_observation_group.csv"
    summary_path = evaluation_dir / "group_evaluation_summary.json"
    _write_group_results(group_results, results_csv)
    _save_group_comparison_plots(group_results, evaluation_dir)

    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_run": folder_name,
        "source_run_path": str(run_data["input_folder"]),
        "sampling": {
            "method": "uniform_random_without_replacement_per_group_and_subrun",
            "observations_per_subrun": observations_per_subrun,
            "num_subruns_per_group": num_subruns,
            "random_seed_per_group": random_seed,
            "prepared_fast_path": prepared_data is not None,
            "max_workers": effective_max_workers,
        },
        "observation_groups": group_details,
        "statistics_by_observation_group": group_results,
        "failed_groups": failed_groups,
        "artifacts": {
            "statistics_csv": str(results_csv),
            "stability_plot": str(
                evaluation_dir / "stability_by_observation_group.png"
            ),
            "quality_plot": str(
                evaluation_dir / "quality_by_observation_group.png"
            ),
            "mean_camera_position_3d_plot": str(
                evaluation_dir
                / "camera_mean_position_by_observation_group_3d.png"
            ),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\nGruppen-Auswertung gespeichert unter: {evaluation_dir}")
    print(
        f"  Erfolgreiche Gruppen: "
        f"{len(group_results)}/{len(observation_groups)}"
    )

    return {
        "group_results": group_results,
        "failed_groups": failed_groups,
        "evaluation_dir": evaluation_dir,
        "results_csv": results_csv,
        "summary_path": summary_path,
    }
