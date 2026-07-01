from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.app.calibration_app import RunOptions, run_calibration_app
from src.evaluation.multirun_evaluation import (
    MultiRunRecord,
    _build_statistics,
    _record_from_result,
    _save_plots,
    _write_results_csv,
)
from src.io.calibration_io import load_calibration_run


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


def run_intrarun_evaluation(
    folder_name: str,
    observations_per_subrun: int,
    num_subruns: int,
    random_seed: int = 42,
    run_options: RunOptions | None = None,
) -> dict:
    if observations_per_subrun <= 0:
        raise ValueError("observations_per_subrun muss positiv sein.")
    if num_subruns <= 0:
        raise ValueError("num_subruns muss positiv sein.")

    run_data = load_calibration_run(folder_name)
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_dir = (
        Path(run_data["input_folder"])
        / "statistical_evaluation"
        / f"evaluation_{timestamp}"
    )
    evaluation_dir.mkdir(parents=True, exist_ok=False)

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

        print(f"\n{'=' * 72}")
        print(
            f"Intra-Run {subrun_index}/{num_subruns}: "
            f"{len(selected_indices)} Beobachtungen"
        )
        print(f"{'=' * 72}")
        try:
            result = run_calibration_app(
                folder_name=folder_name,
                options=run_options,
                observation_frame_indices=selected_indices.tolist(),
                result_output_dir=subrun_dir,
            )
            if result is None:
                raise RuntimeError("Kalibrierung lieferte kein Ergebnis.")
            records.append(_record_from_result(subrun_name, result))
        except Exception as exc:
            failures.append({
                "subrun_index": subrun_index,
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
