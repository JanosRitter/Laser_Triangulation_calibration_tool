from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from src.app.calibration_app import RunOptions, run_calibration_app


RUN_TIMESTAMP_PATTERN = re.compile(r"^(?P<timestamp>\d{8}_\d{6})")


@dataclass
class MultiRunRecord:
    run_name: str
    num_observations: int
    translation_m: np.ndarray
    rotation_matrix: np.ndarray
    euler_xyz_deg: np.ndarray
    mean_ray_distance_m: float
    rmse_ray_distance_m: float
    max_ray_distance_m: float
    solver_success: bool
    solver_cost: float
    solver_nfev: int


def _common_prefix(values: list[str]) -> str:
    if not values:
        return ""

    prefix = values[0]
    for value in values[1:]:
        while not value.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def _build_multirun_output_name(folder_names: list[str]) -> str:
    timestamps = []
    for folder_name in folder_names:
        match = RUN_TIMESTAMP_PATTERN.match(Path(folder_name).name)
        if match is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"multirun_{timestamp}"
        timestamps.append(match.group("timestamp"))

    sorted_timestamps = sorted(timestamps)
    first_timestamp = sorted_timestamps[0]
    last_timestamp = sorted_timestamps[-1]

    if first_timestamp == last_timestamp:
        return f"multirun_{first_timestamp}"

    prefix = _common_prefix([first_timestamp, last_timestamp]).rstrip("_")
    first_suffix = first_timestamp[len(prefix):].lstrip("_")
    last_suffix = last_timestamp[len(prefix):].lstrip("_")

    if not prefix or not first_suffix or not last_suffix:
        return f"multirun_{first_timestamp}_to_{last_timestamp}"

    return f"multirun_{prefix}_{first_suffix}_{last_suffix}"


def _make_unique_output_dir(base_output_dir: Path, folder_names: list[str]) -> Path:
    output_name = _build_multirun_output_name(folder_names)
    candidate = base_output_dir / output_name
    if not candidate.exists():
        return candidate

    for counter in range(2, 1000):
        numbered_candidate = base_output_dir / f"{output_name}_{counter:02d}"
        if not numbered_candidate.exists():
            return numbered_candidate

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_output_dir / f"{output_name}_{timestamp}"


def _record_from_result(run_name: str, result: dict) -> MultiRunRecord:
    optimization = result["camera_pose_optimization_result"]
    pose = result["camera_pose_optimized_R"]

    if optimization is None or pose is None:
        raise RuntimeError(f"Run {run_name!r} lieferte keine optimierte Kamerapose.")

    return MultiRunRecord(
        run_name=run_name,
        num_observations=len(result["calib_observations"]),
        translation_m=pose.translation.copy(),
        rotation_matrix=pose.rotation.copy(),
        euler_xyz_deg=Rotation.from_matrix(pose.rotation).as_euler(
            "xyz", degrees=True
        ),
        mean_ray_distance_m=float(optimization.mean_distance_optimized_m),
        rmse_ray_distance_m=float(optimization.rmse_distance_optimized_m),
        max_ray_distance_m=float(optimization.max_distance_optimized_m),
        solver_success=bool(optimization.solver_result.success),
        solver_cost=float(optimization.solver_result.cost),
        solver_nfev=int(optimization.solver_result.nfev),
    )


def _rotation_statistics(
    records: list[MultiRunRecord],
) -> tuple[Rotation, np.ndarray, np.ndarray]:
    rotations = Rotation.from_matrix(
        np.stack([record.rotation_matrix for record in records])
    )
    mean_rotation = rotations.mean()
    relative_rotations = mean_rotation.inv() * rotations
    rotation_vectors_deg = np.rad2deg(relative_rotations.as_rotvec())
    angular_deviations_deg = np.linalg.norm(rotation_vectors_deg, axis=1)
    return mean_rotation, rotation_vectors_deg, angular_deviations_deg


def _sample_std(values: np.ndarray, axis=0):
    if len(values) < 2:
        return np.zeros_like(np.mean(values, axis=axis))
    return np.std(values, axis=axis, ddof=1)


def _build_statistics(records: list[MultiRunRecord]) -> dict:
    translations_m = np.stack([record.translation_m for record in records])
    euler_deg = np.stack([record.euler_xyz_deg for record in records])
    mean_rotation, rotation_vectors_deg, angular_deviations_deg = (
        _rotation_statistics(records)
    )

    mean_translation_m = np.mean(translations_m, axis=0)
    translation_deviations_mm = (translations_m - mean_translation_m) * 1000.0
    translation_distance_mm = np.linalg.norm(translation_deviations_mm, axis=1)

    return {
        "num_runs": len(records),
        "translation": {
            "mean_m": mean_translation_m.tolist(),
            "sample_std_mm": (
                _sample_std(translations_m, axis=0) * 1000.0
            ).tolist(),
            "range_mm": (
                (np.max(translations_m, axis=0) - np.min(translations_m, axis=0))
                * 1000.0
            ).tolist(),
            "rms_3d_deviation_mm": float(
                np.sqrt(np.mean(translation_distance_mm**2))
            ),
            "max_3d_deviation_mm": float(np.max(translation_distance_mm)),
        },
        "rotation": {
            "mean_matrix_R_R_C": mean_rotation.as_matrix().tolist(),
            "mean_euler_xyz_deg": mean_rotation.as_euler(
                "xyz", degrees=True
            ).tolist(),
            "sample_std_rotation_vector_deg": _sample_std(
                rotation_vectors_deg, axis=0
            ).tolist(),
            "rms_angular_deviation_deg": float(
                np.sqrt(np.mean(angular_deviations_deg**2))
            ),
            "max_angular_deviation_deg": float(
                np.max(angular_deviations_deg)
            ),
            "euler_xyz_deg_for_display": euler_deg.tolist(),
        },
        "fit_quality": {
            "mean_ray_distance_mm": float(
                np.mean([r.mean_ray_distance_m for r in records]) * 1000.0
            ),
            "mean_rmse_ray_distance_mm": float(
                np.mean([r.rmse_ray_distance_m for r in records]) * 1000.0
            ),
            "max_ray_distance_across_runs_mm": float(
                np.max([r.max_ray_distance_m for r in records]) * 1000.0
            ),
        },
    }


def _write_results_csv(
    records: list[MultiRunRecord],
    output_path: Path,
) -> None:
    mean_rotation, rotation_vectors_deg, angular_deviations_deg = (
        _rotation_statistics(records)
    )
    del mean_rotation
    translations_m = np.stack([record.translation_m for record in records])
    translation_deviations_mm = (
        translations_m - np.mean(translations_m, axis=0)
    ) * 1000.0

    fieldnames = [
        "run_name", "num_observations", "solver_success",
        "x_m", "y_m", "z_m", "rx_deg", "ry_deg", "rz_deg",
        "dx_from_mean_mm", "dy_from_mean_mm", "dz_from_mean_mm",
        "position_deviation_3d_mm",
        "rotation_deviation_x_deg", "rotation_deviation_y_deg",
        "rotation_deviation_z_deg", "angular_deviation_deg",
        "mean_ray_distance_mm", "rmse_ray_distance_mm",
        "max_ray_distance_mm", "solver_cost", "solver_nfev",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records):
            t = record.translation_m
            e = record.euler_xyz_deg
            dt = translation_deviations_mm[index]
            dr = rotation_vectors_deg[index]
            writer.writerow({
                "run_name": record.run_name,
                "num_observations": record.num_observations,
                "solver_success": record.solver_success,
                "x_m": t[0], "y_m": t[1], "z_m": t[2],
                "rx_deg": e[0], "ry_deg": e[1], "rz_deg": e[2],
                "dx_from_mean_mm": dt[0],
                "dy_from_mean_mm": dt[1],
                "dz_from_mean_mm": dt[2],
                "position_deviation_3d_mm": np.linalg.norm(dt),
                "rotation_deviation_x_deg": dr[0],
                "rotation_deviation_y_deg": dr[1],
                "rotation_deviation_z_deg": dr[2],
                "angular_deviation_deg": angular_deviations_deg[index],
                "mean_ray_distance_mm": record.mean_ray_distance_m * 1000.0,
                "rmse_ray_distance_mm": record.rmse_ray_distance_m * 1000.0,
                "max_ray_distance_mm": record.max_ray_distance_m * 1000.0,
                "solver_cost": record.solver_cost,
                "solver_nfev": record.solver_nfev,
            })


def _save_plots(records: list[MultiRunRecord], output_dir: Path) -> None:
    labels = [record.run_name for record in records]
    short_labels = [label.split("_")[1] if "_" in label else label for label in labels]
    translations_mm = (
        np.stack([record.translation_m for record in records]) * 1000.0
    )
    translation_deviations_mm = (
        translations_mm - np.mean(translations_mm, axis=0)
    )
    _, rotation_vectors_deg, angular_deviations_deg = _rotation_statistics(records)
    x = np.arange(len(records))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis_index, (axis, component) in enumerate(zip(axes, "xyz")):
        axis.plot(x, translations_mm[:, axis_index], "o-", linewidth=1.5)
        axis.axhline(
            np.mean(translations_mm[:, axis_index]),
            color="black", linestyle="--", linewidth=1,
        )
        axis.set_ylabel(f"{component}_R [mm]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xticks(x, short_labels, rotation=30, ha="right")
    fig.suptitle("Optimierte Kameraposition je Run")
    fig.tight_layout()
    fig.savefig(output_dir / "camera_position_components.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.24
    for index, component in enumerate("xyz"):
        ax.bar(
            x + (index - 1) * width,
            translation_deviations_mm[:, index],
            width,
            label=f"d{component}",
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, short_labels, rotation=30, ha="right")
    ax.set_ylabel("Abweichung vom Mittelwert [mm]")
    ax.set_title("Positionsabweichungen der Runs")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "camera_position_deviations.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        translations_mm[:, 0],
        translations_mm[:, 1],
        translations_mm[:, 2],
        s=55,
    )
    for index, label in enumerate(short_labels):
        ax.text(*translations_mm[index], f" {label}", fontsize=8)
    mean = np.mean(translations_mm, axis=0)
    ax.scatter(*mean, marker="x", s=120, color="black", label="Mittelwert")
    ax.set_xlabel("x_R [mm]")
    ax.set_ylabel("y_R [mm]")
    ax.set_zlabel("z_R [mm]")
    ax.set_title("Streuung des Kamerazentrums im Roboter-KS")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "camera_position_3d.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    width = 0.24
    for index, component in enumerate("xyz"):
        axes[0].bar(
            x + (index - 1) * width,
            rotation_vectors_deg[:, index],
            width,
            label=f"d{component}",
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Rotationsvektor [deg]")
    axes[0].set_title("Orientierungsabweichung von der mittleren Rotation")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x, angular_deviations_deg)
    axes[1].set_ylabel("Gesamtwinkel [deg]")
    axes[1].set_xticks(x, short_labels, rotation=30, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "camera_orientation_deviations.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x,
        [record.rmse_ray_distance_m * 1000.0 for record in records],
    )
    ax.set_xticks(x, short_labels, rotation=30, ha="right")
    ax.set_ylabel("Ray-Pair-RMSE [mm]")
    ax.set_title("Fitqualität je Run")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fit_quality_by_run.png", dpi=180)
    plt.close(fig)


def run_multirun_evaluation(
    folder_names: list[str],
    output_dir: str | Path = "data/multirun_evaluation",
    run_options: RunOptions | None = None,
) -> dict:
    if not folder_names:
        raise ValueError("folder_names darf nicht leer sein.")

    if run_options is None:
        run_options = RunOptions()

    base_output_dir = Path(output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = _make_unique_output_dir(base_output_dir, folder_names)
    output_dir.mkdir(parents=True, exist_ok=False)

    records: list[MultiRunRecord] = []
    failures: list[dict[str, str]] = []

    for index, folder_name in enumerate(folder_names, start=1):
        print(f"\n{'=' * 72}")
        print(f"Multi-Run {index}/{len(folder_names)}: {folder_name}")
        print(f"{'=' * 72}")
        try:
            result = run_calibration_app(
                folder_name,
                options=run_options,
                result_output_dir=output_dir / folder_name,
            )
            if result is None:
                raise RuntimeError("Kalibrierung lieferte kein Ergebnis.")
            records.append(_record_from_result(folder_name, result))
        except Exception as exc:
            failures.append({
                "run_name": folder_name,
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
            print(f"FEHLER in {folder_name}: {type(exc).__name__}: {exc}")

    if not records:
        raise RuntimeError("Keiner der übergebenen Runs konnte ausgewertet werden.")

    statistics = _build_statistics(records)
    results_csv = output_dir / "multirun_results.csv"
    summary_json = output_dir / "multirun_summary.json"
    _write_results_csv(records, results_csv)
    _save_plots(records, output_dir)

    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(output_dir),
        "requested_runs": folder_names,
        "successful_runs": [record.run_name for record in records],
        "failed_runs": failures,
        "statistics": statistics,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\nMulti-Run-Auswertung gespeichert unter: {output_dir.resolve()}")
    print(f"  Erfolgreiche Runs: {len(records)}/{len(folder_names)}")
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
        "output_dir": output_dir,
        "results_csv": results_csv,
        "summary_json": summary_json,
    }
