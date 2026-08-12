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

DEVIATION_HISTOGRAM_FILENAMES = {
    "individual": "camera_pose_deviation_histograms_individual_axes.png",
    "shared": "camera_pose_deviation_histograms_shared_axes.png",
}

CAMERA_POSITION_DIRECTION_3D_FILENAME = (
    "camera_position_and_direction_3d.png"
)


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


def _nice_histogram_step(raw_step: float) -> float:
    """Round a positive bin width up to a readable decimal step."""
    if not np.isfinite(raw_step) or raw_step <= 0.0:
        return 1.0

    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / (10.0 ** exponent)
    for nice_fraction in (1.0, 2.0, 2.5, 5.0, 10.0):
        if fraction <= nice_fraction:
            return float(nice_fraction * (10.0 ** exponent))
    return float(10.0 ** (exponent + 1.0))


def _adaptive_symmetric_histogram_edges(
    value_arrays: list[np.ndarray],
) -> np.ndarray:
    """
    Build readable, symmetric bins that adapt to spread and sample size.

    Freedman-Diaconis supplies the initial bin width. The resulting bin count
    is bounded so that small samples do not create mostly empty classes while
    large samples can show more detail. All observations remain inside the
    displayed range; no quantile clipping is applied.
    """
    finite_values = [
        np.asarray(values, dtype=float).ravel()
        for values in value_arrays
    ]
    finite_values = [values[np.isfinite(values)] for values in finite_values]
    finite_values = [values for values in finite_values if values.size]
    if not finite_values:
        raise ValueError("Keine endlichen Abweichungswerte fuer Histogramme.")

    combined = np.concatenate(finite_values)
    num_values = combined.size
    max_abs = float(np.max(np.abs(combined)))
    if max_abs == 0.0:
        return np.array([-0.5, 0.5], dtype=float)

    q25, q75 = np.percentile(combined, [25.0, 75.0])
    iqr = float(q75 - q25)
    fd_width = 2.0 * iqr / np.cbrt(num_values) if iqr > 0.0 else np.nan
    full_width = 2.0 * max_abs
    sturges_bins = int(np.ceil(np.log2(num_values) + 1.0))
    estimated_bins = (
        int(np.ceil(full_width / fd_width))
        if np.isfinite(fd_width) and fd_width > 0.0
        else sturges_bins
    )

    min_bins = min(num_values, 4)
    max_bins = min(80, max(4, int(np.ceil(2.0 * np.sqrt(num_values)))))
    target_bins = int(np.clip(estimated_bins, min_bins, max_bins))
    step = _nice_histogram_step(full_width / target_bins)
    half_bin_count = max(1, int(np.ceil(max_abs / step)))
    limit = half_bin_count * step
    return np.linspace(
        -limit,
        limit,
        2 * half_bin_count + 1,
        dtype=float,
    )


def _format_histogram_step(step: float) -> str:
    return f"{step:.6g}"


def _draw_deviation_histogram(
    axis,
    values: np.ndarray,
    edges: np.ndarray,
    title: str,
    color: str,
) -> int:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    counts, _, _ = axis.hist(
        finite_values,
        bins=edges,
        color=color,
        edgecolor="white",
        linewidth=0.6,
    )
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_title(
        f"{title}  |  Bin width: "
        f"{_format_histogram_step(edges[1] - edges[0])}"
    )
    axis.grid(True, axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    return int(np.max(counts)) if counts.size else 0


def _save_deviation_histograms(
    translation_deviations_mm: np.ndarray,
    rotation_deviations_deg: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save individual-scale and shared-scale 2x3 pose histograms."""
    translation_deviations_mm = np.asarray(
        translation_deviations_mm,
        dtype=float,
    )
    rotation_deviations_deg = np.asarray(rotation_deviations_deg, dtype=float)
    if (
        translation_deviations_mm.ndim != 2
        or rotation_deviations_deg.ndim != 2
        or translation_deviations_mm.shape[1] != 3
        or rotation_deviations_deg.shape[1] != 3
    ):
        raise ValueError("Translations- und Rotationsdaten muessen Form (N, 3) haben.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    component_colors = ("tab:blue", "tab:orange", "tab:green")
    row_values = (translation_deviations_mm, rotation_deviations_deg)
    row_titles = (
        ("x_R", "y_R", "z_R"),
        ("dRx", "dRy", "dRz"),
    )
    row_units = ("mm", "deg")

    individual_path = output_dir / DEVIATION_HISTOGRAM_FILENAMES["individual"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row_index, values in enumerate(row_values):
        for component_index in range(3):
            axis = axes[row_index, component_index]
            edges = _adaptive_symmetric_histogram_edges(
                [values[:, component_index]]
            )
            max_count = _draw_deviation_histogram(
                axis=axis,
                values=values[:, component_index],
                edges=edges,
                title=row_titles[row_index][component_index],
                color=component_colors[component_index],
            )
            axis.set_xlim(edges[0], edges[-1])
            axis.set_ylim(0.0, max(1.0, np.ceil(max_count * 1.08)))
            axis.set_xlabel(
                f"Deviation from mean in {row_units[row_index]}"
            )
            axis.set_ylabel("Count")
    fig.suptitle("Camera-pose deviation distributions – individual axes")
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.96))
    fig.savefig(individual_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    shared_path = output_dir / DEVIATION_HISTOGRAM_FILENAMES["shared"]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8),
        sharex="row",
        sharey="row",
    )
    for row_index, values in enumerate(row_values):
        shared_edges = _adaptive_symmetric_histogram_edges(
            [values[:, component_index] for component_index in range(3)]
        )
        row_max_count = 0
        for component_index in range(3):
            row_max_count = max(
                row_max_count,
                _draw_deviation_histogram(
                    axis=axes[row_index, component_index],
                    values=values[:, component_index],
                    edges=shared_edges,
                    title=row_titles[row_index][component_index],
                    color=component_colors[component_index],
                ),
            )
            axes[row_index, component_index].set_xlabel(
                f"Deviation from mean in {row_units[row_index]}"
            )
        for component_index in range(3):
            axis = axes[row_index, component_index]
            axis.set_xlim(shared_edges[0], shared_edges[-1])
            axis.set_ylim(0.0, max(1.0, np.ceil(row_max_count * 1.08)))
        axes[row_index, 0].set_ylabel("Count")
    fig.suptitle(
        "Camera-pose deviation distributions – shared axes by unit"
    )
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.96))
    fig.savefig(shared_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return individual_path, shared_path


def save_deviation_histograms_from_csv(
    results_csv: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """
    Regenerate only the pose-deviation histograms from a results CSV.

    This deliberately avoids loading images or repeating any calibration.
    """
    results_csv = Path(results_csv)
    if output_dir is None:
        output_dir = results_csv.parent

    translation_columns = (
        "dx_from_mean_mm",
        "dy_from_mean_mm",
        "dz_from_mean_mm",
    )
    rotation_columns = (
        "rotation_deviation_x_deg",
        "rotation_deviation_y_deg",
        "rotation_deviation_z_deg",
    )
    required_columns = translation_columns + rotation_columns

    rows = []
    with results_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [
            column for column in required_columns
            if column not in (reader.fieldnames or [])
        ]
        if missing_columns:
            raise ValueError(
                "Fehlende Spalten in Ergebnis-CSV: " + ", ".join(missing_columns)
            )
        for row in reader:
            rows.append([float(row[column]) for column in required_columns])

    if not rows:
        raise ValueError(f"Ergebnis-CSV ist leer: {results_csv}")
    values = np.asarray(rows, dtype=float)
    return _save_deviation_histograms(
        translation_deviations_mm=values[:, :3],
        rotation_deviations_deg=values[:, 3:],
        output_dir=Path(output_dir),
    )


def _automatic_camera_direction_length_mm(
    translations_mm: np.ndarray,
) -> float:
    """Use the sample std of 3D position distances as display length."""
    mean_position_mm = np.mean(translations_mm, axis=0)
    distances_from_mean_mm = np.linalg.norm(
        translations_mm - mean_position_mm,
        axis=1,
    )
    if len(distances_from_mean_mm) < 2:
        return 1.0
    direction_length_mm = float(
        np.std(distances_from_mean_mm, ddof=1)
    )
    if not np.isfinite(direction_length_mm):
        raise ValueError("Position-distance standard deviation is not finite.")
    return max(direction_length_mm, 1e-6)


def _set_tight_metric_3d_limits_from_points(
    axis,
    points: np.ndarray,
    padding_ratio: float = 0.04,
) -> None:
    """Fit the 3D frame tightly while retaining one metric scale on all axes."""
    points = np.asarray(points, dtype=float)
    finite_points = points[np.all(np.isfinite(points), axis=1)]
    if not finite_points.size:
        raise ValueError("Keine endlichen Punkte fuer 3D-Achsengrenzen vorhanden.")

    minima = np.min(finite_points, axis=0)
    maxima = np.max(finite_points, axis=0)
    spans = maxima - minima
    largest_span = max(float(np.max(spans)), 1e-6)
    padding = np.maximum(spans * padding_ratio, largest_span * 0.01)
    lower = minima - padding
    upper = maxima + padding
    padded_spans = upper - lower
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(lower[2], upper[2])
    axis.set_box_aspect(
        np.maximum(padded_spans, float(np.max(padded_spans)) * 0.30)
    )


def save_camera_position_and_direction_3d(
    records: list[MultiRunRecord],
    output_dir: str | Path,
    *,
    direction_length_mm: float | None = None,
) -> Path:
    """
    Plot every camera pose as a position-anchored optical-axis vector.

    For this visualization only, the arrow direction is -z_C expressed in the
    robot frame.  The stored camera pose and model convention are not changed.
    ``direction_length_mm`` changes only the visual arrow length.
    """
    if not records:
        raise ValueError("Mindestens ein MultiRunRecord wird benoetigt.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    translations_mm = (
        np.stack([record.translation_m for record in records]) * 1000.0
    )
    rotations_R_R_C = np.stack(
        [record.rotation_matrix for record in records]
    )
    directions_R = -rotations_R_R_C[:, :, 2]
    direction_norms = np.linalg.norm(directions_R, axis=1)
    if np.any(~np.isfinite(direction_norms)) or np.any(direction_norms <= 1e-12):
        raise ValueError("Ungueltige Kamerablickrichtung in den Pose-Daten.")
    directions_R = directions_R / direction_norms[:, None]

    if direction_length_mm is None:
        direction_length_mm = _automatic_camera_direction_length_mm(
            translations_mm,
        )
    elif not np.isfinite(direction_length_mm) or direction_length_mm <= 0.0:
        raise ValueError("direction_length_mm muss endlich und positiv sein.")
    direction_length_mm = float(direction_length_mm)

    mean_rotation, _, _ = _rotation_statistics(records)
    mean_position_mm = np.mean(translations_mm, axis=0)
    mean_direction_R = -mean_rotation.as_matrix()[:, 2]
    endpoints_mm = translations_mm + direction_length_mm * directions_R
    mean_endpoint_mm = (
        mean_position_mm + direction_length_mm * mean_direction_R
    )
    subrun_alpha = float(
        np.clip(4.5 / np.sqrt(len(records)), 0.10, 0.45)
    )

    fig = plt.figure(figsize=(9, 8))
    axis = fig.add_subplot(111, projection="3d")
    axis.quiver(
        translations_mm[:, 0],
        translations_mm[:, 1],
        translations_mm[:, 2],
        directions_R[:, 0],
        directions_R[:, 1],
        directions_R[:, 2],
        length=direction_length_mm,
        normalize=True,
        arrow_length_ratio=0.18,
        color="tab:blue",
        alpha=subrun_alpha,
        linewidth=1.8,
        zorder=2,
        label="Subrun camera poses",
    )
    axis.quiver(
        mean_position_mm[0],
        mean_position_mm[1],
        mean_position_mm[2],
        mean_direction_R[0],
        mean_direction_R[1],
        mean_direction_R[2],
        length=direction_length_mm,
        normalize=True,
        arrow_length_ratio=0.20,
        color="black",
        linewidth=4.0,
        zorder=10,
        label="Mean camera pose",
    )

    _set_tight_metric_3d_limits_from_points(
        axis,
        np.vstack(
            [
                translations_mm,
                endpoints_mm,
                mean_position_mm,
                mean_endpoint_mm,
            ]
        ),
    )
    axis.set_xlabel("x_R in mm", labelpad=10)
    axis.set_ylabel("y_R in mm", labelpad=10)
    axis.set_zlabel("z_R in mm", labelpad=12)
    axis.locator_params(axis="x", nbins=6)
    axis.locator_params(axis="y", nbins=6)
    axis.locator_params(axis="z", nbins=5)
    axis.set_title(
        "Camera position and viewing direction in the robot frame\n"
        f"Direction-vector display length: {direction_length_mm:.3f} mm"
    )
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.subplots_adjust(left=0.04, right=0.88, bottom=0.07, top=0.88)

    output_path = output_dir / CAMERA_POSITION_DIRECTION_3D_FILENAME
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.25,
    )
    plt.close(fig)
    return output_path


def _draw_component_deviations(
    axis,
    x: np.ndarray,
    values: np.ndarray,
    *,
    ylabel: str,
    title: str,
) -> None:
    width = 0.24
    for index, component in enumerate("xyz"):
        axis.bar(
            x + (index - 1) * width,
            values[:, index],
            width,
            label=f"d{component}",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)


def _draw_total_deviations(
    axis,
    x: np.ndarray,
    values: np.ndarray,
    *,
    ylabel: str,
    title: str,
) -> None:
    axis.bar(x, values)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.3)


def _save_pose_deviation_plot_set(
    *,
    x: np.ndarray,
    short_labels: list[str],
    component_values: np.ndarray,
    total_values: np.ndarray,
    component_ylabel: str,
    total_ylabel: str,
    component_title: str,
    total_title: str,
    combined_title: str,
    combined_path: Path,
    components_path: Path,
    total_path: Path,
) -> tuple[Path, Path, Path]:
    """Save combined, component-only, and total-only deviation figures."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    _draw_component_deviations(
        axes[0],
        x,
        component_values,
        ylabel=component_ylabel,
        title=component_title,
    )
    _draw_total_deviations(
        axes[1],
        x,
        total_values,
        ylabel=total_ylabel,
        title=total_title,
    )
    axes[1].set_xticks(x, short_labels, rotation=30, ha="right")
    fig.suptitle(combined_title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(combined_path, dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    _draw_component_deviations(
        axis,
        x,
        component_values,
        ylabel=component_ylabel,
        title=component_title,
    )
    axis.set_xticks(x, short_labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(components_path, dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    _draw_total_deviations(
        axis,
        x,
        total_values,
        ylabel=total_ylabel,
        title=total_title,
    )
    axis.set_xticks(x, short_labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(total_path, dpi=180)
    plt.close(fig)
    return combined_path, components_path, total_path


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
    position_deviations_3d_mm = np.linalg.norm(
        translation_deviations_mm,
        axis=1,
    )
    x = np.arange(len(records))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis_index, (axis, component) in enumerate(zip(axes, "xyz")):
        axis.plot(x, translations_mm[:, axis_index], "o-", linewidth=1.5)
        axis.axhline(
            np.mean(translations_mm[:, axis_index]),
            color="black", linestyle="--", linewidth=1,
        )
        axis.set_ylabel(f"{component}_R in mm")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xticks(x, short_labels, rotation=30, ha="right")
    fig.suptitle("Optimized camera position by run")
    fig.tight_layout()
    fig.savefig(output_dir / "camera_position_components.png", dpi=180)
    plt.close(fig)

    _save_pose_deviation_plot_set(
        x=x,
        short_labels=short_labels,
        component_values=translation_deviations_mm,
        total_values=position_deviations_3d_mm,
        component_ylabel="Position deviation in mm",
        total_ylabel="3D position deviation in mm",
        component_title="Position deviations by component",
        total_title="Total 3D position deviation",
        combined_title="Camera-position deviations by run",
        combined_path=output_dir / "camera_position_deviations.png",
        components_path=(
            output_dir / "camera_position_deviation_components.png"
        ),
        total_path=output_dir / "camera_position_total_deviation.png",
    )

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
    ax.scatter(*mean, marker="x", s=120, color="black", label="Mean")
    ax.set_xlabel("x_R in mm")
    ax.set_ylabel("y_R in mm")
    ax.set_zlabel("z_R in mm")
    ax.set_title("Camera-center distribution in the robot frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "camera_position_3d.png", dpi=180)
    plt.close(fig)

    save_camera_position_and_direction_3d(records, output_dir)

    _save_pose_deviation_plot_set(
        x=x,
        short_labels=short_labels,
        component_values=rotation_vectors_deg,
        total_values=angular_deviations_deg,
        component_ylabel="Rotation vector in deg",
        total_ylabel="Total angle in deg",
        component_title="Orientation deviation from the mean rotation",
        total_title="Total angular deviation",
        combined_title="Camera-orientation deviations by run",
        combined_path=output_dir / "camera_orientation_deviations.png",
        components_path=(
            output_dir / "camera_orientation_deviation_components.png"
        ),
        total_path=output_dir / "camera_orientation_total_deviation.png",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x,
        [record.rmse_ray_distance_m * 1000.0 for record in records],
    )
    ax.set_xticks(x, short_labels, rotation=30, ha="right")
    ax.set_ylabel("Ray-pair RMSE in mm")
    ax.set_title("Fit quality by run")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fit_quality_by_run.png", dpi=180)
    plt.close(fig)

    _save_deviation_histograms(
        translation_deviations_mm=translation_deviations_mm,
        rotation_deviations_deg=rotation_vectors_deg,
        output_dir=output_dir,
    )


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
