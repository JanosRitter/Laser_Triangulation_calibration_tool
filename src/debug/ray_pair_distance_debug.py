from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.calibration.ray_pair_analysis import RayPairDistanceAnalysis

from src.calibration.camera_rays import pixel_to_camera_ray
from src.calibration.camera_pose_in_robot_frame import (
    CameraPoseInRobotFrame,
    transform_camera_ray_to_robot_frame,
)
from src.calibration.calibration_types import CameraIntrinsics


def save_ray_pair_distance_analysis_csv(
    analysis: RayPairDistanceAnalysis,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for c in analysis.closest_approaches:
        rows.append(
            {
                "idx": c.idx,
                "frame_idx": c.frame_idx,
                "laser_x": float(c.laser_point[0]),
                "laser_y": float(c.laser_point[1]),
                "laser_z": float(c.laser_point[2]),
                "camera_x": float(c.camera_point[0]),
                "camera_y": float(c.camera_point[1]),
                "camera_z": float(c.camera_point[2]),
                "midpoint_x": float(c.midpoint[0]),
                "midpoint_y": float(c.midpoint[1]),
                "midpoint_z": float(c.midpoint[2]),
                "distance_m": float(c.distance_m),
                "lambda_laser": float(c.lambda_laser),
                "lambda_camera": float(c.lambda_camera),
                "z_plane_m": float(analysis.fitted_z_m),
                "midpoint_z_residual_m": float(c.midpoint[2] - analysis.fitted_z_m),
            }
        )

    if len(rows) == 0:
        raise ValueError("Keine Analysepunkte zum Speichern übergeben.")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def _extract_ray_pair_xy_arrays(
    analysis: RayPairDistanceAnalysis,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    laser_xy = np.array(
        [[c.laser_point[0], c.laser_point[1]] for c in analysis.closest_approaches],
        dtype=float,
    )
    camera_xy = np.array(
        [[c.camera_point[0], c.camera_point[1]] for c in analysis.closest_approaches],
        dtype=float,
    )
    midpoint_xy = np.array(
        [[c.midpoint[0], c.midpoint[1]] for c in analysis.closest_approaches],
        dtype=float,
    )
    distances = np.array(
        [c.distance_m for c in analysis.closest_approaches],
        dtype=float,
    )

    return laser_xy, camera_xy, midpoint_xy, distances


def _plot_closest_segments_xy(
    ax,
    analysis: RayPairDistanceAnalysis,
) -> None:
    for c in analysis.closest_approaches:
        ax.plot(
            [c.laser_point[0], c.camera_point[0]],
            [c.laser_point[1], c.camera_point[1]],
            linewidth=0.8,
            alpha=0.8,
        )


def _plot_ray_pair_points_xy(
    ax,
    laser_xy: np.ndarray,
    camera_xy: np.ndarray,
    midpoint_xy: np.ndarray,
    distances: np.ndarray,
):
    scatter = ax.scatter(
        midpoint_xy[:, 0],
        midpoint_xy[:, 1],
        c=distances * 1000.0,
        s=35,
        label="midpoint",
    )

    ax.scatter(
        laser_xy[:, 0],
        laser_xy[:, 1],
        s=12,
        marker="x",
        label="closest point laser ray",
    )

    ax.scatter(
        camera_xy[:, 0],
        camera_xy[:, 1],
        s=12,
        marker="+",
        label="closest point camera ray",
    )

    return scatter


def _annotate_ray_pair_indices_xy(
    ax,
    analysis: RayPairDistanceAnalysis,
) -> None:
    for c in analysis.closest_approaches:
        label = str(c.frame_idx) if c.frame_idx is not None else str(c.idx)

        ax.text(
            c.midpoint[0],
            c.midpoint[1],
            label,
            fontsize=7,
        )


def _build_camera_reference_rays_R(
    intrinsics: CameraIntrinsics,
    camera_pose_R: CameraPoseInRobotFrame,
) -> dict[str, object]:
    reference_pixels = {
        "center": np.array([intrinsics.cx, intrinsics.cy], dtype=float),
        "top_left": np.array([0.0, 0.0], dtype=float),
        "top_right": np.array([intrinsics.img_width - 1.0, 0.0], dtype=float),
        "bottom_left": np.array([0.0, intrinsics.img_height - 1.0], dtype=float),
        "bottom_right": np.array(
            [intrinsics.img_width - 1.0, intrinsics.img_height - 1.0],
            dtype=float,
        ),
    }

    rays = {}

    for label, uv in reference_pixels.items():
        ray_C = pixel_to_camera_ray(uv, intrinsics)

        rays[label] = transform_camera_ray_to_robot_frame(
            camera_ray_C=ray_C,
            camera_pose_R=camera_pose_R,
            frame_idx=None,
        )

    return rays


def _plot_camera_reference_rays_xy(
    ax,
    intrinsics: CameraIntrinsics | None,
    camera_pose_R: CameraPoseInRobotFrame | None,
    camera_reference_ray_length: float,
) -> np.ndarray:
    if intrinsics is None or camera_pose_R is None:
        return np.empty((0, 2), dtype=float)

    camera_origin = camera_pose_R.translation

    ax.scatter(
        [camera_origin[0]],
        [camera_origin[1]],
        s=80,
        marker="o",
        label="camera origin",
    )

    ax.text(
        camera_origin[0],
        camera_origin[1],
        "camera",
        fontsize=9,
    )

    reference_rays_R = _build_camera_reference_rays_R(
        intrinsics=intrinsics,
        camera_pose_R=camera_pose_R,
    )

    plotted_points: list[list[float]] = [
        [float(camera_origin[0]), float(camera_origin[1])]
    ]

    for label, ray_R in reference_rays_R.items():
        p0 = ray_R.origin
        p1 = ray_R.origin + camera_reference_ray_length * ray_R.direction

        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            linewidth=1.5,
            linestyle="--",
            label=f"camera {label}" if label == "center" else None,
        )

        ax.text(
            p1[0],
            p1[1],
            label,
            fontsize=8,
        )

        plotted_points.append([float(p0[0]), float(p0[1])])
        plotted_points.append([float(p1[0]), float(p1[1])])

    return np.asarray(plotted_points, dtype=float)


def _set_xy_limits_from_points(
    ax,
    point_sets: list[np.ndarray],
    margin: float = 0.03,
    min_span: float = 0.05,
) -> None:
    valid_sets = [
        points
        for points in point_sets
        if points is not None and len(points) > 0
    ]

    if len(valid_sets) == 0:
        return

    all_xy = np.vstack(valid_sets)

    xy_min = np.min(all_xy, axis=0)
    xy_max = np.max(all_xy, axis=0)
    center = 0.5 * (xy_min + xy_max)
    span = xy_max - xy_min

    span = np.maximum(span + 2.0 * margin, min_span)

    ax.set_xlim(center[0] - span[0] / 2.0, center[0] + span[0] / 2.0)
    ax.set_ylim(center[1] - span[1] / 2.0, center[1] + span[1] / 2.0)


def plot_ray_pair_distance_xy(
    analysis: RayPairDistanceAnalysis,
    output_path: str | Path,
    annotate_indices: bool = True,
    intrinsics: CameraIntrinsics | None = None,
    camera_pose_R: CameraPoseInRobotFrame | None = None,
    camera_reference_ray_length: float = 0.5,
) -> Path:
    """
    XY-Plot der kürzesten Verbindungssegmente zwischen Ray-Paaren.

    Darstellung:
        - Linie: Punkt auf Laser-Ray zu Punkt auf Kamera-Ray
        - Punkt: Mittelpunkt der Verbindungsstrecke
        - Farbe: Länge der Verbindungsstrecke / Ray-Abstand
        - optional: Kameraursprung und Referenzrays für center/eckpixel
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(analysis.closest_approaches) == 0:
        raise ValueError("Keine Analysepunkte zum Plotten übergeben.")

    laser_xy, camera_xy, midpoint_xy, distances = _extract_ray_pair_xy_arrays(
        analysis
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    _plot_closest_segments_xy(ax, analysis)

    scatter = _plot_ray_pair_points_xy(
        ax=ax,
        laser_xy=laser_xy,
        camera_xy=camera_xy,
        midpoint_xy=midpoint_xy,
        distances=distances,
    )

    if annotate_indices:
        _annotate_ray_pair_indices_xy(ax, analysis)

    camera_reference_xy = _plot_camera_reference_rays_xy(
        ax=ax,
        intrinsics=intrinsics,
        camera_pose_R=camera_pose_R,
        camera_reference_ray_length=camera_reference_ray_length,
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Ray-Abstand [mm]")

    ax.set_title(
        "Ray-Pair-Abstände im x-y-Plot\n"
        f"z-Ebene: z = {analysis.fitted_z_m:.6f} m | "
        f"mean distance = {analysis.mean_distance_m * 1000.0:.3f} mm"
    )

    ax.set_xlabel("x_R [m]")
    ax.set_ylabel("y_R [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")

    _set_xy_limits_from_points(
        ax=ax,
        point_sets=[
            laser_xy,
            camera_xy,
            midpoint_xy,
            camera_reference_xy,
        ],
        margin=0.03,
        min_span=0.05,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")
    save_ray_pair_distance_analysis_csv(
        analysis=analysis,
        output_path=csv_path,
    )

    return output_path

    return output_path