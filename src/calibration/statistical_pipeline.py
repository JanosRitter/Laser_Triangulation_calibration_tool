from __future__ import annotations

import json
from dataclasses import dataclass, replace
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.calibration.calibration_types import (
    CalibrationObservation,
    CameraIntrinsics,
)
from src.calibration.camera_pose_in_robot_frame import (
    CameraPoseInRobotFrame,
    camera_pose_from_position_and_axes,
    transform_camera_rays_to_robot_frame,
)
from src.calibration.camera_pose_solver import solve_camera_pose_from_ray_pairs
from src.calibration.camera_rays import (
    build_camera_intrinsics_for_run,
    pixel_to_camera_ray,
)
from src.calibration.laser_rays import Ray3D, build_laser_rays_robot_base_from_run_data
from src.calibration.pipeline import prepare_calibration_observations
from src.calibration.observation_builder import (
    build_calibration_observations_from_fit_table,
)
from src.calibration.ray_pair_analysis import analyze_ray_pair_distances
from src.calibration.robot_to_camera_transform import (
    build_calibration_export_metadata,
    robot_to_camera_transform_from_camera_pose,
    save_robot_to_camera_transform_json,
    transform_rays_R_to_C,
)
from src.io.calibration_io import load_calibration_run
from src.io.io_utils import load_fit_table_csv, save_fit_table_csv


STATISTICAL_FIT_CACHE_VERSION = 1
STATISTICAL_FIT_METHOD = "gaussian"
STATISTICAL_FIT_THRESHOLD_FACTOR = 2.5
STATISTICAL_FIT_SUBTRACT_BACKGROUND = False
STATISTICAL_QUALITY_FILTER = {
    "min_snr": 4.0,
    "min_amplitude": 10.0,
    "sigma_min": 0.8,
    "sigma_max": 40.0,
    "reject_near_border": False,
}


@dataclass
class PreparedStatisticalCalibration:
    folder_name: str
    run_data: dict
    intrinsics: CameraIntrinsics
    initial_pose_R: CameraPoseInRobotFrame
    fit_table_by_frame: dict[int, dict]
    observations_by_frame: dict[int, CalibrationObservation]
    camera_rays_C_by_frame: dict[int, np.ndarray]
    laser_rays_R_by_frame: dict[int, Ray3D]
    preparation_output_dir: Path


def _statistical_fit_cache_dir(run_data: dict) -> Path:
    return Path(run_data["input_folder"]) / "statistical_evaluation" / "_fit_cache"


def _statistical_fit_cache_paths(run_data: dict) -> tuple[Path, Path]:
    cache_dir = _statistical_fit_cache_dir(run_data)
    return (
        cache_dir / "gaussian_fit_table.csv",
        cache_dir / "gaussian_fit_cache_metadata.json",
    )


def _crop_file_signature(run_data: dict) -> list[dict]:
    input_folder = Path(run_data["input_folder"])
    signature = []
    for observation in run_data["observations"]:
        rel_path = str(observation["crop_npy_file"])
        crop_path = input_folder / rel_path
        if not crop_path.exists():
            signature.append({
                "frame_idx": int(observation["frame_idx"]),
                "crop_npy_file": rel_path,
                "exists": False,
            })
            continue

        stat = crop_path.stat()
        signature.append({
            "frame_idx": int(observation["frame_idx"]),
            "crop_npy_file": rel_path,
            "exists": True,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    return signature


def _frame_table_signature(run_data: dict) -> dict:
    frame_table_path = Path(run_data["input_folder"]) / "frame_table.csv"
    stat = frame_table_path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _build_statistical_fit_cache_metadata(run_data: dict) -> dict:
    return {
        "cache_version": STATISTICAL_FIT_CACHE_VERSION,
        "method": STATISTICAL_FIT_METHOD,
        "threshold_factor": STATISTICAL_FIT_THRESHOLD_FACTOR,
        "subtract_background": STATISTICAL_FIT_SUBTRACT_BACKGROUND,
        "quality_filter": STATISTICAL_QUALITY_FILTER,
        "num_total_frames": len(run_data["frame_table"]),
        "num_valid_observations": len(run_data["observations"]),
        "frame_table": _frame_table_signature(run_data),
        "crop_files": _crop_file_signature(run_data),
    }


def _load_valid_statistical_fit_cache(
    run_data: dict,
    expected_metadata: dict,
) -> list[dict] | None:
    fit_table_path, metadata_path = _statistical_fit_cache_paths(run_data)
    if not fit_table_path.exists() or not metadata_path.exists():
        return None

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            cached_metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(
            "Fit-Cache konnte nicht gelesen werden; berechne neu: "
            f"{type(exc).__name__}: {exc}"
        )
        return None

    if cached_metadata != expected_metadata:
        print("Fit-Cache ist nicht mehr gueltig; berechne Gaußfits neu.")
        return None

    fit_table = load_fit_table_csv(fit_table_path)
    if len(fit_table) != expected_metadata["num_valid_observations"]:
        print(
            "Fit-Cache hat eine unerwartete Zeilenanzahl; "
            "berechne Gaußfits neu."
        )
        return None

    print(f"\nFit-Cache gefunden und verwendet: {fit_table_path}")
    return fit_table


def _write_statistical_fit_cache(
    run_data: dict,
    fit_table: list[dict],
    metadata: dict,
) -> Path:
    fit_table_path, metadata_path = _statistical_fit_cache_paths(run_data)
    fit_table_path.parent.mkdir(parents=True, exist_ok=True)
    save_fit_table_csv(fit_table, fit_table_path)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"\nFit-Cache gespeichert: {fit_table_path}")
    return fit_table_path


def _fit_table_stats(fit_table: list[dict], calib_observations: list) -> dict:
    return {
        "num_fit_ok": sum(1 for row in fit_table if bool(row.get("fit_ok", False))),
        "num_fit_total": len(fit_table),
        "num_good": sum(
            1 for row in fit_table if bool(row.get("use_for_calibration", False))
        ),
        "num_fit_table": len(fit_table),
        "num_calib_observations": len(calib_observations),
    }


def _prepare_or_load_statistical_fit_table(
    run_data: dict,
    output_dir: Path,
) -> dict:
    expected_metadata = _build_statistical_fit_cache_metadata(run_data)
    fit_table = _load_valid_statistical_fit_cache(run_data, expected_metadata)
    used_cache = fit_table is not None

    if fit_table is None:
        preparation = prepare_calibration_observations(
            run_data=run_data,
            method=STATISTICAL_FIT_METHOD,
            threshold_factor=STATISTICAL_FIT_THRESHOLD_FACTOR,
            subtract_background=STATISTICAL_FIT_SUBTRACT_BACKGROUND,
            save_fit_overlays=False,
            result_output_dir=output_dir,
            print_fit_reject_details=False,
        )
        fit_table = preparation["fit_table"]
        calib_observations = preparation["calib_observations"]
        _write_statistical_fit_cache(
            run_data=run_data,
            fit_table=fit_table,
            metadata=expected_metadata,
        )
        stats = preparation["stats"]
    else:
        save_fit_table_csv(fit_table, output_dir / "laser_point_fit_table.csv")
        calib_observations = build_calibration_observations_from_fit_table(
            run_data=run_data,
            fit_table=fit_table,
            only_usable=True,
        )
        stats = _fit_table_stats(fit_table, calib_observations)

    return {
        "fit_table": fit_table,
        "calib_observations": calib_observations,
        "stats": stats,
        "used_fit_cache": used_cache,
        "fit_cache_dir": _statistical_fit_cache_dir(run_data),
    }


def prepare_statistical_calibration(
    folder_name: str,
    output_dir: str | Path,
) -> PreparedStatisticalCalibration:
    """Fit and reconstruct every observation once for repeated subset solves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_data = load_calibration_run(folder_name)

    preparation = _prepare_or_load_statistical_fit_table(
        run_data=run_data,
        output_dir=output_dir,
    )
    observations = preparation["calib_observations"]
    intrinsics = build_camera_intrinsics_for_run(run_data)
    all_laser_rays = build_laser_rays_robot_base_from_run_data(
        run_data=run_data,
        local_direction=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    initial_pose_R = camera_pose_from_position_and_axes(
        camera_position_R=np.array([-0.07, 0.975, 0.95], dtype=float),
        camera_z_axis_R=np.array([0.0, 0.0, +1.0], dtype=float),
        camera_x_axis_R=np.array([-1.0, 0.0, 0.0], dtype=float),
    )

    observations_by_frame = {obs.frame_idx: obs for obs in observations}
    return PreparedStatisticalCalibration(
        folder_name=folder_name,
        run_data=run_data,
        intrinsics=intrinsics,
        initial_pose_R=initial_pose_R,
        fit_table_by_frame={
            int(row["frame_idx"]): row for row in preparation["fit_table"]
        },
        observations_by_frame=observations_by_frame,
        camera_rays_C_by_frame={
            frame_idx: pixel_to_camera_ray(obs.uv, intrinsics)
            for frame_idx, obs in observations_by_frame.items()
        },
        laser_rays_R_by_frame={
            frame_idx: all_laser_rays[frame_idx]
            for frame_idx in observations_by_frame
        },
        preparation_output_dir=output_dir,
    )


def prepared_statistical_calibration_with_tool_offset(
    prepared: PreparedStatisticalCalibration,
    tool_offset: dict,
) -> PreparedStatisticalCalibration:
    """
    Reuses the expensive image/crop preparation and rebuilds only laser rays.

    This is useful for statistical offset studies where the image observations
    and camera rays stay fixed, but the laser origin/orientation relative to
    the robot flange is varied.
    """
    run_data = deepcopy(prepared.run_data)
    run_data["run_metadata"]["tool_offset"] = deepcopy(tool_offset)
    all_laser_rays = build_laser_rays_robot_base_from_run_data(
        run_data=run_data,
        local_direction=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    return replace(
        prepared,
        run_data=run_data,
        laser_rays_R_by_frame={
            frame_idx: all_laser_rays[frame_idx]
            for frame_idx in prepared.observations_by_frame
        },
    )


def run_prepared_statistical_calibration(
    prepared: PreparedStatisticalCalibration,
    selected_frame_indices: list[int],
    result_output_dir: str | Path,
    verbose: int = 0,
) -> dict:
    """Solve one subset without repeating crop fitting or ray reconstruction."""
    result_output_dir = Path(result_output_dir)
    result_output_dir.mkdir(parents=True, exist_ok=True)

    requested_indices = [int(value) for value in selected_frame_indices]
    frame_indices = [
        frame_idx
        for frame_idx in requested_indices
        if frame_idx in prepared.observations_by_frame
    ]
    if not frame_indices:
        raise ValueError("Keine verwendbaren vorbereiteten Beobachtungen ausgewählt.")

    calib_observations = [
        prepared.observations_by_frame[frame_idx] for frame_idx in frame_indices
    ]
    camera_rays_C = [
        prepared.camera_rays_C_by_frame[frame_idx] for frame_idx in frame_indices
    ]
    laser_rays_R = [
        prepared.laser_rays_R_by_frame[frame_idx] for frame_idx in frame_indices
    ]

    subset_fit_table = [
        prepared.fit_table_by_frame[frame_idx]
        for frame_idx in requested_indices
        if frame_idx in prepared.fit_table_by_frame
    ]
    save_fit_table_csv(
        subset_fit_table,
        result_output_dir / "laser_point_fit_table.csv",
    )

    initial_pose = prepared.initial_pose_R
    optimization = solve_camera_pose_from_ray_pairs(
        laser_rays_R=laser_rays_R,
        camera_rays_C=camera_rays_C,
        initial_pose_R=initial_pose,
        frame_indices=frame_indices,
        verbose=verbose,
    )
    optimized_pose = optimization.optimized_pose_R
    camera_rays_optimized_R = transform_camera_rays_to_robot_frame(
        camera_rays_C=camera_rays_C,
        camera_pose_R=optimized_pose,
        frame_indices=frame_indices,
    )
    distance_analysis = analyze_ray_pair_distances(
        laser_rays_R=laser_rays_R,
        camera_rays_R=camera_rays_optimized_R,
    )
    transform_C_R = robot_to_camera_transform_from_camera_pose(optimized_pose)
    transform_path = result_output_dir / "robot_to_camera_transform.json"
    metadata = build_calibration_export_metadata(
        output_dir=result_output_dir,
        calib_observations=calib_observations,
        intrinsics=prepared.intrinsics,
        camera_pose_optimization_result=optimization,
        ray_pair_distance_analysis_optimized=distance_analysis,
    )
    save_robot_to_camera_transform_json(
        T_C_R=transform_C_R,
        output_path=transform_path,
        metadata=metadata,
    )

    return {
        "run_data": prepared.run_data,
        "calib_observations": calib_observations,
        "intrinsics": prepared.intrinsics,
        "camera_rays_C": camera_rays_C,
        "camera_pose_initial_R": initial_pose,
        "laser_rays_R": laser_rays_R,
        "camera_pose_optimization_result": optimization,
        "camera_pose_optimized_R": optimized_pose,
        "camera_rays_optimized_R": camera_rays_optimized_R,
        "ray_pair_distance_analysis_optimized": distance_analysis,
        "T_C_R": transform_C_R,
        "laser_rays_C": transform_rays_R_to_C(laser_rays_R, transform_C_R),
        "paths": {
            "output_dir": result_output_dir,
            "robot_to_camera_transform": transform_path,
        },
    }
