from __future__ import annotations

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
from src.calibration.ray_pair_analysis import analyze_ray_pair_distances
from src.calibration.robot_to_camera_transform import (
    build_calibration_export_metadata,
    robot_to_camera_transform_from_camera_pose,
    save_robot_to_camera_transform_json,
    transform_rays_R_to_C,
)
from src.io.calibration_io import load_calibration_run
from src.io.io_utils import save_fit_table_csv


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


def prepare_statistical_calibration(
    folder_name: str,
    output_dir: str | Path,
) -> PreparedStatisticalCalibration:
    """Fit and reconstruct every observation once for repeated subset solves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_data = load_calibration_run(folder_name)

    preparation = prepare_calibration_observations(
        run_data=run_data,
        method="gaussian",
        threshold_factor=2.5,
        subtract_background=False,
        save_fit_overlays=False,
        result_output_dir=output_dir,
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
