from __future__ import annotations

import numpy as np

from src.calibration.calibration_types import RelativePose

from src.calibration.laser_kinematics.transforms import (
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    euler_xyz_deg_to_matrix,
    make_transform,
    invert_transform,
    compose_transforms,
    pose_xyzrpy_deg_to_transform,
    build_extrinsic_pose_from_xyzrpy_deg,
    laser_pose_in_camera,
    laser_ray_in_camera,
)

from src.calibration.laser_kinematics.local_increments import (
    local_increment_to_transform,
    build_absolute_transforms_from_local_increments,
    build_relative_poses_from_local_increments,
)
from src.calibration.laser_kinematics.robot_base_offsets import (
    build_absolute_transforms_from_robot_base_offsets,
    build_relative_poses_from_robot_base_offsets,
)

def build_absolute_transforms_from_trajectory_config(
    trajectory_config: dict,
) -> list[np.ndarray]:
    trajectory_type = trajectory_config["type"]

    if trajectory_type == "local_increments":
        return build_absolute_transforms_from_local_increments(trajectory_config)

    if trajectory_type == "robot_base_absolute_offsets":
        return build_absolute_transforms_from_robot_base_offsets(trajectory_config)

    raise NotImplementedError(
        f"Trajektorientyp aktuell nicht unterstützt: {trajectory_type}"
    )


def build_relative_poses_from_trajectory_config(
    trajectory_config: dict,
) -> list[RelativePose]:
    trajectory_type = trajectory_config["type"]

    if trajectory_type == "local_increments":
        return build_relative_poses_from_local_increments(trajectory_config)

    if trajectory_type == "robot_base_absolute_offsets":
        return build_relative_poses_from_robot_base_offsets(trajectory_config)

    raise NotImplementedError(
        f"Trajektorientyp aktuell nicht unterstützt: {trajectory_type}"
    )


__all__ = [
    "rotation_matrix_x",
    "rotation_matrix_y",
    "rotation_matrix_z",
    "euler_xyz_deg_to_matrix",
    "make_transform",
    "invert_transform",
    "compose_transforms",
    "pose_xyzrpy_deg_to_transform",
    "build_extrinsic_pose_from_xyzrpy_deg",
    "laser_pose_in_camera",
    "laser_ray_in_camera",
    "local_increment_to_transform",
    "build_absolute_transforms_from_local_increments",
    "build_relative_poses_from_local_increments",
    "build_absolute_transforms_from_trajectory_config",
    "build_relative_poses_from_trajectory_config",
    "build_absolute_transforms_from_robot_base_offsets",
    "build_relative_poses_from_robot_base_offsets",
]