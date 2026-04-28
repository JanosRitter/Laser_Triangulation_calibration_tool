from __future__ import annotations

import numpy as np

from src.calibration.calibration_types import RelativePose
from src.calibration.laser_kinematics.transforms import (
    pose_xyzrpy_deg_to_transform,
    invert_transform,
    compose_transforms,
)


def local_increment_to_transform(
    dx: float,
    dy: float,
    dz: float,
    drx_deg: float,
    dry_deg: float,
    drz_deg: float,
) -> np.ndarray:
    return pose_xyzrpy_deg_to_transform(
        dx, dy, dz,
        drx_deg, dry_deg, drz_deg,
    )


def build_absolute_transforms_from_local_increments(
    trajectory_config: dict,
) -> list[np.ndarray]:
    start_pose = trajectory_config["start_pose"]
    include_start_pose = bool(trajectory_config.get("include_start_pose", True))
    increments = trajectory_config["increments"]

    T_current = pose_xyzrpy_deg_to_transform(
        x=float(start_pose[0]),
        y=float(start_pose[1]),
        z=float(start_pose[2]),
        rx_deg=float(start_pose[3]),
        ry_deg=float(start_pose[4]),
        rz_deg=float(start_pose[5]),
    )

    transforms: list[np.ndarray] = []

    if include_start_pose:
        transforms.append(T_current.copy())

    for step in increments:
        dx = float(step["dx_m"])
        dy = float(step["dy_m"])
        dz = float(step["dz_m"])
        drx = float(step["drx_deg"])
        dry = float(step["dry_deg"])
        drz = float(step["drz_deg"])
        repeat = int(step["repeat"])

        T_increment = local_increment_to_transform(
            dx=dx,
            dy=dy,
            dz=dz,
            drx_deg=drx,
            dry_deg=dry,
            drz_deg=drz,
        )

        for _ in range(repeat):
            T_current = compose_transforms(T_current, T_increment)
            transforms.append(T_current.copy())

    return transforms


def build_relative_poses_from_local_increments(
    trajectory_config: dict,
) -> list[RelativePose]:
    absolute_transforms = build_absolute_transforms_from_local_increments(
        trajectory_config
    )

    if len(absolute_transforms) == 0:
        return []

    T_0 = absolute_transforms[0]
    T_0_inv = invert_transform(T_0)

    relative_poses: list[RelativePose] = []

    for T_i in absolute_transforms:
        T_rel = compose_transforms(T_0_inv, T_i)

        relative_poses.append(
            RelativePose(
                translation=T_rel[:3, 3].copy(),
                rotation=T_rel[:3, :3].copy(),
            )
        )

    return relative_poses