from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.calibration.calibration_types import RelativePose
from src.calibration.laser_kinematics.robot_base_offsets import (
    build_absolute_transforms_from_robot_base_offsets,
)
from src.calibration.observation_builder import (
    build_relative_pose_list_from_run_data,
)


@dataclass
class Ray3D:
    origin: np.ndarray
    direction: np.ndarray
    frame_idx: int | None = None

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)
        self.direction = np.asarray(self.direction, dtype=float).reshape(3)

        norm = np.linalg.norm(self.direction)
        if norm <= 1e-15:
            raise ValueError("Ray direction must not be zero.")

        self.direction = self.direction / norm


def ray_from_transform(
    T: np.ndarray,
    local_direction: np.ndarray,
    frame_idx: int | None = None,
) -> Ray3D:
    """
    Baut einen Ray aus einer absoluten Pose.

    Konvention:
        p_parent = R_parent_child @ p_child + t_parent_child

    Der Ray-Ursprung ist der Ursprung des Child-KS im Parent-KS.
    Die Ray-Richtung ist local_direction aus dem Child-KS in Parent-KS gedreht.
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    local_direction = np.asarray(local_direction, dtype=float).reshape(3)

    origin = T[:3, 3].copy()
    direction = T[:3, :3] @ local_direction

    return Ray3D(
        origin=origin,
        direction=direction,
        frame_idx=frame_idx,
    )


def laser_ray_from_relative_pose(
    relative_pose: RelativePose,
    local_direction: np.ndarray,
    frame_idx: int | None = None,
) -> Ray3D:
    """
    Baut einen Laserray im Start-Laser-KS L0 aus ^L0 T_Li.

    RelativePose-Konvention:
        p_L0 = R @ p_Li + t

    Daher:
        origin_L0    = t
        direction_L0 = R @ local_direction_Li
    """
    local_direction = np.asarray(local_direction, dtype=float).reshape(3)

    origin = relative_pose.translation
    direction = relative_pose.rotation @ local_direction

    return Ray3D(
        origin=origin,
        direction=direction,
        frame_idx=frame_idx,
    )


def build_laser_rays_from_relative_poses(
    relative_poses: list[RelativePose],
    local_direction: np.ndarray | None = None,
) -> list[Ray3D]:
    """
    Baut Laserrays im L0-KS aus einer Liste relativer Laserposen.
    """
    if local_direction is None:
        local_direction = np.array([0.0, 1.0, 0.0], dtype=float)

    return [
        laser_ray_from_relative_pose(
            relative_pose=pose,
            local_direction=local_direction,
            frame_idx=i,
        )
        for i, pose in enumerate(relative_poses)
    ]


def build_laser_rays_from_observations(
    observations: list,
    local_direction: np.ndarray | None = None,
) -> list[Ray3D]:
    """
    Baut genau die Laserrays, die zu den verwendeten CalibrationObservations gehören.

    Vorteil:
    Nicht alle Frames, sondern nur kalibrierrelevante Beobachtungen.
    """
    if local_direction is None:
        local_direction = np.array([0.0, 1.0, 0.0], dtype=float)

    rays: list[Ray3D] = []

    for obs in observations:
        rays.append(
            laser_ray_from_relative_pose(
                relative_pose=obs.relative_pose,
                local_direction=local_direction,
                frame_idx=obs.frame_idx,
            )
        )

    return rays


def build_laser_rays_L0_from_run_data(
    run_data: dict,
    local_direction: np.ndarray | None = None,
) -> list[Ray3D]:
    """
    Baut Laserrays im Start-Laser-KS L0 für alle Frames des Runs.
    """
    relative_poses = build_relative_pose_list_from_run_data(run_data)

    return build_laser_rays_from_relative_poses(
        relative_poses=relative_poses,
        local_direction=local_direction,
    )


def build_laser_rays_robot_base_from_run_data(
    run_data: dict,
    local_direction: np.ndarray | None = None,
) -> list[Ray3D]:
    """
    Baut Laserrays im Roboter-Basis-KS.

    Das ersetzt die bisherige Kernlogik aus robot_ray_debug.py.
    Sinnvoll für Debug/Visualisierung absoluter Roboterposen.
    """
    if local_direction is None:
        local_direction = np.array([0.0, 1.0, 0.0], dtype=float)

    trajectory_config = run_data["run_metadata"]["scan"]["trajectory_config"]
    transforms = build_absolute_transforms_from_robot_base_offsets(trajectory_config)

    return [
        ray_from_transform(
            T=T,
            local_direction=local_direction,
            frame_idx=i,
        )
        for i, T in enumerate(transforms)
    ]