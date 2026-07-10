from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.calibration.calibration_types import RelativePose
from src.calibration.laser_kinematics.transforms import (
    pose_xyzrpy_deg_to_transform,
    compose_transforms,
)


DEFAULT_LOCAL_LASER_DIRECTION_L = np.array([1.0, 0.0, 0.0], dtype=float)

DEFAULT_TOOL_OFFSET = {
    # Position des Laser-KS-Ursprungs relativ zum Flansch-/Montageplatten-KS.
    # Konvention:
    #   T_F_L: p_F = R_F_L @ p_L + t_F_L
    #
    # Bedeutet:
    #   translation_m ist der Ursprung des Laser-KS, ausgedrückt im Flansch-KS.
    "translation_m": [0.0, 0.0, 0.042],
    "rotation_deg": [0.0, 0.0, 0.0],
}


FRAME_TABLE_ABSOLUTE_POSE_TRAJECTORY_TYPES = {
    "mixed_cartesian_and_joint_offsets",
    "camera_laser_mixed_offsets",
}


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


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    norm = np.linalg.norm(v)

    if norm <= 1e-15:
        raise ValueError("Vektor darf nicht null sein.")

    return v / norm


def _get_value(data: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in data:
            return float(data[key])
    return float(default)


def tool_offset_to_transform(tool_offset: dict | None) -> np.ndarray:
    """
    Baut T_F_L, also Laser-KS relativ zum Flansch-/Montageplatten-KS.

    Konvention:
        p_F = R_F_L @ p_L + t_F_L

    Der Translationsanteil ist damit der Laserursprung im Flansch-KS.
    """
    if tool_offset is None:
        tool_offset = DEFAULT_TOOL_OFFSET

    translation = tool_offset.get("translation_m", [0.0, 0.0, 0.0])
    rotation = tool_offset.get("rotation_deg", [0.0, 0.0, 0.0])

    return pose_xyzrpy_deg_to_transform(
        x=float(translation[0]),
        y=float(translation[1]),
        z=float(translation[2]),
        rx_deg=float(rotation[0]),
        ry_deg=float(rotation[1]),
        rz_deg=float(rotation[2]),
    )


def flange_pose_xyzrpy_deg_to_transform(pose: list | tuple | np.ndarray) -> np.ndarray:
    """
    Baut T_R_F aus einer Roboter-/Flanschpose:

        [x, y, z, rx, ry, rz]

    Einheit:
        x, y, z in m
        rx, ry, rz in deg
    """
    if len(pose) != 6:
        raise ValueError(
            "Roboterpose muss 6 Werte enthalten: [x, y, z, rx, ry, rz]."
        )

    x, y, z, rx, ry, rz = [float(v) for v in pose]

    return pose_xyzrpy_deg_to_transform(
        x=x,
        y=y,
        z=z,
        rx_deg=rx,
        ry_deg=ry,
        rz_deg=rz,
    )


def flange_transform_to_laser_transform(
    T_R_F: np.ndarray,
    tool_offset: dict | None = None,
) -> np.ndarray:
    """
    Baut aus Flanschpose und Tool-Offset die echte Laserpose:

        T_R_L = T_R_F @ T_F_L
    """
    T_R_F = np.asarray(T_R_F, dtype=float).reshape(4, 4)
    T_F_L = tool_offset_to_transform(tool_offset)

    return compose_transforms(T_R_F, T_F_L)


def ray_from_transform(
    T: np.ndarray,
    local_direction: np.ndarray,
    frame_idx: int | None = None,
) -> Ray3D:
    """
    Baut einen Ray aus einer absoluten Laserpose T_R_L.

    Konvention:
        p_R = R_R_L @ p_L + t_R_L

    Ursprung:
        Laser-KS-Ursprung im Roboter-KS.

    Richtung:
        local_direction aus dem Laser-KS ins Roboter-KS gedreht.
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    local_direction = _normalize(local_direction)

    origin = T[:3, 3].copy()
    direction = T[:3, :3] @ local_direction

    return Ray3D(
        origin=origin,
        direction=direction,
        frame_idx=frame_idx,
    )


def laser_ray_from_flange_pose(
    flange_pose_xyzrpy_deg: list | tuple | np.ndarray,
    local_direction: np.ndarray | None = None,
    tool_offset: dict | None = None,
    frame_idx: int | None = None,
) -> Ray3D:
    """
    Zentraler Pfad:

        Roboter-/Flanschpose
        + Tool-Offset
        + lokale Laserstrahlrichtung
        -> Ray3D im Roboter-KS
    """
    if local_direction is None:
        local_direction = DEFAULT_LOCAL_LASER_DIRECTION_L

    T_R_F = flange_pose_xyzrpy_deg_to_transform(flange_pose_xyzrpy_deg)
    T_R_L = flange_transform_to_laser_transform(
        T_R_F=T_R_F,
        tool_offset=tool_offset,
    )

    return ray_from_transform(
        T=T_R_L,
        local_direction=local_direction,
        frame_idx=frame_idx,
    )


def build_flange_poses_from_robot_base_absolute_offsets(
    trajectory_config: dict,
) -> list[np.ndarray]:
    """
    Kompatibilität zum bisherigen trajectory_config-Format:

        {
            "type": "robot_base_absolute_offsets",
            "start_pose": [x, y, z, rx, ry, rz],
            "offsets": [
                {"dx": ..., "dy": ..., "dz": ..., "drx": ..., "dry": ..., "drz": ...},
                ...
            ],
            "tool_offset": {...}
        }

    Wichtig:
        Die Offsets sind absolute Offsets relativ zur Startpose.
        Sie werden nicht sequentiell verkettet.
    """
    start_pose = trajectory_config["start_pose"]
    offsets = trajectory_config["offsets"]

    sx, sy, sz, srx, sry, srz = [float(v) for v in start_pose]

    poses: list[np.ndarray] = []

    for offset in offsets:
        dx = _get_value(offset, "dx", "dx_m")
        dy = _get_value(offset, "dy", "dy_m")
        dz = _get_value(offset, "dz", "dz_m")

        drx = _get_value(offset, "drx", "drx_deg")
        dry = _get_value(offset, "dry", "dry_deg")
        drz = _get_value(offset, "drz", "drz_deg")

        poses.append(
            np.array(
                [
                    sx + dx,
                    sy + dy,
                    sz + dz,
                    srx + drx,
                    sry + dry,
                    srz + drz,
                ],
                dtype=float,
            )
        )

    return poses


def build_flange_poses_from_trajectory_config(
    trajectory_config: dict,
) -> list[np.ndarray]:
    """
    Liest Roboter-/Flanschposen aus trajectory_config.

    Unterstützte Formate:

    1) Neues direktes Format:
        {
            "type": "robot_base_absolute_poses",
            "poses": [
                [x, y, z, rx, ry, rz],
                ...
            ],
            "tool_offset": {...}
        }

    2) Bisheriges Format:
        {
            "type": "robot_base_absolute_offsets",
            "start_pose": [...],
            "offsets": [...]
        }
    """
    trajectory_type = trajectory_config.get("type")

    if trajectory_type == "robot_base_absolute_poses":
        poses = trajectory_config["poses"]
        return [
            np.asarray(pose, dtype=float).reshape(6)
            for pose in poses
        ]

    if trajectory_type == "robot_base_absolute_offsets":
        return build_flange_poses_from_robot_base_absolute_offsets(
            trajectory_config
        )

    # Fallback: Falls du später ohne type arbeitest, aber poses direkt vorhanden sind.
    if "poses" in trajectory_config:
        return [
            np.asarray(pose, dtype=float).reshape(6)
            for pose in trajectory_config["poses"]
        ]

    raise NotImplementedError(
        f"Trajektorientyp aktuell nicht unterstützt: {trajectory_type!r}"
    )


def _pose_dict_to_xyzrpy_deg(pose: dict) -> np.ndarray:
    return np.array(
        [
            float(pose["laser_x"]),
            float(pose["laser_y"]),
            float(pose["laser_z"]),
            float(pose["laser_rx"]),
            float(pose["laser_ry"]),
            float(pose["laser_rz"]),
        ],
        dtype=float,
    )


def build_flange_poses_from_frame_table_ground_truth(
    run_data: dict,
) -> list[np.ndarray]:
    """
    Baut Flansch-/Montageplattenposen aus den pro Frame gespeicherten Posen.

    Dieses Format wird für Trajektorien verwendet, bei denen die Bewegung über
    Joint-Offsets definiert wurde. Die Kalibrierung soll keine Roboter-
    Vorwärtskinematik nachbilden, sondern die während der Aufnahme gespeicherte
    absolute Pose verwenden.
    """
    frame_poses = run_data.get("ground_truth", {}).get("frame_poses", [])
    if not frame_poses:
        raise ValueError(
            "Keine laser_x/laser_y/laser_z/laser_rx/laser_ry/laser_rz "
            "Posen in frame_table.csv gefunden."
        )

    poses_by_frame = {
        int(pose["frame_idx"]): _pose_dict_to_xyzrpy_deg(pose)
        for pose in frame_poses
    }
    max_frame_idx = max(poses_by_frame)
    missing = [
        frame_idx
        for frame_idx in range(max_frame_idx + 1)
        if frame_idx not in poses_by_frame
    ]
    if missing:
        raise ValueError(
            "frame_table.csv enthält keine durchgehenden Frame-Posen. "
            f"Erste fehlende frame_idx: {missing[:10]}"
        )

    return [
        poses_by_frame[frame_idx]
        for frame_idx in range(max_frame_idx + 1)
    ]


def build_laser_rays_from_flange_poses(
    flange_poses_xyzrpy_deg: list[np.ndarray],
    local_direction: np.ndarray | None = None,
    tool_offset: dict | None = None,
) -> list[Ray3D]:
    """
    Baut Laserrays im Roboter-Basis-KS direkt aus Flanschposen.
    """
    if local_direction is None:
        local_direction = DEFAULT_LOCAL_LASER_DIRECTION_L

    rays: list[Ray3D] = []

    for i, pose in enumerate(flange_poses_xyzrpy_deg):
        rays.append(
            laser_ray_from_flange_pose(
                flange_pose_xyzrpy_deg=pose,
                local_direction=local_direction,
                tool_offset=tool_offset,
                frame_idx=i,
            )
        )

    return rays


def resolve_tool_offset_from_run_metadata(run_metadata: dict) -> dict:
    """
    Ermittelt den Tool-Offset für reale Roboter-Runs.

    Aktuelles Capture-Format:
        run_metadata["tool_offset"]

    Älteres/alternatives Format:
        run_metadata["scan"]["trajectory_config"]["tool_offset"]

    Wenn keine der beiden Angaben existiert, wird aus Legacy-Kompatibilität
    der bisherige DEFAULT_TOOL_OFFSET verwendet.
    """
    if "tool_offset" in run_metadata and run_metadata["tool_offset"] is not None:
        return run_metadata["tool_offset"]

    trajectory_config = (
        run_metadata
        .get("scan", {})
        .get("trajectory_config", {})
    )

    if (
        "tool_offset" in trajectory_config
        and trajectory_config["tool_offset"] is not None
    ):
        return trajectory_config["tool_offset"]

    return DEFAULT_TOOL_OFFSET


def build_laser_rays_robot_base_from_run_data(
    run_data: dict,
    local_direction: np.ndarray | None = None,
) -> list[Ray3D]:
    """
    Baut Laserrays im Roboter-Basis-KS.

    Neuer zentraler Pfad:
        run_data
        -> trajectory_config
        -> Flanschposen
        -> Tool-Offset
        -> Laserposen
        -> Ray3D

    Rückgabe bleibt wie bisher:
        list[Ray3D]
    """
    if local_direction is None:
        local_direction = DEFAULT_LOCAL_LASER_DIRECTION_L

    run_metadata = run_data["run_metadata"]
    trajectory_config = run_metadata["scan"]["trajectory_config"]
    tool_offset = resolve_tool_offset_from_run_metadata(run_metadata)

    trajectory_type = trajectory_config.get("type")
    if trajectory_type in FRAME_TABLE_ABSOLUTE_POSE_TRAJECTORY_TYPES:
        flange_poses = build_flange_poses_from_frame_table_ground_truth(
            run_data
        )
    else:
        flange_poses = build_flange_poses_from_trajectory_config(
            trajectory_config
        )

    return build_laser_rays_from_flange_poses(
        flange_poses_xyzrpy_deg=flange_poses,
        local_direction=local_direction,
        tool_offset=tool_offset,
    )


# -------------------------------------------------------------------------
# Legacy-Kompatibilität für alte Kalibrier-/Simulationspfade
# -------------------------------------------------------------------------

def laser_ray_from_relative_pose(
    relative_pose: RelativePose,
    local_direction: np.ndarray,
    frame_idx: int | None = None,
) -> Ray3D:
    """
    Legacy-Pfad für alte Simulationsdaten mit RelativePose.

    Achtung:
        Hier ist kein Tool-Offset enthalten.
        Für reale Roboterposen sollte build_laser_rays_robot_base_from_run_data()
        verwendet werden.
    """
    local_direction = _normalize(local_direction)

    origin = relative_pose.translation
    direction = relative_pose.rotation @ local_direction

    return Ray3D(
        origin=origin,
        direction=direction,
        frame_idx=frame_idx,
    )


def build_laser_rays_from_observations(
    observations: list,
    local_direction: np.ndarray | None = None,
) -> list[Ray3D]:
    """
    Legacy-Pfad für CalibrationObservations mit obs.relative_pose.

    Für reale Roboterposen ist dieser Pfad nicht empfohlen, weil der Offset
    dort nur korrekt ist, wenn relative_pose bereits eine echte Laserpose ist.
    """
    if local_direction is None:
        local_direction = DEFAULT_LOCAL_LASER_DIRECTION_L

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
