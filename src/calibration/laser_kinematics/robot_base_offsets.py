from __future__ import annotations

import numpy as np

from src.calibration.calibration_types import RelativePose
from src.calibration.laser_kinematics.transforms import (
    pose_xyzrpy_deg_to_transform,
    invert_transform,
    compose_transforms,
)


DEFAULT_TOOL_OFFSET = {
    # Grobe Schätzung:
    # Laserursprung 10 cm vor Montageplatte, 3 cm darüber.
    #
    # ACHTUNG:
    # Welche Achse "vor" und "oben" ist, hängt vom Flansch-KS ab.
    # Diese Werte sind daher bewusst leicht anpassbar.
    "translation_m": [0.0, 0.10, 0.03],
    "rotation_deg": [0.0, 0.0, 0.0],
}


def _get_offset_value(offset: dict, key_short: str, key_long: str) -> float:
    if key_short in offset:
        return float(offset[key_short])
    if key_long in offset:
        return float(offset[key_long])
    return 0.0


def tool_offset_to_transform(tool_offset: dict | None) -> np.ndarray:
    """
    Baut T_F_L:
        Transformation vom Roboterflansch / Montageplatten-KS zum Laser-KS.

    Konvention:
        p_F = R_F_L @ p_L + t_F_L

    Also:
        ^F T_L = [R, t]

    Wenn tool_offset None ist, wird DEFAULT_TOOL_OFFSET verwendet.
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


def build_absolute_flange_transforms_from_robot_base_offsets(
    trajectory_config: dict,
) -> list[np.ndarray]:
    """
    Baut absolute Flanschposen T_R_Fi aus robot_base_absolute_offsets.

    Erwartete Struktur:
        {
            "type": "robot_base_absolute_offsets",
            "start_pose": [x, y, z, rx, ry, rz],
            "offsets": [
                {"dx": ..., "dy": ..., "dz": ..., "drx": ..., "dry": ..., "drz": ...},
                ...
            ],
            "tool_offset": {
                "translation_m": [...],
                "rotation_deg": [...]
            }
        }

    Wichtig:
    - Die Offsets sind absolute Offsets relativ zur Startpose.
    - Sie werden NICHT sequentiell verkettet.
    - Diese Funktion erzeugt erst nur Flanschposen.
    """
    start_pose = trajectory_config["start_pose"]
    offsets = trajectory_config["offsets"]

    sx, sy, sz, srx, sry, srz = [float(v) for v in start_pose]

    transforms: list[np.ndarray] = []

    for offset in offsets:
        dx = _get_offset_value(offset, "dx", "dx_m")
        dy = _get_offset_value(offset, "dy", "dy_m")
        dz = _get_offset_value(offset, "dz", "dz_m")

        drx = _get_offset_value(offset, "drx", "drx_deg")
        dry = _get_offset_value(offset, "dry", "dry_deg")
        drz = _get_offset_value(offset, "drz", "drz_deg")

        T_r_fi = pose_xyzrpy_deg_to_transform(
            x=sx + dx,
            y=sy + dy,
            z=sz + dz,
            rx_deg=srx + drx,
            ry_deg=sry + dry,
            rz_deg=srz + drz,
        )

        transforms.append(T_r_fi)

    return transforms


def build_absolute_transforms_from_robot_base_offsets(
    trajectory_config: dict,
) -> list[np.ndarray]:
    """
    Baut absolute Laserposen T_R_Li.

    Intern:
        T_R_Fi = Roboter-Flanschpose
        T_F_L  = Tool-Offset Flansch -> Laser
        T_R_Li = T_R_Fi @ T_F_L
    """
    flange_transforms = build_absolute_flange_transforms_from_robot_base_offsets(
        trajectory_config
    )

    tool_offset = trajectory_config.get("tool_offset", None)
    T_f_l = tool_offset_to_transform(tool_offset)

    laser_transforms: list[np.ndarray] = []

    for T_r_fi in flange_transforms:
        T_r_li = compose_transforms(T_r_fi, T_f_l)
        laser_transforms.append(T_r_li)

    return laser_transforms


def build_relative_poses_from_robot_base_offsets(
    trajectory_config: dict,
) -> list[RelativePose]:
    """
    Baut relative Laserposen ^L0 T_Li aus absoluten Laserposen T_R_Li.

        ^L0 T_Li = inv(T_R_L0) @ T_R_Li
    """
    absolute_transforms = build_absolute_transforms_from_robot_base_offsets(
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