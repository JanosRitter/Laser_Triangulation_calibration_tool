from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.calibration.laser_rays import Ray3D


@dataclass
class CameraPoseInRobotFrame:
    """
    Pose der Kamera im Roboterbasis-/Roboter-Laser-KS R.

    Konvention:
        p_R = R_R_C @ p_C + t_R_C

    Dabei ist:
        t_R_C: Kamerazentrum im Roboter-KS
        R_R_C: Orientierung des Kamera-KS im Roboter-KS

    Kamera-KS:
        +x_C rechts im Bild
        +y_C oben im Bild
        +z_C Blickrichtung
    """
    translation: np.ndarray
    rotation: np.ndarray

    def __post_init__(self):
        self.translation = np.asarray(self.translation, dtype=float).reshape(3)
        self.rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)

        _validate_rotation_matrix(self.rotation)


def _normalize(v: np.ndarray, name: str = "vector") -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    norm = np.linalg.norm(v)

    if norm <= 1e-15:
        raise ValueError(f"{name} must not be zero.")

    return v / norm


def _validate_rotation_matrix(R: np.ndarray) -> None:
    R = np.asarray(R, dtype=float).reshape(3, 3)

    should_be_I = R.T @ R
    det = np.linalg.det(R)

    if not np.allclose(should_be_I, np.eye(3), atol=1e-6):
        raise ValueError("rotation is not orthonormal.")

    if not np.isclose(det, 1.0, atol=1e-6):
        raise ValueError(f"rotation determinant must be +1, got {det}.")


def build_camera_rotation_from_axes(
    camera_z_axis_R: np.ndarray,
    camera_x_axis_R: np.ndarray,
) -> np.ndarray:
    """
    Baut R_R_C aus grob angegebenen Kameraachsen im Roboter-KS.

    Eingabe:
        camera_z_axis_R:
            Blickrichtung der Kamera im Roboter-KS.

        camera_x_axis_R:
            Richtung von +x_C im Roboter-KS.
            Also: rechts im Bild.

    Kamera-Konvention:
        +x_C rechts im Bild
        +y_C oben im Bild
        +z_C Blickrichtung

    Rückgabe:
        R_R_C mit Spalten:
            [x_C_in_R, y_C_in_R, z_C_in_R]

    Wichtig:
        Die Eingabeachsen müssen nicht perfekt orthogonal sein.
        x wird gegen z orthogonalisiert.
    """
    z_R = _normalize(camera_z_axis_R, name="camera_z_axis_R")

    x_guess_R = _normalize(camera_x_axis_R, name="camera_x_axis_R")

    # x-Anteil entlang z entfernen, damit x senkrecht zu z wird.
    x_R = x_guess_R - np.dot(x_guess_R, z_R) * z_R
    x_R = _normalize(x_R, name="orthogonalized camera_x_axis_R")

    # Rechtshändiges Kamera-KS:
    # x × y = z  =>  y = z × x
    y_R = np.cross(z_R, x_R)
    y_R = _normalize(y_R, name="camera_y_axis_R")

    R_R_C = np.column_stack([x_R, y_R, z_R])

    _validate_rotation_matrix(R_R_C)

    return R_R_C


def camera_pose_from_position_and_axes(
    camera_position_R: np.ndarray,
    camera_z_axis_R: np.ndarray,
    camera_x_axis_R: np.ndarray,
) -> CameraPoseInRobotFrame:
    """
    Baut eine Kamerapose im Roboter-KS aus anschaulichen Startwerten.
    """
    R_R_C = build_camera_rotation_from_axes(
        camera_z_axis_R=camera_z_axis_R,
        camera_x_axis_R=camera_x_axis_R,
    )

    return CameraPoseInRobotFrame(
        translation=np.asarray(camera_position_R, dtype=float).reshape(3),
        rotation=R_R_C,
    )


def transform_camera_ray_to_robot_frame(
    camera_ray_C: np.ndarray,
    camera_pose_R: CameraPoseInRobotFrame,
    frame_idx: int | None = None,
) -> Ray3D:
    """
    Transformiert einen Kameraray aus dem Kamera-KS C ins Roboter-KS R.

    Kamera-Ray in C:
        origin_C = (0, 0, 0)
        direction_C = camera_ray_C

    Transformiert:
        origin_R = t_R_C
        direction_R = R_R_C @ direction_C
    """
    direction_C = _normalize(camera_ray_C, name="camera_ray_C")

    origin_R = camera_pose_R.translation.copy()
    direction_R = camera_pose_R.rotation @ direction_C

    return Ray3D(
        origin=origin_R,
        direction=direction_R,
        frame_idx=frame_idx,
    )


def transform_camera_rays_to_robot_frame(
    camera_rays_C: list[np.ndarray],
    camera_pose_R: CameraPoseInRobotFrame,
    frame_indices: list[int] | None = None,
) -> list[Ray3D]:
    """
    Transformiert mehrere Kamerarays ins Roboter-KS.
    """
    if frame_indices is not None and len(frame_indices) != len(camera_rays_C):
        raise ValueError(
            "frame_indices muss dieselbe Länge wie camera_rays_C haben."
        )

    rays_R: list[Ray3D] = []

    for i, ray_C in enumerate(camera_rays_C):
        frame_idx = frame_indices[i] if frame_indices is not None else i

        rays_R.append(
            transform_camera_ray_to_robot_frame(
                camera_ray_C=ray_C,
                camera_pose_R=camera_pose_R,
                frame_idx=frame_idx,
            )
        )

    return rays_R


def camera_pose_to_matrix(
    camera_pose_R: CameraPoseInRobotFrame,
) -> np.ndarray:
    """
    Baut die 4x4-Matrix ^R T_C.

    Konvention:
        p_R = R_R_C @ p_C + t_R_C
    """
    T_R_C = np.eye(4, dtype=float)
    T_R_C[:3, :3] = camera_pose_R.rotation
    T_R_C[:3, 3] = camera_pose_R.translation

    return T_R_C


def print_camera_pose_in_robot_frame(
    camera_pose_R: CameraPoseInRobotFrame,
    label: str = "Kamerapose im Roboter-KS",
) -> None:
    """
    Kleine Debug-Ausgabe ohne Plotting.
    """
    R = camera_pose_R.rotation
    t = camera_pose_R.translation

    print(f"\n📷 {label}:")
    print(f"  position_R = ({t[0]:+.6f}, {t[1]:+.6f}, {t[2]:+.6f}) m")
    print("  Achsen im Roboter-KS:")
    print(f"    x_C rechts       = ({R[0,0]:+.6f}, {R[1,0]:+.6f}, {R[2,0]:+.6f})")
    print(f"    y_C oben         = ({R[0,1]:+.6f}, {R[1,1]:+.6f}, {R[2,1]:+.6f})")
    print(f"    z_C Blickrichtung= ({R[0,2]:+.6f}, {R[1,2]:+.6f}, {R[2,2]:+.6f})")