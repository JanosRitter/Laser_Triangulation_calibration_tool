from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    img_width: int
    img_height: int

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )


@dataclass
class RelativePose:
    """
    Relative Pose des Lasers im Frame i relativ zum Startframe L0.

    Transformationskonvention:
        p_L0 = R @ p_Li + t

    Das heißt:
        ^L0 T_Li = [R, t]
    """
    translation: np.ndarray   # shape (3,)
    rotation: np.ndarray      # shape (3, 3)

    def __post_init__(self):
        self.translation = np.asarray(self.translation, dtype=float).reshape(3)
        self.rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)


@dataclass
class ExtrinsicPose:
    """
    Pose des Lasers im Kamera-KS zum Startzeitpunkt.

    Transformationskonvention:
        p_C = R @ p_L0 + t

    Das heißt:
        ^C T_L0 = [R, t]
    """
    translation: np.ndarray   # shape (3,)
    rotation: np.ndarray      # shape (3, 3)

    def __post_init__(self):
        self.translation = np.asarray(self.translation, dtype=float).reshape(3)
        self.rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)


@dataclass
class CalibrationObservation:
    frame_idx: int
    uv: np.ndarray            # shape (2,)
    relative_pose: RelativePose
    weight: float = 1.0

    def __post_init__(self):
        self.uv = np.asarray(self.uv, dtype=float).reshape(2)