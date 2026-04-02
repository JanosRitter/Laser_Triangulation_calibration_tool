from __future__ import annotations

import numpy as np

from src.calibration.calibration_types import CameraIntrinsics


def build_camera_intrinsics_from_metadata(run_metadata: dict) -> CameraIntrinsics:
    """
    Baut Kamera-Intrinsics aus run_metadata["camera"].

    Konventionen:
    -------------
    Für Simulationsdaten:
    - Principal Point wird als (img_width / 2, img_height / 2) gesetzt
    - Diese Konvention MUSS mit der Vorwärtsprojektion in der Simulation übereinstimmen

    Für reale Kameras (zukünftig):
    - cx, cy sollten aus einer echten Kamerakalibrierung stammen
    - nicht aus Bildgröße abgeleitet werden

    Annahmen:
    ----------
    - quadratische Pixel
    - pinhole camera model
    """
    camera_cfg = run_metadata["camera"]

    img_width = int(camera_cfg["img_width"])
    img_height = int(camera_cfg["img_height"])
    focal_length = float(camera_cfg["focal_length"])
    pixel_size = float(camera_cfg["pixel_size"])

    # Brennweite in Pixel
    fx = focal_length / pixel_size
    fy = focal_length / pixel_size

    # ---------------------------------------------------------
    # Principal Point
    # ---------------------------------------------------------
    # WICHTIG:
    # Hier wird die gleiche Konvention verwendet wie in der Simulation:
    #   cx = img_width / 2
    #   cy = img_height / 2
    #
    # Alternative (nicht hier verwendet):
    #   (img_width - 1) / 2
    #
    # Diese Wahl beeinflusst die Geometrie messbar (~0.1 mm Unterschied!)
    # und muss daher konsistent zur Projektion sein.
    cx = img_width / 2.0
    cy = img_height / 2.0

    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_width=img_width,
        img_height=img_height,
    )


def pixel_to_camera_ray(uv: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    uv = np.asarray(uv, dtype=float).reshape(2)

    u, v = uv
    x = (u - intrinsics.cx) / intrinsics.fx
    y = (intrinsics.cy - v) / intrinsics.fy

    ray = np.array([x, y, 1.0], dtype=float)
    ray /= np.linalg.norm(ray)

    return ray


def pixels_to_camera_rays(uv_list: list[np.ndarray], intrinsics: CameraIntrinsics) -> list[np.ndarray]:
    """
    Wandelt mehrere Pixelpunkte in normierte Kamerastrahlen um.
    """
    return [pixel_to_camera_ray(uv, intrinsics) for uv in uv_list]