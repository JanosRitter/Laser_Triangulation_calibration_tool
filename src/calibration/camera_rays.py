from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.calibration.calibration_types import CameraIntrinsics


def _attach_camera_metadata(
    intrinsics: CameraIntrinsics,
    dist_coeffs: np.ndarray | None,
    source: str,
    calibration_path: Path | None = None,
) -> CameraIntrinsics:
    """
    Hängt optionale Zusatzinfos an CameraIntrinsics an, ohne die bestehende
    Dataclass sofort ändern zu müssen.
    """
    intrinsics.dist_coeffs = dist_coeffs
    intrinsics.source = source
    intrinsics.calibration_path = calibration_path
    return intrinsics


def build_camera_intrinsics_from_metadata(run_metadata: dict) -> CameraIntrinsics:
    """
    Baut Kamera-Intrinsics aus run_metadata["camera"].

    Fallback-Modell:
    - einfache Lochkamera
    - keine Verzerrung
    - fx = fy = focal_length / pixel_size
    - cx = img_width / 2
    - cy = img_height / 2
    """
    camera_cfg = run_metadata["camera"]

    img_width = int(camera_cfg["img_width"])
    img_height = int(camera_cfg["img_height"])
    focal_length = float(camera_cfg["focal_length"])
    pixel_size = float(camera_cfg["pixel_size"])

    fx = focal_length / pixel_size
    fy = focal_length / pixel_size

    cx = img_width / 2.0
    cy = img_height / 2.0

    intrinsics = CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_width=img_width,
        img_height=img_height,
    )

    return _attach_camera_metadata(
        intrinsics=intrinsics,
        dist_coeffs=None,
        source="metadata_pinhole",
        calibration_path=None,
    )


def find_intrinsic_calibration_json(run_dir: str | Path) -> Path | None:
    """
    Sucht eine Intrinsic-Kalibrierungsdatei unter:

        run_dir / "intrinsic_camera_calibration" / "*.json"

    Wenn mehrere Dateien vorhanden sind, wird die zuletzt geänderte verwendet.
    """
    run_dir = Path(run_dir)
    calib_dir = run_dir / "intrinsic_camera_calibration"

    if not calib_dir.exists():
        return None

    json_files = sorted(
        calib_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not json_files:
        return None

    return json_files[0]


def build_camera_intrinsics_from_calibration_json(path: str | Path) -> CameraIntrinsics:
    """
    Baut CameraIntrinsics aus einer OpenCV-Kalibrierungs-JSON.

    Erwartet Felder:
    - camera_matrix
    - dist_coeffs
    - image_size.width
    - image_size.height
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    camera_matrix = np.asarray(data["camera_matrix"], dtype=float).reshape(3, 3)
    dist_coeffs = np.asarray(data.get("dist_coeffs", []), dtype=float).reshape(-1)

    image_size = data.get("image_size", {})
    img_width = int(image_size.get("width", 0))
    img_height = int(image_size.get("height", 0))

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    intrinsics = CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_width=img_width,
        img_height=img_height,
    )

    return _attach_camera_metadata(
        intrinsics=intrinsics,
        dist_coeffs=dist_coeffs,
        source="opencv_calibration_json",
        calibration_path=path,
    )


def build_camera_intrinsics_for_run(run_data: dict) -> CameraIntrinsics:
    """
    Zentrale Kamera-Intrinsic-Auswahl.

    Priorität:
    1. Wenn unter run_data["input_folder"]/intrinsic_camera_calibration/*.json
       eine Kalibrierungsdatei liegt, wird diese verwendet.
    2. Sonst wird auf run_metadata zurückgefallen.
    """
    run_dir = Path(run_data["input_folder"])
    calibration_path = find_intrinsic_calibration_json(run_dir)

    if calibration_path is not None:
        intrinsics = build_camera_intrinsics_from_calibration_json(calibration_path)

        print("\n📷 Kamera-Kalibrierungsdatei gefunden:")
        print(f"  Datei: {calibration_path}")
        print("  Modell: OpenCV-Kalibrierung mit optionaler Verzerrungskorrektur")
        print_camera_intrinsics_details(intrinsics)

        return intrinsics

    intrinsics = build_camera_intrinsics_from_metadata(run_data["run_metadata"])

    print("\n📷 Keine Kamera-Kalibrierungsdatei gefunden.")
    print("  Nutze Nennwerte aus run_metadata.json.")
    print("  Modell: einfaches Pinhole-Modell ohne Verzerrung")
    print_camera_intrinsics_details(intrinsics)

    return intrinsics


def print_camera_intrinsics_details(intrinsics: CameraIntrinsics) -> None:
    dist_coeffs = getattr(intrinsics, "dist_coeffs", None)
    source = getattr(intrinsics, "source", "unknown")

    print(f"  Quelle: {source}")
    print(f"  fx = {intrinsics.fx:.10f}")
    print(f"  fy = {intrinsics.fy:.10f}")
    print(f"  cx = {intrinsics.cx:.10f}")
    print(f"  cy = {intrinsics.cy:.10f}")
    print(f"  Bildgröße = {intrinsics.img_width} x {intrinsics.img_height}")

    if dist_coeffs is None or len(dist_coeffs) == 0:
        print("  dist_coeffs = None")
    else:
        print(f"  dist_coeffs = {np.asarray(dist_coeffs, dtype=float).tolist()}")


def _has_nonzero_distortion(intrinsics: CameraIntrinsics) -> bool:
    dist_coeffs = getattr(intrinsics, "dist_coeffs", None)

    if dist_coeffs is None:
        return False

    dist_coeffs = np.asarray(dist_coeffs, dtype=float).reshape(-1)

    if len(dist_coeffs) == 0:
        return False

    return bool(np.any(np.abs(dist_coeffs) > 1e-15))


def _pixel_to_normalized_pinhole_coordinates(
    uv: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    """
    Ideales Pinhole-Modell ohne Verzerrung.

    Achtung zur Achskonvention:
    - Pixel-v wächst im Bild nach unten.
    - Kamera-y soll wie bisher im Projekt nach oben positiv sein.
    - Daher: y = (cy - v) / fy
    """
    uv = np.asarray(uv, dtype=float).reshape(2)

    u, v = uv
    x = (u - intrinsics.cx) / intrinsics.fx
    y = (intrinsics.cy - v) / intrinsics.fy

    return float(x), float(y)


def _pixel_to_normalized_opencv_coordinates(
    uv: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    """
    OpenCV-basierte Umrechnung verzerrter Pixel in ideale normierte Koordinaten.

    OpenCV liefert normierte Koordinaten mit y nach unten:
        x_cv = (u - cx) / fx
        y_cv = (v - cy) / fy

    Das Projekt verwendet bisher y nach oben:
        y_project = -y_cv
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV ist erforderlich, um dist_coeffs zu verwenden. "
            "Installiere z. B. opencv-python oder entferne/zero dist_coeffs."
        ) from exc

    uv = np.asarray(uv, dtype=float).reshape(1, 1, 2)

    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    dist_coeffs = np.asarray(
        getattr(intrinsics, "dist_coeffs", None),
        dtype=float,
    ).reshape(-1)

    undistorted = cv2.undistortPoints(
        src=uv,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        P=None,
    )

    x_cv = float(undistorted[0, 0, 0])
    y_cv = float(undistorted[0, 0, 1])

    x = x_cv
    y = -y_cv

    return x, y


def pixel_to_camera_ray(
    uv: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """
    Wandelt einen Bildpunkt in einen normierten Kamerastrahl um.

    Wenn intrinsics.dist_coeffs vorhanden und nicht null ist:
        OpenCV-undistortPoints wird verwendet.

    Sonst:
        einfaches Pinhole-Modell.
    """
    if _has_nonzero_distortion(intrinsics):
        x, y = _pixel_to_normalized_opencv_coordinates(uv, intrinsics)
    else:
        x, y = _pixel_to_normalized_pinhole_coordinates(uv, intrinsics)

    ray = np.array([x, y, 1.0], dtype=float)
    ray /= np.linalg.norm(ray)

    return ray


def pixels_to_camera_rays(
    uv_list: list[np.ndarray],
    intrinsics: CameraIntrinsics,
) -> list[np.ndarray]:
    """
    Wandelt mehrere Pixelpunkte in normierte Kamerastrahlen um.
    """
    return [pixel_to_camera_ray(uv, intrinsics) for uv in uv_list]

def pixel_to_normalized_camera_coordinates(
    uv: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    """
    Zentrale Umrechnung von Pixelkoordinaten in normierte Kamerakoordinaten.

    Nutzt OpenCV-undistortPoints, wenn dist_coeffs vorhanden sind.
    Sonst Pinhole-Fallback.
    """
    if _has_nonzero_distortion(intrinsics):
        return _pixel_to_normalized_opencv_coordinates(uv, intrinsics)

    return _pixel_to_normalized_pinhole_coordinates(uv, intrinsics)