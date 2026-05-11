from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.calibration.camera_rays import (
    pixel_to_camera_ray,
    pixel_to_normalized_camera_coordinates,
)
from src.calibration.calibration_types import CameraIntrinsics


def build_camera_ray_debug_rows(
    uv_list: list[np.ndarray],
    intrinsics: CameraIntrinsics,
    ray_length: float = 1.0,
) -> list[dict]:
    rows: list[dict] = []

    for i, uv in enumerate(uv_list):
        uv = np.asarray(uv, dtype=float).reshape(2)
        ray = pixel_to_camera_ray(uv, intrinsics)
        x_norm, y_norm = pixel_to_normalized_camera_coordinates(uv, intrinsics)

        origin = np.zeros(3, dtype=float)
        end = origin + ray_length * ray

        rows.append(
            {
                "idx": i,
                "u_px": float(uv[0]),
                "v_px": float(uv[1]),
                "x_norm": float(x_norm),
                "y_norm": float(y_norm),
                "origin_x": 0.0,
                "origin_y": 0.0,
                "origin_z": 0.0,
                "dir_x": float(ray[0]),
                "dir_y": float(ray[1]),
                "dir_z": float(ray[2]),
                "end_x": float(end[0]),
                "end_y": float(end[1]),
                "end_z": float(end[2]),
            }
        )

    return rows

def _camera_ray_intersection_with_z_plane(
    uv: np.ndarray,
    intrinsics: CameraIntrinsics,
    z_plane: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Schneidet den Kameraray mit der Ebene z = z_plane.

    Da der Ray im Kamera-KS vom Ursprung startet:
        p(s) = s * direction

    Für z = z_plane gilt:
        s = z_plane / direction_z

    Bei deinem Modell entspricht z=1 praktisch den normierten
    Kamerakoordinaten:
        x = (u - cx) / fx
        y = (cy - v) / fy
    """
    uv = np.asarray(uv, dtype=float).reshape(2)
    ray = pixel_to_camera_ray(uv, intrinsics)

    if abs(ray[2]) <= 1e-15:
        raise ValueError("Kameraray ist parallel zur z-Ebene.")

    scale = z_plane / ray[2]
    point = scale * ray

    return ray, point


def save_camera_ray_debug_csv(
    uv_list: list[np.ndarray],
    intrinsics: CameraIntrinsics,
    output_path: str | Path,
    ray_length: float = 1.0,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_camera_ray_debug_rows(
        uv_list=uv_list,
        intrinsics=intrinsics,
        ray_length=ray_length,
    )

    if len(rows) == 0:
        raise ValueError("Keine Kamerarays zum Speichern übergeben.")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def plot_camera_rays_in_camera_frame(
    uv_list: list[np.ndarray],
    intrinsics: CameraIntrinsics,
    output_path: str | Path,
    ray_length: float = 1.0,
    max_rays: int | None = 200,
    annotate_indices: bool = True,
) -> Path:
    """
    Plottet Kamerarays im Kamera-KS.

    Konvention:
    - Ursprung aller Rays: (0, 0, 0)
    - +x_C: rechts im Bild
    - +y_C: oben im Bild
    - +z_C: Blickrichtung
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if max_rays is not None:
        uv_list_plot = uv_list[:max_rays]
    else:
        uv_list_plot = uv_list

    if len(uv_list_plot) == 0:
        raise ValueError("Keine Kamerarays zum Plotten übergeben.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    origin = np.zeros(3, dtype=float)
    points_for_limits = [origin]

    for i, uv in enumerate(uv_list_plot):
        uv = np.asarray(uv, dtype=float).reshape(2)
        ray = pixel_to_camera_ray(uv, intrinsics)
        end = origin + ray_length * ray

        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            linewidth=1,
        )

        ax.scatter([end[0]], [end[1]], [end[2]], s=12)

        if annotate_indices:
            ax.text(end[0], end[1], end[2], str(i), fontsize=7)

        points_for_limits.append(end)

    # Referenzstrahlen für Bildmitte und Bildecken
    reference_pixels = {
        "center": np.array([intrinsics.cx, intrinsics.cy], dtype=float),
        "top_left": np.array([0.0, 0.0], dtype=float),
        "top_right": np.array([intrinsics.img_width - 1.0, 0.0], dtype=float),
        "bottom_left": np.array([0.0, intrinsics.img_height - 1.0], dtype=float),
        "bottom_right": np.array(
            [intrinsics.img_width - 1.0, intrinsics.img_height - 1.0],
            dtype=float,
        ),
    }

    for label, uv_ref in reference_pixels.items():
        ray_ref = pixel_to_camera_ray(uv_ref, intrinsics)
        end_ref = origin + ray_length * ray_ref

        ax.plot(
            [origin[0], end_ref[0]],
            [origin[1], end_ref[1]],
            [origin[2], end_ref[2]],
            linewidth=2,
            linestyle="--",
        )
        ax.text(end_ref[0], end_ref[1], end_ref[2], label, fontsize=9)
        points_for_limits.append(end_ref)

    ax.scatter([0.0], [0.0], [0.0], s=40)
    ax.text(0.0, 0.0, 0.0, "camera origin", fontsize=9)

    ax.set_title("Kamerarays im Kamera-KS")
    ax.set_xlabel("x_C [rechts im Bild]")
    ax.set_ylabel("y_C [oben im Bild]")
    ax.set_zlabel("z_C [Blickrichtung]")

    points = np.asarray(points_for_limits, dtype=float)
    center = np.mean(points, axis=0)
    radius = float(np.max(np.linalg.norm(points - center, axis=1)))

    if radius <= 1e-12:
        radius = 1.0
    
    scale_factor = 0.1
    ax.set_xlim(center[0] - scale_factor * radius, center[0] + scale_factor * radius)
    ax.set_ylim(center[1] - scale_factor * radius, center[1] + scale_factor * radius)
    ax.set_zlim(center[2] - radius, center[2] + 0.5 * radius)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")
    save_camera_ray_debug_csv(
        uv_list=uv_list,
        intrinsics=intrinsics,
        output_path=csv_path,
        ray_length=ray_length,
    )

    return output_path


def print_camera_ray_reference_points(
    intrinsics: CameraIntrinsics,
) -> None:
    """
    Gibt wichtige Referenzpunkte aus, um Vorzeichen und Orientierung zu prüfen.
    """
    reference_pixels = {
        "center": np.array([intrinsics.cx, intrinsics.cy], dtype=float),
        "top_left": np.array([0.0, 0.0], dtype=float),
        "top_right": np.array([intrinsics.img_width - 1.0, 0.0], dtype=float),
        "bottom_left": np.array([0.0, intrinsics.img_height - 1.0], dtype=float),
        "bottom_right": np.array(
            [intrinsics.img_width - 1.0, intrinsics.img_height - 1.0],
            dtype=float,
        ),
        "pixel_1_1": np.array([1.0, 1.0], dtype=float),
    }

    print("\n📷 Kamera-Ray-Referenzpunkte im Kamera-KS:")
    print("  Konvention: +x rechts, +y oben, +z Blickrichtung")

    for label, uv in reference_pixels.items():
        x_norm, y_norm = pixel_to_normalized_camera_coordinates(uv, intrinsics)
        ray = pixel_to_camera_ray(uv, intrinsics)

        print(
            f"  {label:12s} "
            f"uv=({uv[0]:.3f}, {uv[1]:.3f}) | "
            f"norm=({x_norm:+.8f}, {y_norm:+.8f}) | "
            f"ray=({ray[0]:+.8f}, {ray[1]:+.8f}, {ray[2]:+.8f})"
        )
        
def plot_camera_rays_on_z_plane(
    uv_list: list[np.ndarray],
    intrinsics: CameraIntrinsics,
    output_path: str | Path,
    z_plane: float = 1.0,
    max_rays: int | None = 200,
    annotate_indices: bool = True,
) -> Path:
    """
    2D-Debug der Kamerarays auf der Ebene z = z_plane.

    Darstellung im Kamera-KS:
        x_C nach rechts im Bild
        y_C nach oben im Bild

    Bei z_plane=1 entspricht der Punkt praktisch den normierten
    Bildkoordinaten:
        x = (u - cx) / fx
        y = (cy - v) / fy

    Erwartung:
        top_left     -> x negativ, y positiv
        top_right    -> x positiv, y positiv
        bottom_left  -> x negativ, y negativ
        bottom_right -> x positiv, y negativ
        center       -> ungefähr (0, 0)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if max_rays is not None:
        uv_list_plot = uv_list[:max_rays]
    else:
        uv_list_plot = uv_list

    if len(uv_list_plot) == 0:
        raise ValueError("Keine Kamerarays zum Plotten übergeben.")

    points = []
    rows = []

    for i, uv in enumerate(uv_list_plot):
        uv = np.asarray(uv, dtype=float).reshape(2)
        ray, point = _camera_ray_intersection_with_z_plane(
            uv=uv,
            intrinsics=intrinsics,
            z_plane=z_plane,
        )

        points.append(point)

        rows.append(
            {
                "idx": i,
                "u_px": float(uv[0]),
                "v_px": float(uv[1]),
                "x_on_z_plane": float(point[0]),
                "y_on_z_plane": float(point[1]),
                "z_plane": float(z_plane),
                "dir_x": float(ray[0]),
                "dir_y": float(ray[1]),
                "dir_z": float(ray[2]),
            }
        )

    points = np.asarray(points, dtype=float)

    reference_pixels = {
        "center": np.array([intrinsics.cx, intrinsics.cy], dtype=float),
        "top_left": np.array([0.0, 0.0], dtype=float),
        "top_right": np.array([intrinsics.img_width - 1.0, 0.0], dtype=float),
        "bottom_left": np.array([0.0, intrinsics.img_height - 1.0], dtype=float),
        "bottom_right": np.array(
            [intrinsics.img_width - 1.0, intrinsics.img_height - 1.0],
            dtype=float,
        ),
        "pixel_1_1": np.array([1.0, 1.0], dtype=float),
    }

    reference_points = {}

    for label, uv_ref in reference_pixels.items():
        _, point_ref = _camera_ray_intersection_with_z_plane(
            uv=uv_ref,
            intrinsics=intrinsics,
            z_plane=z_plane,
        )
        reference_points[label] = point_ref

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=25,
        label="detected laser points projected to z-plane",
    )

    if annotate_indices:
        for i, point in enumerate(points):
            ax.text(point[0], point[1], str(i), fontsize=7)

    for label, point_ref in reference_points.items():
        ax.scatter(
            [point_ref[0]],
            [point_ref[1]],
            s=60,
            marker="x",
        )
        ax.text(
            point_ref[0],
            point_ref[1],
            label,
            fontsize=9,
        )

    # Bildrahmen auf z-Ebene verbinden
    corner_order = ["top_left", "top_right", "bottom_right", "bottom_left", "top_left"]
    frame_xy = np.array(
        [[reference_points[name][0], reference_points[name][1]] for name in corner_order],
        dtype=float,
    )

    ax.plot(
        frame_xy[:, 0],
        frame_xy[:, 1],
        linestyle="--",
        linewidth=1.2,
        label="image frame on z-plane",
    )

    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)

    ax.set_title(f"Kamerarays auf Ebene z_C = {z_plane:.3f}")
    ax.set_xlabel("x_C [rechts im Bild]")
    ax.set_ylabel("y_C [oben im Bild]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")

    all_points_xy = np.vstack(
        [
            points[:, :2],
            np.array([[p[0], p[1]] for p in reference_points.values()], dtype=float),
        ]
    )

    xy_min = np.min(all_points_xy, axis=0)
    xy_max = np.max(all_points_xy, axis=0)
    center = 0.5 * (xy_min + xy_max)
    span = xy_max - xy_min

    margin = 0.05
    span = np.maximum(span + 2.0 * margin, 0.1)

    ax.set_xlim(center[0] - span[0] / 2.0, center[0] + span[0] / 2.0)
    ax.set_ylim(center[1] - span[1] / 2.0, center[1] + span[1] / 2.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path