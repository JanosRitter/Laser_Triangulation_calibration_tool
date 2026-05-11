from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def load_fit_uv_table(path: str | Path) -> list[dict]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Fit-Tabelle nicht gefunden: {path}")

    rows = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("use_for_calibration", "False") != "True":
                continue

            rows.append({
                "frame_idx": int(row["frame_idx"]),
                "u": float(row["u"]),
                "v": float(row["v"]),
            })

    if not rows:
        raise RuntimeError("Keine verwendbaren UV-Punkte in fit_table gefunden.")

    return rows


def save_plane_debug_csv(
    path: str | Path,
    uv_points_robot: list[dict],
    laser_intersections: list[dict],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    laser_by_frame = {r["frame_idx"]: r for r in laser_intersections}
    rows = []

    for uv in uv_points_robot:
        frame_idx = uv["frame_idx"]
        laser = laser_by_frame.get(frame_idx)

        if laser is None:
            continue

        dx = laser["x"] - uv["x"] if laser["valid"] else np.nan
        dy = laser["y"] - uv["y"] if laser["valid"] else np.nan
        dz = laser["z"] - uv["z"] if laser["valid"] else np.nan

        dist_3d = (
            float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if laser["valid"]
            else np.nan
        )

        dist_xy = (
            float(np.sqrt(dx * dx + dy * dy))
            if laser["valid"]
            else np.nan
        )

        rows.append({
            "frame_idx": frame_idx,
            "u": uv["u"],
            "v": uv["v"],

            "uv_proj_x": uv["x"],
            "uv_proj_y": uv["y"],
            "uv_proj_z": uv["z"],
            "uv_x_local": uv.get("x_local", np.nan),
            "uv_y_local": uv.get("y_local", np.nan),

            "laser_x": laser["x"],
            "laser_y": laser["y"],
            "laser_z": laser["z"],
            "laser_valid": laser["valid"],
            "laser_ray_t": laser["ray_t"],

            "dx_laser_minus_uv": dx,
            "dy_laser_minus_uv": dy,
            "dz_laser_minus_uv": dz,
            "dist_xy": dist_xy,
            "dist_3d": dist_3d,

            "laser_origin_x": laser["origin_x"],
            "laser_origin_y": laser["origin_y"],
            "laser_origin_z": laser["origin_z"],
            "laser_dir_x": laser["dir_x"],
            "laser_dir_y": laser["dir_y"],
            "laser_dir_z": laser["dir_z"],
        })

    if not rows:
        raise RuntimeError("Keine Debug-Zeilen erzeugt.")

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return path


def save_result_json(path: str | Path, result_dict: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    return path