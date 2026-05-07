from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from src.calibration.laser_kinematics.robot_base_offsets import (
    build_absolute_transforms_from_robot_base_offsets,
)


def rotation_matrix_x(rx_rad: float) -> np.ndarray:
    c = np.cos(rx_rad)
    s = np.sin(rx_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=float)


def rotation_matrix_y(ry_rad: float) -> np.ndarray:
    c = np.cos(ry_rad)
    s = np.sin(ry_rad)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=float)


def rotation_matrix_z(rz_rad: float) -> np.ndarray:
    c = np.cos(rz_rad)
    s = np.sin(rz_rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def euler_xyz_deg_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    return (
        rotation_matrix_z(rz)
        @ rotation_matrix_y(ry)
        @ rotation_matrix_x(rx)
    )


def intersect_ray_with_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = np.asarray(direction, dtype=float).reshape(3)
    plane_point = np.asarray(plane_point, dtype=float).reshape(3)
    plane_normal = np.asarray(plane_normal, dtype=float).reshape(3)

    denom = float(np.dot(plane_normal, direction))

    if abs(denom) < 1e-12:
        return None, None

    t = float(np.dot(plane_normal, plane_point - origin) / denom)
    p = origin + t * direction

    return p, t


def project_uv_to_rotated_plane(
    uv_rows: list[dict],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    center: np.ndarray,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    projection_distance_m: float,
    base_x_axis_robot: np.ndarray = np.array([-1.0, 0.0, 0.0]),
    base_y_axis_robot: np.ndarray = np.array([0.0, -1.0, 0.0]),
) -> list[dict]:
    center = np.asarray(center, dtype=float).reshape(3)

    R = euler_xyz_deg_to_matrix(rx_deg, ry_deg, rz_deg)

    base_x = np.asarray(base_x_axis_robot, dtype=float).reshape(3)
    base_y = np.asarray(base_y_axis_robot, dtype=float).reshape(3)

    base_x = base_x / np.linalg.norm(base_x)
    base_y = base_y / np.linalg.norm(base_y)

    x_axis = R @ base_x
    y_axis = R @ base_y

    points = []

    for row in uv_rows:
        u = float(row["u"])
        v = float(row["v"])

        x_local = (u - cx) / fx * projection_distance_m
        y_local = (cy - v) / fy * projection_distance_m

        p_robot = center + x_local * x_axis + y_local * y_axis

        points.append({
            "frame_idx": int(row["frame_idx"]),
            "u": u,
            "v": v,
            "x": float(p_robot[0]),
            "y": float(p_robot[1]),
            "z": float(p_robot[2]),
            "x_local": float(x_local),
            "y_local": float(y_local),
        })

    return points


def build_laser_intersections_with_rotated_plane(
    run_data: dict,
    frame_indices: list[int],
    center: np.ndarray,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    local_ray_direction: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> list[dict]:
    trajectory_config = run_data["run_metadata"]["scan"]["trajectory_config"]
    transforms = build_absolute_transforms_from_robot_base_offsets(trajectory_config)

    R = euler_xyz_deg_to_matrix(rx_deg, ry_deg, rz_deg)
    plane_normal = R @ np.array([0.0, 0.0, 1.0], dtype=float)

    intersections = []

    for frame_idx in frame_indices:
        T = transforms[frame_idx]
        origin, direction = ray_from_transform(T, local_ray_direction)

        p, t = intersect_ray_with_plane(
            origin=origin,
            direction=direction,
            plane_point=center,
            plane_normal=plane_normal,
        )

        if p is None:
            intersections.append({
                "frame_idx": int(frame_idx),
                "valid": False,
                "ray_t": np.nan,
                "x": np.nan,
                "y": np.nan,
                "z": np.nan,
                "origin_x": float(origin[0]),
                "origin_y": float(origin[1]),
                "origin_z": float(origin[2]),
                "dir_x": float(direction[0]),
                "dir_y": float(direction[1]),
                "dir_z": float(direction[2]),
            })
            continue

        intersections.append({
            "frame_idx": int(frame_idx),
            "valid": True,
            "ray_t": float(t),
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(p[2]),
            "origin_x": float(origin[0]),
            "origin_y": float(origin[1]),
            "origin_z": float(origin[2]),
            "dir_x": float(direction[0]),
            "dir_y": float(direction[1]),
            "dir_z": float(direction[2]),
        })

    return intersections

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


def image_points_to_robot_plane_points(
    uv_rows: list[dict],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    projection_distance_m: float,
    plane_center_robot: np.ndarray,
    image_x_axis_robot: np.ndarray = np.array([-1.0, 0.0, 0.0]),
    image_y_axis_robot: np.ndarray = np.array([0.0, -1.0, 0.0]),
) -> list[dict]:
    """
    Projiziert UV-Bildpunkte näherungsweise auf eine Ebene im Roboter-KS.

    Annahme:
    - Ebene ist parallel zur Roboter-xy-Ebene, also z = konstant.
    - Die Kamera schaut näherungsweise senkrecht auf diese Ebene.
    - projection_distance_m entspricht ungefähr Kamera-Ebene-Abstand.
    - x/y-Skalierung folgt Lochkamera:
        X = (u - cx) / fx * Z
        Y = (cy - v) / fy * Z
    """
    plane_center_robot = np.asarray(plane_center_robot, dtype=float).reshape(3)

    image_x_axis_robot = np.asarray(image_x_axis_robot, dtype=float).reshape(3)
    image_y_axis_robot = np.asarray(image_y_axis_robot, dtype=float).reshape(3)

    image_x_axis_robot = image_x_axis_robot / np.linalg.norm(image_x_axis_robot)
    image_y_axis_robot = image_y_axis_robot / np.linalg.norm(image_y_axis_robot)

    points = []

    for row in uv_rows:
        u = float(row["u"])
        v = float(row["v"])

        x_local = (u - cx) / fx * projection_distance_m
        y_local = (cy - v) / fy * projection_distance_m

        p_robot = (
            plane_center_robot
            + x_local * image_x_axis_robot
            + y_local * image_y_axis_robot
        )

        # harte Sicherung: Ebene bleibt z-konstant
        p_robot[2] = plane_center_robot[2]

        points.append({
            "frame_idx": int(row["frame_idx"]),
            "u": u,
            "v": v,
            "x": float(p_robot[0]),
            "y": float(p_robot[1]),
            "z": float(p_robot[2]),
            "x_local": float(x_local),
            "y_local": float(y_local),
        })

    return points


def ray_from_transform(
    T: np.ndarray,
    local_ray_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origin = T[:3, 3].copy()
    R = T[:3, :3]

    direction = R @ np.asarray(local_ray_direction, dtype=float).reshape(3)
    direction = direction / np.linalg.norm(direction)

    return origin, direction


def intersect_ray_with_z_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    z_plane: float,
) -> tuple[np.ndarray, float] | tuple[None, None]:
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = np.asarray(direction, dtype=float).reshape(3)

    dz = direction[2]

    if abs(dz) < 1e-12:
        return None, None

    t = (z_plane - origin[2]) / dz
    p = origin + t * direction

    return p, float(t)


def build_laser_plane_intersections(
    run_data: dict,
    frame_indices: list[int],
    z_plane: float,
    local_ray_direction: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> list[dict]:
    trajectory_config = run_data["run_metadata"]["scan"]["trajectory_config"]
    transforms = build_absolute_transforms_from_robot_base_offsets(trajectory_config)

    intersections = []

    for frame_idx in frame_indices:
        T = transforms[frame_idx]
        origin, direction = ray_from_transform(T, local_ray_direction)

        p, t = intersect_ray_with_z_plane(
            origin=origin,
            direction=direction,
            z_plane=z_plane,
        )

        if p is None:
            intersections.append({
                "frame_idx": int(frame_idx),
                "valid": False,
                "ray_t": np.nan,
                "x": np.nan,
                "y": np.nan,
                "z": np.nan,
                "origin_x": float(origin[0]),
                "origin_y": float(origin[1]),
                "origin_z": float(origin[2]),
                "dir_x": float(direction[0]),
                "dir_y": float(direction[1]),
                "dir_z": float(direction[2]),
            })
            continue

        intersections.append({
            "frame_idx": int(frame_idx),
            "valid": True,
            "ray_t": float(t),
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(p[2]),
            "origin_x": float(origin[0]),
            "origin_y": float(origin[1]),
            "origin_z": float(origin[2]),
            "dir_x": float(direction[0]),
            "dir_y": float(direction[1]),
            "dir_z": float(direction[2]),
        })

    return intersections


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
        dist_xy = float(np.sqrt(dx * dx + dy * dy)) if laser["valid"] else np.nan

        rows.append({
            "frame_idx": frame_idx,

            "u": uv["u"],
            "v": uv["v"],

            "uv_proj_x": uv["x"],
            "uv_proj_y": uv["y"],
            "uv_proj_z": uv["z"],

            "laser_x": laser["x"],
            "laser_y": laser["y"],
            "laser_z": laser["z"],
            "laser_valid": laser["valid"],
            "laser_ray_t": laser["ray_t"],

            "dx_laser_minus_uv": dx,
            "dy_laser_minus_uv": dy,
            "dist_xy": dist_xy,

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


def plot_uv_projection_vs_laser_intersections(
    output_path: str | Path,
    uv_points_robot: list[dict],
    laser_intersections: list[dict],
    xlim: tuple[float, float] = (-0.09, -0.05),
    ylim: tuple[float, float] = (0.94, 1.0),
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    laser_by_frame = {r["frame_idx"]: r for r in laser_intersections}

    fig, ax = plt.subplots(figsize=(8, 8))

    uv_x = []
    uv_y = []
    laser_x = []
    laser_y = []

    for uv in uv_points_robot:
        frame_idx = uv["frame_idx"]
        laser = laser_by_frame.get(frame_idx)

        if laser is None or not laser["valid"]:
            continue

        uv_x.append(uv["x"])
        uv_y.append(uv["y"])
        laser_x.append(laser["x"])
        laser_y.append(laser["y"])

        ax.plot(
            [uv["x"], laser["x"]],
            [uv["y"], laser["y"]],
            linewidth=0.8,
            alpha=0.6,
        )

        ax.text(
            uv["x"],
            uv["y"],
            str(frame_idx),
            fontsize=7,
        )

    ax.scatter(uv_x, uv_y, marker="o", s=35, label="UV → projizierte Ebene")
    ax.scatter(laser_x, laser_y, marker="x", s=45, label="Laser-Ray ∩ Ebene")

    ax.set_title("UV-Projektion vs. Laserray-Schnittpunkte auf z-Ebene")
    ax.set_xlabel("x_R [m]")
    ax.set_ylabel("y_R [m]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def run_plane_projection_debug(
    run_data: dict,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    projection_distance_m: float = 0.40,
    plane_center_robot: np.ndarray = np.array([-0.075, 0.975, 0.525]),
    local_ray_direction: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> dict:
    """
    Hauptfunktion für den 2D-Ebenen-Debug.

    Erwartet im Run-Ordner:
        laser_point_fit_table.csv

    Output:
        plane_projection_debug.csv
        plane_projection_debug.png
    """
    run_dir = Path(run_data["input_folder"])

    fit_table_path = run_dir / "laser_point_fit_table.csv"
    uv_rows = load_fit_uv_table(fit_table_path)

    z_plane = float(plane_center_robot[2])
    frame_indices = [row["frame_idx"] for row in uv_rows]

    uv_points_robot = image_points_to_robot_plane_points(
        uv_rows=uv_rows,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        projection_distance_m=projection_distance_m,
        plane_center_robot=plane_center_robot,
    )

    laser_intersections = build_laser_plane_intersections(
        run_data=run_data,
        frame_indices=frame_indices,
        z_plane=z_plane,
        local_ray_direction=local_ray_direction,
    )

    csv_path = save_plane_debug_csv(
        run_dir / "plane_projection_debug.csv",
        uv_points_robot=uv_points_robot,
        laser_intersections=laser_intersections,
    )

    plot_path = plot_uv_projection_vs_laser_intersections(
        run_dir / "plane_projection_debug.png",
        uv_points_robot=uv_points_robot,
        laser_intersections=laser_intersections,
    )

    print("\n🧪 Plane-Projection-Debug:")
    print(f"  Projektionsabstand: {projection_distance_m:.3f} m")
    print(f"  Ebenenzentrum:      {plane_center_robot}")
    print(f"  z-Ebene:            {z_plane:.3f} m")
    print(f"  CSV:                {csv_path}")
    print(f"  Plot:               {plot_path}")

    return {
        "csv_path": csv_path,
        "plot_path": plot_path,
        "num_points": len(uv_points_robot),
    }

def fit_plane_projection_debug(
    run_data: dict,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    initial_center: np.ndarray = np.array([-0.075, 0.975, 0.525]),
    initial_projection_distance_m: float = 0.40,
    local_ray_direction: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> dict:
    run_dir = Path(run_data["input_folder"])
    fit_table_path = run_dir / "laser_point_fit_table.csv"

    uv_rows = load_fit_uv_table(fit_table_path)
    frame_indices = [row["frame_idx"] for row in uv_rows]

    def evaluate(params: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict]]:
        x0, y0, z0, rx, ry, rz, scale = params
        center = np.array([x0, y0, z0], dtype=float)

        uv_points = project_uv_to_rotated_plane(
            uv_rows=uv_rows,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            center=center,
            rx_deg=rx,
            ry_deg=ry,
            rz_deg=rz,
            projection_distance_m=scale,
            base_x_axis_robot=np.array([-1.0, 0.0, 0.0], dtype=float),
            base_y_axis_robot=np.array([0.0, -1.0, 0.0], dtype=float),
        )

        laser_points = build_laser_intersections_with_rotated_plane(
            run_data=run_data,
            frame_indices=frame_indices,
            center=center,
            rx_deg=rx,
            ry_deg=ry,
            rz_deg=rz,
            local_ray_direction=local_ray_direction,
        )

        laser_by_frame = {p["frame_idx"]: p for p in laser_points}

        residuals = []

        for uv in uv_points:
            lp = laser_by_frame[uv["frame_idx"]]

            if not lp["valid"]:
                residuals.extend([1.0, 1.0, 1.0])
                continue

            residuals.extend([
                lp["x"] - uv["x"],
                lp["y"] - uv["y"],
                lp["z"] - uv["z"],
            ])

        return np.asarray(residuals, dtype=float), uv_points, laser_points

    x0 = np.array([
        float(initial_center[0]),
        float(initial_center[1]),
        float(initial_center[2]),
        0.0,
        0.0,
        0.0,
        float(initial_projection_distance_m),
    ], dtype=float)

    lower = np.array([
        -0.20,   # x0
         0.85,   # y0
         0.45,   # z0
        -2.0,    # rx
        -2.0,    # ry
        -10.0,   # rz
         0.30,   # projection distance
    ], dtype=float)

    upper = np.array([
         0.05,   # x0
         1.10,   # y0
         0.60,   # z0
         2.0,    # rx
         2.0,    # ry
         10.0,   # rz
         0.50,   # projection distance
    ], dtype=float)

    eps = 1e-9
    x0 = np.clip(x0, lower + eps, upper - eps)

    def objective(params: np.ndarray) -> np.ndarray:
        residuals, _, _ = evaluate(params)
        return residuals

    result = least_squares(
        objective,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        verbose=0,
    )

    residuals, uv_points, laser_points = evaluate(result.x)

    residual_norms = residuals.reshape(-1, 3)
    residual_norms = np.linalg.norm(residual_norms, axis=1)

    hit_lower = np.isclose(result.x, lower, atol=1e-6)
    hit_upper = np.isclose(result.x, upper, atol=1e-6)

    names = ["x0", "y0", "z0", "rx_deg", "ry_deg", "rz_deg", "projection_distance_m"]

    print("\n🧪 Fit Plane-Projection-Debug:")
    print("  Parameter:")
    for name, value, lo, hi, hl, hu in zip(names, result.x, lower, upper, hit_lower, hit_upper):
        bound_info = ""
        if hl:
            bound_info = "  <-- lower bound"
        elif hu:
            bound_info = "  <-- upper bound"

        print(f"    {name:24s}: {value:+.8f}   [{lo:+.3f}, {hi:+.3f}]{bound_info}")

    print("  Qualität:")
    print(f"    success: {result.success}")
    print(f"    cost:    {result.cost:.12e}")
    print(f"    mean:    {np.mean(residual_norms):.8f} m")
    print(f"    median:  {np.median(residual_norms):.8f} m")
    print(f"    p95:     {np.percentile(residual_norms, 95):.8f} m")
    print(f"    max:     {np.max(residual_norms):.8f} m")

    csv_path = save_plane_debug_csv(
        run_dir / "plane_projection_fit_debug.csv",
        uv_points_robot=uv_points,
        laser_intersections=laser_points,
    )

    plot_path = plot_uv_projection_vs_laser_intersections(
        run_dir / "plane_projection_fit_debug.png",
        uv_points_robot=uv_points,
        laser_intersections=laser_points,
    )

    result_dict = {
        "params": {
            name: float(value)
            for name, value in zip(names, result.x)
        },
        "bounds": {
            name: {
                "lower": float(lo),
                "upper": float(hi),
                "hit_lower": bool(hl),
                "hit_upper": bool(hu),
            }
            for name, lo, hi, hl, hu in zip(names, lower, upper, hit_lower, hit_upper)
        },
        "quality": {
            "success": bool(result.success),
            "cost": float(result.cost),
            "mean_residual_m": float(np.mean(residual_norms)),
            "median_residual_m": float(np.median(residual_norms)),
            "p95_residual_m": float(np.percentile(residual_norms, 95)),
            "max_residual_m": float(np.max(residual_norms)),
        },
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
    }

    with open(run_dir / "plane_projection_fit_debug_result.json", "w", encoding="utf-8") as f:
        import json
        json.dump(result_dict, f, indent=2)

    return result_dict