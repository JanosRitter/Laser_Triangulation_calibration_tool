from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from src.debug.plane_projection_debug.geometry import (
    project_uv_rows_to_debug_plane,
    build_laser_intersections_with_debug_plane,
)
from src.debug.plane_projection_debug.io import (
    load_fit_uv_table,
    save_plane_debug_csv,
    save_result_json,
)
from src.debug.plane_projection_debug.plotting import (
    plot_uv_projection_vs_laser_intersections,
)


def _prepare_output_dir(
    run_data: dict,
    output_dir: str | Path | None,
) -> tuple[Path, Path]:
    run_dir = Path(run_data["input_folder"])

    if output_dir is None:
        output_dir = run_dir
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, output_dir


def run_plane_projection_debug(
    run_data: dict,
    intrinsics,
    projection_distance_m: float = 0.40,
    plane_center_robot: np.ndarray = np.array([-0.075, 0.975, 0.525]),
    local_ray_direction: np.ndarray = np.array([0.0, 1.0, 0.0]),
    output_dir: str | Path | None = None,
) -> dict:
    run_dir, output_dir = _prepare_output_dir(run_data, output_dir)

    fit_table_path = run_dir / "laser_point_fit_table.csv"
    uv_rows = load_fit_uv_table(fit_table_path)
    frame_indices = [row["frame_idx"] for row in uv_rows]

    uv_points_robot = project_uv_rows_to_debug_plane(
        uv_rows=uv_rows,
        intrinsics=intrinsics,
        center=plane_center_robot,
        rx_deg=0.0,
        ry_deg=0.0,
        rz_deg=0.0,
        projection_distance_m=projection_distance_m,
    )

    laser_intersections = build_laser_intersections_with_debug_plane(
        run_data=run_data,
        frame_indices=frame_indices,
        center=plane_center_robot,
        rx_deg=0.0,
        ry_deg=0.0,
        rz_deg=0.0,
        local_ray_direction=local_ray_direction,
    )

    csv_path = save_plane_debug_csv(
        output_dir / "plane_projection_debug.csv",
        uv_points_robot=uv_points_robot,
        laser_intersections=laser_intersections,
    )

    plot_path = plot_uv_projection_vs_laser_intersections(
        output_dir / "plane_projection_debug.png",
        uv_points_robot=uv_points_robot,
        laser_intersections=laser_intersections,
    )

    print("\n🧪 Plane-Projection-Debug:")
    print(f"  Projektionsabstand: {projection_distance_m:.3f} m")
    print(f"  Ebenenzentrum:      {plane_center_robot}")
    print(f"  CSV:                {csv_path}")
    print(f"  Plot:               {plot_path}")

    return {
        "csv_path": csv_path,
        "plot_path": plot_path,
        "num_points": len(uv_points_robot),
    }


def fit_plane_projection_debug(
    run_data: dict,
    intrinsics,
    initial_center: np.ndarray = np.array([-0.075, 0.975, 0.525]),
    initial_projection_distance_m: float = 0.40,
    local_ray_direction: np.ndarray = np.array([0.0, 1.0, 0.0]),
    output_dir: str | Path | None = None,
) -> dict:
    run_dir, output_dir = _prepare_output_dir(run_data, output_dir)

    fit_table_path = run_dir / "laser_point_fit_table.csv"
    uv_rows = load_fit_uv_table(fit_table_path)
    frame_indices = [row["frame_idx"] for row in uv_rows]

    def evaluate(params: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict]]:
        x0, y0, z0, rx_deg, ry_deg, rz_deg, projection_distance_m = params
        center = np.array([x0, y0, z0], dtype=float)

        uv_points = project_uv_rows_to_debug_plane(
            uv_rows=uv_rows,
            intrinsics=intrinsics,
            center=center,
            rx_deg=rx_deg,
            ry_deg=ry_deg,
            rz_deg=rz_deg,
            projection_distance_m=projection_distance_m,
        )

        laser_points = build_laser_intersections_with_debug_plane(
            run_data=run_data,
            frame_indices=frame_indices,
            center=center,
            rx_deg=rx_deg,
            ry_deg=ry_deg,
            rz_deg=rz_deg,
            local_ray_direction=local_ray_direction,
        )

        laser_by_frame = {p["frame_idx"]: p for p in laser_points}

        residuals = []

        for uv in uv_points:
            laser = laser_by_frame[uv["frame_idx"]]

            if not laser["valid"]:
                residuals.extend([1.0, 1.0, 1.0])
                continue

            residuals.extend([
                laser["x"] - uv["x"],
                laser["y"] - uv["y"],
                laser["z"] - uv["z"],
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
        -0.20,
         0.85,
         0.45,
        -2.0,
        -2.0,
        -10.0,
         0.30,
    ], dtype=float)

    upper = np.array([
         0.05,
         1.10,
         0.60,
         2.0,
         2.0,
         10.0,
         0.50,
    ], dtype=float)

    eps = 1e-9
    x0 = np.clip(x0, lower + eps, upper - eps)

    result = least_squares(
        fun=lambda params: evaluate(params)[0],
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        verbose=0,
    )

    residuals, uv_points, laser_points = evaluate(result.x)

    residual_vectors = residuals.reshape(-1, 3)
    residual_norms = np.linalg.norm(residual_vectors, axis=1)

    hit_lower = np.isclose(result.x, lower, atol=1e-6)
    hit_upper = np.isclose(result.x, upper, atol=1e-6)

    names = [
        "x0",
        "y0",
        "z0",
        "rx_deg",
        "ry_deg",
        "rz_deg",
        "projection_distance_m",
    ]

    print("\n🧪 Fit Plane-Projection-Debug:")
    print("  Parameter:")

    for name, value, lo, hi, hl, hu in zip(
        names,
        result.x,
        lower,
        upper,
        hit_lower,
        hit_upper,
    ):
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
        output_dir / "plane_projection_fit_debug.csv",
        uv_points_robot=uv_points,
        laser_intersections=laser_points,
    )

    plot_path = plot_uv_projection_vs_laser_intersections(
        output_dir / "plane_projection_fit_debug.png",
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
            for name, lo, hi, hl, hu in zip(
                names,
                lower,
                upper,
                hit_lower,
                hit_upper,
            )
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

    result_json_path = save_result_json(
        output_dir / "plane_projection_fit_debug_result.json",
        result_dict,
    )

    result_dict["result_json_path"] = str(result_json_path)

    return result_dict