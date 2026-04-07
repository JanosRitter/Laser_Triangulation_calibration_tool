from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def _to_serializable(obj):
    """
    Wandelt numpy-/Path-Objekte rekursiv in JSON-kompatible Typen um.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return [_to_serializable(v) for v in obj]

    return obj


def build_calibration_result_dict(
    run_data: dict,
    prep_result: dict,
    intrinsics,
    calibration_result: dict,
    gt_ray_debug: dict | None = None,
    solver_debug: dict | None = None,
) -> dict:
    """
    Baut ein kompaktes, speicherbares Ergebnis-Dictionary.

    Enthält:
    - Run-Zusammenfassung
    - Kamera-Intrinsics
    - finalen GT-freien Kalibrierlauf
    - optional GT-Debug-Infos
    """
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_folder": str(run_data["input_folder"]),
        "num_total_frames": len(run_data["frame_table"]),
        "num_valid_observations_raw": len(run_data["observations"]),
        "num_calibration_observations": len(prep_result["calib_observations"]),
        "fit_stats": prep_result["stats"],
    }

    intrinsics_dict = {
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
        "img_width": int(intrinsics.img_width),
        "img_height": int(intrinsics.img_height),
    }

    calibration_dict = {
        "initial_params": np.asarray(calibration_result["params_initial"], dtype=float),
        "optimized_params": np.asarray(calibration_result["params_optimized"], dtype=float),
        "residual_summary_initial": calibration_result["residual_summary_initial"],
        "residual_summary_optimized": calibration_result["residual_summary_optimized"],
        "solver_result": {
            "success": bool(calibration_result["solver_result"].success),
            "status": int(calibration_result["solver_result"].status),
            "message": str(calibration_result["solver_result"].message),
            "nfev": int(calibration_result["solver_result"].nfev),
            "njev": (
                int(calibration_result["solver_result"].njev)
                if calibration_result["solver_result"].njev is not None
                else None
            ),
            "cost": float(calibration_result["solver_result"].cost),
            "optimality": float(calibration_result["solver_result"].optimality),
        },
    }

    result = {
        "summary": summary,
        "intrinsics": intrinsics_dict,
        "calibration_without_gt": calibration_dict,
        "has_ground_truth_debug": gt_ray_debug is not None,
    }

    if gt_ray_debug is not None:
        result["ground_truth_ray_debug"] = {
            "mean_distance_m": float(gt_ray_debug["mean_distance_m"]),
            "max_distance_m": float(gt_ray_debug["max_distance_m"]),
            "mean_lambda_cam_m": float(gt_ray_debug["mean_lambda_cam_m"]),
            "mean_lambda_laser_m": float(gt_ray_debug["mean_lambda_laser_m"]),
        }

    if solver_debug is not None:
        result["ground_truth_solver_debug"] = {
            "params_gt": np.asarray(solver_debug["params_gt"], dtype=float),
            "params_perturbed": np.asarray(solver_debug["params_perturbed"], dtype=float),
            "params_optimized": np.asarray(solver_debug["params_optimized"], dtype=float),
            "diff_init": solver_debug["diff_init"],
            "diff_final": solver_debug["diff_final"],
            "residual_summary_gt": solver_debug["residual_summary_gt"],
            "residual_summary_perturbed": solver_debug["residual_summary_perturbed"],
            "residual_summary_optimized": solver_debug["residual_summary_optimized"],
            "solver_result": {
                "success": bool(solver_debug["solver_result"].success),
                "status": int(solver_debug["solver_result"].status),
                "message": str(solver_debug["solver_result"].message),
                "nfev": int(solver_debug["solver_result"].nfev),
                "njev": (
                    int(solver_debug["solver_result"].njev)
                    if solver_debug["solver_result"].njev is not None
                    else None
                ),
                "cost": float(solver_debug["solver_result"].cost),
                "optimality": float(solver_debug["solver_result"].optimality),
            },
        }

    return _to_serializable(result)


def save_calibration_result_json(result_dict: dict, output_path: str | Path) -> Path:
    """
    Speichert das Ergebnis als JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    return output_path


def save_frame_residuals_csv(
    frame_results: list[dict],
    output_path: str | Path,
) -> Path:
    """
    Speichert detaillierte frameweise Residuen als CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_idx",
        "residual_norm",
        "residual_x",
        "residual_y",
        "residual_z",
        "lambda_cam",
        "lambda_laser",
        "point_cam_x",
        "point_cam_y",
        "point_cam_z",
        "point_laser_x",
        "point_laser_y",
        "point_laser_z",
        "midpoint_x",
        "midpoint_y",
        "midpoint_z",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in frame_results:
            rv = np.asarray(row["residual_vector"], dtype=float).reshape(3)
            point_cam = np.asarray(row["point_cam"], dtype=float).reshape(3)
            point_laser = np.asarray(row["point_laser"], dtype=float).reshape(3)
            midpoint = np.asarray(row["midpoint"], dtype=float).reshape(3)

            writer.writerow({
                "frame_idx": int(row["frame_idx"]),
                "residual_norm": float(row["residual_norm"]),
                "residual_x": float(rv[0]),
                "residual_y": float(rv[1]),
                "residual_z": float(rv[2]),
                "lambda_cam": float(row["lambda_cam"]),
                "lambda_laser": float(row["lambda_laser"]),
                "point_cam_x": float(point_cam[0]),
                "point_cam_y": float(point_cam[1]),
                "point_cam_z": float(point_cam[2]),
                "point_laser_x": float(point_laser[0]),
                "point_laser_y": float(point_laser[1]),
                "point_laser_z": float(point_laser[2]),
                "midpoint_x": float(midpoint[0]),
                "midpoint_y": float(midpoint[1]),
                "midpoint_z": float(midpoint[2]),
            })

    return output_path