from __future__ import annotations

import numpy as np

from src.calibration.laser_kinematics import (
    build_absolute_transforms_from_trajectory_config,
    pose_xyzrpy_deg_to_transform,
)


def build_ground_truth_transforms_from_frame_table(frame_table: list[dict]) -> list[np.ndarray]:
    """
    Baut absolute GT-Transformationen aus frame_table.

    Erwartete Spalten:
    - laser_x, laser_y, laser_z
    - laser_rx, laser_ry, laser_rz
    """
    transforms = []

    for row in frame_table:
        T = pose_xyzrpy_deg_to_transform(
            x=float(row["laser_x"]),
            y=float(row["laser_y"]),
            z=float(row["laser_z"]),
            rx_deg=float(row["laser_rx"]),
            ry_deg=float(row["laser_ry"]),
            rz_deg=float(row["laser_rz"]),
        )
        transforms.append(T)

    return transforms


def rotation_angle_deg_from_matrix(R: np.ndarray) -> float:
    """
    Berechnet den Rotationswinkel einer Rotationsmatrix in Grad.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    trace_val = np.trace(R)
    cos_theta = 0.5 * (trace_val - 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    return float(np.rad2deg(theta_rad))


def compare_transform_lists(
    reference_transforms: list[np.ndarray],
    test_transforms: list[np.ndarray],
) -> list[dict]:
    """
    Vergleicht zwei Folgen absoluter 4x4-Transformationen frameweise.
    """
    if len(reference_transforms) != len(test_transforms):
        raise ValueError(
            "Transformationslisten haben unterschiedliche Länge: "
            f"{len(reference_transforms)} != {len(test_transforms)}"
        )

    results = []

    for frame_idx, (T_ref, T_test) in enumerate(zip(reference_transforms, test_transforms)):
        t_ref = T_ref[:3, 3]
        t_test = T_test[:3, 3]

        pos_diff = t_test - t_ref
        pos_err = float(np.linalg.norm(pos_diff))

        R_ref = T_ref[:3, :3]
        R_test = T_test[:3, :3]

        R_delta = R_ref.T @ R_test
        rot_err_deg = rotation_angle_deg_from_matrix(R_delta)

        results.append({
            "frame_idx": frame_idx,
            "pos_err_m": pos_err,
            "dx_err_m": float(pos_diff[0]),
            "dy_err_m": float(pos_diff[1]),
            "dz_err_m": float(pos_diff[2]),
            "rot_err_deg": rot_err_deg,
        })

    return results


def summarize_transform_comparison(results: list[dict]) -> dict:
    """
    Kompakte Statistik der Poseabweichungen.
    """
    if len(results) == 0:
        return {
            "num_frames": 0,
            "mean_pos_err_m": np.nan,
            "max_pos_err_m": np.nan,
            "mean_rot_err_deg": np.nan,
            "max_rot_err_deg": np.nan,
            "worst_frame_pos": None,
            "worst_frame_rot": None,
        }

    pos_errs = np.array([r["pos_err_m"] for r in results], dtype=float)
    rot_errs = np.array([r["rot_err_deg"] for r in results], dtype=float)

    worst_frame_pos = int(results[int(np.argmax(pos_errs))]["frame_idx"])
    worst_frame_rot = int(results[int(np.argmax(rot_errs))]["frame_idx"])

    return {
        "num_frames": len(results),
        "mean_pos_err_m": float(np.mean(pos_errs)),
        "max_pos_err_m": float(np.max(pos_errs)),
        "mean_rot_err_deg": float(np.mean(rot_errs)),
        "max_rot_err_deg": float(np.max(rot_errs)),
        "worst_frame_pos": worst_frame_pos,
        "worst_frame_rot": worst_frame_rot,
    }


def can_run_trajectory_debug(run_data: dict) -> bool:
    capabilities = run_data.get("capabilities", {})
    has_gt_pose = bool(capabilities.get("has_ground_truth_pose", False))

    run_metadata = run_data.get("run_metadata", {})
    has_trajectory_config = (
        isinstance(run_metadata.get("scan"), dict)
        and "trajectory_config" in run_metadata["scan"]
    )

    return has_gt_pose and has_trajectory_config


def run_trajectory_debug(run_data: dict) -> dict:
    """
    Führt den Trajectory-Debuglauf aus.

    Vergleicht:
    - rekonstruierte absolute Posen aus trajectory_config
    - Ground-Truth-Posen aus frame_table
    """
    trajectory_config = run_data["run_metadata"]["scan"]["trajectory_config"]
    frame_table = run_data["frame_table"]

    reconstructed = build_absolute_transforms_from_trajectory_config(trajectory_config)
    ground_truth = build_ground_truth_transforms_from_frame_table(frame_table)

    comparison = compare_transform_lists(
        reference_transforms=ground_truth,
        test_transforms=reconstructed,
    )

    summary = summarize_transform_comparison(comparison)

    return {
        "reconstructed_transforms": reconstructed,
        "ground_truth_transforms": ground_truth,
        "comparison": comparison,
        "summary": summary,
    }


def run_optional_trajectory_debug(run_data: dict) -> dict | None:
    """
    Führt den Trajectory-Debug nur aus, wenn GT-Pose-Daten vorhanden sind.
    """
    if not can_run_trajectory_debug(run_data):
        return None

    return run_trajectory_debug(run_data)


def print_trajectory_debug_report(debug_result: dict | None, first_n: int = 12) -> None:
    """
    Schöne Konsolenausgabe für den Trajectory-Debug.
    """
    if debug_result is None:
        print("\nℹ️ Kein Trajectory-Debug: keine Ground-Truth-Pose vorhanden.")
        return

    summary = debug_result["summary"]
    comparison = debug_result["comparison"]

    print("\n🛤️ Trajectory-Debug:")
    print(f"  Frames verglichen: {summary['num_frames']}")
    print(f"  Mittlerer Positionsfehler: {summary['mean_pos_err_m']:.10f} m")
    print(f"  Maximaler Positionsfehler: {summary['max_pos_err_m']:.10f} m")
    print(f"  Mittlerer Rotationsfehler: {summary['mean_rot_err_deg']:.10f} deg")
    print(f"  Maximaler Rotationsfehler: {summary['max_rot_err_deg']:.10f} deg")
    print(f"  Schlimmster Frame (Position): {summary['worst_frame_pos']}")
    print(f"  Schlimmster Frame (Rotation): {summary['worst_frame_rot']}")

    print("\n🔎 Erste Frames im Detail:")
    n = min(first_n, len(comparison))
    for row in comparison[:n]:
        print(
            f"  frame {row['frame_idx']:3d} | "
            f"pos_err={row['pos_err_m']:.10f} m | "
            f"dx={row['dx_err_m']:+.10f}, "
            f"dy={row['dy_err_m']:+.10f}, "
            f"dz={row['dz_err_m']:+.10f} | "
            f"rot_err={row['rot_err_deg']:.10f} deg"
        )