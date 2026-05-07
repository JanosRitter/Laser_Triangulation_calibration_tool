from __future__ import annotations

import numpy as np

from src.calibration.camera_rays import pixel_to_camera_ray
from src.calibration.laser_kinematics import (
    build_extrinsic_pose_from_xyzrpy_deg,
    laser_ray_in_camera,
)
from src.calibration.ray_geometry import closest_points_between_rays
from src.calibration.parameterization import (
    initial_guess_from_ground_truth_start_pose,
    perturb_parameter_vector,
)
from src.calibration.residuals import (
    compute_frame_residuals_detailed,
    summarize_residuals,
    print_residual_summary,
    print_first_frame_residuals,
)
from src.calibration.solver import (
    solve_extrinsic_pose,
    parameter_difference,
    print_parameter_vector,
    print_parameter_difference,
    print_solver_report,
)


def run_gt_ray_debug(
    run_data: dict,
    calib_observations: list,
    intrinsics,
) -> dict | None:
    """
    Optionaler GT-Strahl-Debug.
    Gibt None zurück, wenn keine GT-Startpose vorhanden ist.
    """
    start_pose_gt = run_data["ground_truth"]["start_pose"]
    if start_pose_gt is None:
        return None

    extrinsic_gt = build_extrinsic_pose_from_xyzrpy_deg(
        x=float(start_pose_gt["laser_x"]),
        y=float(start_pose_gt["laser_y"]),
        z=float(start_pose_gt["laser_z"]),
        rx_deg=float(start_pose_gt["laser_rx"]),
        ry_deg=float(start_pose_gt["laser_ry"]),
        rz_deg=float(start_pose_gt["laser_rz"]),
    )

    distances = []
    lambda_cam_list = []
    lambda_laser_list = []

    for obs in calib_observations:
        cam_origin = np.zeros(3, dtype=float)
        cam_direction = pixel_to_camera_ray(obs.uv, intrinsics)

        laser_origin, laser_direction = laser_ray_in_camera(
            extrinsic_pose=extrinsic_gt,
            relative_pose=obs.relative_pose,
        )

        result = closest_points_between_rays(
            origin_a=cam_origin,
            direction_a=cam_direction,
            origin_b=laser_origin,
            direction_b=laser_direction,
        )

        distances.append(result["distance"])
        lambda_cam_list.append(result["lambda_a"])
        lambda_laser_list.append(result["lambda_b"])

    distances = np.asarray(distances, dtype=float)
    lambda_cam_list = np.asarray(lambda_cam_list, dtype=float)
    lambda_laser_list = np.asarray(lambda_laser_list, dtype=float)

    return {
        "mean_distance_m": float(np.mean(distances)),
        "max_distance_m": float(np.max(distances)),
        "mean_lambda_cam_m": float(np.mean(lambda_cam_list)),
        "mean_lambda_laser_m": float(np.mean(lambda_laser_list)),
        "distances": distances,
    }


def print_gt_ray_debug_report(gt_debug: dict | None, first_n: int = 10) -> None:
    if gt_debug is None:
        print("\nℹ️ Kein GT-Strahl-Debug: keine Ground-Truth-Startpose vorhanden.")
        return

    print("\n🧪 Strahl-Debug mit Ground Truth:")
    print(f"  mittlerer Strahlabstand: {gt_debug['mean_distance_m']:.10f} m")
    print(f"  maximaler Strahlabstand: {gt_debug['max_distance_m']:.10f} m")
    print(f"  mittlere Kamera-Tiefe λ: {gt_debug['mean_lambda_cam_m']:.10f} m")
    print(f"  mittlere Laser-Tiefe μ:  {gt_debug['mean_lambda_laser_m']:.10f} m")

    print(f"\n🔎 Erste {min(first_n, len(gt_debug['distances']))} Strahlabstände:")
    for i, d in enumerate(gt_debug["distances"][:first_n]):
        print(f"  frame {i:3d}: {d:.10f} m")


def run_solver_debug(
    run_data: dict,
    calib_observations: list,
    intrinsics,
    use_weight: bool = False,
) -> dict | None:
    """
    Optionaler GT-basierter Solver-Debug.
    Gibt None zurück, wenn keine GT vorhanden ist.
    """
    start_pose_gt = run_data["ground_truth"]["start_pose"]
    if start_pose_gt is None:
        return None

    params_gt = initial_guess_from_ground_truth_start_pose(run_data)
    params_perturbed = perturb_parameter_vector(
        params_gt,
        translation_offset=np.array([0.005, -0.003, 0.004], dtype=float),
        rotation_offset_deg=np.array([0.2, -0.3, 0.15], dtype=float),
    )

    frame_results_gt = compute_frame_residuals_detailed(
        params=params_gt,
        observations=calib_observations,
        intrinsics=intrinsics,
        use_weight=use_weight,
    )
    residual_summary_gt = summarize_residuals(frame_results_gt)

    frame_results_perturbed = compute_frame_residuals_detailed(
        params=params_perturbed,
        observations=calib_observations,
        intrinsics=intrinsics,
        use_weight=use_weight,
    )
    residual_summary_perturbed = summarize_residuals(frame_results_perturbed)

    result = solve_extrinsic_pose(
        initial_params=params_perturbed,
        observations=calib_observations,
        intrinsics=intrinsics,
        use_weight=use_weight,
        method="trf",
        verbose=0,
    )
    params_optimized = result.x

    frame_results_optimized = compute_frame_residuals_detailed(
        params=params_optimized,
        observations=calib_observations,
        intrinsics=intrinsics,
        use_weight=use_weight,
    )
    residual_summary_optimized = summarize_residuals(frame_results_optimized)

    diff_init = parameter_difference(params_perturbed, params_gt)
    diff_final = parameter_difference(params_optimized, params_gt)

    return {
        "params_gt": params_gt,
        "params_perturbed": params_perturbed,
        "params_optimized": params_optimized,
        "diff_init": diff_init,
        "diff_final": diff_final,
        "frame_results_gt": frame_results_gt,
        "frame_results_perturbed": frame_results_perturbed,
        "frame_results_optimized": frame_results_optimized,
        "residual_summary_gt": residual_summary_gt,
        "residual_summary_perturbed": residual_summary_perturbed,
        "residual_summary_optimized": residual_summary_optimized,
        "solver_result": result,
    }


def print_solver_debug_report(solver_debug: dict | None, first_n: int = 8) -> None:
    if solver_debug is None:
        print("\nℹ️ Kein Solver-GT-Debug: keine Ground-Truth-Startpose vorhanden.")
        return

    print_parameter_vector(solver_debug["params_gt"], label="Ground-Truth-Startpose")
    print_parameter_vector(solver_debug["params_perturbed"], label="Gestörte Startpose")
    print_parameter_difference(solver_debug["diff_init"], label="Initiale Abweichung zur GT")

    print_residual_summary(
        solver_debug["residual_summary_gt"],
        label="Residuale mit GT-Startpose",
    )
    print_first_frame_residuals(
        solver_debug["frame_results_gt"],
        first_n=first_n,
        label="Erste Residuen mit GT-Startpose",
    )

    print_residual_summary(
        solver_debug["residual_summary_perturbed"],
        label="Residuale mit gestörter Startpose",
    )
    print_first_frame_residuals(
        solver_debug["frame_results_perturbed"],
        first_n=first_n,
        label="Erste Residuen mit gestörter Startpose",
    )

    print("\n🚀 Starte least_squares-Optimierung ...")
    print_solver_report(solver_debug["solver_result"])
    print_parameter_vector(solver_debug["params_optimized"], label="Optimierte Startpose")
    print_parameter_difference(solver_debug["diff_final"], label="Abweichung optimierte Pose zur GT")

    print_residual_summary(
        solver_debug["residual_summary_optimized"],
        label="Residuale nach Optimierung",
    )
    print_first_frame_residuals(
        solver_debug["frame_results_optimized"],
        first_n=first_n,
        label="Erste Residuen nach Optimierung",
    )

    print("\n📈 Vergleich gestört -> optimiert:")
    print(
        f"  mittlerer Residualbetrag: "
        f"{solver_debug['residual_summary_perturbed']['mean_residual_norm_m']:.10f} m  ->  "
        f"{solver_debug['residual_summary_optimized']['mean_residual_norm_m']:.10f} m"
    )
    print(
        f"  maximaler Residualbetrag: "
        f"{solver_debug['residual_summary_perturbed']['max_residual_norm_m']:.10f} m  ->  "
        f"{solver_debug['residual_summary_optimized']['max_residual_norm_m']:.10f} m"
    )
    print(
        f"  RMSE: "
        f"{solver_debug['residual_summary_perturbed']['rmse_residual_m']:.10f} m  ->  "
        f"{solver_debug['residual_summary_optimized']['rmse_residual_m']:.10f} m"
    )