from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.io.calibration_io import (
    load_calibration_run,
    summarize_calibration_run,
)

from src.calibration.camera_rays import (
    build_camera_intrinsics_from_metadata,
)

from src.calibration.pipeline import (
    prepare_calibration_observations,
    run_calibration_without_gt,
)

from src.debug.gt_debug import (
    run_gt_ray_debug,
    run_solver_debug,
)

from src.debug.trajectory_debug import (
    run_optional_trajectory_debug,
)

from src.calibration.result_io import (
    build_calibration_result_dict,
    save_calibration_result_json,
    save_frame_residuals_csv,
)

from src.calibration.reporting import (
    print_run_header,
    print_intrinsics_summary,
    print_compact_preparation_summary,
    print_compact_gt_ray_summary,
    print_compact_solver_debug_summary,
    print_compact_calibration_summary,
    write_full_debug_log,
)

from src.debug.ray_visualization_debug import (
    plot_camera_and_laser_rays,
    save_ray_debug_csv,
)

from src.debug.robot_ray_debug import plot_laser_rays_in_robot_base

from src.debug.plane_projection_debug import (
    run_plane_projection_debug,
    fit_plane_projection_debug,
)


@dataclass
class RunOptions:
    run_trajectory_debug: bool = True
    run_robot_ray_debug: bool = True
    run_expected_pose_ray_debug: bool = True
    run_plane_projection_debug: bool = True

    run_gt_ray_debug: bool = True
    run_gt_solver_debug: bool = True
    run_optimized_pose_ray_debug: bool = True

    write_debug_log: bool = True


def main():
    print("🔧 Calibration Tool – Kalibrierlauf gestartet")

    options = RunOptions()

    folder_name = "20260504_082923_robot_calibration"

    # ---------------------------------------------------------
    # 1) Run laden
    # ---------------------------------------------------------
    run_data = load_calibration_run(folder_name)
    summary = summarize_calibration_run(run_data)
    print_run_header(summary)

    # ---------------------------------------------------------
    # 2) Optionale Debugs vor der Kalibrierung
    # ---------------------------------------------------------
    trajectory_debug = None

    if options.run_trajectory_debug:
        trajectory_debug = run_optional_trajectory_debug(run_data)

    if options.run_robot_ray_debug:
        robot_ray_debug_path = run_data["input_folder"] / "robot_base_laser_rays.png"

        plot_laser_rays_in_robot_base(
            run_data=run_data,
            output_path=robot_ray_debug_path,
            local_ray_direction=np.array([0.0, 1.0, 0.0], dtype=float),
            ray_length=0.3,
        )

        print(f"\n🤖 Roboter-Ray-Debug gespeichert: {robot_ray_debug_path}")

    # ---------------------------------------------------------
    # 3) Beobachtungen vorbereiten
    # ---------------------------------------------------------
    prep_result = prepare_calibration_observations(
        run_data=run_data,
        method="gaussian",
        threshold_factor=2.5,
        subtract_background=False,
    )
    print_compact_preparation_summary(prep_result)

    calib_observations = prep_result["calib_observations"]
    if len(calib_observations) == 0:
        print("\n⚠️ Keine Kalibrierbeobachtungen verfügbar.")
        return

    # ---------------------------------------------------------
    # 4) Kamera-Intrinsics
    # ---------------------------------------------------------
    intrinsics = build_camera_intrinsics_from_metadata(run_data["run_metadata"])
    print_intrinsics_summary(intrinsics)

    # ---------------------------------------------------------
    # 4b) Optionaler Ray-Visualisierungs-Debug mit erwarteter Grobpose
    # ---------------------------------------------------------
    if options.run_expected_pose_ray_debug:
        expected_params = np.array([
            0.0,    # x [m]
            0.20,   # y [m]
            0.20,   # z [m]
            30.0,   # rx [deg]
            0.0,    # ry [deg]
            0.0,    # rz [deg]
        ], dtype=float)

        ray_debug_path = run_data["input_folder"] / "ray_debug_expected_pose.png"

        plot_camera_and_laser_rays(
            calib_observations=calib_observations,
            intrinsics=intrinsics,
            params=expected_params,
            output_path=ray_debug_path,
            max_rays=100,
            ray_length=1.0,
        )

        ray_debug_csv_path = run_data["input_folder"] / "ray_debug_expected_pose.csv"

        save_ray_debug_csv(
            calib_observations=calib_observations,
            intrinsics=intrinsics,
            params=expected_params,
            output_path=ray_debug_csv_path,
        )

        print(f"🧭 Ray-Debug-CSV gespeichert: {ray_debug_csv_path}")

    # ---------------------------------------------------------
    # 4c) Optionaler Plane-Projection-Debug
    # ---------------------------------------------------------
    if options.run_plane_projection_debug:
        run_plane_projection_debug(
            run_data=run_data,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            projection_distance_m=0.40,
            plane_center_robot=np.array([-0.075, 0.975, 0.525], dtype=float),
            local_ray_direction=np.array([0.0, 1.0, 0.0], dtype=float),
        )

        fit_plane_projection_debug(
            run_data=run_data,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            initial_center=np.array([-0.075, 0.975, 0.525], dtype=float),
            initial_projection_distance_m=0.40,
            local_ray_direction=np.array([0.0, 1.0, 0.0], dtype=float),
        )

    # ---------------------------------------------------------
    # 5) Optionaler GT-Strahl-Debug
    # ---------------------------------------------------------
    gt_ray_debug = None

    if options.run_gt_ray_debug:
        gt_ray_debug = run_gt_ray_debug(
            run_data=run_data,
            calib_observations=calib_observations,
            intrinsics=intrinsics,
        )
        print_compact_gt_ray_summary(gt_ray_debug)

    # ---------------------------------------------------------
    # 6) Optionaler GT-Solver-Debug
    # ---------------------------------------------------------
    solver_debug = None

    if options.run_gt_solver_debug:
        solver_debug = run_solver_debug(
            run_data=run_data,
            calib_observations=calib_observations,
            intrinsics=intrinsics,
            use_weight=False,
        )
        print_compact_solver_debug_summary(solver_debug)

    # ---------------------------------------------------------
    # 7) Echter GT-freier Kalibrierlauf
    # ---------------------------------------------------------
    calibration_result = run_calibration_without_gt(
        run_data=run_data,
        calib_observations=calib_observations,
        intrinsics=intrinsics,
        initial_params=None,
        use_weight=False,
    )
    print_compact_calibration_summary(calibration_result)

    optimized_params = calibration_result["params_optimized"]

    # ---------------------------------------------------------
    # 7b) Optionaler Ray-Debug mit optimierter Pose
    # ---------------------------------------------------------
    if options.run_optimized_pose_ray_debug:
        ray_debug_opt_path = run_data["input_folder"] / "ray_debug_optimized_pose.png"

        plot_camera_and_laser_rays(
            calib_observations=calib_observations,
            intrinsics=intrinsics,
            params=optimized_params,
            output_path=ray_debug_opt_path,
            max_rays=50,
            ray_length=0.5,
        )

        print(f"\n🧭 Ray-Debug optimierte Pose gespeichert: {ray_debug_opt_path}")

    # ---------------------------------------------------------
    # 8) Ergebnisse speichern
    # ---------------------------------------------------------
    result_dict = build_calibration_result_dict(
        run_data=run_data,
        prep_result=prep_result,
        intrinsics=intrinsics,
        calibration_result=calibration_result,
        gt_ray_debug=gt_ray_debug,
        solver_debug=solver_debug,
    )

    json_path = save_calibration_result_json(
        result_dict,
        run_data["input_folder"] / "calibration_result.json",
    )

    residual_csv_path = save_frame_residuals_csv(
        calibration_result["frame_results_optimized"],
        run_data["input_folder"] / "calibration_frame_residuals.csv",
    )

    log_path = None

    if options.write_debug_log:
        log_path = write_full_debug_log(
            run_data["input_folder"] / "calibration_debug.log",
            trajectory_debug=trajectory_debug,
            prep_result=prep_result,
            gt_ray_debug=gt_ray_debug,
            solver_debug=solver_debug,
            calibration_result=calibration_result,
        )

    print("\n💾 Ergebnisse gespeichert:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {residual_csv_path}")

    if log_path is not None:
        print(f"  LOG:  {log_path}")

    print("\n✅ Kalibrierlauf abgeschlossen")


if __name__ == "__main__":
    main()