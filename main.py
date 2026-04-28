from __future__ import annotations

from src.io.calibration_io import (
    load_calibration_run,
    summarize_calibration_run,
)

from src.calibration.camera_rays import (
    build_camera_intrinsics_from_metadata,
)
from src.calibration.trajectory_debug import (
    run_optional_trajectory_debug,
)
from src.calibration.pipeline_debug import (
    prepare_calibration_observations,
    run_gt_ray_debug,
    run_solver_debug,
    run_calibration_without_gt,
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


def main():
    print("🔧 Calibration Tool – Kalibrierlauf gestartet")

    folder_name = "20260427_085225_robot_calibration"

    # ---------------------------------------------------------
    # 1) Run laden
    # ---------------------------------------------------------
    run_data = load_calibration_run(folder_name)
    summary = summarize_calibration_run(run_data)
    print_run_header(summary)

    # ---------------------------------------------------------
    # 2) Optionalen Trajectory-Debug vorbereiten
    # ---------------------------------------------------------
    trajectory_debug = run_optional_trajectory_debug(run_data)

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
    # 5) Optionaler GT-Strahl-Debug
    # ---------------------------------------------------------
    gt_ray_debug = run_gt_ray_debug(
        run_data=run_data,
        calib_observations=calib_observations,
        intrinsics=intrinsics,
    )
    print_compact_gt_ray_summary(gt_ray_debug)

    # ---------------------------------------------------------
    # 6) Optionaler GT-Solver-Debug
    # ---------------------------------------------------------
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
    print(f"  LOG:  {log_path}")

    print("\n✅ Kalibrierlauf abgeschlossen")


if __name__ == "__main__":
    main()