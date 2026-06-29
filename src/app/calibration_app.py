from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.io.calibration_io import (
    load_calibration_run,
    summarize_calibration_run,
)

from src.calibration.camera_pose_in_robot_frame import (
    camera_pose_from_position_and_axes,
    print_camera_pose_in_robot_frame,
    transform_camera_rays_to_robot_frame,
)
from src.calibration.camera_pose_solver import (
    print_camera_pose_optimization_result,
    solve_camera_pose_from_ray_pairs,
)
from src.calibration.camera_rays import (
    build_camera_intrinsics_for_run,
    pixel_to_camera_ray,
)
from src.calibration.laser_rays import build_laser_rays_robot_base_from_run_data
from src.calibration.pipeline import prepare_calibration_observations
from src.calibration.ray_pair_analysis import (
    analyze_ray_pair_distances,
    print_ray_pair_distance_summary,
)
from src.calibration.reporting import (
    print_compact_preparation_summary,
    print_intrinsics_summary,
    print_run_header,
)
from src.calibration.robot_to_camera_transform import (
    build_calibration_export_metadata,
    robot_to_camera_transform_from_camera_pose,
    save_robot_to_camera_transform_json,
    transform_rays_R_to_C,
)

from src.debug.camera_ray_debug import (
    plot_camera_rays_in_camera_frame,
    plot_camera_rays_on_z_plane,
    print_camera_ray_reference_points,
)
from src.debug.ray_pair_debug import plot_ray_pairs_in_robot_frame
from src.debug.ray_pair_distance_debug import plot_ray_pair_distance_xy
from src.debug.robot_ray_debug import (
    plot_laser_rays,
    plot_laser_rays_in_robot_base,
    plot_laser_ray_offset_debug_in_robot_base,
    plot_laser_offset_top_view_in_robot_base,
    plot_laser_ray_intersections_with_z_plane_in_robot_base,
)
from src.debug.trajectory_debug import run_optional_trajectory_debug


@dataclass
class RunOptions:
    save_fit_crop_overlays: bool = True

    run_trajectory_debug: bool = True
    run_robot_ray_debug: bool = True
    run_initial_ray_pair_debug: bool = True
    run_camera_pose_optimization: bool = True
    run_optimized_ray_pair_debug: bool = True


def run_calibration_app(
    folder_name: str,
    options: RunOptions | None = None,
) -> dict | None:
    if options is None:
        options = RunOptions()

    print("🔧 Calibration Tool – Ray-Rekonstruktion und Debug gestartet")

    # ---------------------------------------------------------
    # 1) Run laden
    # ---------------------------------------------------------
    run_data = load_calibration_run(folder_name)
    summary = summarize_calibration_run(run_data)
    print_run_header(summary)

    output_dir = run_data["input_folder"]
    debug_output_dir = output_dir / "debug_outputs"
    debug_output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 2) Optional: Trajektorie prüfen
    # ---------------------------------------------------------
    trajectory_debug = None

    if options.run_trajectory_debug:
        trajectory_debug = run_optional_trajectory_debug(run_data)

    # ---------------------------------------------------------
    # 3) Absolute Laserrays im Roboter-Basis-KS visualisieren
    # ---------------------------------------------------------
    robot_ray_debug_path = None

    if options.run_robot_ray_debug:
        robot_ray_debug_path = debug_output_dir / "robot_base_laser_rays.png"

        plot_laser_rays_in_robot_base(
            run_data=run_data,
            output_path=robot_ray_debug_path,
            local_ray_direction=np.array([1.0, 0.0, 0.0], dtype=float),
            ray_length=0.5,
        )
        
        plot_laser_ray_offset_debug_in_robot_base(
            run_data=run_data,
            output_path=debug_output_dir / "laser_rays_robot_base_offset_debug.png",
            offset_arrow_scale=1.0,
        )
        plot_laser_offset_top_view_in_robot_base(
            run_data=run_data,
            output_path=debug_output_dir / "laser_offset_top_view_robot_base.png",
        )

        print(f"\n🤖 Roboter-Ray-Debug gespeichert: {robot_ray_debug_path}")

    # ---------------------------------------------------------
    # 4) Crops fitten und CalibrationObservations aufbauen
    # ---------------------------------------------------------
    prep_result = prepare_calibration_observations(
        run_data=run_data,
        method="gaussian",
        threshold_factor=2.5,
        subtract_background=False,
        save_fit_overlays=options.save_fit_crop_overlays,
        fit_overlay_output_dir=debug_output_dir / "fitted_crops",
    )
    print_compact_preparation_summary(prep_result)

    calib_observations = prep_result["calib_observations"]
    if len(calib_observations) == 0:
        print("\n⚠️ Keine Kalibrierbeobachtungen verfügbar.")
        return None

    # ---------------------------------------------------------
    # 5) Kamera-Intrinsics laden
    # ---------------------------------------------------------
    intrinsics = build_camera_intrinsics_for_run(run_data)
    print_intrinsics_summary(intrinsics)

    # ---------------------------------------------------------
    # 6) Kamerarays im Kamera-KS rekonstruieren und debuggen
    # ---------------------------------------------------------
    camera_rays_C = [
        pixel_to_camera_ray(obs.uv, intrinsics)
        for obs in calib_observations
    ]

    uv_list = [obs.uv for obs in calib_observations]

    camera_ray_debug_path = debug_output_dir / "camera_rays_camera_frame.png"
    camera_ray_z_plane_debug_path = debug_output_dir / "camera_rays_z1_plane.png"

    print_camera_ray_reference_points(intrinsics)

    plot_camera_rays_in_camera_frame(
        uv_list=uv_list,
        intrinsics=intrinsics,
        output_path=camera_ray_debug_path,
        ray_length=1.0,
        max_rays=200,
        annotate_indices=True,
    )

    plot_camera_rays_on_z_plane(
        uv_list=uv_list,
        intrinsics=intrinsics,
        output_path=camera_ray_z_plane_debug_path,
        z_plane=1.0,
        max_rays=200,
        annotate_indices=True,
    )

    print(f"📷 Kamera-Ray-Debug gespeichert: {camera_ray_debug_path}")
    print(f"📷 Kamera-Ray-z=1-Debug gespeichert: {camera_ray_z_plane_debug_path}")
    print(f"\n📷 Kamerarays im Kamera-KS rekonstruiert: {len(camera_rays_C)}")

    # ---------------------------------------------------------
    # 7) Grobe Kamerapose im Roboter-KS definieren
    # ---------------------------------------------------------
    camera_pose_initial_R = camera_pose_from_position_and_axes(
        camera_position_R=np.array([-0.07, 0.975, 0.95], dtype=float),
        camera_z_axis_R=np.array([0.0, 0.0, +1.0], dtype=float),
        camera_x_axis_R=np.array([-1.0, 0.0, 0.0], dtype=float),
    )

    print_camera_pose_in_robot_frame(
        camera_pose_initial_R,
        label="Initiale Kamerapose im Roboter-KS",
    )

    # ---------------------------------------------------------
    # 8) Passende Laserrays im Roboter-KS auswählen
    # ---------------------------------------------------------
    all_laser_rays_R = build_laser_rays_robot_base_from_run_data(
        run_data=run_data,
        local_direction=np.array([1.0, 0.0, 0.0], dtype=float),
    )

    laser_rays_R = [
        all_laser_rays_R[obs.frame_idx]
        for obs in calib_observations
    ]

    print(
        f"\n🔦 Laserrays im Roboter-KS für Beobachtungen ausgewählt: "
        f"{len(laser_rays_R)}"
    )

    # ---------------------------------------------------------
    # 9) Kamerarays mit Startpose ins Roboter-KS transformieren
    # ---------------------------------------------------------
    frame_indices = [
        obs.frame_idx
        for obs in calib_observations
    ]

    camera_rays_initial_R = transform_camera_rays_to_robot_frame(
        camera_rays_C=camera_rays_C,
        camera_pose_R=camera_pose_initial_R,
        frame_indices=frame_indices,
    )

    print(
        f"📷 Kamerarays mit initialer Pose ins Roboter-KS transformiert: "
        f"{len(camera_rays_initial_R)}"
    )

    # ---------------------------------------------------------
    # 10) Initiale Ray-Paare gemeinsam plotten
    # ---------------------------------------------------------
    initial_ray_pair_debug_path = None

    if options.run_initial_ray_pair_debug:
        initial_ray_pair_debug_path = (
            debug_output_dir / "ray_pairs_initial_camera_pose_robot_frame.png"
        )

        plot_ray_pairs_in_robot_frame(
            laser_rays_R=laser_rays_R,
            camera_rays_R=camera_rays_initial_R,
            output_path=initial_ray_pair_debug_path,
            laser_ray_length=0.3,
            camera_ray_length=0.5,
            max_pairs=100,
            draw_closest_segments=True,
            annotate_indices=True,
        )

        print(f"🧭 Initialer Ray-Pair-Debug gespeichert: {initial_ray_pair_debug_path}")

    # ---------------------------------------------------------
    # 11) Ray-Pair-Abstände für initiale Kamerapose analysieren
    # ---------------------------------------------------------
    ray_pair_distance_analysis_initial = analyze_ray_pair_distances(
        laser_rays_R=laser_rays_R,
        camera_rays_R=camera_rays_initial_R,
    )

    print_ray_pair_distance_summary(
        ray_pair_distance_analysis_initial,
        label="Initiale Kamerapose",
    )

    ray_pair_distance_xy_debug_path = (
        debug_output_dir / "ray_pair_distances_initial_xy.png"
    )

    plot_ray_pair_distance_xy(
        analysis=ray_pair_distance_analysis_initial,
        output_path=ray_pair_distance_xy_debug_path,
        annotate_indices=True,
        intrinsics=intrinsics,
        camera_pose_R=camera_pose_initial_R,
        camera_reference_ray_length=0.5,
    )

    print(
        f"📏 Initialer Ray-Pair-Abstands-Debug gespeichert: "
        f"{ray_pair_distance_xy_debug_path}"
    )

    # ---------------------------------------------------------
    # 12) Kamerapose im Roboter-KS optimieren
    # ---------------------------------------------------------
    camera_pose_optimization_result = None
    camera_pose_optimized_R = None
    camera_rays_optimized_R = None
    optimized_ray_pair_debug_path = None
    optimized_ray_pair_distance_xy_debug_path = None
    ray_pair_distance_analysis_optimized = None
    T_C_R = None
    laser_rays_C = None
    transform_output_path = None
    laser_rays_camera_frame_debug_path = None

    if options.run_camera_pose_optimization:
        camera_pose_optimization_result = solve_camera_pose_from_ray_pairs(
            laser_rays_R=laser_rays_R,
            camera_rays_C=camera_rays_C,
            initial_pose_R=camera_pose_initial_R,
            frame_indices=frame_indices,
            position_bounds_m=(
                camera_pose_initial_R.translation
                - np.array([0.20, 0.20, 0.20], dtype=float),
                camera_pose_initial_R.translation
                + np.array([0.20, 0.20, 0.20], dtype=float),
            ),
            rotation_bounds_deg=(
                np.array([-20, -20.0, +150.0], dtype=float),
                np.array([+20.0, +20.0, +210.0], dtype=float),
            ),
            verbose=1,
        )

        print_camera_pose_optimization_result(camera_pose_optimization_result)

        camera_pose_optimized_R = camera_pose_optimization_result.optimized_pose_R

        camera_rays_optimized_R = transform_camera_rays_to_robot_frame(
            camera_rays_C=camera_rays_C,
            camera_pose_R=camera_pose_optimized_R,
            frame_indices=frame_indices,
        )

        # -----------------------------------------------------
        # 13) Optimierte Ray-Paare gemeinsam plotten
        # -----------------------------------------------------
        if options.run_optimized_ray_pair_debug:
            optimized_ray_pair_debug_path = (
                debug_output_dir / "ray_pairs_optimized_camera_pose_robot_frame.png"
            )

            plot_ray_pairs_in_robot_frame(
                laser_rays_R=laser_rays_R,
                camera_rays_R=camera_rays_optimized_R,
                output_path=optimized_ray_pair_debug_path,
                laser_ray_length=0.3,
                camera_ray_length=0.5,
                max_pairs=100,
                draw_closest_segments=True,
                annotate_indices=True,
            )

            print(
                f"🧭 Optimierter Ray-Pair-Debug gespeichert: "
                f"{optimized_ray_pair_debug_path}"
            )

            ray_pair_distance_analysis_optimized = analyze_ray_pair_distances(
                laser_rays_R=laser_rays_R,
                camera_rays_R=camera_rays_optimized_R,
            )

            print_ray_pair_distance_summary(
                ray_pair_distance_analysis_optimized,
                label="Optimierte Kamerapose",
            )

            optimized_ray_pair_distance_xy_debug_path = (
                debug_output_dir / "ray_pair_distances_optimized_xy.png"
            )

            plot_ray_pair_distance_xy(
                analysis=ray_pair_distance_analysis_optimized,
                output_path=optimized_ray_pair_distance_xy_debug_path,
                annotate_indices=True,
                intrinsics=intrinsics,
                camera_pose_R=camera_pose_optimized_R,
                camera_reference_ray_length=0.5,
            )

            print(
                f"📏 Optimierter Ray-Pair-Abstands-Debug gespeichert: "
                f"{optimized_ray_pair_distance_xy_debug_path}"
            )

        # -----------------------------------------------------
        # 14) Roboter-KS -> Kamera-KS Transformation exportieren
        # -----------------------------------------------------
        T_C_R = robot_to_camera_transform_from_camera_pose(
            camera_pose_R=camera_pose_optimized_R,
        )

        transform_output_path = output_dir / "robot_to_camera_transform.json"

        export_metadata = build_calibration_export_metadata(
            output_dir=output_dir,
            calib_observations=calib_observations,
            intrinsics=intrinsics,
            camera_pose_optimization_result=camera_pose_optimization_result,
            ray_pair_distance_analysis_optimized=ray_pair_distance_analysis_optimized,
        )

        save_robot_to_camera_transform_json(
            T_C_R=T_C_R,
            output_path=transform_output_path,
            metadata=export_metadata,
        )

        print(f"💾 Roboter->Kamera-Transformation gespeichert: {transform_output_path}")

        # -----------------------------------------------------
        # 15) Laserrays ins Kamera-KS transformieren und plotten
        # -----------------------------------------------------
        laser_rays_C = transform_rays_R_to_C(
            rays_R=laser_rays_R,
            T_C_R=T_C_R,
        )

        laser_rays_camera_frame_debug_path = (
            output_dir / "laser_rays_in_camera_frame.png"
        )

        plot_laser_rays(
            rays=laser_rays_C,
            output_path=laser_rays_camera_frame_debug_path,
            title="Laser-Rays im Kamera-KS",
            axis_labels=("x_C [rechts]", "y_C [oben]", "z_C [Blickrichtung]"),
            ray_length=0.3,
        )

        print(f"📷 Laserrays im Kamera-KS gespeichert: {laser_rays_camera_frame_debug_path}")

    print("\n✅ Ray-Rekonstruktion, Optimierung und Export abgeschlossen")

    return {
        "run_data": run_data,
        "prep_result": prep_result,
        "calib_observations": calib_observations,
        "intrinsics": intrinsics,
        "camera_rays_C": camera_rays_C,
        "trajectory_debug": trajectory_debug,
        "camera_pose_initial_R": camera_pose_initial_R,
        "laser_rays_R": laser_rays_R,
        "camera_rays_initial_R": camera_rays_initial_R,
        "ray_pair_distance_analysis_initial": ray_pair_distance_analysis_initial,
        "camera_pose_optimization_result": camera_pose_optimization_result,
        "camera_pose_optimized_R": camera_pose_optimized_R,
        "camera_rays_optimized_R": camera_rays_optimized_R,
        "ray_pair_distance_analysis_optimized": ray_pair_distance_analysis_optimized,
        "T_C_R": T_C_R,
        "laser_rays_C": laser_rays_C,
        "paths": {
            "output_dir": output_dir,
            "debug_output_dir": debug_output_dir,
            "robot_ray_debug": robot_ray_debug_path,
            "camera_ray_debug": camera_ray_debug_path,
            "camera_ray_z_plane_debug": camera_ray_z_plane_debug_path,
            "fitted_crops": debug_output_dir / "fitted_crops",
            "initial_ray_pair_debug": initial_ray_pair_debug_path,
            "ray_pair_distance_xy_debug": ray_pair_distance_xy_debug_path,
            "optimized_ray_pair_debug": optimized_ray_pair_debug_path,
            "optimized_ray_pair_distance_xy_debug": optimized_ray_pair_distance_xy_debug_path,
            "robot_to_camera_transform": transform_output_path,
            "laser_rays_camera_frame_debug": laser_rays_camera_frame_debug_path,
        },
    }