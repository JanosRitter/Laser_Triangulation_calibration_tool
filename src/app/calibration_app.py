from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.io.calibration_io import (
    load_calibration_run,
    summarize_calibration_run,
)
from src.calibration.camera_rays import (
    build_camera_intrinsics_for_run,
    pixel_to_camera_ray,
)
from src.calibration.pipeline import prepare_calibration_observations
from src.calibration.reporting import (
    print_run_header,
    print_intrinsics_summary,
    print_compact_preparation_summary,
)

from src.debug.trajectory_debug import run_optional_trajectory_debug
from src.debug.robot_ray_debug import (
    plot_laser_rays_in_robot_base,
)
from src.debug.camera_ray_debug import (
    plot_camera_rays_in_camera_frame,
    print_camera_ray_reference_points,
)
from src.calibration.camera_pose_in_robot_frame import (
    camera_pose_from_position_and_axes,
    transform_camera_rays_to_robot_frame,
    print_camera_pose_in_robot_frame,
)
from src.calibration.laser_rays import (
    build_laser_rays_robot_base_from_run_data,
)
from src.debug.ray_pair_debug import plot_ray_pairs_in_robot_frame
from src.calibration.ray_pair_analysis import (
    analyze_ray_pair_distances,
    print_ray_pair_distance_summary,
)
from src.debug.ray_pair_distance_debug import plot_ray_pair_distance_xy
from src.calibration.camera_pose_solver import (
    solve_camera_pose_from_ray_pairs,
    print_camera_pose_optimization_result,
)
from src.debug.camera_ray_debug import (
    plot_camera_rays_in_camera_frame,
    plot_camera_rays_on_z_plane,
    print_camera_ray_reference_points,
)


@dataclass
@dataclass
class RunOptions:
    save_fit_crop_overlays: bool = True

    run_trajectory_debug: bool = True
    run_robot_ray_debug: bool = True
    #run_camera_ray_debug: bool = True
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
            local_ray_direction=np.array([0.0, 1.0, 0.0], dtype=float),
            ray_length=0.3,
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
    # 6) Kamera-Intrinsics laden
    # ---------------------------------------------------------
    intrinsics = build_camera_intrinsics_for_run(run_data)
    print_intrinsics_summary(intrinsics)

    # ---------------------------------------------------------
    # 7) Kamerarays im Kamera-KS rekonstruieren
    # ---------------------------------------------------------
    camera_rays_C = [
        pixel_to_camera_ray(obs.uv, intrinsics)
        for obs in calib_observations
    ]
    camera_ray_debug_path = debug_output_dir / "camera_rays_camera_frame.png"

    uv_list = [obs.uv for obs in calib_observations]
    
    print_camera_ray_reference_points(intrinsics)
    
    plot_camera_rays_in_camera_frame(
        uv_list=uv_list,
        intrinsics=intrinsics,
        output_path=camera_ray_debug_path,
        ray_length=1.0,
        max_rays=200,
        annotate_indices=True,
    )
    camera_ray_z_plane_debug_path = debug_output_dir / "camera_rays_z1_plane.png"

    plot_camera_rays_on_z_plane(
        uv_list=uv_list,
        intrinsics=intrinsics,
        output_path=camera_ray_z_plane_debug_path,
        z_plane=1.0,
        max_rays=200,
        annotate_indices=True,
    )
    
    print(f"📷 Kamera-Ray-z=1-Debug gespeichert: {camera_ray_z_plane_debug_path}")
    
    print(f"📷 Kamera-Ray-Debug gespeichert: {camera_ray_debug_path}")
    print(f"\n📷 Kamerarays im Kamera-KS rekonstruiert: {len(camera_rays_C)}")

    print("\n✅ Ray-Rekonstruktion und Debug abgeschlossen")
    
    # ---------------------------------------------------------
    # 8) Grobe Kamerapose im Roboter-KS definieren
    # ---------------------------------------------------------
    camera_pose_initial_R = camera_pose_from_position_and_axes(
        camera_position_R=np.array([-0.07, 0.97, 0.9], dtype=float),
        camera_z_axis_R=np.array([0.0, 0.0, +1.0], dtype=float),
        camera_x_axis_R=np.array([+1.0, 0.0, 0.0], dtype=float),
    )
    
    print_camera_pose_in_robot_frame(
        camera_pose_initial_R,
        label="Initiale Kamerapose im Roboter-KS",
    )
    
    # ---------------------------------------------------------
    # 9) Passende Laserrays im Roboter-KS für die Beobachtungen auswählen
    # ---------------------------------------------------------
    all_laser_rays_R = build_laser_rays_robot_base_from_run_data(
        run_data=run_data,
        local_direction=np.array([0.0, 1.0, 0.0], dtype=float),
    )
    
    laser_rays_R = [
        all_laser_rays_R[obs.frame_idx]
        for obs in calib_observations
    ]
    
    print(f"\n🔦 Laserrays im Roboter-KS für Beobachtungen ausgewählt: {len(laser_rays_R)}")
    
    # ---------------------------------------------------------
    # 10) Kamerarays mit Startpose ins Roboter-KS transformieren
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
    # 11) Initiale Ray-Paare gemeinsam plotten
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
            camera_ray_length=-0.5,
            max_pairs=100,
            draw_closest_segments=True,
            annotate_indices=True,
        )
    
        print(f"🧭 Initialer Ray-Pair-Debug gespeichert: {initial_ray_pair_debug_path}")
        
    # ---------------------------------------------------------
    # 12) Ray-Pair-Abstände für initiale Kamerapose analysieren
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
    # 13) Kamerapose im Roboter-KS optimieren
    # ---------------------------------------------------------
    camera_pose_optimization_result = None
    camera_pose_optimized_R = None
    camera_rays_optimized_R = None
    optimized_ray_pair_debug_path = None
    optimized_ray_pair_distance_xy_debug_path = None
    ray_pair_distance_analysis_optimized = None
    
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
                np.array([-210.0, -210.0, -210.0], dtype=float),
                np.array([+210.0, +210.0, +210.0], dtype=float),
            ),
            verbose=1,
        )
    
        print_camera_pose_optimization_result(
            camera_pose_optimization_result
        )
    
        camera_pose_optimized_R = (
            camera_pose_optimization_result.optimized_pose_R
        )
    
        camera_rays_optimized_R = transform_camera_rays_to_robot_frame(
            camera_rays_C=camera_rays_C,
            camera_pose_R=camera_pose_optimized_R,
            frame_indices=frame_indices,
        )
    
        # -----------------------------------------------------
        # 14) Optimierte Ray-Paare gemeinsam plotten
        # -----------------------------------------------------
        if options.run_optimized_ray_pair_debug:
            optimized_ray_pair_debug_path = (
                debug_output_dir
                / "ray_pairs_optimized_camera_pose_robot_frame.png"
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
        "paths": {
            "output_dir": output_dir,
            "debug_output_dir": debug_output_dir,
            "robot_ray_debug": robot_ray_debug_path,
            "camera_ray_debug": camera_ray_debug_path,
            "fitted_crops": debug_output_dir / "fitted_crops",
            "initial_ray_pair_debug": initial_ray_pair_debug_path,
            "ray_pair_distance_xy_debug": ray_pair_distance_xy_debug_path,
            "optimized_ray_pair_debug": optimized_ray_pair_debug_path,
            "optimized_ray_pair_distance_xy_debug": optimized_ray_pair_distance_xy_debug_path,
        },
    }