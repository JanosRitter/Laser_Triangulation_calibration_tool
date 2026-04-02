from __future__ import annotations

import numpy as np

from src.io.calibration_io import (
    load_calibration_run,
    summarize_calibration_run,
    load_crop_array,
)

from src.fitting.crop_fitting import fit_multiple_observations
from src.evaluation.fit_summary import (
    build_fit_results_table,
    apply_quality_flags,
)

from src.calibration.camera_rays import (
    build_camera_intrinsics_from_metadata,
    pixel_to_camera_ray,
)
from src.calibration.laser_kinematics import (
    build_extrinsic_pose_from_xyzrpy_deg,
    laser_ray_in_camera,
)
from src.calibration.ray_geometry import closest_points_between_rays
from src.calibration.observation_builder import (
    build_calibration_observations_from_fit_table,
)
from src.calibration.trajectory_debug import (
    run_trajectory_debug,
    print_trajectory_debug_report,
)




def main():
    print("🔧 Calibration Tool – Trajectory- und Geometrie-Debug gestartet")

    folder_name = "2026-04-01_14-06-13"

    # ---------------------------------------------------------
    # 1) Run laden
    # ---------------------------------------------------------
    run_data = load_calibration_run(folder_name)
    summary = summarize_calibration_run(run_data)

    print("\n📂 Eingelesener Datensatz:")
    print(f"  Input-Ordner: {summary['input_folder']}")
    print(f"  Gesamtframes: {summary['num_total_frames']}")
    print(f"  Gültige Beobachtungen: {summary['num_valid_observations']}")
    print(f"  Ungültige Frames: {summary['num_invalid_frames']}")

    # ---------------------------------------------------------
    # 2) Trajectory-Debug gegen Ground Truth
    # ---------------------------------------------------------
    trajectory_debug = run_trajectory_debug(run_data)
    print_trajectory_debug_report(trajectory_debug, first_n=15)

    # ---------------------------------------------------------
    # 3) Beobachtungen fitten
    # ---------------------------------------------------------
    observations_raw = run_data["observations"]
    if len(observations_raw) == 0:
        print("\n⚠️ Keine gültigen Beobachtungen gefunden.")
        return

    def crop_loader(obs):
        return load_crop_array(run_data["input_folder"], obs["crop_npy_file"])

    fit_results = fit_multiple_observations(
        observations=observations_raw,
        crop_loader=crop_loader,
        method="gaussian",
        threshold_factor=2.5,
        subtract_background=False,
    )

    num_fit_ok = sum(1 for r in fit_results if r["fit_ok"])
    print("\n📍 Fit-Auswertung:")
    print(f"  Erfolgreiche Fits: {num_fit_ok}/{len(fit_results)}")

    fit_table = build_fit_results_table(
        observations=observations_raw,
        fit_results=fit_results,
        crop_loader=crop_loader,
    )

    fit_table = apply_quality_flags(
        fit_table,
        min_snr=4.0,
        min_amplitude=10.0,
        sigma_min=0.8,
        sigma_max=8.0,
        reject_near_border=True,
    )

    num_good = sum(1 for row in fit_table if row["use_for_calibration"])
    print("\n📊 Qualitätsbewertung:")
    print(f"  Verwendbar für Kalibrierung: {num_good}/{len(fit_table)}")

    calib_observations = build_calibration_observations_from_fit_table(
        run_data=run_data,
        fit_table=fit_table,
        only_usable=True,
    )

    print("\n🧩 Kalibrierbeobachtungen aufgebaut:")
    print(f"  Anzahl: {len(calib_observations)}")

    if len(calib_observations) == 0:
        print("\n⚠️ Keine Kalibrierbeobachtungen verfügbar.")
        return

    # ---------------------------------------------------------
    # 4) Kamera-Intrinsics
    # ---------------------------------------------------------
    intrinsics = build_camera_intrinsics_from_metadata(run_data["run_metadata"])

    print("\n📷 Kamera-Intrinsics:")
    print(f"  fx={intrinsics.fx:.3f}")
    print(f"  fy={intrinsics.fy:.3f}")
    print(f"  cx={intrinsics.cx:.3f}")
    print(f"  cy={intrinsics.cy:.3f}")

    # ---------------------------------------------------------
    # 5) Strahl-Debug mit Ground Truth
    # ---------------------------------------------------------
    start_pose_gt = run_data["ground_truth"]["start_pose"]

    if start_pose_gt is not None:
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

        print("\n🧪 Strahl-Debug mit Ground Truth:")
        print(f"  mittlerer Strahlabstand: {np.mean(distances):.10f} m")
        print(f"  maximaler Strahlabstand: {np.max(distances):.10f} m")
        print(f"  mittlere Kamera-Tiefe λ: {np.mean(lambda_cam_list):.10f} m")
        print(f"  mittlere Laser-Tiefe μ:  {np.mean(lambda_laser_list):.10f} m")

        print("\n🔎 Erste 10 Strahlabstände:")
        for i, d in enumerate(distances[:10]):
            print(f"  frame {i:3d}: {d:.10f} m")

    print("\n✅ Debuglauf abgeschlossen")


if __name__ == "__main__":
    main()