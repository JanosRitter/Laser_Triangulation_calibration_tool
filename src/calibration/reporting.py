from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path

from src.debug.trajectory_debug import print_trajectory_debug_report
from src.debug.gt_debug import (
    print_gt_ray_debug_report,
    print_solver_debug_report,
)


def print_run_header(summary: dict) -> None:
    print("\n📂 Eingelesener Datensatz:")
    print(f"  Input-Ordner: {summary['input_folder']}")
    print(f"  Gesamtframes: {summary['num_total_frames']}")
    print(f"  Gültige Beobachtungen: {summary['num_valid_observations']}")
    print(f"  Ungültige Frames: {summary['num_invalid_frames']}")


def print_intrinsics_summary(intrinsics) -> None:
    print("\n📷 Kamera-Intrinsics:")
    print(f"  fx={intrinsics.fx:.3f}")
    print(f"  fy={intrinsics.fy:.3f}")
    print(f"  cx={intrinsics.cx:.3f}")
    print(f"  cy={intrinsics.cy:.3f}")


def print_compact_preparation_summary(prep_result: dict) -> None:
    stats = prep_result["stats"]

    print("\n📍 Fit-Auswertung:")
    print(f"  Erfolgreiche Fits: {stats['num_fit_ok']}/{stats['num_fit_total']}")

    print("\n📊 Qualitätsbewertung:")
    print(f"  Verwendbar für Kalibrierung: {stats['num_good']}/{stats['num_fit_table']}")

    print("\n🧩 Kalibrierbeobachtungen aufgebaut:")
    print(f"  Anzahl: {stats['num_calib_observations']}")
    
def print_preparation_report(prep_result: dict) -> None:
    stats = prep_result["stats"]

    print("\n📍 Fit-Auswertung:")
    print(f"  Erfolgreiche Fits: {stats['num_fit_ok']}/{stats['num_fit_total']}")

    print("\n📊 Qualitätsbewertung:")
    print(f"  Verwendbar für Kalibrierung: {stats['num_good']}/{stats['num_fit_table']}")

    print("\n🧩 Kalibrierbeobachtungen aufgebaut:")
    print(f"  Anzahl: {stats['num_calib_observations']}")
    
def print_calibration_without_gt_report(
    calibration_result: dict,
    first_n: int = 8,
) -> None:
    from src.calibration.residuals import (
        print_residual_summary,
        print_first_frame_residuals,
    )
    from src.calibration.solver import (
        print_parameter_vector,
        print_solver_report,
    )

    print("\n🎯 GT-freier Kalibrierlauf:")

    print_parameter_vector(
        calibration_result["params_initial"],
        label="Initiale Startpose",
    )
    print_residual_summary(
        calibration_result["residual_summary_initial"],
        label="Residuale vor Optimierung",
    )
    print_first_frame_residuals(
        calibration_result["frame_results_initial"],
        first_n=first_n,
        label="Erste Residuen vor Optimierung",
    )

    print_solver_report(calibration_result["solver_result"])
    print_parameter_vector(
        calibration_result["params_optimized"],
        label="Optimierte Startpose",
    )

    print_residual_summary(
        calibration_result["residual_summary_optimized"],
        label="Residuale nach Optimierung",
    )
    print_first_frame_residuals(
        calibration_result["frame_results_optimized"],
        first_n=first_n,
        label="Erste Residuen nach Optimierung",
    )

    print("\n📈 Vergleich vor -> nach Optimierung:")
    print(
        f"  mittlerer Residualbetrag: "
        f"{calibration_result['residual_summary_initial']['mean_residual_norm_m']:.10f} m  ->  "
        f"{calibration_result['residual_summary_optimized']['mean_residual_norm_m']:.10f} m"
    )
    print(
        f"  maximaler Residualbetrag: "
        f"{calibration_result['residual_summary_initial']['max_residual_norm_m']:.10f} m  ->  "
        f"{calibration_result['residual_summary_optimized']['max_residual_norm_m']:.10f} m"
    )
    print(
        f"  RMSE: "
        f"{calibration_result['residual_summary_initial']['rmse_residual_m']:.10f} m  ->  "
        f"{calibration_result['residual_summary_optimized']['rmse_residual_m']:.10f} m"
    )


def print_compact_gt_ray_summary(gt_ray_debug: dict | None) -> None:
    if gt_ray_debug is None:
        print("\nℹ️ Kein GT-Strahl-Debug verfügbar.")
        return

    print("\n🧪 Strahl-Debug mit Ground Truth:")
    print(f"  mittlerer Strahlabstand: {gt_ray_debug['mean_distance_m']:.10f} m")
    print(f"  maximaler Strahlabstand: {gt_ray_debug['max_distance_m']:.10f} m")
    print(f"  mittlere Kamera-Tiefe λ: {gt_ray_debug['mean_lambda_cam_m']:.10f} m")
    print(f"  mittlere Laser-Tiefe μ:  {gt_ray_debug['mean_lambda_laser_m']:.10f} m")


def print_compact_solver_debug_summary(solver_debug: dict | None) -> None:
    if solver_debug is None:
        print("\nℹ️ Kein GT-Solver-Debug verfügbar.")
        return

    diff_final = solver_debug["diff_final"]
    residual_summary_gt = solver_debug["residual_summary_gt"]
    residual_summary_optimized = solver_debug["residual_summary_optimized"]

    print("\n🧪 GT-Solver-Debug:")
    print(f"  GT-Residuum (mittel): {residual_summary_gt['mean_residual_norm_m']:.10f} m")
    print(f"  Optimiertes Residuum: {residual_summary_optimized['mean_residual_norm_m']:.10f} m")
    print(f"  Abweichung zur GT |dt|: {diff_final['dtx_norm_m']:.10f} m")
    print(f"  Abweichung zur GT |drot|: {diff_final['drot_norm_deg']:.10f} deg")


def print_compact_calibration_summary(calibration_result: dict) -> None:
    params = calibration_result["params_optimized"]
    residual_summary = calibration_result["residual_summary_optimized"]
    solver_result = calibration_result["solver_result"]

    print("\n🎯 Kalibrierergebnis:")
    print(f"  x  = {params[0]:+.10f} m")
    print(f"  y  = {params[1]:+.10f} m")
    print(f"  z  = {params[2]:+.10f} m")
    print(f"  rx = {params[3]:+.10f} deg")
    print(f"  ry = {params[4]:+.10f} deg")
    print(f"  rz = {params[5]:+.10f} deg")

    print("\n📐 Endgültige Residuen:")
    print(f"  Mittelwert: {residual_summary['mean_residual_norm_m']:.10f} m")
    print(f"  Maximum:    {residual_summary['max_residual_norm_m']:.10f} m")
    print(f"  RMSE:       {residual_summary['rmse_residual_m']:.10f} m")

    print("\n🧠 Solver:")
    print(f"  success: {solver_result.success}")
    print(f"  nfev:    {solver_result.nfev}")
    print(f"  cost:    {solver_result.cost:.12e}")


def write_full_debug_log(
    output_path: str | Path,
    trajectory_debug: dict,
    prep_result: dict,
    gt_ray_debug: dict | None,
    solver_debug: dict | None,
    calibration_result: dict,
) -> Path:
    """
    Schreibt einen ausführlichen Debug-Log mit allen langen Reports.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f, redirect_stdout(f):
        print("🔧 Calibration Tool – Ausführlicher Debug-Log")

        print_trajectory_debug_report(trajectory_debug, first_n=15)
        print_preparation_report(prep_result)
        print_gt_ray_debug_report(gt_ray_debug, first_n=10)
        print_solver_debug_report(solver_debug, first_n=8)
        print_calibration_without_gt_report(calibration_result, first_n=8)

    return output_path