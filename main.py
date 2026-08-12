from __future__ import annotations

from typing import Literal

from src.app.calibration_app import RunOptions, run_calibration_app
from src.evaluation.grouped_intrarun_evaluation import (
    ObservationGroup,
    run_grouped_intrarun_evaluation,
)
from src.evaluation.intrarun_evaluation import run_intrarun_evaluation
from src.evaluation.multirun_evaluation import run_multirun_evaluation
from src.evaluation.offset_optimization_evaluation import (
    run_offset_dz_intrarun_optimization,
)


EvaluationMode = Literal[
    "single_run",
    "multi_run",
    "intra_run",
    "grouped_intra_run",
    "offset_dz_optimization",
]


# Waehle hier, welche Auswertung gestartet werden soll:
# - "single_run": ein kompletter Kalibrier-Run
# - "multi_run": mehrere komplette Runs mit statistischem Vergleich
# - "intra_run": zufaellige Teilmengen eines Runs mit statistischem Vergleich
# - "grouped_intra_run": gleich grosse Teilmengen aus benannten Frame-Bereichen
# - "offset_dz_optimization": Intra-Run-Statistik fuer variierenden Laser-dz-Offset
MODE: EvaluationMode = "intra_run"


# Gemeinsame Pipeline-Optionen fuer einen einzelnen vollstaendigen Run.
SINGLE_RUN_OPTIONS = RunOptions(
    save_fit_crop_overlays=True,
    run_trajectory_debug=True,
    run_robot_ray_debug=True,
    run_initial_ray_pair_debug=True,
    run_camera_pose_optimization=True,
    run_optimized_ray_pair_debug=True,
)

# Schlankere Optionen fuer wiederholte statistische Auswertungen.
# Die Fitting- und Optimierungslogik bleibt identisch; nur teure Debug-Plots
# werden reduziert, damit Multi-/Intra-Run-Auswertungen nicht unnoetig wachsen.
STATISTICAL_RUN_OPTIONS = RunOptions(
    save_fit_crop_overlays=False,
    run_trajectory_debug=False,
    run_robot_ray_debug=False,
    run_initial_ray_pair_debug=False,
    run_camera_pose_optimization=True,
    run_optimized_ray_pair_debug=False,
)
# Parallele Solver fuer vorbereitete Sub-Runs. Bei aktivierten Einzel-Debugplots
# faellt die Auswertung automatisch auf die vollstaendige sequenzielle Pipeline
# zurueck.
STATISTICAL_MAX_WORKERS = 4


# ---------------------------------------------------------------------------
# single_run
# ---------------------------------------------------------------------------
SINGLE_RUN_FOLDER = "20260727_103247_robot_calibration"


# ---------------------------------------------------------------------------
# multi_run
# ---------------------------------------------------------------------------
MULTI_RUN_FOLDERS = [
    "20260630_103748_robot_calibration",
    "20260630_103951_robot_calibration",
    "20260630_104154_robot_calibration",
    "20260630_104356_robot_calibration",
    "20260630_104559_robot_calibration",
    "20260630_104801_robot_calibration",
    "20260630_105005_robot_calibration",
    "20260630_105208_robot_calibration",
    "20260630_105416_robot_calibration",
    "20260630_105625_robot_calibration",
]
MULTI_RUN_OUTPUT_DIR = "data/multirun_evaluation"


# ---------------------------------------------------------------------------
# intra_run
# ---------------------------------------------------------------------------
INTRA_RUN_FOLDER = "20260727_124907_robot_calibration"
# Einzelwert fuer die bisherige Auswertung oder mehrere Bildanzahlen fuer
# eine vergleichende Stabilitaetsanalyse, z. B. [20, 40, 60, 80].
OBSERVATIONS_PER_SUBRUN: int | list[int] = [150]
NUM_SUBRUNS = 1000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# grouped_intra_run
# ---------------------------------------------------------------------------
GROUPED_INTRA_RUN_FOLDER = "20260710_120750_robot_calibration"
GROUPED_OBSERVATIONS_PER_SUBRUN = 60
GROUPED_NUM_SUBRUNS = 100
GROUPED_RANDOM_SEED = 42

# Alle Bereichsgrenzen sind inklusiv. Mehrere Tupel in einer Gruppe werden
# vereinigt. Durch die Aufnahme gefilterte Frames werden automatisch
# ausgelassen und in group_evaluation_summary.json dokumentiert.
#OBSERVATION_GROUPS = [
#    ObservationGroup(name="5cm", frame_ranges=((0, 95),)),
#    ObservationGroup(name="10cm", frame_ranges=((0, 31), (96, 159))),
#   ObservationGroup(name="15cm", frame_ranges=((0, 31), (160, 223))),
#    ObservationGroup(name="20cm", frame_ranges=((0, 31), (224, 287))),
#    ObservationGroup(name="25cm", frame_ranges=((0, 31), (288, 351))),
#    ObservationGroup(name="5+10cm", frame_ranges=((0, 159),)),
#    ObservationGroup(name="5+15cm", frame_ranges=((0, 95), (160, 223))),
#    ObservationGroup(name="5+20cm", frame_ranges=((0, 95), (224, 287))),
#    ObservationGroup(name="5+25cm", frame_ranges=((0, 95), (288, 351))),
#]

OBSERVATION_GROUPS = [
    ObservationGroup(name="main_01_02", frame_ranges=((0, 127),)),
    ObservationGroup(name="main_01_03", frame_ranges=((0, 63), (128, 191))),
    ObservationGroup(name="main_01_04", frame_ranges=((0, 63), (192, 255))),
    ObservationGroup(name="main_01_05", frame_ranges=((0, 63), (256, 319))),
    ObservationGroup(name="main_01_06", frame_ranges=((0, 63), (320, 383))),
    ObservationGroup(name="main_01_07", frame_ranges=((0, 63), (384, 447))),
    ObservationGroup(name="main_01_08", frame_ranges=((0, 63), (448, 511))),
    ObservationGroup(name="main_01_09", frame_ranges=((0, 63), (512, 575))),
    ObservationGroup(name="main_01_10", frame_ranges=((0, 63), (576, 639))),
    ObservationGroup(name="main_01_11", frame_ranges=((0, 63), (640, 703))),
]

# ---------------------------------------------------------------------------
# offset_dz_optimization
# ---------------------------------------------------------------------------
OFFSET_OPTIMIZATION_FOLDER = "20260701_143829_robot_calibration"
OFFSET_DZ_VALUES_M = [
    0.037,
    0.0395,
    0.042,
    0.0445,
    0.047,
]
OFFSET_OBSERVATIONS_PER_SUBRUN = 40
OFFSET_NUM_SUBRUNS = 8
OFFSET_RANDOM_SEED = 42
OFFSET_OBJECTIVE_METRIC = "translation_mean_variance_mm2"


def main() -> None:
    if MODE == "single_run":
        run_calibration_app(
            folder_name=SINGLE_RUN_FOLDER,
            options=SINGLE_RUN_OPTIONS,
        )
        return

    if MODE == "multi_run":
        run_multirun_evaluation(
            folder_names=MULTI_RUN_FOLDERS,
            output_dir=MULTI_RUN_OUTPUT_DIR,
            run_options=STATISTICAL_RUN_OPTIONS,
        )
        return

    if MODE == "intra_run":
        run_intrarun_evaluation(
            folder_name=INTRA_RUN_FOLDER,
            observations_per_subrun=OBSERVATIONS_PER_SUBRUN,
            num_subruns=NUM_SUBRUNS,
            random_seed=RANDOM_SEED,
            run_options=STATISTICAL_RUN_OPTIONS,
            max_workers=STATISTICAL_MAX_WORKERS,
        )
        return

    if MODE == "grouped_intra_run":
        run_grouped_intrarun_evaluation(
            folder_name=GROUPED_INTRA_RUN_FOLDER,
            observation_groups=OBSERVATION_GROUPS,
            observations_per_subrun=GROUPED_OBSERVATIONS_PER_SUBRUN,
            num_subruns=GROUPED_NUM_SUBRUNS,
            random_seed=GROUPED_RANDOM_SEED,
            run_options=STATISTICAL_RUN_OPTIONS,
            max_workers=STATISTICAL_MAX_WORKERS,
        )
        return

    if MODE == "offset_dz_optimization":
        run_offset_dz_intrarun_optimization(
            folder_name=OFFSET_OPTIMIZATION_FOLDER,
            dz_values_m=OFFSET_DZ_VALUES_M,
            observations_per_subrun=OFFSET_OBSERVATIONS_PER_SUBRUN,
            num_subruns=OFFSET_NUM_SUBRUNS,
            random_seed=OFFSET_RANDOM_SEED,
            run_options=STATISTICAL_RUN_OPTIONS,
            max_workers=STATISTICAL_MAX_WORKERS,
            objective_metric=OFFSET_OBJECTIVE_METRIC,
        )
        return

    raise ValueError(f"Unbekannter MODE: {MODE!r}")


if __name__ == "__main__":
    main()
