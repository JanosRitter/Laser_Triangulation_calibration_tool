from __future__ import annotations

from typing import Literal

from src.app.calibration_app import RunOptions, run_calibration_app
from src.evaluation.grouped_intrarun_evaluation import (
    ObservationGroup,
    run_grouped_intrarun_evaluation,
)
from src.evaluation.intrarun_evaluation import run_intrarun_evaluation
from src.evaluation.multirun_evaluation import run_multirun_evaluation


EvaluationMode = Literal[
    "single_run",
    "multi_run",
    "intra_run",
    "grouped_intra_run",
]


# Waehle hier, welche Auswertung gestartet werden soll:
# - "single_run": ein kompletter Kalibrier-Run
# - "multi_run": mehrere komplette Runs mit statistischem Vergleich
# - "intra_run": zufaellige Teilmengen eines Runs mit statistischem Vergleich
# - "grouped_intra_run": gleich grosse Teilmengen aus benannten Frame-Bereichen
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
SINGLE_RUN_FOLDER = "20260520_094027_robot_calibration"


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
INTRA_RUN_FOLDER = "20260701_143829_robot_calibration"
# Einzelwert fuer die bisherige Auswertung oder mehrere Bildanzahlen fuer
# eine vergleichende Stabilitaetsanalyse, z. B. [20, 40, 60, 80].
OBSERVATIONS_PER_SUBRUN: int | list[int] = [20, 30, 40, 50, 60, 80, 100, 120, 160, 200]
NUM_SUBRUNS = 40
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# grouped_intra_run
# ---------------------------------------------------------------------------
GROUPED_INTRA_RUN_FOLDER = "20260701_143829_robot_calibration"
GROUPED_OBSERVATIONS_PER_SUBRUN = 30
GROUPED_NUM_SUBRUNS = 1
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
    ObservationGroup(name="ref_0_19_vs_100_109", frame_ranges=((0, 19), (100, 109))),
    ObservationGroup(name="ref_0_19_vs_110_119", frame_ranges=((0, 19), (110, 119))),
    ObservationGroup(name="ref_0_19_vs_120_129", frame_ranges=((0, 19), (120, 129))),
    ObservationGroup(name="ref_0_19_vs_130_139", frame_ranges=((0, 19), (130, 139))),
    ObservationGroup(name="ref_0_19_vs_140_149", frame_ranges=((0, 19), (140, 149))),
    ObservationGroup(name="ref_0_19_vs_150_159", frame_ranges=((0, 19), (150, 159))),
    ObservationGroup(name="ref_0_19_vs_160_169", frame_ranges=((0, 19), (160, 169))),
    ObservationGroup(name="ref_0_19_vs_170_179", frame_ranges=((0, 19), (170, 179))),
    ObservationGroup(name="ref_0_19_vs_180_189", frame_ranges=((0, 19), (180, 189))),
    ObservationGroup(name="ref_0_19_vs_190_199", frame_ranges=((0, 19), (190, 199))),

   
]


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

    raise ValueError(f"Unbekannter MODE: {MODE!r}")


if __name__ == "__main__":
    main()
