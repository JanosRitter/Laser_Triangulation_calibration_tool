from __future__ import annotations

import numpy as np

from src.calibration.calibration_types import CalibrationObservation
from src.calibration.laser_kinematics import build_relative_poses_from_trajectory_config


def build_relative_pose_list_from_run_data(run_data: dict):
    """
    Rekonstruiert die relativen Posen ^L0 T_Li aus run_data["run_metadata"].
    """
    trajectory_config = run_data["run_metadata"]["scan"]["trajectory_config"]
    relative_poses = build_relative_poses_from_trajectory_config(trajectory_config)

    num_total_frames = len(run_data["frame_table"])
    if len(relative_poses) != num_total_frames:
        raise ValueError(
            "Anzahl rekonstruierter relativer Posen passt nicht zur Anzahl Frames: "
            f"{len(relative_poses)} != {num_total_frames}"
        )

    return relative_poses


def build_calibration_observations_from_fit_table(
    run_data: dict,
    fit_table: list[dict],
    only_usable: bool = True,
) -> list[CalibrationObservation]:
    """
    Baut aus der Fit-Tabelle eine Liste von CalibrationObservation-Objekten.

    Erwartete Felder pro Zeile:
    - frame_idx
    - u
    - v
    - use_for_calibration (optional)
    - fit_ok (optional)
    """
    relative_poses = build_relative_pose_list_from_run_data(run_data)

    observations: list[CalibrationObservation] = []

    for row in fit_table:
        if only_usable and ("use_for_calibration" in row):
            if not bool(row["use_for_calibration"]):
                continue
        elif only_usable and ("fit_ok" in row):
            if not bool(row["fit_ok"]):
                continue

        frame_idx = int(row["frame_idx"])
        u = float(row["u"])
        v = float(row["v"])

        if not np.isfinite(u) or not np.isfinite(v):
            continue

        weight = 1.0
        if "snr_estimate" in row and np.isfinite(row["snr_estimate"]):
            weight = float(max(row["snr_estimate"], 1e-6))

        obs = CalibrationObservation(
            frame_idx=frame_idx,
            uv=np.array([u, v], dtype=float),
            relative_pose=relative_poses[frame_idx],
            weight=weight,
        )
        observations.append(obs)

    return observations