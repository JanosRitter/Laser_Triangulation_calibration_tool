from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================
# PFLICHT- UND OPTIONALE SPALTEN
# ============================================================
REQUIRED_OBSERVATION_COLUMNS = {
    "frame_idx",
    "status",
    "crop_npy_file",
    "crop_x0",
    "crop_y0",
}

OPTIONAL_OBSERVATION_COLUMNS = {
    "crop_png_file",
    "crop_width",
    "crop_height",
}

GROUND_TRUTH_POSE_COLUMNS = {
    "laser_x",
    "laser_y",
    "laser_z",
    "laser_rx",
    "laser_ry",
    "laser_rz",
}

GROUND_TRUTH_UV_COLUMNS = {
    "u",
    "v",
}


# ============================================================
# PFADAUFLÖSUNG / BASISLADEN
# ============================================================
def resolve_input_folder(folder_name_or_path: str | Path) -> Path:
    """
    Löst einen Input-Ordner auf.

    Regeln:
    - absoluter Pfad bleibt unverändert
    - existierender relativer Pfad bleibt unverändert
    - sonst wird relativ zu data/ interpretiert
    """
    folder = Path(folder_name_or_path)

    if folder.is_absolute():
        resolved = folder
    elif folder.exists():
        resolved = folder
    else:
        resolved = Path("data") / folder

    if not resolved.exists():
        raise FileNotFoundError(f"Input-Ordner nicht gefunden: {resolved}")

    if not resolved.is_dir():
        raise NotADirectoryError(f"Pfad ist kein Ordner: {resolved}")

    return resolved.resolve()


def load_run_metadata(input_folder: str | Path) -> dict[str, Any]:
    """
    Lädt run_metadata.json.
    """
    folder = resolve_input_folder(input_folder)
    metadata_path = folder / "run_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"run_metadata.json nicht gefunden: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _convert_csv_value(value: str):
    """
    Wandelt CSV-Werte möglichst sinnvoll um:
    - leer -> ""
    - true/false -> bool
    - int/float -> numerisch
    - sonst string
    """
    if value is None:
        return ""

    value = value.strip()

    if value == "":
        return ""

    if value.lower() == "true":
        return True

    if value.lower() == "false":
        return False

    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_frame_table(input_folder: str | Path) -> list[dict[str, Any]]:
    """
    Lädt frame_table.csv als Liste von Dictionaries.

    WICHTIG:
    - frame_table.csv bleibt aktuell die Pflichtdatei für die
      Beobachtungsbeschreibung pro Frame
    - Ground-Truth-Spalten darin sind optional
    """
    folder = resolve_input_folder(input_folder)
    frame_table_path = folder / "frame_table.csv"

    if not frame_table_path.exists():
        raise FileNotFoundError(
            f"frame_table.csv nicht gefunden: {frame_table_path}\n"
            "Für das Kalibrierungstool wird aktuell eine frame_table.csv "
            "mit mindestens den Beobachtungsspalten benötigt."
        )

    rows = []
    with open(frame_table_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {k: _convert_csv_value(v) for k, v in row.items()}
            rows.append(converted)

    return rows


def load_crop_array(input_folder: str | Path, crop_rel_path: str) -> np.ndarray:
    """
    Lädt ein Crop-Array relativ zum Run-Ordner.
    """
    folder = resolve_input_folder(input_folder)
    crop_path = folder / crop_rel_path

    if not crop_path.exists():
        raise FileNotFoundError(f"Crop-Datei nicht gefunden: {crop_path}")

    return np.load(crop_path)


# ============================================================
# SPALTENERKENNUNG / CAPABILITIES
# ============================================================
def get_frame_table_columns(frame_table: list[dict[str, Any]]) -> set[str]:
    """
    Gibt die vorhandenen Spaltennamen der frame_table zurück.
    """
    if len(frame_table) == 0:
        return set()

    return set(frame_table[0].keys())


def has_required_observation_columns(frame_table: list[dict[str, Any]]) -> bool:
    """
    Prüft, ob alle für die Beobachtung nötigen Spalten vorhanden sind.
    """
    columns = get_frame_table_columns(frame_table)
    return REQUIRED_OBSERVATION_COLUMNS.issubset(columns)


def has_ground_truth_pose_columns(frame_table: list[dict[str, Any]]) -> bool:
    """
    Prüft, ob GT-Pose-Spalten vorhanden sind.
    """
    columns = get_frame_table_columns(frame_table)
    return GROUND_TRUTH_POSE_COLUMNS.issubset(columns)


def has_ground_truth_uv_columns(frame_table: list[dict[str, Any]]) -> bool:
    """
    Prüft, ob GT-UV-Spalten vorhanden sind.
    """
    columns = get_frame_table_columns(frame_table)
    return GROUND_TRUTH_UV_COLUMNS.issubset(columns)


def detect_frame_table_capabilities(frame_table: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analysiert, welche Informationen in der frame_table vorhanden sind.
    """
    columns = get_frame_table_columns(frame_table)

    return {
        "columns": sorted(columns),
        "has_required_observation_columns": has_required_observation_columns(frame_table),
        "has_ground_truth_pose": has_ground_truth_pose_columns(frame_table),
        "has_ground_truth_uv": has_ground_truth_uv_columns(frame_table),
    }


# ============================================================
# BEOBACHTUNGSDATEN
# ============================================================
def is_valid_crop_row(frame_row: dict[str, Any]) -> bool:
    """
    Prüft, ob ein Frame für die Kalibrierung als Beobachtung
    grundsätzlich verwendbar ist.
    """
    if frame_row.get("status") != "valid":
        return False

    crop_file = frame_row.get("crop_npy_file", "")
    if not crop_file:
        return False

    return True


def build_observation_from_frame_row(frame_row: dict[str, Any]) -> dict[str, Any]:
    """
    Baut aus einer frame_table-Zeile eine reduzierte Beobachtungssicht.

    Diese Sicht enthält nur Informationen, die das Kalibrierungstool
    für reale Daten auch sehen darf.
    """
    observation = {
        "frame_idx": int(frame_row["frame_idx"]),
        "crop_npy_file": str(frame_row["crop_npy_file"]),
        "crop_png_file": str(frame_row.get("crop_png_file", "")),
        "crop_x0": frame_row.get("crop_x0", ""),
        "crop_y0": frame_row.get("crop_y0", ""),
        "crop_width": frame_row.get("crop_width", ""),
        "crop_height": frame_row.get("crop_height", ""),
        "status": frame_row.get("status", ""),
    }
    return observation


# ============================================================
# GROUND TRUTH (OPTIONAL)
# ============================================================
def build_ground_truth_pose_from_frame_row(frame_row: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extrahiert die GT-Pose aus einer frame_table-Zeile.

    Gibt None zurück, wenn die benötigten Spalten nicht vorhanden sind.
    """
    missing = [k for k in GROUND_TRUTH_POSE_COLUMNS if k not in frame_row]
    if missing:
        return None

    return {
        "frame_idx": int(frame_row["frame_idx"]),
        "laser_x": float(frame_row["laser_x"]),
        "laser_y": float(frame_row["laser_y"]),
        "laser_z": float(frame_row["laser_z"]),
        "laser_rx": float(frame_row["laser_rx"]),
        "laser_ry": float(frame_row["laser_ry"]),
        "laser_rz": float(frame_row["laser_rz"]),
    }


def build_ground_truth_uv_from_frame_row(frame_row: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extrahiert GT-UV aus einer frame_table-Zeile.

    Gibt None zurück, wenn u/v nicht vorhanden sind.
    """
    missing = [k for k in GROUND_TRUTH_UV_COLUMNS if k not in frame_row]
    if missing:
        return None

    return {
        "frame_idx": int(frame_row["frame_idx"]),
        "u": float(frame_row["u"]),
        "v": float(frame_row["v"]),
    }


# ============================================================
# KOMPLETTER RUN-LOAD
# ============================================================
def load_calibration_run(input_folder: str | Path) -> dict[str, Any]:
    """
    Lädt einen Run in einer für das Kalibrierungstool sinnvollen Struktur.

    WICHTIG:
    - frame_table.csv bleibt Pflicht, weil sie die Beobachtungsbeschreibung
      (crop_npy_file, crop_x0, crop_y0, status, ...) enthält
    - Ground-Truth-Spalten in der CSV sind optional
    """
    folder = resolve_input_folder(input_folder)
    run_metadata = load_run_metadata(folder)
    frame_table = load_frame_table(folder)

    capabilities = detect_frame_table_capabilities(frame_table)

    if not capabilities["has_required_observation_columns"]:
        missing = sorted(REQUIRED_OBSERVATION_COLUMNS - get_frame_table_columns(frame_table))
        raise ValueError(
            "frame_table.csv enthält nicht alle benötigten Beobachtungsspalten.\n"
            f"Fehlende Spalten: {missing}"
        )

    observations = []
    frame_poses_gt = []
    frame_uv_gt = []

    for row in frame_table:
        gt_pose = build_ground_truth_pose_from_frame_row(row)
        if gt_pose is not None:
            frame_poses_gt.append(gt_pose)

        gt_uv = build_ground_truth_uv_from_frame_row(row)
        if gt_uv is not None:
            frame_uv_gt.append(gt_uv)

        if is_valid_crop_row(row):
            observations.append(build_observation_from_frame_row(row))

    start_pose_gt = frame_poses_gt[0] if len(frame_poses_gt) > 0 else None

    return {
        "input_folder": folder,
        "run_metadata": run_metadata,
        "frame_table": frame_table,
        "observations": observations,
        "ground_truth": {
            "start_pose": start_pose_gt,
            "frame_poses": frame_poses_gt,
            "frame_uv": frame_uv_gt,
        },
        "capabilities": capabilities,
    }


# ============================================================
# SUMMARY
# ============================================================
def summarize_calibration_run(run_data: dict[str, Any]) -> dict[str, Any]:
    """
    Erzeugt eine kompakte Zusammenfassung für Debug-Ausgaben.
    """
    frame_table = run_data["frame_table"]
    observations = run_data["observations"]
    capabilities = run_data.get("capabilities", {})

    num_total_frames = len(frame_table)
    num_valid_observations = len(observations)
    num_invalid_frames = num_total_frames - num_valid_observations

    summary = {
        "input_folder": str(run_data["input_folder"]),
        "num_total_frames": num_total_frames,
        "num_valid_observations": num_valid_observations,
        "num_invalid_frames": num_invalid_frames,
        "has_start_pose_gt": run_data["ground_truth"]["start_pose"] is not None,
        "has_ground_truth_pose": bool(capabilities.get("has_ground_truth_pose", False)),
        "has_ground_truth_uv": bool(capabilities.get("has_ground_truth_uv", False)),
    }

    return summary