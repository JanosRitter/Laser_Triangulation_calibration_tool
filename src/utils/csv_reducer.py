from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------
# Minimal benötigte Spalten für Kalibrierung
# ---------------------------------------------------------
MINIMAL_COLUMNS = [
    "frame_idx",
    "crop_npy_file",
    "crop_png_file",
    "crop_x0",
    "crop_y0",
    "crop_width",
    "crop_height",
    "status",
]


# ---------------------------------------------------------
# Optionale GT-Spalten (werden entfernt)
# ---------------------------------------------------------
GT_COLUMNS = [
    "laser_x",
    "laser_y",
    "laser_z",
    "laser_rx",
    "laser_ry",
    "laser_rz",
]


def reduce_frame_table_csv(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    keep_columns: Iterable[str] = MINIMAL_COLUMNS,
    drop_gt: bool = True,
) -> Path:
    """
    Reduziert eine frame_table.csv auf ein Minimal-Subset.

    Parameter:
    ----------
    input_csv_path : Pfad zur Original-CSV
    output_csv_path : Zielpfad
    keep_columns : Spalten, die behalten werden sollen
    drop_gt : entfernt explizit bekannte GT-Spalten

    Rückgabe:
    ----------
    Path zur erzeugten Datei
    """
    input_csv_path = Path(input_csv_path)
    output_csv_path = Path(output_csv_path)

    if not input_csv_path.exists():
        raise FileNotFoundError(f"Input CSV nicht gefunden: {input_csv_path}")

    with open(input_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        input_columns = reader.fieldnames or []

        # -------------------------------------------------
        # Zielspalten bestimmen
        # -------------------------------------------------
        selected_columns = [col for col in keep_columns if col in input_columns]

        if not selected_columns:
            raise ValueError("Keine der gewünschten Spalten in CSV gefunden.")

        rows_out = []

        for row in reader:
            new_row = {col: row.get(col, "") for col in selected_columns}

            # Optional: GT-Spalten explizit entfernen (falls vorhanden)
            if drop_gt:
                for gt_col in GT_COLUMNS:
                    new_row.pop(gt_col, None)

            rows_out.append(new_row)

    # -----------------------------------------------------
    # Schreiben
    # -----------------------------------------------------
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=selected_columns)
        writer.writeheader()
        writer.writerows(rows_out)

    return output_csv_path


# ---------------------------------------------------------
# Convenience-Funktion für deinen Workflow
# ---------------------------------------------------------
def reduce_run_folder(
    input_folder: str | Path,
    output_folder: str | Path,
    csv_name: str = "frame_table.csv",
) -> Path:
    """
    Kopiert einen Run-Ordner und reduziert die CSV darin.

    Achtung:
    - andere Dateien werden NICHT kopiert
    - nur CSV wird erzeugt → ideal für Tests

    (wenn du vollständiges Kopieren willst, sag Bescheid → shutil-Version)
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    input_csv = input_folder / csv_name
    output_csv = output_folder / csv_name

    return reduce_frame_table_csv(
        input_csv_path=input_csv,
        output_csv_path=output_csv,
        keep_columns=MINIMAL_COLUMNS,
        drop_gt=True,
    )

reduce_frame_table_csv(
    input_csv_path=r"C:\Users\JRI\Documents\Robotercode_new\laser_calibration\data\2026-04-07_09-01-51_Kopie\frame_table.csv",
    output_csv_path=r"C:\Users\JRI\Documents\Robotercode_new\laser_calibration\data\2026-04-07_09-01-51_Kopie\frame_table_reduced.csv",
)