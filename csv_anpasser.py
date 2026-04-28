from pathlib import Path
import shutil
from datetime import datetime

import pandas as pd


def patch_frame_table(csv_path: str):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Datei ist keine CSV: {csv_path}")

    # Backup-Ordner im gleichen Verzeichnis wie die CSV
    backup_dir = csv_path.parent / "frame_table_backups"
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{csv_path.stem}_backup_{timestamp}{csv_path.suffix}"

    shutil.copy2(csv_path, backup_path)
    print(f"Backup erstellt: {backup_path}")

    df = pd.read_csv(csv_path)

    # crop_npy_file aus crop_png_file erzeugen
    if "crop_png_file" not in df.columns:
        raise KeyError("Spalte 'crop_png_file' fehlt.")

    df["crop_npy_file"] = (
        df["crop_png_file"]
        .fillna("")
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.replace(".png", ".npy", regex=False)
    )

    # image_npy_file optional aus image_png_file erzeugen
    if "image_png_file" in df.columns:
        df["image_npy_file"] = (
            df["image_png_file"]
            .fillna("")
            .astype(str)
            .str.replace("\\", "/", regex=False)
            .str.replace(".png", ".npy", regex=False)
        )

    # ground_truth_file leer lassen, falls keine GT-Dateien existieren
    if "ground_truth_file" in df.columns:
        df["ground_truth_file"] = df["ground_truth_file"].fillna("")

    # num_visible / num_missing sinnvoll setzen, falls leer
    if "num_visible" in df.columns:
        df["num_visible"] = df["num_visible"].fillna(1).astype(int)

    if "num_missing" in df.columns:
        df["num_missing"] = df["num_missing"].fillna(0).astype(int)

    # Sicherstellen, dass status gültig ist
    if "status" in df.columns:
        df["status"] = df["status"].fillna("valid")
    else:
        df["status"] = "valid"

    df.to_csv(csv_path, index=False)
    print(f"CSV überschrieben: {csv_path}")

    print("\nKontrollausgabe:")
    print(df[[
        "frame_idx",
        "status",
        "crop_npy_file",
        "crop_png_file",
        "crop_x0",
        "crop_y0",
        "crop_width",
        "crop_height",
    ]].head())


if __name__ == "__main__":
    path = r"C:\Users\JRI\Documents\Robotercode_new\laser_calibration\data\20260427_085225_robot_calibration\frame_table.csv"
    patch_frame_table(path)