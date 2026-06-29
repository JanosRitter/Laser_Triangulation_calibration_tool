from __future__ import annotations

from pathlib import Path
import argparse
import csv

import cv2
import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_RUN_DIR = r"C:\Users\JRI\Documents\Robotercode_new\laser_calibration\data\20260513_133615_robot_calibration"

CROP_SIZE = 400
OVERWRITE_FRAME_TABLE = True

# ============================================================
# Parameter
# ============================================================

DEFAULT_CROP_SIZE = 400
MIN_LASER_INTENSITY = 100


# ============================================================
# Laser detection
# ============================================================

def find_laser_point(image: np.ndarray) -> tuple[int, int, float]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    y, x = np.unravel_index(np.argmax(blur), blur.shape)
    max_val = float(blur[y, x])

    return int(x), int(y), max_val


def extract_crop_clamped(
    image: np.ndarray,
    x: int,
    y: int,
    crop_size: int,
) -> tuple[np.ndarray, int, int, bool]:
    """
    Schneidet Crop um x/y aus.

    Wenn der Crop am Bildrand liegen würde, wird er ins Bild hineingeschoben.
    Dadurch entsteht immer ein vollständiger Crop, solange crop_size <= Bildgröße.
    """

    h, w = image.shape[:2]
    half = crop_size // 2

    x0 = x - half
    y0 = y - half

    x0 = max(0, min(x0, w - crop_size))
    y0 = max(0, min(y0, h - crop_size))

    x1 = x0 + crop_size
    y1 = y0 + crop_size

    crop = image[y0:y1, x0:x1]

    was_clamped = not (x0 == x - half and y0 == y - half)

    return crop, x0, y0, was_clamped


# ============================================================
# Main
# ============================================================

def process_run_manual_cropping(
    run_dir: Path,
    crop_size: int = DEFAULT_CROP_SIZE,
    overwrite_frame_table: bool = True,
) -> None:
    run_dir = Path(run_dir)

    images_dir = run_dir / "images"
    crops_dir = run_dir / "crops"
    frame_table_path = run_dir / "frame_table.csv"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images-Ordner nicht gefunden: {images_dir}")

    if not frame_table_path.exists():
        raise FileNotFoundError(f"frame_table.csv nicht gefunden: {frame_table_path}")

    crops_dir.mkdir(parents=True, exist_ok=True)

    with open(frame_table_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    print("\nStarte manuelles Re-Cropping...")
    print(f"Run: {run_dir}")
    print(f"Crop-Größe: {crop_size}x{crop_size}")
    print(f"Bilder laut frame_table.csv: {len(rows)}")

    updated_rows = []

    for row in rows:
        frame_idx = int(row["frame_idx"])

        image_rel = row.get("image_png_file", "")
        image_name = Path(image_rel).name if image_rel else f"frame_{frame_idx:06d}.png"
        image_path = images_dir / image_name

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            row["status"] = "invalid"
            row["crop_clipped"] = True
            updated_rows.append(row)
            print(f"✖ {image_name} → image_load_failed")
            continue

        h, w = image.shape[:2]

        if crop_size > w or crop_size > h:
            row["status"] = "invalid"
            row["crop_clipped"] = True
            updated_rows.append(row)
            print(f"✖ {image_name} → crop_size_larger_than_image")
            continue

        x, y, intensity = find_laser_point(image)

        row["full_px"] = x
        row["full_py"] = y

        if intensity < MIN_LASER_INTENSITY:
            row["status"] = "invalid"
            row["crop_clipped"] = True
            updated_rows.append(row)
            print(f"✖ {image_name} → no_laser intensity={intensity:.1f}")
            continue

        crop, x0, y0, was_clamped = extract_crop_clamped(
            image=image,
            x=x,
            y=y,
            crop_size=crop_size,
        )

        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            row["status"] = "invalid"
            row["crop_clipped"] = True
            updated_rows.append(row)
            print(f"✖ {image_name} → invalid_crop_size {crop.shape}")
            continue

        crop_png_name = f"crop_{frame_idx:06d}.png"
        crop_npy_name = f"crop_{frame_idx:06d}.npy"

        crop_png_path = crops_dir / crop_png_name
        crop_npy_path = crops_dir / crop_npy_name

        cv2.imwrite(str(crop_png_path), crop)

        if crop.ndim == 3:
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            crop_gray = crop

        np.save(crop_npy_path, crop_gray.astype(np.float64))

        row["crop_png_file"] = f"crops/{crop_png_name}"
        row["crop_npy_file"] = f"crops/{crop_npy_name}"

        row["crop_x0"] = x0
        row["crop_y0"] = y0
        row["crop_width"] = crop_size
        row["crop_height"] = crop_size

        row["local_px"] = x - x0
        row["local_py"] = y - y0

        row["status"] = "valid"
        row["crop_clipped"] = was_clamped

        updated_rows.append(row)

        print(
            f"✔ {image_name} → {crop_png_name}, "
            f"laser=({x}, {y}), local=({x - x0}, {y - y0}), "
            f"clamped={was_clamped}"
        )

    if overwrite_frame_table:
        backup_path = frame_table_path.with_suffix(".before_manual_cropping.csv")
        if not backup_path.exists():
            backup_path.write_text(
                frame_table_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        with open(frame_table_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

        print(f"\nBackup gespeichert: {backup_path}")
        print(f"frame_table.csv aktualisiert: {frame_table_path}")

    print(f"\n✅ Manuelles Re-Cropping abgeschlossen")
    print(f"Crops gespeichert in: {crops_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Pfad zum Run-Ordner, der images/ und frame_table.csv enthält.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=DEFAULT_CROP_SIZE,
        help=f"Crop-Größe in Pixeln. Default: {DEFAULT_CROP_SIZE}",
    )
    parser.add_argument(
        "--no-overwrite-frame-table",
        action="store_true",
        help="frame_table.csv nicht überschreiben.",
    )

    args = parser.parse_args()

    process_run_manual_cropping(
        run_dir=args.run_dir,
        crop_size=args.crop_size,
        overwrite_frame_table=not args.no_overwrite_frame_table,
    )


if __name__ == "__main__":
    process_run_manual_cropping(
        run_dir=Path(INPUT_RUN_DIR),
        crop_size=CROP_SIZE,
        overwrite_frame_table=OVERWRITE_FRAME_TABLE,
    )