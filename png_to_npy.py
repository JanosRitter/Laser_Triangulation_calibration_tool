from pathlib import Path
import numpy as np
import imageio.v3 as iio


def convert_pngs_to_npy(folder_path: str):
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Ungültiger Ordner: {folder}")

    png_files = list(folder.glob("*.png"))

    if len(png_files) == 0:
        print("Keine PNG-Dateien gefunden.")
        return

    print(f"Gefundene PNGs: {len(png_files)}\n")

    for png_path in png_files:
        try:
            # Bild laden
            img = iio.imread(png_path)

            # Falls doch RGB → Graustufen nehmen (erste Channel reicht hier)
            if img.ndim == 3:
                img = img[..., 0]

            # Sicherstellen: float64 (wie im Rest deiner Pipeline)
            img = img.astype(np.float64)

            # Zielpfad
            npy_path = png_path.with_suffix(".npy")

            # Speichern
            np.save(npy_path, img)

            print(f"✔ {png_path.name} → {npy_path.name}")

        except Exception as e:
            print(f"✖ Fehler bei {png_path.name}: {e}")

    print("\nFertig.")


if __name__ == "__main__":
    folder_input = r"C:\Users\JRI\Documents\Robotercode_new\laser_calibration\data\20260427_085225_robot_calibration\crops"
    convert_pngs_to_npy(folder_input)
