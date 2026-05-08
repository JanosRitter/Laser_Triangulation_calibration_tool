from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.fitting.fit_methods import (
    fit_single_slice_gaussian,
    fit_single_slice_threshold_centroid,
)
from src.fitting.preprocessing import subtract_mean_background


def fit_crop(
    crop_array: np.ndarray,
    method: str = "gaussian",
    threshold_factor: float = 2.5,
    subtract_background: bool = False,
) -> dict:
    """
    Fittet genau einen Peak in einem Crop-Bild.

    Parameters
    ----------
    crop_array : np.ndarray
        2D-Crop mit genau einem Laserspot
    method : str
        "gaussian" oder "threshold_centroid"
    threshold_factor : float
        Nur relevant für threshold_centroid
    subtract_background : bool
        Mittelwert des Crops abziehen und negative Werte auf 0 setzen

    Returns
    -------
    dict
        Enthält mindestens:
        - "method"
        - "crop_array"
        - "local_center"
        - "fit_ok"

        Zusätzlich je nach Methode:
        - Gaussian:
            "deviations", "amplitude", "fitted_image"
        - Threshold centroid:
            "uncertainties", "filtered_image"
    """
    crop_array = np.asarray(crop_array, dtype=np.float64)

    if crop_array.ndim != 2:
        raise ValueError("crop_array muss ein 2D-Array sein.")

    working_crop = crop_array.copy()

    if subtract_background:
        working_crop = subtract_mean_background(working_crop)

    if method == "gaussian":
        center, deviations, amplitude, fitted = fit_single_slice_gaussian(working_crop)

        fit_ok = not np.any(np.isnan(center))

        return {
            "method": method,
            "crop_array": working_crop,
            "local_center": center,
            "fit_ok": fit_ok,
            "deviations": deviations,
            "amplitude": amplitude,
            "fitted_image": fitted,
        }

    if method == "threshold_centroid":
        center, uncertainties, filtered = fit_single_slice_threshold_centroid(
            working_crop,
            threshold_factor=threshold_factor,
        )

        fit_ok = not np.any(np.isnan(center))

        return {
            "method": method,
            "crop_array": working_crop,
            "local_center": center,
            "fit_ok": fit_ok,
            "uncertainties": uncertainties,
            "filtered_image": filtered,
        }

    raise ValueError(f"Unbekannte Methode: {method}")


def local_crop_center_to_global_uv(local_center: np.ndarray, observation: dict) -> np.ndarray:
    """
    Rechnet lokale Crop-Koordinaten in globale Vollbildkoordinaten um.

    Erwartet in observation:
    - crop_x0
    - crop_y0

    Returns
    -------
    np.ndarray, shape (2,)
        [u, v] in globalen Bildkoordinaten
    """
    local_center = np.asarray(local_center, dtype=float)

    if local_center.shape != (2,):
        raise ValueError("local_center muss die Form (2,) haben.")

    crop_x0 = float(observation["crop_x0"])
    crop_y0 = float(observation["crop_y0"])

    u = crop_x0 + local_center[0]
    v = crop_y0 + local_center[1]

    return np.array([u, v], dtype=float)


def save_fitted_crop_overlay(
    crop_array: np.ndarray,
    local_center: np.ndarray,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """
    Speichert ein Crop-Bild mit eingezeichneter Fit-Position.

    Die Koordinate local_center ist im lokalen Crop-Koordinatensystem:
        x nach rechts
        y nach unten
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crop_array = np.asarray(crop_array, dtype=float)
    local_center = np.asarray(local_center, dtype=float)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(crop_array, cmap="gray", origin="upper")

    if local_center.shape == (2,) and np.all(np.isfinite(local_center)):
        x, y = float(local_center[0]), float(local_center[1])

        ax.scatter([x], [y], marker="x", s=80)
        ax.axvline(x, linewidth=0.8)
        ax.axhline(y, linewidth=0.8)

        ax.text(
            x + 0.5,
            y + 0.5,
            f"({x:.2f}, {y:.2f})",
            fontsize=8,
        )
    else:
        ax.text(
            0.02,
            0.98,
            "fit failed",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel("x crop [px]")
    ax.set_ylabel("y crop [px]")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def fit_observation_crop(
    crop_array: np.ndarray,
    observation: dict,
    method: str = "gaussian",
    threshold_factor: float = 2.5,
    subtract_background: bool = False,
) -> dict:
    """
    Führt den Crop-Fit für eine Beobachtung aus und ergänzt globale UV-Koordinaten.

    Returns
    -------
    dict
        Enthält:
        - frame_idx
        - method
        - fit_ok
        - local_center
        - global_uv
        - methodenspezifische Zusatzinfos
    """
    fit_result = fit_crop(
        crop_array=crop_array,
        method=method,
        threshold_factor=threshold_factor,
        subtract_background=subtract_background,
    )

    if fit_result["fit_ok"]:
        global_uv = local_crop_center_to_global_uv(
            fit_result["local_center"],
            observation,
        )
    else:
        global_uv = np.array([np.nan, np.nan], dtype=float)

    result = {
        "frame_idx": int(observation["frame_idx"]),
        "method": fit_result["method"],
        "fit_ok": fit_result["fit_ok"],
        "local_center": fit_result["local_center"],
        "global_uv": global_uv,
    }

    if fit_result["method"] == "gaussian":
        result["deviations"] = fit_result["deviations"]
        result["amplitude"] = fit_result["amplitude"]
        result["fitted_image"] = fit_result["fitted_image"]

    elif fit_result["method"] == "threshold_centroid":
        result["uncertainties"] = fit_result["uncertainties"]
        result["filtered_image"] = fit_result["filtered_image"]

    return result


def fit_multiple_observations(
    observations: list[dict],
    crop_loader,
    method: str = "gaussian",
    threshold_factor: float = 2.5,
    subtract_background: bool = False,
    save_fit_overlays: bool = False,
    fit_overlay_output_dir: str | Path | None = None,
) -> list[dict]:
    """
    Führt den Fit für mehrere Beobachtungen aus.

    Parameters
    ----------
    observations : list[dict]
        Beobachtungen aus calibration_io
    crop_loader : callable
        Funktion mit Signatur:
            crop_loader(observation) -> np.ndarray
    method : str
    threshold_factor : float
    subtract_background : bool
    save_fit_overlays : bool
        Wenn True, wird pro Crop ein PNG mit eingezeichneter Fit-Position gespeichert.
    fit_overlay_output_dir : str | Path | None
        Zielordner für Overlay-PNGs. Muss gesetzt sein, wenn save_fit_overlays=True.

    Returns
    -------
    list[dict]
        Liste von Fit-Ergebnissen pro Beobachtung
    """
    if save_fit_overlays and fit_overlay_output_dir is None:
        raise ValueError(
            "fit_overlay_output_dir muss gesetzt sein, wenn save_fit_overlays=True."
        )

    overlay_dir = Path(fit_overlay_output_dir) if fit_overlay_output_dir is not None else None

    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for obs in observations:
        crop_array = crop_loader(obs)

        fit_result = fit_observation_crop(
            crop_array=crop_array,
            observation=obs,
            method=method,
            threshold_factor=threshold_factor,
            subtract_background=subtract_background,
        )

        if save_fit_overlays and overlay_dir is not None:
            frame_idx = int(obs["frame_idx"])
            overlay_path = overlay_dir / f"frame_{frame_idx:04d}_fit_overlay.png"

            save_fitted_crop_overlay(
                crop_array=crop_array,
                local_center=fit_result["local_center"],
                output_path=overlay_path,
                title=(
                    f"frame {frame_idx} | "
                    f"method={fit_result['method']} | "
                    f"fit_ok={fit_result['fit_ok']}"
                ),
            )

            fit_result["fit_overlay_path"] = overlay_path

        results.append(fit_result)

    return results