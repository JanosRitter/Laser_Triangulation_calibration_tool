from __future__ import annotations

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
            threshold_factor=threshold_factor
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
            observation
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

    # methodenspezifische Zusatzinfos weiterreichen
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

    Returns
    -------
    list[dict]
        Liste von Fit-Ergebnissen pro Beobachtung
    """
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
        results.append(fit_result)

    return results