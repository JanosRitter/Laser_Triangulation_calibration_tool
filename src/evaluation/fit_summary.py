from __future__ import annotations

import math
from typing import Callable

import numpy as np


def _safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _estimate_snr(crop_array: np.ndarray) -> float:
    crop_array = np.asarray(crop_array, dtype=float)
    mean_val = float(np.mean(crop_array))
    std_val = float(np.std(crop_array))
    peak_val = float(np.max(crop_array))

    if std_val <= 0:
        return np.nan

    return (peak_val - mean_val) / std_val


def _is_near_border(
    local_center: np.ndarray,
    crop_shape: tuple[int, int],
    border_margin_px: float = 3.0,
) -> bool:
    """
    Prüft, ob das Fit-Zentrum zu nah am Crop-Rand liegt.
    """
    local_center = np.asarray(local_center, dtype=float)
    if local_center.shape != (2,) or np.any(~np.isfinite(local_center)):
        return True

    x, y = local_center
    height, width = crop_shape

    if x < border_margin_px:
        return True
    if y < border_margin_px:
        return True
    if x > (width - 1 - border_margin_px):
        return True
    if y > (height - 1 - border_margin_px):
        return True

    return False


def summarize_fit_result(
    observation: dict,
    fit_result: dict,
    crop_array: np.ndarray,
) -> dict:
    """
    Erstellt eine kompakte Tabellenzeile für genau einen Frame.
    """
    crop_array = np.asarray(crop_array, dtype=float)

    row = {
        "frame_idx": int(observation["frame_idx"]),
        "method": str(fit_result.get("method", "")),
        "fit_ok": bool(fit_result.get("fit_ok", False)),
        "crop_file": str(observation.get("crop_npy_file", "")),
        "crop_x0": _safe_float(observation.get("crop_x0")),
        "crop_y0": _safe_float(observation.get("crop_y0")),
        "crop_width": int(crop_array.shape[1]),
        "crop_height": int(crop_array.shape[0]),
        "local_x": np.nan,
        "local_y": np.nan,
        "u": np.nan,
        "v": np.nan,
        "sigma_x": np.nan,
        "sigma_y": np.nan,
        "sigma_mean": np.nan,
        "amplitude": np.nan,
        "peak_value": float(np.max(crop_array)),
        "crop_mean": float(np.mean(crop_array)),
        "crop_std": float(np.std(crop_array)),
        "snr_estimate": _estimate_snr(crop_array),
        "is_near_crop_border": True,
        "use_for_calibration": False,
        "reject_reason": "",
    }

    local_center = np.asarray(fit_result.get("local_center", [np.nan, np.nan]), dtype=float)
    global_uv = np.asarray(fit_result.get("global_uv", [np.nan, np.nan]), dtype=float)

    if local_center.shape == (2,):
        row["local_x"] = float(local_center[0])
        row["local_y"] = float(local_center[1])

    if global_uv.shape == (2,):
        row["u"] = float(global_uv[0])
        row["v"] = float(global_uv[1])

    row["is_near_crop_border"] = _is_near_border(
        local_center=local_center,
        crop_shape=crop_array.shape,
        border_margin_px=3.0,
    )

    if row["method"] == "gaussian":
        deviations = np.asarray(fit_result.get("deviations", [np.nan, np.nan]), dtype=float)
        if deviations.shape == (2,):
            row["sigma_x"] = float(deviations[0])
            row["sigma_y"] = float(deviations[1])
            row["sigma_mean"] = float(np.mean(deviations))

        row["amplitude"] = _safe_float(fit_result.get("amplitude"))

    return row


def build_fit_results_table(
    observations: list[dict],
    fit_results: list[dict],
    crop_loader: Callable[[dict], np.ndarray],
) -> list[dict]:
    """
    Baut die vollständige Fit-Result-Tabelle für alle Beobachtungen auf.
    """
    if len(observations) != len(fit_results):
        raise ValueError("observations und fit_results müssen gleich lang sein.")

    table = []
    for obs, fit in zip(observations, fit_results):
        crop_array = crop_loader(obs)
        row = summarize_fit_result(
            observation=obs,
            fit_result=fit,
            crop_array=crop_array,
        )
        table.append(row)

    return table


def apply_quality_flags(
    fit_table: list[dict],
    min_snr: float = 4.0,
    min_amplitude: float = 10.0,
    sigma_min: float = 0.8,
    sigma_max: float = 8.0,
    reject_near_border: bool = True,
) -> list[dict]:
    """
    Ergänzt use_for_calibration und reject_reason.
    Arbeitet bewusst konservativ und nachvollziehbar.
    """
    flagged = []

    for row in fit_table:
        new_row = dict(row)
        reasons = []

        if not bool(new_row["fit_ok"]):
            reasons.append("fit_failed")

        snr = _safe_float(new_row["snr_estimate"])
        if np.isfinite(snr) and snr < min_snr:
            reasons.append("low_snr")

        amp = _safe_float(new_row["amplitude"])
        if np.isfinite(amp) and amp < min_amplitude:
            reasons.append("low_amplitude")

        sigma_mean = _safe_float(new_row["sigma_mean"])
        if np.isfinite(sigma_mean):
            if sigma_mean < sigma_min:
                reasons.append("spot_too_small")
            if sigma_mean > sigma_max:
                reasons.append("spot_too_large")

        if reject_near_border and bool(new_row["is_near_crop_border"]):
            reasons.append("near_crop_border")

        new_row["use_for_calibration"] = len(reasons) == 0
        new_row["reject_reason"] = ";".join(reasons)

        flagged.append(new_row)

    return flagged