from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def _extract_valid_uv(fit_results: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrahiert globale UV-Koordinaten aus Fit-Ergebnissen.

    Returns
    -------
    frame_indices : np.ndarray
    uv : np.ndarray, shape (N, 2)
    valid_mask : np.ndarray, shape (N,)
    """
    frame_indices = np.array([int(r["frame_idx"]) for r in fit_results], dtype=int)
    uv = np.array([r["global_uv"] for r in fit_results], dtype=float)
    valid_mask = np.array(
        [
            bool(r.get("fit_ok", False)) and np.all(np.isfinite(r["global_uv"]))
            for r in fit_results
        ],
        dtype=bool,
    )
    return frame_indices, uv, valid_mask


def plot_global_uv_trajectory(
    fit_results: list[dict],
    image_width: int | None = None,
    image_height: int | None = None,
    show_frame_labels: bool = False,
    title: str = "Erkannte Laserpunkte im globalen Bildkoordinatensystem",
):
    """
    Zeigt alle globalen Spot-Positionen (u, v) als Übersicht.
    """
    if len(fit_results) == 0:
        raise ValueError("fit_results ist leer.")

    frame_indices, uv, valid_mask = _extract_valid_uv(fit_results)

    plt.figure(figsize=(10, 6))

    if np.any(valid_mask):
        plt.scatter(
            uv[valid_mask, 0],
            uv[valid_mask, 1],
            s=25,
            label="gültige Fits",
        )

        plt.plot(
            uv[valid_mask, 0],
            uv[valid_mask, 1],
            linewidth=1,
            alpha=0.8,
            label="Trajektorie",
        )

        if show_frame_labels:
            for idx, (u, v) in zip(frame_indices[valid_mask], uv[valid_mask]):
                plt.text(u + 2, v + 2, str(idx), fontsize=8)

    if np.any(~valid_mask):
        plt.scatter(
            uv[~valid_mask, 0],
            uv[~valid_mask, 1],
            marker="x",
            s=40,
            label="ungültige Fits",
        )

    plt.title(title)
    plt.xlabel("u [px]")
    plt.ylabel("v [px]")
    plt.grid(True)
    plt.legend()

    # Kamerabild-Konvention: Ursprung links oben, v nach unten
    if image_width is not None:
        plt.xlim(0, image_width)
    if image_height is not None:
        plt.ylim(image_height, 0)
    else:
        plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


def plot_crop_fit_diagnostic(
    crop_array: np.ndarray,
    fit_result: dict,
    observation: dict | None = None,
    title_prefix: str = "Crop-Fit-Diagnose",
):
    """
    Zeigt einen einzelnen Crop mit erkanntem Zentrum.
    Bei Gauß-Fit zusätzlich das rekonstruierten Fit-Bild.
    """
    crop_array = np.asarray(crop_array, dtype=float)

    method = fit_result.get("method", "unknown")
    fit_ok = bool(fit_result.get("fit_ok", False))
    center = np.asarray(fit_result.get("local_center", [np.nan, np.nan]), dtype=float)

    if method == "gaussian" and "fitted_image" in fit_result:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax0, ax1 = axes

        ax0.imshow(crop_array, origin="upper")
        if fit_ok and np.all(np.isfinite(center)):
            ax0.plot(center[0], center[1], marker="+", markersize=12, markeredgewidth=2)
        ax0.set_title(f"{title_prefix} – Original")
        ax0.set_xlabel("x [px]")
        ax0.set_ylabel("y [px]")

        ax1.imshow(fit_result["fitted_image"], origin="upper")
        if fit_ok and np.all(np.isfinite(center)):
            ax1.plot(center[0], center[1], marker="+", markersize=12, markeredgewidth=2)
        ax1.set_title(f"{title_prefix} – Gauß-Fit")
        ax1.set_xlabel("x [px]")
        ax1.set_ylabel("y [px]")

        if observation is not None:
            frame_idx = observation.get("frame_idx", "?")
            fig.suptitle(f"Frame {frame_idx} | Methode: {method} | fit_ok={fit_ok}")

        plt.tight_layout()
        plt.show()
        return

    # threshold_centroid oder Fallback
    plt.figure(figsize=(5, 4))
    plt.imshow(crop_array, origin="upper")
    if fit_ok and np.all(np.isfinite(center)):
        plt.plot(center[0], center[1], marker="+", markersize=12, markeredgewidth=2)
    plt.title(f"{title_prefix} – Methode: {method} | fit_ok={fit_ok}")
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.tight_layout()
    plt.show()


def plot_multiple_crop_diagnostics(
    observations: list[dict],
    fit_results: list[dict],
    crop_loader,
    max_plots: int = 9,
):
    """
    Zeigt mehrere Crops als Rasteransicht mit markiertem Fit-Zentrum.
    Gut für schnelle manuelle Plausibilitätskontrolle.
    """
    if len(observations) != len(fit_results):
        raise ValueError("observations und fit_results müssen gleich lang sein.")

    n = min(len(observations), max_plots)
    if n == 0:
        raise ValueError("Keine Beobachtungen zum Plotten vorhanden.")

    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes[n:]:
        ax.axis("off")

    for i in range(n):
        obs = observations[i]
        fit = fit_results[i]
        crop = crop_loader(obs)

        ax = axes[i]
        ax.imshow(crop, origin="upper")

        center = np.asarray(fit.get("local_center", [np.nan, np.nan]), dtype=float)
        fit_ok = bool(fit.get("fit_ok", False))

        if fit_ok and np.all(np.isfinite(center)):
            ax.plot(center[0], center[1], marker="+", markersize=10, markeredgewidth=2)

        ax.set_title(f"Frame {obs['frame_idx']} | ok={fit_ok}")
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")

    plt.tight_layout()
    plt.show()


def save_global_uv_plot(
    fit_results: list[dict],
    output_path: str | Path,
    image_width: int | None = None,
    image_height: int | None = None,
    show_frame_labels: bool = False,
    title: str = "Erkannte Laserpunkte im globalen Bildkoordinatensystem",
):
    """
    Speichert den Übersichtsplot als PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_indices, uv, valid_mask = _extract_valid_uv(fit_results)

    plt.figure(figsize=(10, 6))

    if np.any(valid_mask):
        plt.scatter(uv[valid_mask, 0], uv[valid_mask, 1], s=25, label="gültige Fits")
        plt.plot(uv[valid_mask, 0], uv[valid_mask, 1], linewidth=1, alpha=0.8, label="Trajektorie")

        if show_frame_labels:
            for idx, (u, v) in zip(frame_indices[valid_mask], uv[valid_mask]):
                plt.text(u + 2, v + 2, str(idx), fontsize=8)

    if np.any(~valid_mask):
        plt.scatter(uv[~valid_mask, 0], uv[~valid_mask, 1], marker="x", s=40, label="ungültige Fits")

    plt.title(title)
    plt.xlabel("u [px]")
    plt.ylabel("v [px]")
    plt.grid(True)
    plt.legend()

    if image_width is not None:
        plt.xlim(0, image_width)
    if image_height is not None:
        plt.ylim(image_height, 0)
    else:
        plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()