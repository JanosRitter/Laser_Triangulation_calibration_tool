import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter


# ============================================================
# 2D-GAUSS FIT
# ============================================================
def gaussian_2d_model(xy, mu_x, mu_y, sigma_x, sigma_y, amplitude):
    """
    2D-Gaußmodell auf einem Gitter (x, y).
    """
    x, y = xy
    return amplitude * np.exp(
        -(((x - mu_x) ** 2) / (2 * sigma_x ** 2) +
          ((y - mu_y) ** 2) / (2 * sigma_y ** 2))
    )


def gaussian_2d_residuals(params, xy, slice_2d):
    """
    Fehlerfunktion für den Gauß-Fit.
    Verwendet gewichtete quadratische Residuen.
    """
    mu_x, mu_y, sigma_x, sigma_y, amplitude = params
    model = gaussian_2d_model(xy, mu_x, mu_y, sigma_x, sigma_y, amplitude)
    residuals = (model - slice_2d.ravel()) ** 2
    weights = slice_2d.ravel() + 1e-6
    return np.sum(residuals * weights)


def prepare_meshgrid(width, height):
    """
    Erzeugt ein x/y-Gitter für ein 2D-Bild.
    """
    x = np.arange(width)
    y = np.arange(height)
    return np.meshgrid(x, y)


def fit_single_slice_gaussian(slice_2d: np.ndarray):
    """
    Fit einer 2D-Gaußfunktion auf ein einzelnes Crop/Subarray.

    Returns
    -------
    center : np.ndarray
        [mu_x, mu_y]
    deviations : np.ndarray
        [sigma_x, sigma_y]
    amplitude : float
    fitted_slice : np.ndarray
    """
    if not isinstance(slice_2d, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    slice_2d = np.asarray(slice_2d, dtype=np.float64)
    if slice_2d.ndim != 2:
        raise ValueError("slice_2d must be a 2D array.")

    height, width = slice_2d.shape

    smoothed = gaussian_filter(slice_2d, sigma=1.5)
    y_max, x_max = np.unravel_index(np.argmax(smoothed), smoothed.shape)

    initial_guess = [
        x_max,
        y_max,
        max(width / 8, 1),
        max(height / 8, 1),
        smoothed.max()
    ]

    bounds = [
        (0, width),
        (0, height),
        (1, width),
        (1, height),
        (0.1, None)
    ]

    x_value, y_value = prepare_meshgrid(width, height)
    xy_grid = np.vstack([x_value.ravel(), y_value.ravel()])

    result = minimize(
        gaussian_2d_residuals,
        initial_guess,
        args=(xy_grid, slice_2d),
        bounds=bounds,
        method="L-BFGS-B"
    )

    if not result.success:
        return (
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            np.nan,
            np.full((height, width), np.nan)
        )

    mu_x, mu_y, sigma_x, sigma_y, amplitude = result.x
    fitted = gaussian_2d_model(
        xy_grid, mu_x, mu_y, sigma_x, sigma_y, amplitude
    ).reshape(height, width)

    return (
        np.array([mu_x, mu_y]),
        np.array([sigma_x, sigma_y]),
        amplitude,
        fitted
    )


# ============================================================
# THRESHOLD-CENTROID FIT
# ============================================================
def fit_single_slice_threshold_centroid(
    slice_2d: np.ndarray,
    threshold_factor: float = 2.5
):
    """
    Schwerpunkt-Fit nach Thresholding.

    Returns
    -------
    center : np.ndarray
        [x_center, y_center]
    uncertainties : np.ndarray
        [sigma_x, sigma_y]
    filtered_slice : np.ndarray
    """
    if not isinstance(slice_2d, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    slice_2d = np.asarray(slice_2d, dtype=np.float64)
    if slice_2d.ndim != 2:
        raise ValueError("slice_2d must be a 2D array.")

    height, width = slice_2d.shape

    mean_val = np.mean(slice_2d)
    std_val = np.std(slice_2d)
    threshold = mean_val + threshold_factor * std_val

    filtered_slice = np.where(slice_2d >= threshold, slice_2d, 0)

    total_mass = np.sum(filtered_slice)
    if total_mass == 0:
        return (
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            filtered_slice
        )

    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))

    x_center = np.sum(x_indices * filtered_slice) / total_mass
    y_center = np.sum(y_indices * filtered_slice) / total_mass

    x_uncertainty = np.sqrt(
        np.sum((x_indices - x_center) ** 2 * filtered_slice) / total_mass
    )
    y_uncertainty = np.sqrt(
        np.sum((y_indices - y_center) ** 2 * filtered_slice) / total_mass
    )

    return (
        np.array([x_center, y_center]),
        np.array([x_uncertainty, y_uncertainty]),
        filtered_slice
    )