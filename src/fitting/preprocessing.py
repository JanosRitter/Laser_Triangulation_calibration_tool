import numpy as np


def subtract_mean_background(array: np.ndarray) -> np.ndarray:
    """
    Subtrahiert den Mittelwert eines Arrays und setzt negative Werte auf 0.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    adjusted = array.astype(np.float64) - np.mean(array)
    adjusted[adjusted < 0] = 0
    return adjusted