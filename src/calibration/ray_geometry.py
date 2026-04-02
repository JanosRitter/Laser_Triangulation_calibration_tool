from __future__ import annotations

import numpy as np


def point_on_ray(origin: np.ndarray, direction: np.ndarray, scale: float) -> np.ndarray:
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = np.asarray(direction, dtype=float).reshape(3)
    direction = direction / np.linalg.norm(direction)
    return origin + scale * direction


def closest_points_between_rays(
    origin_a: np.ndarray,
    direction_a: np.ndarray,
    origin_b: np.ndarray,
    direction_b: np.ndarray,
) -> dict:
    """
    Berechnet die nächstgelegenen Punkte zweier 3D-Geraden/Rays.

    Achtung:
    - mathematisch werden zunächst unendliche Geraden behandelt
    - die zurückgegebenen Parameter lambda_a / lambda_b können negativ sein

    Returns
    -------
    dict mit:
        point_a
        point_b
        lambda_a
        lambda_b
        midpoint
        distance
    """
    origin_a = np.asarray(origin_a, dtype=float).reshape(3)
    origin_b = np.asarray(origin_b, dtype=float).reshape(3)

    direction_a = np.asarray(direction_a, dtype=float).reshape(3)
    direction_b = np.asarray(direction_b, dtype=float).reshape(3)

    direction_a = direction_a / np.linalg.norm(direction_a)
    direction_b = direction_b / np.linalg.norm(direction_b)

    w0 = origin_a - origin_b

    a = float(np.dot(direction_a, direction_a))
    b = float(np.dot(direction_a, direction_b))
    c = float(np.dot(direction_b, direction_b))
    d = float(np.dot(direction_a, w0))
    e = float(np.dot(direction_b, w0))

    denom = a * c - b * b

    if abs(denom) < 1e-12:
        # Fast parallel
        lambda_a = 0.0
        lambda_b = e / c if abs(c) > 1e-12 else 0.0
    else:
        lambda_a = (b * e - c * d) / denom
        lambda_b = (a * e - b * d) / denom

    point_a = point_on_ray(origin_a, direction_a, lambda_a)
    point_b = point_on_ray(origin_b, direction_b, lambda_b)
    midpoint = 0.5 * (point_a + point_b)
    distance = float(np.linalg.norm(point_a - point_b))

    return {
        "point_a": point_a,
        "point_b": point_b,
        "lambda_a": float(lambda_a),
        "lambda_b": float(lambda_b),
        "midpoint": midpoint,
        "distance": distance,
    }


def distance_between_rays(
    origin_a: np.ndarray,
    direction_a: np.ndarray,
    origin_b: np.ndarray,
    direction_b: np.ndarray,
) -> float:
    """
    Kürzester Abstand zweier Geraden/Rays.
    """
    result = closest_points_between_rays(
        origin_a=origin_a,
        direction_a=direction_a,
        origin_b=origin_b,
        direction_b=direction_b,
    )
    return float(result["distance"])