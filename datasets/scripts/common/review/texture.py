"""Gabor/FFT texture boundary for the review ground estimator.

The line-based estimator needs two reliable ground directions and a horizon.
Dark floors, reflections and low-texture walls can starve that estimator even
when a usable floor strip is visible. This module measures the floor/wall
transition in the frequency domain with a small Gabor filter bank and a
one-dimensional change-point search; it is a fallback signal, not a new ground
truth. A returned row still has to pass the geometry and placement constraints.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import cv2
import numpy as np

__all__ = ["DEFAULT_PARAMS", "texture_floor_top"]

DEFAULT_PARAMS: dict[str, Any] = {
    "max_width": 640,
    "wavelengths": (8.0, 16.0, 32.0),
    "orientations_deg": (0.0, 45.0, 90.0, 135.0),
    "window_fraction": 0.04,
    "search_above_horizon_fraction": 0.20,
    "min_floor_fraction": 0.06,
    "min_score": 1.0,
    "confidence_scale": 3.0,
}


def _params(value: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_PARAMS)
    if value:
        unknown = set(value) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError("unknown texture parameters: %s" % ", ".join(sorted(unknown)))
        merged.update(value)
    return merged


def _as_gray(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("image must not be empty")
    if array.dtype != np.uint8:
        array = np.clip(np.nan_to_num(array.astype(np.float64), nan=0.0), 0, 255).astype(np.uint8)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        elif array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        elif array.shape[2] == 1:
            array = array[:, :, 0]
        else:
            raise ValueError("image must have 1, 3 or 4 channels")
    if array.ndim != 2:
        raise ValueError("image must be H×W or H×W×C")
    return np.ascontiguousarray(array)


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return (values - median) / max(mad, 1e-6)


def _gabor_profile(gray: np.ndarray, params: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row median Gabor energy and luma in the downscaled image."""

    energies = []
    for wavelength in params["wavelengths"]:
        wavelength = float(wavelength)
        sigma = 0.56 * wavelength
        size = max(3, int(round(4.0 * sigma)) | 1)
        for angle in params["orientations_deg"]:
            kernel = cv2.getGaborKernel(
                (size, size), sigma, math.radians(float(angle)), wavelength, 0.5, 0.0,
                cv2.CV_32F,
            )
            response = cv2.filter2D(gray, cv2.CV_32F, kernel)
            energies.append(np.abs(response))
    if not energies:
        raise ValueError("texture profile needs at least one Gabor filter")
    energy = np.median(np.max(np.stack(energies, axis=0), axis=0), axis=1)
    luma = np.median(gray.astype(np.float32), axis=1)
    return energy, luma


def texture_floor_top(
    image_bgr: Any,
    *,
    horizon_y: float | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate the top row of the visibly textured floor.

    The search starts above the supplied horizon (a fraction of image height) and
    ends above the bottom margin. It picks the strongest low-to-high transition
    of a combined luma/Gabor score, so a bright ceiling or a dark wall does not
    by itself become floor. The result is a fallback candidate: the caller must
    still enforce floor extent, patch depth and placement constraints.
    """

    options = _params(params)
    image = np.asarray(image_bgr)
    if image.size == 0:
        raise ValueError("image must not be empty")
    height, width = image.shape[:2]
    max_width = max(64, int(options["max_width"]))
    scale = min(1.0, max_width / float(width))
    if scale < 1.0:
        small = cv2.resize(
            image, (max_width, max(2, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image
    gray = _as_gray(small)
    rows, width_small = gray.shape[:2]
    energy, luma = _gabor_profile(gray, options)
    profile = _robust_z(energy) + _robust_z(luma)
    window = max(5, int(round(float(options["window_fraction"]) * rows)))
    min_floor = max(4, int(round(float(options["min_floor_fraction"]) * rows)))
    search_above = max(0, int(round(float(options["search_above_horizon_fraction"]) * rows)))
    horizon_small = None if horizon_y is None else int(round(float(horizon_y) * rows / height))
    start = max(int(0.12 * rows), (horizon_small or 0) - search_above, window)
    end = rows - max(min_floor, window)
    if end <= start:
        return {
            "row": None,
            "confidence": 0.0,
            "score": 0.0,
            "floor_fraction": 0.0,
            "method": "gabor_texture_v1",
            "reason": "no_search_range",
        }
    best_score = -float("inf")
    best_row = None
    for row in range(start, end):
        before = float(np.median(profile[max(0, row - window):row]))
        after = float(np.median(profile[row:min(rows, row + window)]))
        score = after - before
        if score > best_score:
            best_score = score
            best_row = row
    floor_fraction = (rows - best_row) / float(rows) if best_row is not None else 0.0
    score = float(best_score)
    if best_row is None or score < float(options["min_score"]) or floor_fraction < float(
        options["min_floor_fraction"]
    ):
        return {
            "row": None,
            "confidence": 0.0,
            "score": round(score, 3),
            "floor_fraction": round(floor_fraction, 4),
            "method": "gabor_texture_v1",
            "reason": "weak_or_short_floor_boundary",
        }
    confidence = min(1.0, score / float(options["confidence_scale"]))
    return {
        "row": int(round(best_row * height / float(rows))),
        "confidence": round(float(confidence), 4),
        "score": round(score, 3),
        "floor_fraction": round(float(floor_fraction), 4),
        "method": "gabor_texture_v1",
        "reason": "ok",
    }
