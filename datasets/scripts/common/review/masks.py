"""Geometry quality of an alpha/silhouette mask.

Review needs a cheap, deterministic answer to "is this mask usable as a target
mask?" before a frame may enter a dataset.  The measured quantities are purely
geometric:

* connected components after 8-neighbour labelling, ignoring specks smaller
  than ``MIN_COMPONENT_AREA`` pixels;
* enclosed hole area relative to the foreground area;
* the share of foreground pixels that touch the image border, because such a
  silhouette is cut off by the frame.

A mask without foreground carries no geometry: it returns zero metrics with the
reason code ``geometry_rejected`` instead of a hole or border complaint, which
would misdescribe an empty mask.

Input tolerance: a single-channel uint8 mask, a 0/1 float mask, or a ``H×W×1``
array.  Any non-zero value counts as foreground.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

__all__ = [
    "ALPHA_DEFAULTS",
    "MIN_COMPONENT_AREA",
    "alpha_quality",
    "border_fraction",
    "component_count",
    "hole_area_fraction",
]

#: Components below this many pixels are specks, not geometry.
MIN_COMPONENT_AREA = 16

ALPHA_DEFAULTS: dict[str, Any] = {
    "hole_max": 0.02,
    "border_max": 0.15,
    "components_max": 3,
    "border_margin_px": 2,
}


def _foreground(mask: Any, *, name: str = "mask") -> np.ndarray:
    """Return a boolean foreground map; any non-zero value is foreground."""

    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError("%s must be H×W or H×W×1" % name)
    if array.size == 0:
        raise ValueError("%s must not be empty" % name)
    if array.dtype == np.bool_:
        return array
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError("%s must be finite" % name)
    return array != 0


def _component_count_of(foreground: np.ndarray) -> int:
    if not foreground.any():
        return 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(np.count_nonzero(areas >= MIN_COMPONENT_AREA))


def _hole_fraction_of(foreground: np.ndarray) -> float:
    total = int(np.count_nonzero(foreground))
    if total == 0:
        return 0.0
    background = (~foreground).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
    height, width = foreground.shape[:2]
    holes = 0
    for label in range(1, count):
        left, top, box_width, box_height, area = stats[label]
        # A background region touching the image border is outside the
        # silhouette; only fully enclosed regions are holes.
        if left == 0 or top == 0 or left + box_width == width or top + box_height == height:
            continue
        holes += int(area)
    return float(holes) / float(total)


def _border_fraction_of(foreground: np.ndarray, margin_px: int) -> float:
    margin = int(margin_px)
    if margin < 0:
        raise ValueError("margin_px must be >= 0")
    total = int(np.count_nonzero(foreground))
    if total == 0:
        return 0.0
    if margin == 0:
        return 0.0
    height, width = foreground.shape[:2]
    margin = min(margin, height, width)
    band = np.zeros_like(foreground)
    band[:margin, :] = True
    band[height - margin:, :] = True
    band[:, :margin] = True
    band[:, width - margin:] = True
    return float(np.count_nonzero(foreground & band)) / float(total)


def component_count(mask: np.ndarray) -> int:
    """Number of 8-connected foreground components of at least 16 pixels."""

    return _component_count_of(_foreground(mask))


def hole_area_fraction(mask: np.ndarray) -> float:
    """Enclosed hole area divided by foreground area; 0.0 without foreground."""

    return _hole_fraction_of(_foreground(mask))


def border_fraction(mask: np.ndarray, margin_px: int = 2) -> float:
    """Share of foreground pixels lying inside the ``margin_px`` border band."""

    foreground = _foreground(mask)
    return _border_fraction_of(foreground, margin_px)


def alpha_quality(
    mask_alpha: np.ndarray,
    *,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Measure a mask and name every threshold it violates.

    Reason codes are emitted in a fixed order: ``mask_hole``, ``mask_border``,
    then ``geometry_rejected`` for too many components.  An empty mask returns
    zero metrics with ``geometry_rejected`` only.
    """

    merged = dict(ALPHA_DEFAULTS)
    if params:
        merged.update(params)
    hole_max = float(merged["hole_max"])
    border_max = float(merged["border_max"])
    components_max = int(merged["components_max"])
    border_margin = int(merged["border_margin_px"])
    for name, value in (("hole_max", hole_max), ("border_max", border_max)):
        if not np.isfinite(value):
            raise ValueError("%s must be finite" % name)
    if border_margin < 0:
        raise ValueError("border_margin_px must be >= 0")

    foreground = _foreground(mask_alpha, name="mask_alpha")
    height, width = foreground.shape[:2]
    foreground_pixels = int(np.count_nonzero(foreground))
    quality: dict[str, Any] = {
        "components": 0,
        "hole_fraction": 0.0,
        "border_fraction": 0.0,
        "foreground_fraction": float(foreground_pixels) / float(height * width),
        "reason_codes": [],
    }
    if foreground_pixels == 0:
        # No geometry to describe: report the rejection, not a hole or border.
        quality["reason_codes"] = ["geometry_rejected"]
        return quality

    quality["components"] = _component_count_of(foreground)
    quality["hole_fraction"] = _hole_fraction_of(foreground)
    quality["border_fraction"] = _border_fraction_of(foreground, border_margin)

    reason_codes: list[str] = []
    if quality["hole_fraction"] > hole_max:
        reason_codes.append("mask_hole")
    if quality["border_fraction"] > border_max:
        reason_codes.append("mask_border")
    if quality["components"] > components_max:
        reason_codes.append("geometry_rejected")
    quality["reason_codes"] = reason_codes
    return quality
