"""Persist placement observations in one image's pixel coordinate system."""

from __future__ import annotations

import base64
import binascii
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .ground_space import GroundSpace
from .placement import GroundPlacement


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "near_floor_local_support_v2"


def encode_placement_record(
    assessment: GroundPlacement,
    *,
    reference_quad_px: Sequence[Sequence[float]] | None = None,
    horizon_y: float | None = None,
) -> dict[str, Any]:
    """Encode unknown, candidate, and blocked pixels as lossless PNG values 0, 1, and 2.

    Bind the returned payload to the exact source image identity and task in
    the enclosing record. The optional quad defines local coordinates only.
    Its corners run far-left, far-right, near-right, near-left.
    """
    candidate = assessment.candidate_mask
    blocked = assessment.obstacle_mask
    if (candidate.ndim != 2 or candidate.size == 0 or candidate.shape != blocked.shape
            or candidate.dtype != np.bool_ or blocked.dtype != np.bool_ or np.any(candidate & blocked)):
        raise ValueError("expected disjoint image-sized boolean placement masks")
    expected = "candidate" if candidate.any() else "unknown"
    if assessment.status != expected or not isinstance(assessment.basis, str) or not assessment.basis:
        raise ValueError("placement status must agree with its candidate mask and have a basis")
    shape = list(candidate.shape)
    record = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "image_shape": shape,
        "status": assessment.status,
        "basis": assessment.basis,
        "reference_quad_px": None,
        "horizon_y": None,
    }
    if horizon_y is not None:
        if not math.isfinite(horizon_y):
            raise ValueError("horizon_y must be finite")
        record["horizon_y"] = float(horizon_y)
    if reference_quad_px is not None:
        space = GroundSpace(reference_quad_px, image_shape=shape, horizon_y=horizon_y)
        record["reference_quad_px"] = [list(point) for point in space.polygon]
    pixels = candidate.astype(np.uint8)
    pixels[blocked] = 2
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise ValueError("cannot encode placement pixels")
    record["contact_mask_png_base64"] = base64.b64encode(encoded.tobytes()).decode("ascii")
    return record


def decode_placement_record(record: Mapping[str, Any]) -> GroundPlacement:
    """Restore placement evidence without reading or analyzing the background."""
    algorithm = record.get("algorithm_version")
    if (type(record.get("schema_version")) is not int or record["schema_version"] != SCHEMA_VERSION
            or not isinstance(algorithm, str) or not algorithm):
        raise ValueError("unsupported or unversioned placement record")
    shape = record.get("image_shape")
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(type(size) is not int or size < 1 for size in shape)):
        raise ValueError("placement image_shape must contain positive height and width")
    try:
        raw = base64.b64decode(record["contact_mask_png_base64"], validate=True)
    except (KeyError, TypeError, ValueError, binascii.Error) as error:
        raise ValueError("invalid placement mask encoding") from error
    height, width = shape
    if (raw[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            or int.from_bytes(raw[16:20], "big") != width
            or int.from_bytes(raw[20:24], "big") != height):
        raise ValueError("placement PNG dimensions differ from image_shape")
    pixels = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.shape != (height, width) or pixels.dtype != np.uint8 or np.any(pixels > 2):
        raise ValueError("placement mask must contain image-sized uint8 values 0 to 2")
    candidate = pixels == 1
    status = "candidate" if candidate.any() else "unknown"
    basis = record.get("basis")
    if status != record.get("status") or not isinstance(basis, str) or not basis:
        raise ValueError("placement status must agree with its candidate mask and have a basis")
    return GroundPlacement(status, basis, candidate, pixels == 2)


def contact_status_in_view(
    assessment: GroundPlacement,
    foot_xy: Sequence[float],
    *,
    contact_width: float,
    crop_xyxy: Sequence[int],
    content_xyxy: Sequence[int],
) -> str:
    """Check output pixels against a saved mask after crop, resize, and padding.

    The source crop and the output content rectangle use half-open bounds.
    Footpoints use the OpenCV pixel-center resize convention. Output padding
    and contacts that straddle the crop boundary stay unknown.
    """
    sx0, sy0, sx1, sy1 = crop_xyxy
    dx0, dy0, dx1, dy1 = content_xyxy
    height, width = assessment.candidate_mask.shape
    if (any(type(value) is not int for value in (*crop_xyxy, *content_xyxy))
            or not (0 <= sx0 < sx1 <= width and 0 <= sy0 < sy1 <= height)
            or not (0 <= dx0 < dx1 and 0 <= dy0 < dy1)):
        raise ValueError("expected a valid source crop and output content rectangle")
    x, y = foot_xy
    if not all(math.isfinite(value) for value in (x, y, contact_width)) or contact_width < 0:
        raise ValueError("contact requires finite coordinates and a nonnegative width")
    if not (dx0 <= x - contact_width / 2 <= x + contact_width / 2 <= dx1 - 1 and dy0 <= y <= dy1 - 1):
        return "unknown"
    source_per_x = (sx1 - sx0) / (dx1 - dx0)
    source_per_y = (sy1 - sy0) / (dy1 - dy0)
    px = sx0 + (x - dx0 + 0.5) * source_per_x - 0.5
    py = sy0 + (y - dy0 + 0.5) * source_per_y - 0.5
    length = contact_width * source_per_x
    if not (sx0 <= px - length / 2 <= px + length / 2 <= sx1 - 1 and sy0 <= py <= sy1 - 1):
        return "unknown"
    return assessment.contact_status((px, py), contact_width=length)
