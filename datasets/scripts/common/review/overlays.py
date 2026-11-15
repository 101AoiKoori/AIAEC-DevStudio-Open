"""Visual evidence for candidate review.

This module draws only what the measurement modules reported. It never decides
whether a frame is usable and never mutates its input: `draw_overlay` always
works on a copy and `write_overlay` only serialises the returned image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
AMBER = (0, 191, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.5
THICKNESS = 1


def _canvas(image_bgr: np.ndarray) -> np.ndarray:
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim not in (2, 3):
        raise ValueError("draw_overlay expects a BGR/gray numpy image")
    canvas = image_bgr.copy()
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return canvas


def _point(value: Any) -> tuple[int, int] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        try:
            return int(round(float(value[0]))), int(round(float(value[1])))
        except (TypeError, ValueError):
            return None
    return None


def _xyxy(value: Any) -> tuple[int, int, int, int] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 4:
        try:
            x1, y1, x2, y2 = (int(round(float(part))) for part in value[:4])
        except (TypeError, ValueError):
            return None
        return x1, y1, x2, y2
    return None


def _suspect_box(suspect: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    for key in ("xyxy", "box", "bbox", "rect", "bounds", "region"):
        box = _xyxy(suspect.get(key))
        if box is not None:
            return box
    if all(key in suspect for key in ("x1", "y1", "x2", "y2")):
        return _xyxy([suspect["x1"], suspect["y1"], suspect["x2"], suspect["y2"]])
    center = _point(suspect.get("center") or suspect.get("xy"))
    if center is not None:
        try:
            half_w = int(round(float(suspect.get("w", suspect.get("width", 0))))) // 2
            half_h = int(round(float(suspect.get("h", suspect.get("height", 0))))) // 2
        except (TypeError, ValueError):
            return None
        if half_w > 0 and half_h > 0:
            x, y = center
            return x - half_w, y - half_h, x + half_w, y + half_h
    return None


def _suspect_score(suspect: Mapping[str, Any]) -> float | None:
    for key in ("score", "confidence", "probability", "value"):
        value = suspect.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _draw_ground(canvas: np.ndarray, ground: Any) -> None:
    polygon = getattr(ground, "polygon", None) or []
    points = [_point(item) for item in polygon]
    points = [point for point in points if point is not None]
    if len(points) >= 3:
        array = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [array], isClosed=True, color=GREEN, thickness=2)
    patch = getattr(ground, "placement_patch", None) or []
    patch_points = [_point(item) for item in patch]
    patch_points = [point for point in patch_points if point is not None]
    if len(patch_points) >= 3:
        patch_array = np.array(patch_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [patch_array], isClosed=True, color=AMBER, thickness=2)
    foot = _point(getattr(ground, "foot_xy", None))
    if foot is not None:
        cv2.circle(canvas, foot, 5, YELLOW, -1)


def _draw_emptiness(canvas: np.ndarray, emptiness: Any) -> None:
    for suspect in getattr(emptiness, "suspects", None) or []:
        if not isinstance(suspect, Mapping):
            continue
        box = _suspect_box(suspect)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), RED, THICKNESS)
        score = _suspect_score(suspect)
        if score is not None:
            cv2.putText(canvas, f"{score:.2f}", (x1, max(y1 - 4, 10)),
                        FONT, SCALE, RED, THICKNESS, cv2.LINE_AA)


def _draw_mask_contours(canvas: np.ndarray, mask: Any, color: tuple[int, int, int]) -> None:
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, color, 2)


def _draw_placement(canvas: np.ndarray, masks: Mapping[str, Any], clearance: Mapping[str, Any]) -> None:
    """Draw object/marking/target masks and target-footpoint evidence."""

    _draw_mask_contours(canvas, masks.get("object"), RED)
    _draw_mask_contours(canvas, masks.get("marking"), YELLOW)
    _draw_mask_contours(canvas, masks.get("target"), GREEN)
    for value in masks.get("valid_footpoints") or []:
        point = _point(value)
        if point is not None:
            cv2.circle(canvas, point, 2, GREEN, -1)
    best = _point(masks.get("best_footpoint"))
    if best is not None:
        cv2.circle(canvas, best, 5, WHITE, -1)
    body = _xyxy(masks.get("best_body_box"))
    if body is not None:
        cv2.rectangle(canvas, body[:2], body[2:], CYAN, 1)
    values = [
        ("obj", clearance.get("object_ratio")),
        ("max", clearance.get("object_max_component_ratio")),
        ("vert", clearance.get("vertical_structure_ratio")),
        ("lfr", clearance.get("largest_free_rect_ratio")),
        ("foot", clearance.get("valid_footpoint_ratio")),
        ("conf", clearance.get("confidence")),
    ]
    text = " ".join(f"{name}={value}" for name, value in values if value is not None)
    if text:
        _put_text(canvas, text, (5, canvas.shape[0] - 22), WHITE)


def _put_text(canvas: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    if text:
        cv2.putText(canvas, text, origin, FONT, SCALE, color, THICKNESS, cv2.LINE_AA)


def draw_overlay(
    image_bgr: np.ndarray,
    *,
    ground: Any = None,
    emptiness: Any = None,
    duplicates: Any = None,
    placement_masks: Mapping[str, Any] | None = None,
    caption: str = "",
) -> np.ndarray:
    """Return a copy of ``image_bgr`` with ground, emptiness and duplicate evidence.

    All evidence arguments are optional and independent: any combination of
    ``None`` values simply draws nothing for that layer.
    """

    canvas = _canvas(image_bgr)
    if ground is not None:
        _draw_ground(canvas, ground)
    if emptiness is not None:
        _draw_emptiness(canvas, emptiness)
    if placement_masks:
        _draw_placement(canvas, placement_masks, getattr(ground, "clearance", {}) or {})
    if duplicates is not None and getattr(duplicates, "group", None):
        _put_text(canvas, f"duplicate {duplicates.group}", (5, 18), CYAN)
    if caption:
        height = canvas.shape[0]
        _put_text(canvas, caption, (5, max(height - 8, 14)), WHITE)
    return canvas


def write_overlay(image_bgr: np.ndarray, path: str | Path) -> str:
    """Write an overlay image, creating the parent directory when missing."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = cv2.imwrite(str(target), image_bgr)
    except cv2.error as exc:
        raise RuntimeError(f"cannot write overlay image: {target}") from exc
    if not written:
        raise RuntimeError(f"cannot write overlay image: {target}")
    return str(target)
