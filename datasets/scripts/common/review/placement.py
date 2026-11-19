"""Placement-patch clutter assessment for candidate review.

The geometry stage decides whether a ground placement patch exists. This module
asks a different question: is that patch clean enough to receive a composited
target? It measures non-target objects inside the patch, separates thin ground
markings from standing objects, and checks whether a target body box can be
placed without overlapping clutter.

The output is evidence, not a verdict. ``judge`` applies the task plan's
thresholds and maps the result to ``ready``/``excluded``/``unknown``. A frame
with a generated patch and failed clutter gate is not ready; a low-confidence
measurement stays ``unknown`` and is never promoted.

``assess_placeable_ground`` combines this clutter evidence with positive floor
support to screen pixel contacts for a compositor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .geometry import ground_clearance

__all__ = [
    "DEFAULT_PARAMS",
    "GroundPlacement",
    "PlacementAssessment",
    "assess_placeable_ground",
    "assess_placement",
    "placement_decision",
]

DEFAULT_PARAMS: dict[str, Any] = {
    "rectified_size": 512,
    "canny_low": 50,
    "canny_high": 150,
    "close_kernel": 7,
    "close_iterations": 2,
    "dilate_kernel": 5,
    "vertical_threshold": 80.0,
    "min_component_area_ratio": 0.01,
    "marking_aspect_min": 5.0,
    "marking_vertical_max": 0.20,
    "floor_grid": 8,
    "floor_sample_fraction": 0.5,
    "floor_mahalanobis_max": 5.0,
    "grid_rows": 9,
    "grid_cols": 15,
    "target_body_free_min": 0.95,
    "edge_density_win": 24,
    "edge_density_max": 0.06,
    "stability_probes": 2,
}

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "max_object_ratio": 0.10,
    "max_object_component_ratio": 0.08,
    "max_vertical_ratio": 0.008,
    "min_largest_free_rect_ratio": 0.50,
    "min_valid_footpoint_ratio": 0.05,
    "max_body_obstacle_ratio": 0.05,
    "clutter_exclude_ratio": 0.15,
    "confidence_min": 0.60,
    "borderline_margin": 0.10,
}

DEFAULT_TARGET: dict[str, Any] = {
    "height_fraction": 0.30,
    "aspect": 0.42,
}

REQUIRED_METRICS = (
    "object_ratio",
    "object_max_component_ratio",
    "vertical_structure_ratio",
    "largest_free_rect_ratio",
    "valid_footpoint_ratio",
    "best_body_obstacle_ratio",
    "confidence",
)


@dataclass
class PlacementAssessment:
    """JSON-safe metrics plus transient masks for overlays."""

    metrics: dict[str, Any] = field(default_factory=dict)
    masks: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundPlacement:
    """Image-sized masks for a placement candidate and suspected obstacles."""

    status: str
    basis: str
    candidate_mask: np.ndarray
    obstacle_mask: np.ndarray

    def contact_status(self, foot_xy: Sequence[float], *, contact_width: float = 0) -> str:
        """Return candidate, blocked or unknown for the entire pixel contact segment."""
        x, y = foot_xy
        if not all(math.isfinite(v) for v in (x, y, contact_width)) or contact_width < 0:
            raise ValueError("contact requires finite coordinates and a nonnegative width")
        left = math.floor(x - contact_width / 2)
        right = math.ceil(x + contact_width / 2)
        top, bottom = math.floor(y), math.ceil(y)
        height, width = self.candidate_mask.shape
        if left < 0 or right >= width or top < 0 or bottom >= height:
            return "unknown"
        contact = np.s_[top:bottom + 1, left:right + 1]
        if self.obstacle_mask[contact].any():
            return "blocked"
        return "candidate" if self.candidate_mask[contact].all() else "unknown"


def _bgr_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("image must not be empty")
    if array.dtype != np.uint8:
        array = np.clip(np.nan_to_num(array.astype(np.float64), nan=0.0), 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must be H×W, H×W×1, H×W×3 or H×W×4")
    return np.ascontiguousarray(array)


def _params(value: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_PARAMS)
    if value:
        unknown = set(value) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError("unknown placement parameters: %s" % ", ".join(sorted(unknown)))
        merged.update(value)
    merged["rectified_size"] = int(merged["rectified_size"])
    merged["grid_rows"] = max(3, int(merged["grid_rows"]))
    merged["grid_cols"] = max(3, int(merged["grid_cols"]))
    return merged


def _target(value: Mapping[str, Any] | None) -> dict[str, float]:
    merged = dict(DEFAULT_TARGET)
    if value:
        for key in ("height_fraction", "target_height_fraction"):
            if key in value:
                merged["height_fraction"] = float(value[key])
        for key in ("aspect", "target_aspect"):
            if key in value:
                merged["aspect"] = float(value[key])
    if not (0.0 < merged["height_fraction"] <= 1.0):
        raise ValueError("target height_fraction must be in (0, 1]")
    if merged["aspect"] <= 0.0:
        raise ValueError("target aspect must be positive")
    return merged


def _patch_mask(shape: tuple[int, int], patch: Sequence[Sequence[float]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(patch) >= 3:
        cv2.fillPoly(mask, [np.asarray(patch, dtype=np.int32).reshape(-1, 1, 2)], 255)
    return mask


def _boxes_mask(shape: tuple[int, int], boxes: Iterable[Any]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for box in boxes or []:
        xyxy = getattr(box, "xyxy", None)
        if xyxy is None and isinstance(box, Mapping):
            xyxy = box.get("xyxy") or box.get("box")
        if not isinstance(xyxy, Sequence) or len(xyxy) < 4:
            continue
        try:
            x1, y1, x2, y2 = (int(round(float(value))) for value in xyxy[:4])
        except (TypeError, ValueError):
            continue
        x1, y1 = max(0, min(x1, shape[1] - 1)), max(0, min(y1, shape[0] - 1))
        x2, y2 = max(x1 + 1, min(x2, shape[1])), max(y1 + 1, min(y2, shape[0]))
        mask[y1:y2, x1:x2] = 255
    return mask


def _edge_density(gray: np.ndarray, window: int) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    win = max(3, int(window))
    return cv2.boxFilter(edges, -1, (win, win), normalize=True)


def _legacy_clearance(image: np.ndarray, patch: Sequence[Sequence[float]]) -> dict[str, Any]:
    try:
        return dict(ground_clearance(image, patch) or {})
    except (TypeError, ValueError):
        return {}


def _floor_model(image: np.ndarray, patch_mask: np.ndarray, params: dict[str, Any]):
    patch_pixels = int(np.count_nonzero(patch_mask))
    if patch_pixels < 50:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    density = _edge_density(gray, params["edge_density_win"])
    height, width = gray.shape
    grid = max(2, int(params["floor_grid"]))
    cells = []
    for row in range(grid):
        y1, y2 = int(height * row / grid), int(height * (row + 1) / grid)
        for col in range(grid):
            x1, x2 = int(width * col / grid), int(width * (col + 1) / grid)
            region = patch_mask[y1:y2, x1:x2] > 0
            if int(np.count_nonzero(region)) < 20:
                continue
            cells.append(
                {
                    "density": float(density[y1:y2, x1:x2][region].mean()),
                    "pixels": lab[y1:y2, x1:x2][region],
                }
            )
    if len(cells) < 3:
        return None
    cells.sort(key=lambda cell: cell["density"])
    keep = max(1, int(math.ceil(len(cells) * float(params["floor_sample_fraction"]))))
    samples = np.concatenate([cell["pixels"] for cell in cells[:keep]], axis=0)
    if samples.shape[0] < 20:
        return None
    mean = samples.mean(axis=0)
    centered = samples - mean
    covariance = centered.T @ centered / max(1, samples.shape[0] - 1)
    covariance += np.eye(3, dtype=np.float32) * 1e-3
    try:
        inverse = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        return None
    delta = lab[patch_mask > 0] - mean
    distance = np.sqrt(np.einsum("ij,jk,ik->i", delta, inverse, delta))
    threshold = float(params["floor_mahalanobis_max"])
    support = float(np.mean(distance <= threshold)) if distance.size else 0.0
    return {
        "mean": mean,
        "inverse": inverse,
        "threshold": threshold,
        "support": support,
        "density": density,
    }


def _vertical_map(gray: np.ndarray, threshold: float) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return ((np.abs(gx) > threshold) & (np.abs(gx) > np.abs(gy))).astype(np.uint8)


def _mask_components(
    image: np.ndarray,
    patch_mask: np.ndarray,
    target_mask: np.ndarray,
    floor: dict[str, Any] | None,
    params: dict[str, Any],
    *,
    canny_low: int | None = None,
    canny_high: int | None = None,
    close_kernel: int | None = None,
) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low = int(canny_low if canny_low is not None else params["canny_low"])
    high = int(canny_high if canny_high is not None else params["canny_high"])
    kernel = int(close_kernel if close_kernel is not None else params["close_kernel"])
    kernel = max(3, kernel | 1)
    edges = cv2.Canny(gray, low, high)
    closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8),
        iterations=int(params["close_iterations"]),
    )
    closed = cv2.dilate(closed, np.ones((int(params["dilate_kernel"]),) * 2, np.uint8), iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vertical = _vertical_map(gray, float(params["vertical_threshold"]))
    patch_pixels = int(np.count_nonzero(patch_mask))
    min_area = max(1, int(float(params["min_component_area_ratio"]) * patch_pixels))
    object_mask = np.zeros_like(patch_mask)
    marking_mask = np.zeros_like(patch_mask)
    target_components = np.zeros_like(patch_mask)
    components: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 3 or height < 3:
            continue
        component = np.zeros_like(patch_mask)
        cv2.drawContours(component, [contour], -1, 255, -1)
        inside = (component > 0) & (patch_mask > 0)
        if int(np.count_nonzero(inside)) < min_area:
            continue
        overlap_target = float(np.count_nonzero(inside & (target_mask > 0))) / max(
            1, int(np.count_nonzero(inside))
        )
        support = float(vertical[inside].mean()) if np.any(inside) else 0.0
        aspect = max(width, height) / max(1, min(width, height))
        if overlap_target >= 0.50:
            classification = "task_target"
            cv2.drawContours(target_components, [contour], -1, 255, -1)
        elif aspect >= float(params["marking_aspect_min"]) and support < float(
            params["marking_vertical_max"]
        ):
            classification = "marking"
            cv2.drawContours(marking_mask, [contour], -1, 255, -1)
        else:
            classification = "object"
            cv2.drawContours(object_mask, [contour], -1, 255, -1)
        components.append(
            {
                "bbox": [int(x), int(y), int(width), int(height)],
                "area_ratio": round(float(np.count_nonzero(inside)) / max(1, patch_pixels), 4),
                "vertical": round(support, 4),
                "aspect": round(aspect, 3),
                "class": classification,
            }
        )
    if floor is not None:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        delta = lab - floor["mean"]
        distance = np.sqrt(np.einsum("...j,jk,...k->...", delta, floor["inverse"], delta))
        color_outlier = (
            (distance > float(floor["threshold"]))
            & (patch_mask > 0)
            & (target_mask == 0)
            & (marking_mask == 0)
        ).astype(np.uint8) * 255
        kernel_size = max(3, int(params["dilate_kernel"]))
        color_outlier = cv2.morphologyEx(
            color_outlier, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8)
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(color_outlier, connectivity=8)
        filtered = np.zeros_like(color_outlier)
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) >= min_area:
                filtered[labels == index] = 255
        object_mask = cv2.bitwise_or(object_mask, filtered)
    object_mask = cv2.bitwise_and(object_mask, patch_mask)
    marking_mask = cv2.bitwise_and(marking_mask, patch_mask)
    target_mask = cv2.bitwise_or(target_mask, target_components)
    return {
        "object": object_mask,
        "marking": marking_mask,
        "target": cv2.bitwise_and(target_mask, patch_mask),
        "components": components,
    }


def _jaccard(first: np.ndarray, second: np.ndarray) -> float:
    a = first > 0
    b = second > 0
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b)) / union


def _rectify(image: np.ndarray, patch: Sequence[Sequence[float]], size: int):
    if len(patch) != 4:
        return None, None
    src = np.asarray(patch, dtype=np.float32)
    dst = np.array(
        [[0, size - 1], [size - 1, size - 1], [size - 1, 0], [0, 0]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        return None, None
    warped = cv2.warpPerspective(image, matrix, (size, size), flags=cv2.INTER_LINEAR)
    return warped, matrix


def _warp_mask(mask: np.ndarray, matrix: np.ndarray, size: int) -> np.ndarray:
    return cv2.warpPerspective(mask, matrix, (size, size), flags=cv2.INTER_NEAREST)


def _largest_free_rect_ratio(object_mask: np.ndarray, target_mask: np.ndarray, size: int) -> float:
    obstacle = cv2.bitwise_or(object_mask, target_mask) > 0
    free = (~obstacle).astype(np.uint8) * 255
    cell = max(1, size // 64)
    height, width = size // cell, size // cell
    if height < 2 or width < 2:
        return 0.0
    small = cv2.resize(free, (width, height), interpolation=cv2.INTER_AREA) > 220
    heights = np.zeros(width, dtype=np.int32)
    best = 0
    for row in range(height):
        heights = np.where(small[row], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for column in range(width + 1):
            current = int(heights[column]) if column < width else 0
            start = column
            while stack and stack[-1][1] >= current:
                index, value = stack.pop()
                best = max(best, value * (column - index))
                start = index
            stack.append((start, current))
    return float(best) / float(height * width) if height * width else 0.0


def _footpoints(
    image: np.ndarray,
    patch: Sequence[Sequence[float]],
    obstacle_mask: np.ndarray,
    target: dict[str, float],
    params: dict[str, Any],
):
    height, width = image.shape[:2]
    points = [np.asarray(point, dtype=np.float64) for point in patch]
    if len(points) != 4:
        return [], None, 0.0, None
    near = (points[0] + points[1]) / 2.0
    far = (points[3] + points[2]) / 2.0
    box_height = float(target["height_fraction"]) * height
    box_width = float(target["aspect"]) * box_height
    valid: list[tuple[float, float, float]] = []
    best: tuple[float, float, float] | None = None
    best_box = None
    rows = max(3, int(params["grid_rows"]))
    cols = max(3, int(params["grid_cols"]))
    patch_array = np.asarray(patch, dtype=np.int32).reshape(-1, 1, 2)
    for row_index in range(rows):
        fraction = 0.08 + 0.84 * row_index / max(1, rows - 1)
        left = points[0] * (1.0 - fraction) + points[3] * fraction
        right = points[1] * (1.0 - fraction) + points[2] * fraction
        row_y = near[1] * (1.0 - fraction) + far[1] * fraction
        for col_index in range(cols):
            column = 0.05 + 0.90 * col_index / max(1, cols - 1)
            x = left[0] * (1.0 - column) + right[0] * column
            y = row_y
            if cv2.pointPolygonTest(patch_array, (float(x), float(y)), False) < 0:
                continue
            x1 = max(0, int(round(x - box_width / 2.0)))
            x2 = min(width, int(round(x + box_width / 2.0)))
            y1 = max(0, int(round(y - box_height)))
            y2 = min(height, int(round(y)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            region = obstacle_mask[y1:y2, x1:x2]
            free = 1.0 - float(np.count_nonzero(region)) / float(region.size)
            foot_clear = obstacle_mask[min(max(int(round(y)) - 1, 0), height - 1),
                                       min(max(int(round(x)) - 1, 0), width - 1)] == 0
            if free >= float(params["target_body_free_min"]) and foot_clear:
                valid.append((float(x), float(y), free))
                if best is None or free > best[2]:
                    best = (float(x), float(y), free)
                    best_box = [x1, y1, x2, y2]
    ratio = len(valid) / float(rows * cols)
    best_obstacle = 1.0 - best[2] if best is not None else 1.0
    return valid, best, best_obstacle, best_box


def assess_placement(
    image_bgr: Any,
    patch: Sequence[Sequence[float]],
    *,
    target: Mapping[str, Any] | None = None,
    boxes: Iterable[Any] = (),
    params: Mapping[str, Any] | None = None,
) -> PlacementAssessment:
    """Measure clutter and target placement capacity for one patch."""

    image = _bgr_uint8(image_bgr)
    options = _params(params)
    target_spec = _target(target)
    height, width = image.shape[:2]
    patch_mask = _patch_mask((height, width), patch)
    patch_pixels = int(np.count_nonzero(patch_mask))
    if patch_pixels <= 0:
        return PlacementAssessment(
            metrics={"method": "placement_clutter_v1", "patch_area_ratio": 0.0},
            masks={},
        )
    target_mask = _boxes_mask((height, width), boxes)
    floor = _floor_model(image, patch_mask, options)
    base = _mask_components(image, patch_mask, target_mask, floor, options)
    object_mask = base["object"]
    marking_mask = base["marking"]
    target_mask = base["target"]
    components = base["components"]
    object_ratio = float(np.count_nonzero(object_mask)) / patch_pixels
    marking_ratio = float(np.count_nonzero(marking_mask)) / patch_pixels
    target_ratio = float(np.count_nonzero(target_mask)) / patch_pixels
    object_components = [item for item in components if item["class"] == "object"]
    max_component = max((item["area_ratio"] for item in object_components), default=0.0)
    vertical = _vertical_map(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), float(options["vertical_threshold"]))
    vertical_ratio = float(np.count_nonzero(vertical & (patch_mask > 0))) / patch_pixels
    legacy = _legacy_clearance(image, patch)
    size = int(options["rectified_size"])
    rectified, matrix = _rectify(image, patch, size)
    if matrix is not None and rectified is not None:
        rectified_object = _warp_mask(object_mask, matrix, size)
        rectified_marking = _warp_mask(marking_mask, matrix, size)
        rectified_target = _warp_mask(target_mask, matrix, size)
        lfr = _largest_free_rect_ratio(rectified_object, rectified_target, size)
        lfr_method = "rectified_patch"
    else:
        lfr = _largest_free_rect_ratio(object_mask, target_mask, max(height, width))
        lfr_method = "image_fallback"
        rectified_object = rectified_marking = rectified_target = None
    valid, best, best_obstacle, best_box = _footpoints(
        image, patch, cv2.bitwise_or(object_mask, target_mask), target_spec, options
    )
    valid_ratio = len(valid) / float(max(1, int(options["grid_rows"]) * int(options["grid_cols"])))
    stability = 1.0
    probes = int(options["stability_probes"])
    if probes > 0 and matrix is not None and rectified is not None:
        scores = []
        for probe in range(probes):
            factor = 0.8 if probe % 2 == 0 else 1.2
            kernel = int(options["close_kernel"]) + (-2 if probe % 2 == 0 else 2)
            probe_masks = _mask_components(
                rectified,
                np.full((size, size), 255, np.uint8),
                rectified_target,
                floor,
                options,
                canny_low=max(1, int(options["canny_low"] * factor)),
                canny_high=max(2, int(options["canny_high"] * factor)),
                close_kernel=kernel,
            )
            scores.append(_jaccard(rectified_object, probe_masks["object"]))
        stability = float(np.mean(scores)) if scores else 0.0
    floor_support = float(floor["support"]) if floor is not None else 0.0
    area_factor = min(1.0, (patch_pixels / float(height * width)) / 0.05)
    confidence = max(0.0, min(1.0, 0.5 * stability + 0.3 * floor_support + 0.2 * area_factor))
    metrics = {
        "method": "placement_clutter_v1",
        "params": dict(options),
        "patch_area_ratio": round(patch_pixels / float(height * width), 4),
        "free_ratio": legacy.get("free_ratio"),
        "obstacle_ratio": legacy.get("obstacle_ratio"),
        "components": legacy.get("components"),
        "object_ratio": round(object_ratio, 4),
        "object_component_count": len(object_components),
        "object_max_component_ratio": round(float(max_component), 4),
        "vertical_structure_ratio": round(vertical_ratio, 4),
        "marking_ratio": round(marking_ratio, 4),
        "task_target_ratio": round(target_ratio, 4),
        "largest_free_rect_ratio": round(float(lfr), 4),
        "largest_free_rect_method": lfr_method,
        "valid_footpoint_ratio": round(valid_ratio, 4),
        "best_footpoint": [round(best[0], 1), round(best[1], 1)] if best else None,
        "best_body_obstacle_ratio": round(float(best_obstacle), 4),
        "confidence": round(confidence, 4),
        "stability": round(stability, 4),
        "floor_support": round(floor_support, 4),
    }
    masks = {
        "object": object_mask,
        "marking": marking_mask,
        "target": target_mask,
        "rectified_object": rectified_object,
        "rectified_marking": rectified_marking,
        "rectified_target": rectified_target,
        "valid_footpoints": [(round(x, 1), round(y, 1)) for x, y, _ in valid],
        "best_footpoint": metrics["best_footpoint"],
        "best_body_box": best_box,
    }
    return PlacementAssessment(metrics=metrics, masks=masks)


def _near_floor_support(image: np.ndarray) -> np.ndarray | None:
    """Propose foreground floor from a distinct bottom-band appearance."""
    height = image.shape[0]
    if min(image.shape[:2]) < 32:
        return None
    # Use the image's near field, since fitting the whole review polygon can
    # learn the facade when it occupies a large part of that polygon.
    lab = cv2.cvtColor(cv2.medianBlur(image, 5), cv2.COLOR_BGR2LAB).astype(np.float32)
    seed_start = int(0.85 * height)
    seeds = lab[seed_start:]
    median = np.median(seeds, axis=(0, 1))
    mad = np.median(np.abs(seeds - median), axis=(0, 1))
    if mad[0] < 2:
        return None
    tolerance = np.clip(3 * mad, (12, 6, 6), (30, 12, 12))
    similar = np.all(np.abs(lab - median) <= tolerance, axis=2)
    seed_agreement = float(similar[seed_start:].mean())
    if seed_agreement < 0.4 or seed_agreement - float(similar[:height // 3].mean()) < 0.25:
        return None

    # Closing connects narrow gaps from paint/shadows for the connectivity
    # test; only pixels supported by the original appearance mask are returned.
    connected = cv2.morphologyEx(similar.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, labels = cv2.connectedComponents(connected, connectivity=8)
    bottom_labels, counts = np.unique(labels[seed_start:], return_counts=True)
    foreground_labels = bottom_labels[(bottom_labels != 0) & (counts >= 0.03 * seeds.shape[0] * seeds.shape[1])]
    return similar & np.isin(labels, foreground_labels)


def _local_floor_contacts(support: np.ndarray, patch: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    """Find a near-field contact island with an unobstructed route to the foreground."""
    height, width = support.shape
    # Unlike 2-D connectivity, a column cannot turn around the side of a
    # building and then continue upwards into a similar-looking facade.
    appearance = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE,
                                  np.ones((5, 1), np.uint8)) > 0
    route = np.logical_and.accumulate((appearance & ~obstacle)[::-1], axis=0)[::-1]
    near = np.arange(height)[:, None] >= 0.70 * height
    valid = support & patch & route & near
    margin = max(2, round(0.00625 * width))
    diameter = 2 * margin + 1
    valid = cv2.erode(valid.astype(np.uint8), np.ones((diameter, diameter), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(valid, connectivity=8)
    retained = ((stats[:, cv2.CC_STAT_WIDTH] >= max(diameter, round(0.03 * width)))
                & (stats[:, cv2.CC_STAT_HEIGHT] >= diameter)
                & (stats[:, cv2.CC_STAT_AREA] >= diameter ** 2))
    retained[0] = False
    return retained[labels]


def assess_placeable_ground(
    image_bgr: Any,
    polygon: Sequence[Sequence[float]],
    *,
    boxes: Iterable[Any] = (),
    floor_mask: np.ndarray | None = None,
    obstacle_mask: np.ndarray | None = None,
) -> GroundPlacement:
    """Screen the original, unpadded image for possible contact within a review polygon.

    Provide a per-frame floor mask to supply independent positive evidence, or
    use the automatic near-field appearance cue. The latter is a heuristic for
    continuous-looking floor and can confuse surfaces of similar appearance.
    Existing clutter, target masks, and supplied obstacle evidence veto contact. Every unblocked pixel
    without positive support stays unknown. The result assesses sprite contact
    geometry only; a separate emptiness observation covers the entire image.
    """
    image = _bgr_uint8(image_bgr)
    shape = image.shape[:2]
    explicit_obstacle = _evidence_mask(obstacle_mask, shape, "obstacle_mask")
    points = np.asarray(polygon, dtype=float)
    if not points.size:
        return GroundPlacement("unknown", "ground_polygon_missing", np.zeros(shape, bool),
                               (_boxes_mask(shape, boxes) > 0) | explicit_obstacle)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all() or np.any(points < 0)
            or np.any(points > (shape[1] - 1, shape[0] - 1))
            or cv2.contourArea(points.astype(np.float32)) <= 0):
        raise ValueError("polygon must be a nondegenerate region in the input image")
    patch = _patch_mask(shape, points) > 0
    assessment = assess_placement(image, points, boxes=boxes)
    obstacle = (assessment.masks["object"] > 0) | (assessment.masks["target"] > 0) | explicit_obstacle
    if floor_mask is None:
        decision, _ = placement_decision(assessment.metrics)
        floor = _near_floor_support(image)
        if decision != "usable":
            if floor is not None:
                contacts = _local_floor_contacts(floor, patch, obstacle)
                if contacts.any():
                    return GroundPlacement("candidate", "local_floor_contact_support", contacts, obstacle)
            return GroundPlacement("unknown", f"placement_{decision}", np.zeros(shape, bool), obstacle)
        if floor is None:
            return GroundPlacement("unknown", "near_floor_appearance_unresolved", np.zeros(shape, bool), obstacle)
        basis = "near_floor_appearance"
    else:
        floor = _evidence_mask(floor_mask, shape, "floor_mask")
        basis = "supplied_floor_mask"

    supported = floor & patch & ~obstacle
    # Leave a pixel of clearance at the edge of evidence or a detected obstacle.
    supported = cv2.erode(supported.astype(np.uint8), np.ones((3, 3), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return GroundPlacement("candidate" if supported.any() else "unknown", basis, supported, obstacle)


def _evidence_mask(mask: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    array = np.asarray(mask)
    if (array.shape != shape or not (array.dtype == np.bool_ or np.issubdtype(array.dtype, np.integer))
            or not np.isin(array, (0, 1, 255)).all()):
        raise ValueError(f"{name} must be an image-sized binary mask")
    return array > 0


def placement_decision(
    metrics: Mapping[str, Any] | None,
    thresholds: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return ``usable``, ``cluttered`` or ``unsure`` plus decision evidence."""

    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        limits.update(thresholds)
    values = dict(metrics or {})
    missing = []
    numeric: dict[str, float] = {}
    for key in REQUIRED_METRICS:
        if key not in values:
            missing.append(key)
            continue
        try:
            numeric[key] = float(values[key])
        except (TypeError, ValueError):
            missing.append(key)
    values.update(numeric)
    evidence: dict[str, Any] = {
        "judge": "placement",
        "metrics": {key: values.get(key) for key in REQUIRED_METRICS if key in values},
        "thresholds": limits,
        "missing": missing,
    }
    if missing:
        evidence["judge"] = "placement_metrics_missing"
        return "unsure", evidence
    confidence = float(values["confidence"])
    evidence["confidence"] = confidence
    if confidence < float(limits["confidence_min"]):
        evidence["judge"] = "placement_confidence_low"
        return "unsure", evidence
    failures = []
    checks = (
        ("object_ratio", "max_object_ratio", "object_ratio_exceeded"),
        ("object_max_component_ratio", "max_object_component_ratio", "object_component_exceeded"),
        ("vertical_structure_ratio", "max_vertical_ratio", "vertical_structure_exceeded"),
    )
    for value_key, limit_key, reason in checks:
        if float(values[value_key]) > float(limits[limit_key]):
            failures.append({"check": reason, "value": values[value_key], "limit": limits[limit_key]})
    if float(values["largest_free_rect_ratio"]) < float(limits["min_largest_free_rect_ratio"]):
        failures.append(
            {
                "check": "largest_free_rect_too_small",
                "value": values["largest_free_rect_ratio"],
                "limit": limits["min_largest_free_rect_ratio"],
            }
        )
    if float(values["valid_footpoint_ratio"]) < float(limits["min_valid_footpoint_ratio"]):
        failures.append(
            {
                "check": "valid_footpoint_ratio_too_small",
                "value": values["valid_footpoint_ratio"],
                "limit": limits["min_valid_footpoint_ratio"],
            }
        )
    if float(values["best_body_obstacle_ratio"]) > float(limits["max_body_obstacle_ratio"]):
        failures.append(
            {
                "check": "best_body_obstacle_ratio_exceeded",
                "value": values["best_body_obstacle_ratio"],
                "limit": limits["max_body_obstacle_ratio"],
            }
        )
    if not failures:
        margin = float(limits.get("borderline_margin", 0.0) or 0.0)
        borderline = []
        if margin > 0.0:
            for value_key, limit_key in (
                ("object_ratio", "max_object_ratio"),
                ("object_max_component_ratio", "max_object_component_ratio"),
                ("vertical_structure_ratio", "max_vertical_ratio"),
                ("best_body_obstacle_ratio", "max_body_obstacle_ratio"),
            ):
                limit = float(limits[limit_key])
                if limit > 0.0 and float(values[value_key]) > limit * (1.0 - margin):
                    borderline.append(
                        {
                            "check": f"{value_key}_near_limit",
                            "value": values[value_key],
                            "limit": limit,
                        }
                    )
            for value_key, limit_key in (
                ("largest_free_rect_ratio", "min_largest_free_rect_ratio"),
                ("valid_footpoint_ratio", "min_valid_footpoint_ratio"),
            ):
                limit = float(limits[limit_key])
                if limit > 0.0 and float(values[value_key]) < limit * (1.0 + margin):
                    borderline.append(
                        {
                            "check": f"{value_key}_near_limit",
                            "value": values[value_key],
                            "limit": limit,
                        }
                    )
        if borderline:
            evidence["borderline"] = borderline
            evidence["judge"] = "placement_patch_borderline"
            return "unsure", evidence
        evidence["judge"] = "placement_usable"
        return "usable", evidence
    evidence["failures"] = failures
    if float(values["object_ratio"]) >= float(limits["clutter_exclude_ratio"]):
        evidence["judge"] = "placement_patch_cluttered"
        return "cluttered", evidence
    evidence["judge"] = "placement_patch_unsure"
    return "unsure", evidence
