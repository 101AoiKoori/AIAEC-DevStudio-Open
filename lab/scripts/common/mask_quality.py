"""Mask geometry and layer-quality checks with visual evidence.

The shared ``review.masks`` module already owns alpha silhouette metrics. This
module composes those checks for an authoring instance: full versus visible
region, optional occlusion layers, ground contact, and an overlay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from datasets.scripts.common.review.masks import alpha_quality

from .contracts import SCHEMA_VERSION, now_local, resolve_workspace, workspace_relative

DEFAULT_PARAMS: dict[str, Any] = {
    "hole_max": 0.02,
    "border_max": 0.15,
    "components_max": 3,
    "border_margin_px": 2,
    "subset_tolerance": 0.005,
    "hidden_unexplained_max": 0.02,
    "minimum_visible_fraction": 0.01,
}


def _binary_mask(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[:, :, 3]
        elif array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        elif array.shape[2] == 1:
            array = array[:, :, 0]
        else:
            raise ValueError(f"{name} must be H×W, H×W×1, H×W×3 or H×W×4")
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty 2-D mask")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array != 0


def _union(masks: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for mask in masks:
        array = _binary_mask(mask, name="occlusion mask")
        if array.shape != shape:
            raise ValueError("occlusion mask shape differs from full mask")
        result |= array
    return result


def _bbox(mask: np.ndarray) -> list[int] | None:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]


def _contact_inside(contact: Sequence[float], polygon: Sequence[Sequence[float]] | None) -> bool | None:
    if polygon is None:
        return None
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("ground_polygon must contain at least three x/y points")
    return cv2.pointPolygonTest(points.reshape(-1, 1, 2), (float(contact[0]), float(contact[1])), False) >= 0


def audit_mask_arrays(
    full_mask: Any,
    visible_mask: Any,
    *,
    occlusion_masks: Sequence[Any] = (),
    ground_polygon: Sequence[Sequence[float]] | None = None,
    contact_point: Sequence[float] | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit one full/visible/occlusion mask set and return JSON-safe evidence."""

    options = dict(DEFAULT_PARAMS)
    if params:
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError("unknown mask parameters: " + ", ".join(sorted(unknown)))
        options.update(params)
    full = _binary_mask(full_mask, name="full_mask")
    visible = _binary_mask(visible_mask, name="visible_mask")
    if full.shape != visible.shape:
        raise ValueError("full and visible masks must have the same shape")
    full_pixels = int(np.count_nonzero(full))
    visible_pixels = int(np.count_nonzero(visible))
    reasons: list[str] = []
    if full_pixels == 0:
        status = "reject"
        reasons.append("empty_full_mask")
    elif visible_pixels == 0:
        status = "reject"
        reasons.append("empty_visible_mask")
    else:
        status = "pass"

    outside = visible & ~full
    subset_error = float(np.count_nonzero(outside)) / float(max(1, full_pixels))
    if subset_error > float(options["subset_tolerance"]):
        reasons.append("visible_not_subset")
        status = "reject"

    full_quality = alpha_quality(full, params=options)
    visible_quality = alpha_quality(visible, params=options)
    reasons.extend("full_" + code for code in full_quality["reason_codes"])
    reasons.extend("visible_" + code for code in visible_quality["reason_codes"])
    if full_quality["reason_codes"] or visible_quality["reason_codes"]:
        status = "review" if status != "reject" else status

    hidden = full & ~visible
    hidden_pixels = int(np.count_nonzero(hidden))
    occlusion_union = _union(occlusion_masks, full.shape)
    hidden_uncovered = int(np.count_nonzero(hidden & ~occlusion_union))
    hidden_unexplained = (
        float(hidden_uncovered) / float(max(1, hidden_pixels)) if hidden_pixels else 0.0
    )
    if hidden_pixels and not occlusion_masks:
        reasons.append("occlusion_layers_missing")
        status = "review" if status != "reject" else status
    elif hidden_unexplained > float(options["hidden_unexplained_max"]):
        reasons.append("hidden_region_unexplained")
        status = "reject"

    visible_fraction = float(visible_pixels) / float(max(1, full_pixels))
    if visible_fraction < float(options["minimum_visible_fraction"]):
        reasons.append("visible_fraction_too_small")
        status = "reject"

    visible_box = _bbox(visible)
    if contact_point is None and visible_box:
        contact_point = [
            (visible_box[0] + visible_box[2]) / 2.0,
            float(visible_box[3] - 1),
        ]
        contact_basis = "visible_bbox_bottom_center_default"
    else:
        contact_basis = "declared"
    contact_inside = None
    if contact_point is None:
        reasons.append("ground_contact_unavailable")
        status = "review" if status != "reject" else status
    else:
        contact_inside = _contact_inside(contact_point, ground_polygon)
        if contact_inside is False:
            reasons.append("ground_contact_outside")
            status = "reject"
        elif contact_inside is None:
            reasons.append("ground_polygon_unavailable")
            status = "review" if status != "reject" else status

    return {
        "status": status,
        "reason_codes": sorted(set(reasons)),
        "masks": {
            "shape": list(full.shape),
            "full_pixels": full_pixels,
            "visible_pixels": visible_pixels,
            "visible_fraction": round(visible_fraction, 6),
            "subset_outside_pixels": int(np.count_nonzero(outside)),
            "subset_error": round(subset_error, 6),
            "hidden_pixels": hidden_pixels,
            "hidden_uncovered_pixels": hidden_uncovered,
            "hidden_unexplained_fraction": round(hidden_unexplained, 6),
            "occlusion_layer_count": len(occlusion_masks),
            "visible_bbox_xyxy": visible_box,
        },
        "alpha_quality": {
            "full": full_quality,
            "visible": visible_quality,
        },
        "ground": {
            "contact_point": list(contact_point) if contact_point is not None else None,
            "contact_basis": contact_basis,
            "inside": contact_inside,
        },
    }


def load_mask(path: str | Path, *, root: str | Path | None = None) -> np.ndarray:
    """Read a PNG/JPEG mask with alpha as the mask when present."""

    source = resolve_workspace(path, root=root, must_exist=True)
    array = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError(f"cannot read mask: {source}")
    return _binary_mask(array, name=str(source)).astype(np.uint8) * 255


def load_image(path: str | Path, *, root: str | Path | None = None) -> np.ndarray:
    """Read a BGR image for overlay rendering."""

    source = resolve_workspace(path, root=root, must_exist=True)
    array = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError(f"cannot read image: {source}")
    return array


def draw_evidence(
    image_bgr: np.ndarray,
    full_mask: np.ndarray,
    visible_mask: np.ndarray,
    *,
    occlusion_masks: Sequence[np.ndarray] = (),
    ground_polygon: Sequence[Sequence[float]] | None = None,
    contact_point: Sequence[float] | None = None,
) -> np.ndarray:
    """Draw full, visible, occlusion and ground-contact layers on a copy."""

    canvas = image_bgr.copy()
    if canvas.shape[:2] != full_mask.shape:
        raise ValueError("image and masks must have the same height and width")
    layers = [
        (full_mask, (255, 255, 255), 1),
        (visible_mask, (0, 255, 0), 2),
    ]
    for index, mask in enumerate(occlusion_masks):
        layers.append((mask, (0, 191, 255) if index == 0 else (0, 128, 255), 1))
    for mask, color, thickness in layers:
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, color, thickness)
    if ground_polygon:
        points = np.asarray(ground_polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, (255, 0, 255), 1)
    if contact_point is not None:
        cv2.circle(
            canvas,
            (int(round(contact_point[0])), int(round(contact_point[1]))),
            4,
            (0, 0, 255),
            -1,
        )
    return canvas


def audit_manifest(
    manifest: Mapping[str, Any],
    *,
    root: str | Path | None = None,
    overlay_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit a manifest of mask instances and optionally write overlays."""

    task = str(manifest.get("task") or "common")
    batch = str(manifest.get("batch") or "mask_review")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("mask manifest requires a nonempty records list")
    output_dir = resolve_workspace(overlay_dir, root=root) if overlay_dir else None
    results: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"mask record {index} is not an object")
        record_id = str(record.get("id") or "")
        if not record_id:
            raise ValueError(f"mask record {index} has no id")
        full_path = str(record.get("full_mask") or "")
        visible_path = str(record.get("visible_mask") or "")
        if not full_path or not visible_path:
            raise ValueError(f"{record_id}: full_mask and visible_mask are required")
        full = load_mask(full_path, root=root)
        visible = load_mask(visible_path, root=root)
        occlusions = [
            load_mask(path, root=root) for path in record.get("occlusion_masks") or []
        ]
        audit = audit_mask_arrays(
            full,
            visible,
            occlusion_masks=occlusions,
            ground_polygon=record.get("ground_polygon"),
            contact_point=record.get("contact_point"),
            params=record.get("params"),
        )
        evidence: dict[str, Any] = {
            "full_mask": workspace_relative(full_path, root=root),
            "visible_mask": workspace_relative(visible_path, root=root),
            "occlusion_masks": [workspace_relative(path, root=root) for path in record.get("occlusion_masks") or []],
        }
        if output_dir is not None:
            image_path = record.get("image")
            if not image_path:
                raise ValueError(f"{record_id}: image is required when writing overlays")
            image = load_image(image_path, root=root)
            overlay = draw_evidence(
                image,
                full,
                visible,
                occlusion_masks=occlusions,
                ground_polygon=record.get("ground_polygon"),
                contact_point=audit["ground"]["contact_point"],
            )
            target = output_dir / f"{record_id}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), overlay):
                raise RuntimeError(f"cannot write overlay: {target}")
            evidence["overlay"] = workspace_relative(target, root=root)
        results.append({"id": record_id, **audit, "evidence": evidence})
    counts = {state: sum(item["status"] == state for item in results) for state in ("pass", "review", "reject")}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_local(),
        "kind": "mask_quality_report",
        "task": task,
        "batch": batch,
        "records": results,
        "summary": {"total": len(results), **counts},
        "boundary": (
            "Geometry and layer coverage are measured. Semantic occlusion correctness, "
            "mask ownership and human acceptance remain review fields."
        ),
    }
