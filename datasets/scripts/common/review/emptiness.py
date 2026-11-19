"""Evidence that a frame holds no unlabelled target.

A bounding box proves that a target exists and where it is; it says nothing
about the rest of the frame.  A frame may only be used as background or as a
negative sample when an independent check shows that nothing else was left
unlabelled.  This module produces that evidence:

* the annotation is turned into a coverage mask (``mask_boxes``) so that
  "explained" pixels can be separated from unexplained ones;
* each caller-supplied template is matched at several scales, restricted to the
  unexplained part of the frame (``multi_scale_match``);
* the raw candidates of every template and scale are merged with
  non-maximum suppression;
* surviving candidates are classified against the task's target spec: a standee
  below the minimum target scale is not a target, a back-side (white-back)
  standee is a hard negative, and only the rest count as suspects.

Only the suspects decide the verdict.  A frame showing distant small standees
or back-side boards can therefore still be clean background, which is why
``EmptinessReport`` records ``below_scale``, ``hard_negatives`` and the
``target_spec`` used for the decision.

``scan_unlabelled_targets`` never reports ``clean`` without template evidence:
an empty ``templates_bgr`` yields ``clean=False`` with the reason code
``emptiness_unproven``, because absence of measurement is not absence of
targets.  ``variance_map`` is a local-statistics helper for later extensions and
does not take part in the verdict.

All coordinates are pixel coordinates in the input image, with ``x2``/``y2``
treated as exclusive bounds, matching the ``cv2.matchTemplate`` convention.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import cv2
import numpy as np

from .schema import Box, EmptinessReport

__all__ = [
    "DEFAULT_PARAMS",
    "DEFAULT_SPEC",
    "MATCH_SCORE_FLOOR",
    "MIN_BOX_PX",
    "REASON_BELOW_SCALE",
    "REASON_CLEAN",
    "REASON_HARD_NEGATIVE",
    "REASON_SUSPECT",
    "REASON_UNPROVEN",
    "SIDE_BACK",
    "SIDE_FRONTAL",
    "SIDE_UNKNOWN",
    "SPEC_BACK_ACTIONS",
    "SPEC_BELOW_ACTIONS",
    "check_emptiness",
    "classify_standee_side",
    "covered_area_fraction",
    "mask_boxes",
    "multi_scale_match",
    "nms",
    "scan_unlabelled_targets",
    "variance_map",
]

REASON_CLEAN = "no_unlabelled_target"
REASON_SUSPECT = "unlabelled_target_suspected"
REASON_UNPROVEN = "emptiness_unproven"
REASON_BELOW_SCALE = "below_target_scale"
REASON_HARD_NEGATIVE = "hard_negative_present"

SIDE_FRONTAL = "frontal"
SIDE_BACK = "back"
SIDE_UNKNOWN = "unknown"

#: ``multi_scale_match`` reports every candidate above this raw score.
MATCH_SCORE_FLOOR = 0.5
#: Candidates smaller than this on either side are dropped: they carry no shape
#: evidence and would only add noise to the candidate list.
MIN_BOX_PX = 6

#: Target-spec policies.  ``suspect`` keeps the conservative round-one behaviour
#: so that callers without a spec classify nothing away.
SPEC_BACK_ACTIONS = ("hard_negative", "suspect")
SPEC_BELOW_ACTIONS = ("not_a_target", "suspect")

DEFAULT_SPEC: dict[str, Any] = {
    "min_height_fraction": 0.0,
    "back_side": "suspect",
    "below_min_scale": "suspect",
}

DEFAULT_PARAMS: dict[str, Any] = {
    "mask_margin_px": 8,
    "suspect_min_score": 0.62,
    "nms_iou": 0.3,
    "scales": (0.6, 0.8, 1.0, 1.25),
    "drop_explained_candidates": True,
    # Heuristic side classification; see ``classify_standee_side``.
    "side_min_patch_px": 8,
    "side_canny_low": 50,
    "side_canny_high": 150,
    "side_saturation_frontal_min": 0.12,
    "side_saturation_texture_min": 0.05,
    "side_saturation_back_max": 0.08,
    "side_edge_frontal_min": 0.08,
    "side_edge_back_max": 0.12,
    "side_face_band_fraction": 0.4,
    "side_face_texture_min": 0.05,
    "side_spread_ratio_max": 0.25,
    "side_spread_density_min": 0.05,
}


def _params(params: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Merge caller parameters over the defaults and validate the used keys."""

    merged = dict(DEFAULT_PARAMS)
    if params:
        merged.update(params)
    margin = int(merged["mask_margin_px"])
    if margin < 0:
        raise ValueError("mask_margin_px must be >= 0")
    suspect_min = float(merged["suspect_min_score"])
    if not np.isfinite(suspect_min):
        raise ValueError("suspect_min_score must be finite")
    iou_thr = float(merged["nms_iou"])
    if not np.isfinite(iou_thr) or not 0.0 <= iou_thr <= 1.0:
        raise ValueError("nms_iou must be in [0, 1]")
    scales = tuple(float(scale) for scale in merged["scales"])
    if not scales:
        raise ValueError("scales must not be empty")
    for scale in scales:
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be positive and finite")
    merged["mask_margin_px"] = margin
    merged["suspect_min_score"] = suspect_min
    merged["nms_iou"] = iou_thr
    merged["scales"] = scales
    merged["drop_explained_candidates"] = bool(merged["drop_explained_candidates"])

    min_patch = int(merged["side_min_patch_px"])
    if min_patch < 2:
        raise ValueError("side_min_patch_px must be >= 2")
    merged["side_min_patch_px"] = min_patch
    for key in ("side_canny_low", "side_canny_high"):
        value = int(merged[key])
        if value < 0:
            raise ValueError("%s must be >= 0" % key)
        merged[key] = value
    if merged["side_canny_low"] >= merged["side_canny_high"]:
        raise ValueError("side_canny_low must be below side_canny_high")
    for key in (
        "side_saturation_frontal_min",
        "side_saturation_texture_min",
        "side_saturation_back_max",
        "side_edge_frontal_min",
        "side_edge_back_max",
        "side_face_band_fraction",
        "side_face_texture_min",
        "side_spread_ratio_max",
        "side_spread_density_min",
    ):
        value = float(merged[key])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("%s must be in [0, 1]" % key)
        merged[key] = value
    return merged


def _spec(spec: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Merge the caller's target spec over the documented defaults."""

    merged = dict(DEFAULT_SPEC)
    if spec:
        merged.update(spec)
    fraction = float(merged["min_height_fraction"])
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("min_height_fraction must be in [0, 1]")
    back_side = str(merged["back_side"])
    if back_side not in SPEC_BACK_ACTIONS:
        raise ValueError("back_side must be one of %r" % (SPEC_BACK_ACTIONS,))
    below_min_scale = str(merged["below_min_scale"])
    if below_min_scale not in SPEC_BELOW_ACTIONS:
        raise ValueError("below_min_scale must be one of %r" % (SPEC_BELOW_ACTIONS,))
    return {
        "min_height_fraction": fraction,
        "back_side": back_side,
        "below_min_scale": below_min_scale,
    }


def _resolve_spec(
    spec: Optional[dict[str, Any]],
    params: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve the target spec, accepting spec keys embedded in ``params``.

    The review CLI forwards only ``boxes``, ``templates_bgr`` and ``params`` to
    ``scan_unlabelled_targets``.  A caller therefore may place the target spec
    inside ``params`` as an interim channel; an explicit ``spec`` always wins.
    """

    if spec is not None:
        return _spec(spec)
    if params:
        embedded = {key: params[key] for key in DEFAULT_SPEC if key in params}
        if embedded:
            return _spec(embedded)
    return _spec(None)


def _to_gray(image: Any, *, name: str) -> np.ndarray:
    """Return a 2-D uint8/float32 view of ``image`` without reordering data."""

    array = np.asarray(image)
    if array.dtype not in (np.uint8, np.float32):
        if array.dtype == np.bool_:
            array = array.astype(np.uint8) * 255
        elif np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32)
        else:
            array = array.astype(np.uint8)
    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
        elif array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        elif array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("%s must be H×W, H×W×1, H×W×3 or H×W×4" % name)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("%s must be a non-empty 2-D image" % name)
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError("%s must be finite" % name)
    return array


def _to_bgr_uint8(image: Any, *, name: str) -> np.ndarray:
    """Return a 3-channel uint8 BGR view; 2-D input is treated as grey."""

    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("%s must not be empty" % name)
    if array.dtype != np.uint8:
        if array.dtype == np.bool_:
            array = array.astype(np.uint8) * 255
        elif np.issubdtype(array.dtype, np.floating):
            data = array.astype(np.float32)
            if not np.isfinite(data).all():
                raise ValueError("%s must be finite" % name)
            if float(data.max()) <= 1.0 and float(data.min()) >= 0.0:
                data = data * 255.0
            array = np.clip(data, 0.0, 255.0).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim != 3:
        raise ValueError("%s must be H×W, H×W×1, H×W×3 or H×W×4" % name)
    if array.shape[2] == 1:
        return cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2BGR)
    if array.shape[2] == 3:
        return array
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    raise ValueError("%s must be H×W, H×W×1, H×W×3 or H×W×4" % name)


def _ordered_box(box: Sequence[int]) -> tuple[int, int, int, int]:
    """Return ``xyxy`` with ``x1 <= x2`` and ``y1 <= y2`` as integers."""

    values = [int(value) for value in box]
    if len(values) != 4:
        raise ValueError("box must hold four coordinates, got %r" % (box,))
    x1, y1, x2, y2 = values
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _box_area(box: Sequence[int]) -> int:
    x1, y1, x2, y2 = _ordered_box(box)
    return max(x2 - x1, 0) * max(y2 - y1, 0)


def _iou(first: Sequence[int], second: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = _ordered_box(first)
    bx1, by1, bx2, by2 = _ordered_box(second)
    width = min(ax2, bx2) - max(ax1, bx1)
    height = min(ay2, by2) - max(ay1, by1)
    if width <= 0 or height <= 0:
        return 0.0
    intersection = float(width * height)
    union = float(_box_area(first) + _box_area(second)) - intersection
    return intersection / union if union > 0.0 else 0.0


def nms(
    boxes: Sequence[Sequence[int]],
    scores: Sequence[float],
    iou_thr: float = 0.3,
) -> list[int]:
    """Greedy non-maximum suppression; returns surviving indices, best first."""

    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")
    threshold = float(iou_thr)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_thr must be in [0, 1]")
    kept: list[int] = []
    if not boxes:
        return kept
    for score in scores:
        if not np.isfinite(float(score)):
            raise ValueError("scores must be finite")
    ordered = sorted(range(len(boxes)), key=lambda index: (-float(scores[index]), index))
    for index in ordered:
        if all(_iou(boxes[index], boxes[other]) <= threshold for other in kept):
            kept.append(index)
    return kept


def mask_boxes(
    shape: tuple[int, int],
    boxes: Sequence[Sequence[int]],
    margin_px: int = 8,
) -> np.ndarray:
    """Coverage mask (255 = explained by a box) for ``shape == (height, width)``."""

    if len(tuple(shape)) != 2:
        raise ValueError("shape must be (height, width)")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("shape must be positive")
    margin = int(margin_px)
    if margin < 0:
        raise ValueError("margin_px must be >= 0")
    mask = np.zeros((height, width), np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = _ordered_box(box)
        left = max(x1 - margin, 0)
        top = max(y1 - margin, 0)
        right = min(x2 + margin, width)
        bottom = min(y2 + margin, height)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = 255
    return mask


def covered_area_fraction(mask: np.ndarray) -> float:
    """Fraction of the frame that is non-zero in ``mask``."""

    array = np.asarray(mask)
    if array.size == 0:
        return 0.0
    return float(np.count_nonzero(array)) / float(array.size)


def multi_scale_match(
    image_gray: np.ndarray,
    template_gray: np.ndarray,
    *,
    scales: Sequence[float] = (0.6, 0.8, 1.0, 1.25),
    search_mask: Optional[np.ndarray] = None,
) -> tuple[list[list[int]], list[float]]:
    """Match one template at several scales and return candidates above 0.5.

    ``search_mask`` marks the pixels a candidate may start on (non-zero =
    allowed).  The mask is applied to the ``cv2.matchTemplate`` result at the
    candidate's top-left corner; no masked correlation is computed.  Returned
    boxes are clipped to the image and dropped when smaller than
    ``MIN_BOX_PX``; both lists are ordered by descending score.
    """

    image = _to_gray(image_gray, name="image_gray")
    template = _to_gray(template_gray, name="template_gray")
    scale_values = [float(scale) for scale in scales]
    if not scale_values:
        raise ValueError("scales must not be empty")
    for scale in scale_values:
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be positive and finite")

    allowed: Optional[np.ndarray] = None
    if search_mask is not None:
        array = np.asarray(search_mask)
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        if array.ndim != 2 or array.shape[:2] != image.shape[:2]:
            raise ValueError("search_mask must be H×W and match image_gray")
        allowed = array != 0

    if image.dtype != template.dtype:
        image = image.astype(np.float32)
        template = template.astype(np.float32)

    height, width = image.shape[:2]
    boxes: list[list[int]] = []
    scores: list[float] = []
    for scale in scale_values:
        template_h = int(round(template.shape[0] * scale))
        template_w = int(round(template.shape[1] * scale))
        if template_h < MIN_BOX_PX or template_w < MIN_BOX_PX:
            continue
        if template_h > height or template_w > width:
            continue
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(template, (template_w, template_h), interpolation=interpolation)
        result = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        if allowed is not None:
            window = allowed[: result.shape[0], : result.shape[1]]
            result = np.where(window, result, 0.0)
        rows, columns = np.nonzero(result > MATCH_SCORE_FLOOR)
        for row, column in zip(rows.tolist(), columns.tolist()):
            box = _clip_box(
                [column, row, column + template_w, row + template_h],
                height,
                width,
            )
            if box is None:
                continue
            boxes.append(box)
            scores.append(float(result[row, column]))

    order = sorted(range(len(scores)), key=lambda index: (-scores[index], boxes[index]))
    return [boxes[index] for index in order], [scores[index] for index in order]


def _clip_box(
    box: Sequence[int],
    height: int,
    width: int,
) -> Optional[list[int]]:
    """Clip a box to the image; ``None`` when it becomes too small to be useful."""

    x1, y1, x2, y2 = _ordered_box(box)
    left = min(max(x1, 0), width)
    top = min(max(y1, 0), height)
    right = min(max(x2, 0), width)
    bottom = min(max(y2, 0), height)
    if right - left < MIN_BOX_PX or bottom - top < MIN_BOX_PX:
        return None
    return [left, top, right, bottom]


def _centre_is_explained(mask: np.ndarray, box: Sequence[int]) -> bool:
    """True when a candidate's centre already falls inside the annotation mask."""

    x1, y1, x2, y2 = _ordered_box(box)
    centre_x = min(max((x1 + x2) // 2, 0), mask.shape[1] - 1)
    centre_y = min(max((y1 + y2) // 2, 0), mask.shape[0] - 1)
    return bool(mask[centre_y, centre_x])


def classify_standee_side(
    patch_bgr: np.ndarray,
    *,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Guess whether one standee patch shows its printed front or its back.

    HEURISTIC, NOT A CLASSIFIER PROOF.  A standee's front carries a saturated
    printed figure with texture in the upper (face) band, while the back is a
    plain white/grey board with almost no colour and few edges.  The measurement
    therefore uses three observable quantities: mean HSV saturation, Canny edge
    density over the whole patch, and Canny edge density inside the upper
    ``side_face_band_fraction`` of the patch.  A patch whose quadrants disagree
    strongly in texture (half flat, half busy) is contaminated and returns
    ``unknown`` rather than a guess.  Texture alone never promotes a colourless
    patch to frontal: the texture route also needs a minimum saturation, so a
    plain board with a printed outline still counts as a back side.  A 2-D
    (grey) input carries no saturation at all, so it is never declared ``back``;
    it stays ``unknown`` because the decisive evidence is missing.

    The result can only be used to *propose* a class; thresholds live in
    ``DEFAULT_PARAMS`` and must be calibrated per task.  ``unknown`` means the
    evidence is insufficient: callers must keep it as a suspect instead of
    treating it as clean.

    Returns ``{"side", "saturation_mean", "edge_density", "face_texture",
    "reason"}`` with fractions in ``[0, 1]``.
    """

    merged = _params(params)
    incoming = np.asarray(patch_bgr)
    has_colour = incoming.ndim == 3 and incoming.shape[2] in (3, 4)
    patch = _to_bgr_uint8(patch_bgr, name="patch_bgr")
    height, width = patch.shape[:2]
    result: dict[str, Any] = {
        "side": SIDE_UNKNOWN,
        "saturation_mean": 0.0,
        "edge_density": 0.0,
        "face_texture": 0.0,
        "reason": "patch_too_small",
    }
    if min(height, width) < merged["side_min_patch_px"]:
        return result

    saturation_mean = float(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) / 255.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, merged["side_canny_low"], merged["side_canny_high"]) > 0
    edge_density = float(edges.mean())
    band_height = max(int(round(height * merged["side_face_band_fraction"])), 1)
    face_texture = float(edges[:band_height, :].mean())

    mid_height, mid_width = height // 2, width // 2
    quadrant_density = []
    for rows, columns in (
        (slice(0, mid_height), slice(0, mid_width)),
        (slice(0, mid_height), slice(mid_width, width)),
        (slice(mid_height, height), slice(0, mid_width)),
        (slice(mid_height, height), slice(mid_width, width)),
    ):
        cell = edges[rows, columns]
        quadrant_density.append(float(cell.mean()) if cell.size else 0.0)
    peak_density = max(quadrant_density, default=0.0)
    trough_density = min(quadrant_density, default=0.0)
    spread_ratio = trough_density / peak_density if peak_density > 0.0 else 1.0

    result.update(
        saturation_mean=saturation_mean,
        edge_density=edge_density,
        face_texture=face_texture,
    )
    if peak_density >= merged["side_spread_density_min"] and spread_ratio <= merged["side_spread_ratio_max"]:
        result["reason"] = "mixed_texture"
        return result
    if saturation_mean >= merged["side_saturation_frontal_min"]:
        result["side"] = SIDE_FRONTAL
        result["reason"] = "saturated_print"
        return result
    if (
        saturation_mean >= merged["side_saturation_texture_min"]
        and face_texture >= merged["side_face_texture_min"]
        and edge_density >= merged["side_edge_frontal_min"]
    ):
        result["side"] = SIDE_FRONTAL
        result["reason"] = "textured_face_band"
        return result
    if saturation_mean <= merged["side_saturation_back_max"] and edge_density <= merged["side_edge_back_max"]:
        if not has_colour:
            # Saturation cannot be measured on a 2-D patch, so a colourless
            # input must not be demoted to a hard negative.
            result["reason"] = "colour_unavailable"
            return result
        result["side"] = SIDE_BACK
        result["reason"] = "uniform_low_saturation"
        return result
    result["reason"] = "inconclusive"
    return result


def scan_unlabelled_targets(
    image_bgr: np.ndarray,
    *,
    boxes: Sequence[Box],
    templates_bgr: Sequence[np.ndarray],
    params: Optional[dict[str, Any]] = None,
    spec: Optional[dict[str, Any]] = None,
) -> EmptinessReport:
    """Report whether a frame may be treated as free of unlabelled targets.

    A frame is ``clean`` only when at least one template was available and no
    candidate survived as a suspect.  Candidates whose centre is already covered
    by an annotated box are treated as explained by that annotation; without
    this rule a partially shifted match on a correctly labelled target would be
    reported as a missing label.

    ``spec`` declares the task's target definition.  With
    ``{"min_height_fraction": 0.28, "back_side": "hard_negative",
    "below_min_scale": "not_a_target"}`` a standee shorter than 28 % of the frame
    height is recorded in ``below_scale`` and a back-side board in
    ``hard_negatives``; neither enters ``suspects``.  A below-scale candidate
    that the spec keeps (``below_min_scale="suspect"``) still undergoes any
    enabled side check. The result is kept as evidence for the suspect when
    the match window cannot establish the full object's size. The defaults keep the
    round-one behaviour (no scale filter, back side stays a suspect), so callers
    without a spec see unchanged verdicts.  ``unknown`` side evidence is always
    kept as a suspect.  When ``spec`` is omitted, spec keys found inside
    ``params`` are used instead, because the review CLI forwards only ``params``.
    """

    merged = _params(params)
    effective_spec = _resolve_spec(spec, params)
    gray = _to_gray(image_bgr, name="image_bgr")
    frame_height = gray.shape[0]
    shape = (frame_height, gray.shape[1])
    annotated = [box.xyxy for box in boxes]
    mask = mask_boxes(shape, annotated, merged["mask_margin_px"])
    covered = covered_area_fraction(mask)

    if not templates_bgr:
        # No template means no measurement: absence of evidence is not evidence
        # of absence, so the frame must stay unproven instead of clean.
        return EmptinessReport(
            clean=False,
            suspects=[],
            covered_area_fraction=covered,
            reason_codes=[REASON_UNPROVEN],
            target_spec=dict(effective_spec),
        )

    search_mask = np.where(mask == 0, np.uint8(255), np.uint8(0))
    candidates: list[list[int]] = []
    candidate_scores: list[float] = []
    for template in templates_bgr:
        template_gray = _to_gray(template, name="template")
        boxes_found, scores_found = multi_scale_match(
            gray,
            template_gray,
            scales=merged["scales"],
            search_mask=search_mask,
        )
        # Greedy NMS visits the highest scores first. A match below the
        # acceptance score cannot suppress a match that will be reported.
        for box, score in zip(boxes_found, scores_found):
            if score > merged["suspect_min_score"]:
                candidates.append(box)
                candidate_scores.append(score)

    kept = nms(candidates, candidate_scores, merged["nms_iou"]) if candidates else []
    suspects: list[dict[str, Any]] = []
    below_scale: list[dict[str, Any]] = []
    hard_negatives: list[dict[str, Any]] = []
    min_height_fraction = effective_spec["min_height_fraction"]
    for index in kept:
        score = float(candidate_scores[index])
        box = candidates[index]
        if merged["drop_explained_candidates"] and _centre_is_explained(mask, box):
            continue

        height_fraction = float(box[3] - box[1]) / float(frame_height)
        below_min_scale_candidate = (
            min_height_fraction > 0.0 and height_fraction < min_height_fraction
        )
        if below_min_scale_candidate and effective_spec["below_min_scale"] == "not_a_target":
            below_scale.append({"xyxy": box, "score": score, "height_fraction": height_fraction})
            continue

        suspect = {"xyxy": box, "score": score, "method": "template"}
        if effective_spec["back_side"] == "hard_negative":
            x1, y1, x2, y2 = box
            side = classify_standee_side(image_bgr[y1:y2, x1:x2], params=merged)
            if side["side"] == SIDE_BACK and not below_min_scale_candidate:
                hard_negatives.append({"xyxy": box, "score": score, "side": SIDE_BACK})
                continue
            if below_min_scale_candidate:
                suspect["side_observation"] = side

        suspects.append(suspect)

    suspects.sort(key=lambda item: (-item["score"], item["xyxy"]))
    below_scale.sort(key=lambda item: (-item["score"], item["xyxy"]))
    hard_negatives.sort(key=lambda item: (-item["score"], item["xyxy"]))

    clean = not suspects
    reason_codes: list[str] = []
    if below_scale:
        reason_codes.append(REASON_BELOW_SCALE)
    if hard_negatives:
        reason_codes.append(REASON_HARD_NEGATIVE)
    reason_codes.append(REASON_CLEAN if clean else REASON_SUSPECT)
    return EmptinessReport(
        clean=clean,
        suspects=suspects,
        covered_area_fraction=covered,
        reason_codes=reason_codes,
        below_scale=below_scale,
        hard_negatives=hard_negatives,
        target_spec=dict(effective_spec),
    )


def _plan_boxes(items: Any) -> list[Box]:
    """Normalise plan boxes (``Box``, mapping with ``xyxy``, or 4-sequence)."""

    result: list[Box] = []
    for item in items or []:
        if isinstance(item, Box):
            result.append(item)
        elif isinstance(item, dict):
            result.append(Box(class_name=str(item.get("class_name", "")), xyxy=tuple(item["xyxy"])))
        else:
            result.append(Box(class_name="", xyxy=tuple(item)))
    return result


def check_emptiness(
    image_bgr: np.ndarray,
    plan: Optional[dict[str, Any]] = None,
) -> EmptinessReport:
    """CLI adapter: measure one frame from a review plan mapping.

    The plan carries the annotated ``boxes`` of the frame, the in-memory
    ``templates_bgr`` of the task, an optional ``emptiness`` mapping used as
    ``params`` and an optional ``spec`` (or ``target_spec``) used as the target
    definition.  Templates are never loaded from disk here, so a plan without
    templates stays ``emptiness_unproven`` instead of clean.
    """

    payload = dict(plan or {})
    templates = list(payload.get("templates_bgr") or [])
    for template in templates:
        if not isinstance(template, np.ndarray):
            raise ValueError("templates_bgr must hold in-memory arrays; file loading stays with the caller")
    params = payload.get("emptiness")
    spec = payload.get("spec")
    if spec is None:
        spec = payload.get("target_spec")
    return scan_unlabelled_targets(
        image_bgr,
        boxes=_plan_boxes(payload.get("boxes")),
        templates_bgr=templates,
        params=params if isinstance(params, dict) else None,
        spec=spec if isinstance(spec, dict) else None,
    )


def variance_map(gray: np.ndarray, win: int = 16) -> np.ndarray:
    """Local standard deviation of ``gray`` in a ``win`` × ``win`` window.

    Available for later extensions (e.g. texture-free regions); the current
    verdict does not consume it.
    """

    image = _to_gray(gray, name="gray")
    window = int(win)
    if window < 2:
        raise ValueError("win must be >= 2")
    height, width = image.shape[:2]
    window = min(window, height, width)
    if window < 2:
        raise ValueError("image is too small for variance_map")
    data = image.astype(np.float32)
    mean = cv2.boxFilter(data, -1, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_square = cv2.boxFilter(
        data * data, -1, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT
    )
    return np.sqrt(np.clip(mean_square - mean * mean, 0.0, None))
