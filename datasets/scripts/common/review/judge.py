"""Turn measured review quantities into one verdict per base frame.

The judge adds no new measurement: it applies thresholds from the task's
`config/review_plan.json` to a :class:`MetricRecord`. `unknown` is a first-class
result: an unresolved frame is never promoted to `ready`, and an empty
template list can never establish "no target here".
"""

from __future__ import annotations

from typing import Any

from .schema import (
    VERDICT_EXCLUDED,
    VERDICT_READY,
    VERDICT_UNKNOWN,
    EmptinessReport,
    FrameRef,
    MetricRecord,
    Verdict,
)
from .placement import placement_decision

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "ground_min_area_ratio": 0.02,
    "ground_min_plane_support": 0.30,
    "ground_max_horizon_fraction": 0.85,
    "emptiness_required": True,
    "suspect_min_score": 0.62,
    "near_duplicate_phash_max": 8,
    "near_duplicate_ssim_min": 0.95,
    "require_two_duplicate_signals": True,
    "min_ground_area_ratio": 0.0,
    "min_patch_free_ratio": 0.0,
    "max_patch_obstacle_components": 0,
}


def resolve_thresholds(plan: dict[str, Any] | None) -> dict[str, Any]:
    """Merge plan thresholds over the defaults without mutating the inputs."""

    merged = dict(DEFAULT_THRESHOLDS)
    if plan:
        merged.update(plan.get("thresholds") or {})
    return merged


def _polygon_area_ratio(polygon: list[list[int]], shape: tuple[int, int] | None) -> float | None:
    if not polygon or len(polygon) < 3 or not shape:
        return None
    area = 0.0
    points = [(float(x), float(y)) for x, y in polygon]
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2.0
    frame_area = float(shape[0] * shape[1])
    if frame_area <= 0:
        return None
    return area / frame_area


def _is_near_duplicate(metric: MetricRecord, thresholds: dict[str, Any]) -> bool:
    duplicate = metric.duplicates
    if duplicate is None or not duplicate.duplicate_of:
        return False
    distances = duplicate.distances or {}
    hits = 0
    phash = distances.get("phash")
    if phash is not None and float(phash) <= thresholds["near_duplicate_phash_max"]:
        hits += 1
    ssim = distances.get("ssim")
    if ssim is not None and float(ssim) >= thresholds["near_duplicate_ssim_min"]:
        hits += 1
    orb = distances.get("orb")
    if orb is not None and float(orb) >= 0.35:
        hits += 1
    if thresholds.get("require_two_duplicate_signals", True):
        return hits >= 2
    return hits >= 1


def patch_is_valid(patch: list[list[int]], shape: tuple[int, int] | None) -> tuple[bool, str]:
    """Reject deprecated patches: self-intersecting, concave, thin or tiny.

    A background patch is the rectangle a composited target may stand in, so a
    crossed or degenerate quadrilateral is worse than no patch at all.
    """

    if not patch or len(patch) != 4:
        return False, "patch_not_quad"
    points = [(float(x), float(y)) for x, y in patch]
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2.0
    minimum = 1200.0
    maximum = float("inf")
    if shape:
        frame_area = float(shape[0] * shape[1])
        minimum = max(1200.0, 0.0025 * frame_area)
        maximum = 0.45 * frame_area
    if area < minimum:
        return False, "patch_too_small"
    if area > maximum:
        # Sanity cap only: a foot-position band may legitimately cover much of the
        # lower frame, so this rejects scene-wide bands rather than judging area.
        return False, "patch_too_large"
    signs = set()
    for index in range(4):
        ax, ay = points[index]
        bx, by = points[(index + 1) % 4]
        cx, cy = points[(index + 2) % 4]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        signs.add(1 if cross > 0 else (-1 if cross < 0 else 0))
        if 0 in signs:
            return False, "patch_collinear"
    if len(signs) != 1:
        return False, "patch_not_convex"
    def length(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    edges = [length(points[i], points[(i + 1) % 4]) for i in range(4)]
    if min(edges) <= 0:
        return False, "patch_degenerate"
    # Edge aspect is intentionally not judged here: a rectangle lying on the ground
    # plane is wide and shallow in image space, so shape quality belongs to the
    # geometry stage that owns the homography.
    return True, "ok"

def decide(metric: MetricRecord, plan: dict[str, Any] | None = None, *, role: str = "background") -> Verdict:
    """Decide one frame. Returns a Verdict carrying the reasons for the decision."""

    verdict = _decide_core(metric, plan, role=role)
    extra_codes = _extra_codes(metric)
    if extra_codes:
        verdict.reason_codes = sorted(set(list(verdict.reason_codes) + extra_codes))
    search = (metric.extra or {}).get("emptiness_search")
    if search is not None:
        verdict.evidence["emptiness_search"] = search
    return verdict


def _extra_codes(metric: MetricRecord) -> list[str]:
    """Facts about the frame that qualify any verdict without changing it."""

    codes: list[str] = []
    extra = metric.extra or {}
    if extra.get("parent_link_verified") is False:
        codes.append("source_unresolved")
    return codes


def _suspect_verdict(frame: FrameRef, role: str, emptiness: EmptinessReport) -> Verdict:
    suspects = emptiness.suspects or []
    return Verdict(
        frame=frame,
        role=role,
        verdict=VERDICT_UNKNOWN,
        reason_codes=["unlabelled_target_suspected"],
        evidence={
            "judge": "suspects",
            "count": len(suspects),
            "top_score": round(max((float(s.get("score", 0.0)) for s in suspects), default=0.0), 4),
        },
    )


def _decide_core(metric: MetricRecord, plan: dict[str, Any] | None = None, *, role: str = "background") -> Verdict:
    """Apply the thresholds. Separated so the wrapper can attach qualifiers."""

    thresholds = resolve_thresholds(plan)
    frame = metric.frame
    reasons: list[str] = []
    extra = metric.extra or {}
    emptiness = metric.emptiness
    if metric.task == "person" and role == "background":
        existing_targets = extra.get("existing_targets") or []
        if existing_targets:
            return Verdict(
                frame=frame,
                role=role,
                verdict=VERDICT_EXCLUDED,
                reason_codes=["target_annotation_present"],
                existing_targets=list(existing_targets),
                evidence={"judge": "annotated_background_target", "count": len(existing_targets)},
            )
        if emptiness and emptiness.below_scale:
            # Older metrics can say clean despite small matches. Decide this before
            # the placement result, which can be used for provisional backgrounds.
            return Verdict(
                frame=frame,
                role=role,
                verdict=VERDICT_UNKNOWN,
                reason_codes=["below_target_scale", "unlabelled_target_suspected"],
                evidence={"judge": "below_scale_suspects", "below_scale_count": len(emptiness.below_scale)},
            )
        if (emptiness and emptiness.target_spec.get("below_min_scale") == "not_a_target" and
                not (extra.get("emptiness_search") or {}).get("reference_frame_height")):
            return Verdict(
                frame=frame,
                role=role,
                verdict=VERDICT_UNKNOWN,
                reason_codes=["emptiness_unproven"],
                evidence={"judge": "stale_person_emptiness_scope"},
            )
        if emptiness and emptiness.suspects:
            return _suspect_verdict(frame, role, emptiness)

    shape = None
    raw_shape = extra.get("image_shape")
    if isinstance(raw_shape, (list, tuple)) and len(raw_shape) >= 2:
        shape = (int(raw_shape[0]), int(raw_shape[1]))

    ground = metric.ground
    if ground is None or not ground.ok:
        # An estimator failure leaves the frame unproven, not disproven.
        return Verdict(
            frame=frame,
            role=role,
            verdict=VERDICT_UNKNOWN,
            reason_codes=["geometry_rejected"],
            evidence={"judge": "ground_missing"},
        )

    polygon = ground.polygon or []
    area_ratio = _polygon_area_ratio(polygon, shape)
    if area_ratio is None:
        return Verdict(
            frame=frame,
            role=role,
            verdict=VERDICT_EXCLUDED,
            reason_codes=["geometry_rejected"],
            evidence={"judge": "polygon_incomplete"},
        )
    if area_ratio < float(thresholds["ground_min_area_ratio"]):
        return Verdict(
            frame=frame,
            role=role,
            verdict=VERDICT_EXCLUDED,
            reason_codes=["geometry_rejected", "scale_out_of_range"],
            evidence={"judge": "ground_area_ratio", "value": round(area_ratio, 4)},
        )

    residual = ground.residual
    # geometry.residual is the vanishing-point inlier deficit (0..1): 1 - residual is
    # the share of line length supporting the two ground directions. It qualifies the
    # decision instead of gating it, because a rejected estimator means unknown, not bad.
    edge_support = None if residual is None else round(1.0 - float(residual), 4)
    if edge_support is not None and edge_support < float(thresholds["ground_min_plane_support"]):
        support_low = True
    else:
        support_low = False

    horizon = ground.horizon_y
    if horizon is not None and shape is not None and shape[0] > 0:
        horizon_fraction = float(horizon) / float(shape[0])
        if horizon_fraction > float(thresholds["ground_max_horizon_fraction"]):
            return Verdict(
                frame=frame,
                role=role,
                verdict=VERDICT_UNKNOWN,
                reason_codes=["geometry_rejected"],
                evidence={"judge": "horizon_too_low", "fraction": round(horizon_fraction, 4)},
            )

    patch = getattr(ground, "placement_patch", None) or []
    if role == "background":
        if not patch:
            # A background without a placement area cannot receive a composited target.
            return Verdict(
                frame=frame,
                role=role,
                verdict=VERDICT_UNKNOWN,
                reason_codes=["geometry_rejected"],
                evidence={"judge": "placement_patch_missing"},
            )
        valid, why = patch_is_valid(patch, shape)
        if not valid:
            return Verdict(
                frame=frame,
                role=role,
                verdict=VERDICT_UNKNOWN,
                reason_codes=["geometry_rejected"],
                evidence={"judge": "patch_invalid", "detail": why},
            )
        ground_area = _polygon_area_ratio(polygon, shape)
        minimum_area = float(thresholds["min_ground_area_ratio"])
        if ground_area is not None and minimum_area > 0.0 and ground_area < minimum_area:
            return Verdict(
                frame=frame,
                role=role,
                verdict=VERDICT_UNKNOWN,
                reason_codes=["scale_out_of_range"],
                evidence={"judge": "ground_area_too_small", "value": round(ground_area, 4)},
            )
        clearance = getattr(ground, "clearance", None) or {}
        free_ratio = clearance.get("free_ratio")
        minimum_free = float(thresholds["min_patch_free_ratio"])
        if free_ratio is not None and minimum_free > 0.0 and float(free_ratio) < minimum_free:
            return Verdict(
                frame=frame,
                role=role,
                verdict=VERDICT_UNKNOWN,
                reason_codes=["occluded"],
                evidence={"judge": "patch_too_cluttered", "free_ratio": round(float(free_ratio), 4)},
            )
        components = clearance.get("components")
        max_components = int(thresholds["max_patch_obstacle_components"])
        if components is not None and max_components > 0 and int(components) > max_components:
            return Verdict(
                frame=frame,
                role=role,
                verdict=VERDICT_UNKNOWN,
                reason_codes=["occluded"],
                evidence={"judge": "patch_obstacles", "components": int(components)},
            )
        placement = (plan or {}).get("placement") or {}
        if isinstance(placement, dict) and placement.get("required", False):
            decision, placement_evidence = placement_decision(
                clearance, placement.get("thresholds")
            )
            if decision == "cluttered":
                return Verdict(
                    frame=frame,
                    role=role,
                    verdict=VERDICT_EXCLUDED,
                    reason_codes=["placement_patch_cluttered"],
                    evidence=placement_evidence,
                )
            if decision == "unsure":
                return Verdict(
                    frame=frame,
                    role=role,
                    verdict=VERDICT_UNKNOWN,
                    reason_codes=["placement_patch_unsure"],
                    evidence=placement_evidence,
                )

    if thresholds.get("emptiness_required", True):
        if emptiness is None:
            return Verdict(
                frame=frame,
                role=role,
                verdict=VERDICT_UNKNOWN,
                reason_codes=["emptiness_unproven"],
                evidence={"judge": "emptiness_missing"},
            )
        codes = set(emptiness.reason_codes or [])
        if "emptiness_unproven" in codes:
            return Verdict(
                frame=frame,
                role=role,
                verdict=VERDICT_UNKNOWN,
                reason_codes=["emptiness_unproven"],
                evidence={"judge": "emptiness_requires_templates"},
            )
        if not emptiness.clean:
            return _suspect_verdict(frame, role, emptiness)

    if _is_near_duplicate(metric, thresholds):
        duplicate = metric.duplicates
        return Verdict(
            frame=frame,
            role=role,
            verdict=VERDICT_EXCLUDED,
            reason_codes=["near_duplicate"],
            near_duplicate_group=duplicate.group if duplicate else None,
            evidence={"judge": "near_duplicate", "of": duplicate.duplicate_of if duplicate else None},
        )

    reasons.extend(ground.reason_codes or [])
    reasons.append("no_unlabelled_target")
    hard_negatives = list(getattr(emptiness, "hard_negatives", []) or []) if emptiness else []
    below_scale = list(getattr(emptiness, "below_scale", []) or []) if emptiness else []
    if below_scale:
        reasons.append("below_target_scale")
    if hard_negatives:
        reasons.append("hard_negative_present")
    return Verdict(
        frame=frame,
        role=role,
        verdict=VERDICT_READY,
        reason_codes=sorted(set(reasons)),
        ground_polygon=polygon,
        foot_xy=ground.foot_xy,
        near_duplicate_group=metric.duplicates.group if metric.duplicates else None,
        evidence={
            "judge": "ready",
            "ground_area_ratio": round(area_ratio, 4),
            "ground_plane_support": edge_support,
            "ground_support_low": support_low,
            "emptiness_covered_fraction": round(float(emptiness.covered_area_fraction), 4) if emptiness else None,
            "below_scale_count": len(below_scale),
            "hard_negative_count": len(hard_negatives),
        },
    )


def summarize(verdicts: list[Verdict]) -> dict[str, Any]:
    """Count verdicts, reason codes and parent groups for `report`."""

    by_verdict: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    by_group: dict[str, int] = {}
    for item in verdicts:
        by_verdict[item.verdict] = by_verdict.get(item.verdict, 0) + 1
        for code in item.reason_codes:
            by_reason[code] = by_reason.get(code, 0) + 1
        group = item.frame.source_group or item.frame.parent_video or "unknown"
        by_group[group] = by_group.get(group, 0) + 1
    return {
        "total": len(verdicts),
        "by_verdict": dict(sorted(by_verdict.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "by_group": dict(sorted(by_group.items())),
    }


def frame_from_record(record: dict[str, Any]) -> FrameRef:
    """Build a FrameRef from a JSON record produced by evidence/plan."""

    return FrameRef(
        path=str(record.get("path") or record.get("image") or ""),
        parent_video=record.get("parent_video"),
        timestamp_seconds=record.get("timestamp_seconds"),
        record_index=record.get("record_index"),
        source_group=record.get("source_group"),
    )
