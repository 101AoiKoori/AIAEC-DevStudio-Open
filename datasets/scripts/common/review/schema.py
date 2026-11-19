"""Shared record shapes for candidate review.

Pure data only: no file IO, no OpenCV import. Every module in this package
accepts or returns these records so that plans, metrics and verdicts stay
comparable across tasks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = 1

VERDICT_READY = "ready"
VERDICT_EXCLUDED = "excluded"
VERDICT_UNKNOWN = "unknown"
VERDICTS = (VERDICT_READY, VERDICT_EXCLUDED, VERDICT_UNKNOWN)

# Reason codes are a closed vocabulary: statistics and audits count them.
REASON_CODES = frozenset(
    {
        "ground_visible",
        "ground_polygon_fit",
        "horizon_estimable",
        "no_unlabelled_target",
        "unlabelled_target_suspected",
        "target_annotation_present",
        "occluded",
        "truncated",
        "near_duplicate",
        "mask_hole",
        "mask_border",
        "transition_state",
        "scale_out_of_range",
        "crop_incomplete",
        "label_missing",
        "source_unresolved",
        "emptiness_unproven",
        "geometry_rejected",
        "below_target_scale",
        "hard_negative_present",
        "placement_patch_fit",
        "placement_patch_cluttered",
        "placement_patch_unsure",
        "placement_markings_dense",
        "placement_footpoint_unavailable",
        "placement_patch_level_camera",
        "placement_patch_texture_fallback",
        "horizon_texture_fallback",
        "horizon_forced_horizontal",
        "dual_vp_patch",
        "level_model_patch",
    }
)


@dataclass(frozen=True)
class FrameRef:
    """One reviewable frame and its known provenance."""

    path: str
    parent_video: Optional[str] = None
    timestamp_seconds: Optional[float] = None
    record_index: Optional[int] = None
    source_group: Optional[str] = None


@dataclass
class Box:
    class_name: str
    xyxy: tuple[int, int, int, int]
    physical_instance_id: Optional[str] = None


@dataclass
class GroundObservation:
    """Legacy-shaped ground view: floor region plus one placement foot point.

    `polygon` is the visible floor region; `placement_patch` keeps the local
    region where a target of the declared size can stand, so a consumer can
    draw both layers without recomputing the plane.
    """

    ok: bool
    polygon: list[list[int]] = field(default_factory=list)
    foot_xy: Optional[list[int]] = None
    horizon_y: Optional[float] = None
    residual: Optional[float] = None
    reason_codes: list[str] = field(default_factory=list)
    placement_patch: list[list[int]] = field(default_factory=list)
    clearance: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmptinessReport:
    """Evidence about targets that may be missing from the annotation.

    Unresolved candidates count as suspects. Cases resolved by the task's
    size or back-side rule keep their evidence in separate lists.
    """

    clean: bool
    suspects: list[dict[str, Any]] = field(default_factory=list)
    covered_area_fraction: float = 0.0
    reason_codes: list[str] = field(default_factory=list)
    below_scale: list[dict[str, Any]] = field(default_factory=list)
    hard_negatives: list[dict[str, Any]] = field(default_factory=list)
    target_spec: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundPlaneObservation:
    """Visible ground plane estimated from the scene's line structure.

    `ground_plane` covers the visible floor; `placement_patch` is the local
    region where a target of the declared apparent size can actually stand.
    Composition uses placement_patch; the plane only supplies projection and
    scale constraints.
    """

    ok: bool
    horizon_y: Optional[float] = None
    vanishing_points: list[list[float]] = field(default_factory=list)
    homography: Optional[list[list[float]]] = None
    floor_region: list[list[int]] = field(default_factory=list)
    placement_patch: list[list[int]] = field(default_factory=list)
    residual: Optional[float] = None
    reason_codes: list[str] = field(default_factory=list)


@dataclass
class DuplicateReport:
    group: str
    duplicate_of: Optional[str] = None
    distances: dict[str, float] = field(default_factory=dict)


@dataclass
class MetricRecord:
    """One frame's measured quantities. Recomputed from media plus code."""

    frame: FrameRef
    task: str
    ground: Optional[GroundObservation] = None
    emptiness: Optional[EmptinessReport] = None
    duplicates: Optional[DuplicateReport] = None
    extra: dict[str, Any] = field(default_factory=dict)
    code_hash: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Verdict:
    """A decision about one base frame. Tracked in config/review_selection.json."""

    frame: FrameRef
    role: str
    verdict: str
    reason_codes: list[str] = field(default_factory=list)
    ground_polygon: Optional[list[list[int]]] = None
    foot_xy: Optional[list[int]] = None
    existing_targets: list[dict[str, Any]] = field(default_factory=list)
    near_duplicate_group: Optional[str] = None
    reviewed_by: str = "vision"
    accepted_by: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)


def to_jsonable(record: Any) -> Any:
    """Convert a schema record into JSON-safe primitives."""

    if isinstance(record, (FrameRef, GroundObservation, GroundPlaneObservation, EmptinessReport, DuplicateReport, MetricRecord, Verdict)):
        return asdict(record)
    raise TypeError("unsupported record type: %r" % (type(record),))
