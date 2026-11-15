"""Record shapes and closed vocabularies for the background/source ledger.

Pure data only: the ledger tool reads configurations and manifests, validates
their declarations and writes one derived report. These records do not replace
the task ``source.yaml``, the batch manifests, the shared background manifest
or the Blender registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1

PAIRING_STATUSES = frozenset({"draft", "reserved", "executable"})
PARENT_CLASSES = frozenset({"task_real", "background"})
GROUP_CLASSES = frozenset({"task_real", "background", "blender"})
PARENT_KINDS = frozenset(
    {
        "parent_video",
        "parent_photo",
        "parent_photo_batch",
        "physical_object",
        "field_session",
        "blender_scene",
    }
)
DATA_SOURCE_KINDS = frozenset(
    {"fixed_artwork", "real_cutout", "mask", "blender_target", "generated_text"}
)
ROLES = frozenset(
    {
        "direct_target",
        "composite_target",
        "target_foreground",
        "target_scene",
        "mask_source",
        "occluder",
        "composite_background",
        "empty_negative",
        "geometry_reference",
        "material_reference",
        "background_scene",
        "ocr_target",
        "reference",
        "ordinary_pole",
        "surface",
    }
)
SPLITS = frozenset({"train", "val", "test", "shared", "unassigned", "inherit"})
SPLIT_STATUSES = frozenset({"proposed", "reserved", "confirmed"})
ACTIONS = frozenset(
    {"selected", "reserved", "to_review", "to_label", "to_extract", "to_capture", "blocked"}
)
CAPTURE_SESSIONS = frozenset({"planned", "proven", "unknown", "same_session"})
CAPTURE_STATUSES = frozenset({"proposed", "conditional", "confirmed"})
REVIEW_STATES = frozenset(
    {"confirmed_empty", "candidate_empty", "contains_target", "unknown", "rejected"}
)
LABEL_STATES = frozenset(
    {"complete", "partial", "unknown", "blocked", "excluded", "candidate", "ready"}
)
BLENDER_PRIORITIES = frozenset({"background_only", "increased"})
ISSUE_LEVELS = frozenset({"error", "warning", "info"})

# A background may serve a composite scene while it is still a candidate, but
# only a confirmed-empty background may be used as an empty negative. A frame
# that contains the task target may serve as that target's labelled background.
COMPOSITE_BACKGROUND_STATES = frozenset(
    {"confirmed_empty", "candidate_empty", "contains_target"}
)


@dataclass
class ParentGroup:
    """One task-level parent group declared by ``pairing``."""

    task: str
    group_id: str
    group_class: str
    kind: str
    paths: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    split: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] | None = None
    labels: dict[str, Any] | None = None
    axes: dict[str, Any] = field(default_factory=dict)
    capture_session: str | None = None
    capture_id: str | None = None
    origin: str = "pairing"
    raw_key: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "id": self.group_id,
            "class": self.group_class,
            "kind": self.kind,
            "paths": list(self.paths),
            "evidence": [dict(item) for item in self.evidence],
            "roles": list(self.roles),
            "split": dict(self.split),
            "review": dict(self.review) if self.review is not None else None,
            "labels": dict(self.labels) if self.labels is not None else None,
            "axes": dict(self.axes),
            "capture_session": self.capture_session,
            "capture_id": self.capture_id,
            "origin": self.origin,
            "declared_at": self.raw_key,
        }


@dataclass
class DataSource:
    """One task-level data source declared by ``pairing``."""

    task: str
    source_id: str
    kind: str
    path: str | None
    parent_group: str | None
    roles: list[str] = field(default_factory=list)
    split: dict[str, Any] = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)
    origin: str = "pairing"

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "id": self.source_id,
            "kind": self.kind,
            "path": self.path,
            "paths": list(self.paths),
            "parent_group": self.parent_group,
            "roles": list(self.roles),
            "split": dict(self.split),
            "origin": self.origin,
        }


@dataclass
class Issue:
    """One ledger finding. ``code`` is a stable identifier for reports."""

    code: str
    level: str
    message: str
    task: str | None = None
    group: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "level": self.level,
            "task": self.task,
            "group": self.group,
            "message": self.message,
        }


def group_to_dict(group: ParentGroup) -> dict[str, Any]:
    return group.to_json()


def issue_to_dict(issue: Issue) -> dict[str, Any]:
    return issue.to_json()
