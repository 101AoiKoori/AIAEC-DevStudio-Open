"""Cross-task perspective templates derived from review metrics.

The review harness measures one candidate frame at a time; the ground plane,
horizon and placement patch it records are useful beyond that frame and beyond
the task that supplied it. This module turns stored metrics into a compact,
task-tagged document that generators can read for yaw and placement reference
without re-running geometry.

Pure data only: no image IO, no OpenCV import.
"""

from __future__ import annotations

from typing import Any, Mapping

SCHEMA_VERSION = 1


def ground_template(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the reusable perspective fields of one measured frame.

    Returns ``None`` when the frame has no ground observation, because an
    unresolved frame must not be promoted into a placement reference.
    """

    ground = payload.get("ground")
    if not isinstance(ground, Mapping) or not ground.get("ok"):
        return None
    frame = payload.get("frame") or {}
    extra = payload.get("extra") or {}
    return {
        "path": frame.get("path"),
        "source_group": frame.get("source_group"),
        "parent_video": frame.get("parent_video"),
        "timestamp_seconds": frame.get("timestamp_seconds"),
        "role": payload.get("role"),
        "image_shape": extra.get("image_shape"),
        "horizon_y": ground.get("horizon_y"),
        "ground_polygon": ground.get("polygon"),
        "placement_patch": ground.get("placement_patch"),
        "foot_xy": ground.get("foot_xy"),
        "residual": ground.get("residual"),
        "reason_codes": list(ground.get("reason_codes") or []),
        "clearance": ground.get("clearance"),
    }


def build_document(header: Mapping[str, Any], payloads: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the task-tagged perspective document for one review batch."""

    frames = []
    rejected = 0
    groups: dict[str, int] = {}
    for payload in payloads:
        template = ground_template(payload)
        if template is None:
            rejected += 1
            continue
        group = template.get("source_group") or template.get("parent_video") or "unknown"
        groups[group] = groups.get(group, 0) + 1
        frames.append(template)
    return {
        "schema_version": SCHEMA_VERSION,
        "task": header.get("task"),
        "batch": header.get("batch"),
        "generated_at": header.get("generated_at"),
        "code_hash": header.get("code_hash"),
        "params_hash": header.get("params_hash"),
        "source_metrics": header.get("metrics"),
        "frames": frames,
        "summary": {
            "measured": len(payloads),
            "with_ground": len(frames),
            "without_ground": rejected,
            "source_groups": dict(sorted(groups.items())),
        },
    }
