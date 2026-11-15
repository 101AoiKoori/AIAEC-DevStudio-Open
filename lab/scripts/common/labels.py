"""Audit and rebuild complete object labels from explicit authoring records.

A zero-length label file is not evidence of a negative image. This module
requires explicit full-frame review and a named reviewer for confirmed
negatives, keeps partial/unknown images separate, and rebuilds detector labels
only from validated visible boxes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .contracts import (
    LABEL_STATES,
    SCHEMA_VERSION,
    SOURCE_KINDS,
    now_local,
    resolve_workspace,
    workspace_relative,
)
from .mask_quality import audit_mask_arrays, load_mask

BOX_ACCEPTANCE_STATES = frozenset({"pending", "candidate", "human", "programmatic"})


def _image_size(path: Path) -> tuple[int, int]:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise ValueError(f"cannot read image size: {path}")
    return int(image.shape[1]), int(image.shape[0])


def _bbox(value: Any, *, name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    if any(type(item) not in (int, float) for item in value):
        raise ValueError(f"{name} coordinates must be numeric")
    x1, y1, x2, y2 = (float(item) for item in value)
    if not all(np.isfinite([x1, y1, x2, y2])):
        raise ValueError(f"{name} must be finite")
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError(f"{name} must have positive area and ordered xyxy")
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _visible_yolo_row(class_id: int, box: Sequence[int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("visible box lies outside the image")
    return (
        f"{class_id} {(x1 + x2) / (2.0 * width):.6f} "
        f"{(y1 + y2) / (2.0 * height):.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
    )


def _parent_sources(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("parent_sources must be a nonempty list")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"parent_sources[{index}] must be an object")
        path = item.get("path")
        role = item.get("role")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise ValueError(f"parent_sources[{index}].path must be workspace-relative")
        if not isinstance(role, str) or not role:
            raise ValueError(f"parent_sources[{index}].role is required")
        result.append(
            {
                **dict(item),
                "path": path.replace("\\", "/"),
                "role": role,
                "evidence": item.get("evidence"),
            }
        )
    return result


def audit_label_record(record: Mapping[str, Any], *, root: str | Path | None = None) -> dict[str, Any]:
    """Validate one complete/partial/unknown/negative annotation record."""

    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("label record requires a nonempty id")
    image = record.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError(f"{record_id}: image is required")
    image_path = resolve_workspace(image, root=root, must_exist=True)
    width, height = _image_size(image_path)
    source_kind = str(record.get("source_kind") or "")
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"{record_id}: unsupported source_kind {source_kind!r}")
    state = str(record.get("label_state") or "unknown")
    if state not in LABEL_STATES:
        raise ValueError(f"{record_id}: unsupported label_state {state!r}")
    parents = _parent_sources(record.get("parent_sources"))
    review = record.get("review") if isinstance(record.get("review"), Mapping) else {}
    full_frame_checked = review.get("full_frame_checked") is True
    accepted_by = review.get("accepted_by")
    box_accepted_by = review.get("box_accepted_by")
    box_acceptance = review.get("box_acceptance")
    class_names = record.get("class_names")
    if not isinstance(class_names, list) or not class_names or any(
        not isinstance(name, str) or not name for name in class_names
    ):
        raise ValueError(f"{record_id}: class_names must be a nonempty string list")
    instances = record.get("instances")
    if not isinstance(instances, list):
        raise ValueError(f"{record_id}: instances must be a list")

    reasons: list[str] = []
    status = "pass_candidate"
    label_lines: list[str] = []
    audit_instances: list[dict[str, Any]] = []
    instance_ids: set[str] = set()
    for index, instance in enumerate(instances):
        if not isinstance(instance, Mapping):
            raise ValueError(f"{record_id}: instances[{index}] must be an object")
        instance_id = instance.get("id")
        if not isinstance(instance_id, str) or not instance_id or instance_id in instance_ids:
            raise ValueError(f"{record_id}: instance IDs must be unique nonempty strings")
        instance_ids.add(instance_id)
        class_id = instance.get("class_id")
        if type(class_id) is not int or not 0 <= class_id < len(class_names):
            raise ValueError(f"{record_id}/{instance_id}: invalid class_id")
        visible_box = _bbox(instance.get("visible_bbox_xyxy"), name="visible_bbox_xyxy")
        if visible_box[2] > width or visible_box[3] > height:
            raise ValueError(f"{record_id}/{instance_id}: visible box exceeds image")
        visibility = str(instance.get("visibility") or "unknown")
        if visibility not in {"full", "partial", "occluded", "unknown"}:
            raise ValueError(f"{record_id}/{instance_id}: invalid visibility")
        full_box = instance.get("full_bbox_xyxy")
        if full_box is not None:
            normalized_full = _bbox(full_box, name="full_bbox_xyxy")
            if (
                visible_box[0] < normalized_full[0]
                or visible_box[1] < normalized_full[1]
                or visible_box[2] > normalized_full[2]
                or visible_box[3] > normalized_full[3]
            ):
                raise ValueError(f"{record_id}/{instance_id}: visible box exceeds full box")
        else:
            normalized_full = None
        mask_result = None
        visible_mask = instance.get("visible_mask")
        full_mask = instance.get("full_mask")
        occlusion_masks = list(instance.get("occlusion_masks") or [])
        if visible_mask or full_mask or occlusion_masks:
            if not visible_mask or not full_mask:
                raise ValueError(f"{record_id}/{instance_id}: mask review requires full and visible masks")
            mask_result = audit_mask_arrays(
                load_mask(full_mask, root=root),
                load_mask(visible_mask, root=root),
                occlusion_masks=[load_mask(path, root=root) for path in occlusion_masks],
                ground_polygon=instance.get("ground_polygon"),
                contact_point=instance.get("contact_point"),
                params=instance.get("mask_params"),
            )
            if mask_result["status"] == "review":
                reasons.append("mask_review")
                status = "review" if status == "pass_candidate" else status
            elif mask_result["status"] == "reject":
                reasons.append("mask_reject")
                status = "reject"
        if visibility in {"partial", "occluded", "unknown"} and not (
            visible_mask or instance.get("visibility_evidence")
        ):
            reasons.append("visible_region_unproven")
            status = "review" if status == "pass_candidate" else status
        label_lines.append(_visible_yolo_row(class_id, visible_box, width, height))
        audit_instances.append(
            {
                "id": instance_id,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "visibility": visibility,
                "visible_bbox_xyxy": list(visible_box),
                "full_bbox_xyxy": list(normalized_full) if normalized_full else None,
                "mask_audit": mask_result,
            }
        )

    if state == "confirmed_negative":
        if instances:
            raise ValueError(f"{record_id}: confirmed negative cannot contain positive instances")
        if not full_frame_checked:
            raise ValueError(f"{record_id}: confirmed negative requires full_frame_checked")
        if not isinstance(accepted_by, str) or not accepted_by.strip():
            raise ValueError(f"{record_id}: confirmed negative requires accepted_by")
        if not isinstance(review.get("method"), str) or not review.get("method"):
            raise ValueError(f"{record_id}: confirmed negative requires a review method")
        status = "confirmed_negative"
    elif state == "unknown":
        if not full_frame_checked:
            reasons.append("full_frame_unreviewed")
        status = "unknown"
        label_lines = []
    elif state == "excluded":
        status = "excluded"
        label_lines = []
    elif state == "complete":
        if not full_frame_checked:
            raise ValueError(f"{record_id}: complete labels require full_frame_checked")
        if not instances:
            raise ValueError(f"{record_id}: complete positive record needs instances")
        if not isinstance(accepted_by, str) or not accepted_by.strip():
            reasons.append("human_acceptance_missing")
            status = "review" if status == "pass_candidate" else status
        if box_acceptance not in BOX_ACCEPTANCE_STATES:
            reasons.append("box_acceptance_pending")
            status = "review" if status == "pass_candidate" else status
        elif box_acceptance in {"pending", "candidate"}:
            reasons.append("box_acceptance_pending")
            status = "review" if status == "pass_candidate" else status
        elif not isinstance(box_accepted_by, str) or not box_accepted_by.strip():
            reasons.append("box_acceptor_missing")
            status = "review" if status == "pass_candidate" else status
    elif state == "partial":
        if not instances:
            raise ValueError(f"{record_id}: partial record should retain reviewed positive instances")
        if not isinstance(record.get("partial_reason"), str) or not record.get("partial_reason"):
            raise ValueError(f"{record_id}: partial record requires partial_reason")
        reasons.append("partial_label")
        status = "review" if status == "pass_candidate" else status

    return {
        "id": record_id,
        "image": workspace_relative(image_path, root=root),
        "image_size_wh": [width, height],
        "source_kind": source_kind,
        "label_state": state,
        "status": status,
        "reason_codes": sorted(set(reasons)),
        "parent_sources": parents,
        "review": dict(review),
        "box_acceptance": box_acceptance,
        "box_accepted_by": box_accepted_by,
        "instances": audit_instances,
        "label_lines": label_lines,
        "label_text": "\n".join(label_lines) + ("\n" if label_lines else ""),
    }


def audit_manifest(manifest: Mapping[str, Any], *, root: str | Path | None = None) -> dict[str, Any]:
    """Audit all explicit label records in one manifest."""

    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("label manifest requires a nonempty records list")
    audited = [audit_label_record(record, root=root) for record in records]
    ids = [record["id"] for record in audited]
    if len(ids) != len(set(ids)):
        raise ValueError("label manifest contains duplicate IDs")
    counts: dict[str, int] = {}
    for record in audited:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    complete = [record for record in audited if record["label_state"] == "complete"]
    human_complete = [
        record for record in complete if record.get("box_acceptance") == "human"
    ]
    programmatic_complete = [
        record for record in complete if record.get("box_acceptance") == "programmatic"
    ]
    complete_without_box_acceptance = [
        record
        for record in complete
        if "box_acceptance_pending" in record["reason_codes"]
        or "box_acceptor_missing" in record["reason_codes"]
    ]
    candidate_complete = [
        record for record in complete if record.get("box_acceptance") == "candidate"
    ]
    candidate_records = [
        record
        for record in audited
        if record["label_state"] == "partial" or record["id"] in {
            item["id"] for item in candidate_complete
        }
    ]
    human_confirmed_records = [
        record
        for record in audited
        if record["id"] in {item["id"] for item in human_complete}
        or record["label_state"] == "confirmed_negative"
    ]
    auto_validated_records = list(programmatic_complete)
    pending_records = [
        record
        for record in audited
        if record["label_state"] == "unknown"
        or record["id"] in {item["id"] for item in complete_without_box_acceptance}
    ]
    training_admitted_records = [
        record
        for record in audited
        if record.get("review", {}).get("training_admitted") is True
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_local(),
        "kind": "label_audit_report",
        "task": manifest.get("task"),
        "batch": manifest.get("batch"),
        "records": audited,
        "summary": {
            "total": len(audited),
            "by_status": dict(sorted(counts.items())),
            "positive_instances": sum(len(record["instances"]) for record in audited),
            "confirmed_negative": sum(record["label_state"] == "confirmed_negative" for record in audited),
            "unknown_not_negative": sum(record["label_state"] == "unknown" for record in audited),
            "partial_not_complete": sum(record["label_state"] == "partial" for record in audited),
            "complete_frames": len(complete),
            "human_complete_frames": len(human_complete),
            "human_complete_instances": sum(len(record["instances"]) for record in human_complete),
            "programmatic_complete_frames": len(programmatic_complete),
            "programmatic_complete_instances": sum(
                len(record["instances"]) for record in programmatic_complete
            ),
            "candidate_complete_frames": len(candidate_complete),
            "candidate_instances": sum(len(record["instances"]) for record in candidate_complete),
            "complete_without_box_acceptance": len(complete_without_box_acceptance),
            "status_layers": {
                "available_sources": None,
                "annotated_records": len(audited),
                "candidate_records": len(candidate_records),
                "auto_validated_records": len(auto_validated_records),
                "human_confirmed_records": len(human_confirmed_records),
                "training_admitted_records": len(training_admitted_records),
                "pending_records": len(pending_records),
            },
        },
        "boundary": (
            "Rebuilt labels contain only explicit visible detector boxes. An empty "
            "unknown label is not a negative; only confirmed_negative has absence evidence. "
            "A complete frame counts as human-confirmed boxes only when box_acceptance is "
            "human; semantic accepted_by or an AI candidate box alone is insufficient. "
            "Candidate and pending records are not rejected or excluded."
        ),
    }


def rebuild_labels(
    report: Mapping[str, Any],
    output_dir: str | Path,
    *,
    root: str | Path | None = None,
) -> list[str]:
    """Write validated YOLO labels for positive/partial records and return paths."""

    target = resolve_workspace(output_dir, root=root)
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for record in report.get("records") or []:
        state = record.get("label_state")
        if state not in {"complete", "partial"}:
            continue
        path = target / f"{record['id']}.txt"
        path.write_text(str(record.get("label_text") or ""), encoding="utf-8", newline="\n")
        written.append(workspace_relative(path, root=root))
    return written
