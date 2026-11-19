"""Validate that a bounded sample exercises every required source and content branch.

The manifest describes existing candidate files and provenance. This module does
not copy pixels, generate composites, render, or assign human acceptance. It
also emits a manual-review queue with per-image questions for a human reviewer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    LABEL_STATES,
    SCHEMA_VERSION,
    SOURCE_KINDS,
    atomic_write_json,
    now_local,
    resolve_workspace,
    workspace_relative,
)

CONTENT_BRANCHES = frozenset({"empty", "single", "multi", "interference"})
PROGRAMMATIC_ACCEPTORS = frozenset({"programmatic_blender_generator"})
AUTOMATIC_ACCEPTORS = frozenset({"programmatic_blender_generator", "Codex visual review"})

OBJECT_MATRIX_COLUMNS = {
    "row_id": "本图内稳定行标识，例如 obj_001",
    "left_to_right_order": "从左到右顺序；不确定时为 null",
    "category": "community_person / noncommunity_person / person_standee_back / person_standee_nonfront / vehicle / other / unknown",
    "view": "front / back / side / oblique / unknown",
    "visibility": "full / partial / cropped / occluded / unknown",
    "occluder": "none / person / vehicle / object / unknown",
    "is_task_target": "true / false / null",
    "annotation_action": "label_visible_box / hard_negative_no_box / do_not_label / unknown",
    "visible_bbox_xyxy": "只覆盖可见区域的像素框 [x1,y1,x2,y2]；不输出框时为 null",
    "full_bbox_xyxy": "有完整实体依据时的像素框 [x1,y1,x2,y2]；没有依据时为 null",
    "box_accepted_by": "实际确认这些框的人员；程序化生成值只能进入自动验证层",
    "notes": "遮挡、截断、类别疑点或证据说明",
}

MANUAL_REVIEW_ENUMS = {
    "presence": {"none", "single", "multi", "unknown"},
    "background_state": {"confirmed_empty", "contains_target", "unknown"},
    "category": {
        "community_person",
        "noncommunity_person",
        "person_standee_back",
        "person_standee_nonfront",
        "vehicle",
        "other",
        "unknown",
    },
    "view": {"front", "back", "side", "oblique", "unknown"},
    "visibility": {"full", "partial", "cropped", "occluded", "unknown"},
    "occluder": {"none", "person", "vehicle", "object", "unknown"},
    "annotation_action": {
        "label_visible_box",
        "hard_negative_no_box",
        "do_not_label",
        "unknown",
    },
}

REQUIRED_PARENT_ROLES = {
    "direct_camera_capture": {"camera_source"},
    "real_image_enhancement": {"camera_source"},
    "real_image_composite": {"real_background", "target_source"},
    "blender_render": {"blender_scene"},
    "render_real_mix": {"real_background", "blender_scene"},
}


def _object_rows(instance_count: Any) -> list[dict[str, Any]]:
    """Create empty matrix rows without inventing an object when count is unknown."""

    if instance_count is None:
        return []
    if type(instance_count) is not int or instance_count < 0:
        raise ValueError("instance_count must be a nonnegative integer or null")
    return [
        {
            "row_id": f"obj_{index:03d}",
            "left_to_right_order": index,
            "category": None,
            "view": None,
            "visibility": None,
            "occluder": None,
            "is_task_target": None,
            "annotation_action": None,
            "visible_bbox_xyxy": None,
            "full_bbox_xyxy": None,
            "box_accepted_by": None,
            "notes": None,
        }
        for index in range(1, instance_count + 1)
    ]


def _candidate_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build pending rows and copy only explicit candidate box coordinates."""

    rows = _object_rows(record.get("instance_count"))
    candidates = record.get("instances")
    if not isinstance(candidates, list):
        candidates = record.get("boxes")
    if not isinstance(candidates, list):
        return rows
    class_names = record.get("class_names")
    while len(rows) < len(candidates):
        rows.extend(_object_rows(len(rows) + 1)[len(rows):])
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        row = rows[index]
        row["row_id"] = str(candidate.get("row_id") or candidate.get("id") or row["row_id"])
        category = candidate.get("category")
        if isinstance(category, str) and category:
            row["category"] = category
        elif type(candidate.get("class_id")) is int and isinstance(class_names, list):
            class_id = candidate["class_id"]
            if 0 <= class_id < len(class_names):
                row["category"] = f"{class_names[class_id]}_person"
        visible = candidate.get("visible_bbox_xyxy") or candidate.get("xyxy")
        if _valid_bbox(visible):
            row["visible_bbox_xyxy"] = copy.deepcopy(visible)
        full = candidate.get("full_bbox_xyxy")
        if _valid_bbox(full):
            row["full_bbox_xyxy"] = copy.deepcopy(full)
        for key in ("view", "visibility", "occluder", "annotation_action"):
            value = candidate.get(key)
            if isinstance(value, str) and value:
                row[key] = value
        if candidate.get("is_task_target") is not None:
            row["is_task_target"] = candidate["is_task_target"]
        if candidate.get("box_accepted_by") is not None:
            row["box_accepted_by"] = candidate["box_accepted_by"]
    return rows


def _matrix_template(record: Mapping[str, Any]) -> dict[str, Any]:
    """Create the appropriate single-frame or contact-sheet review shape."""

    if record.get("image_role") == "review_overlay_not_raw_render":
        return {
            "image_role": "contact_sheet",
            "panels": [],
            "reviewer": None,
            "accepted_by": None,
            "review_state": "pending",
            "notes": None,
        }
    return {
        "image_role": "single_frame",
        "presence": None,
        "task_target_count": None,
        "background_state": None,
        "objects": _candidate_rows(record),
        "other_objects": [],
        "reviewer": None,
        "accepted_by": None,
        "full_frame_checked": False,
        "review_state": "pending",
        "notes": None,
    }


def _exists(value: Any, *, root: Path) -> bool:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return False
    try:
        resolve_workspace(value, root=root, must_exist=True)
    except (OSError, ValueError):
        return False
    return True


def _parents(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("sample parent_sources must be a nonempty list")
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
        result.append(dict(item))
    return result


def _acceptance_method(review: Mapping[str, Any]) -> str | None:
    accepted_by = review.get("accepted_by")
    if not isinstance(accepted_by, str) or not accepted_by:
        return None
    return "programmatic" if accepted_by in PROGRAMMATIC_ACCEPTORS else "human"


def validate_sample(
    manifest: Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate branch coverage, provenance and pending manual questions."""

    base = Path(root or Path(__file__).resolve().parents[3]).resolve()
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("sample manifest requires a nonempty records list")
    required_sources = list(manifest.get("required_source_kinds") or [])
    required_content = list(manifest.get("required_content_branches") or [])
    if not required_sources or any(value not in SOURCE_KINDS for value in required_sources):
        raise ValueError("required_source_kinds must contain supported source kinds")
    if not required_content or any(value not in CONTENT_BRANCHES for value in required_content):
        raise ValueError("required_content_branches must contain supported content branches")

    errors: list[str] = []
    warnings: list[str] = []
    source_counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()
    results: list[dict[str, Any]] = []
    manual_review: list[dict[str, Any]] = []
    blender_image_roles: list[str] = []
    ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"records[{index}] must be an object")
        record_id = str(record.get("id") or "")
        if not record_id or record_id in ids:
            errors.append(f"bad_record_id:{index}")
            continue
        ids.add(record_id)
        image = record.get("image")
        if not _exists(image, root=base):
            errors.append(f"{record_id}:image_missing")
        source_kind = str(record.get("source_kind") or "")
        if source_kind not in SOURCE_KINDS:
            errors.append(f"{record_id}:source_kind_invalid")
            continue
        source_counts[source_kind] += 1
        parents = _parents(record.get("parent_sources"))
        parent_roles = {str(parent["role"]) for parent in parents}
        missing_roles = REQUIRED_PARENT_ROLES[source_kind] - parent_roles
        if missing_roles:
            errors.append(f"{record_id}:parent_roles_missing:{','.join(sorted(missing_roles))}")
        for parent in parents:
            if not _exists(parent["path"], root=base):
                errors.append(f"{record_id}:parent_missing:{parent['path']}")
        label_state = str(record.get("label_state") or "unknown")
        if label_state not in LABEL_STATES:
            errors.append(f"{record_id}:label_state_invalid")
        review = record.get("review") if isinstance(record.get("review"), Mapping) else {}
        if label_state == "confirmed_negative":
            if not review.get("full_frame_checked"):
                errors.append(f"{record_id}:negative_without_full_frame_check")
        tags = record.get("content_branches")
        if not isinstance(tags, list) or not tags:
            errors.append(f"{record_id}:content_branches_missing")
            tags = []
        for tag in tags:
            if tag not in CONTENT_BRANCHES:
                errors.append(f"{record_id}:content_branch_invalid:{tag}")
            else:
                content_counts[str(tag)] += 1
        if "empty" in tags:
            if label_state not in {"unknown", "confirmed_negative"}:
                errors.append(f"{record_id}:empty_branch_has_positive_label_state")
            if not record.get("background_state"):
                warnings.append(f"{record_id}:background_state_missing")
        if "single" in tags and record.get("instance_count") not in {None, 1}:
            errors.append(f"{record_id}:single_branch_count_mismatch")
        if "multi" in tags and record.get("instance_count") is not None and int(record["instance_count"]) < 2:
            errors.append(f"{record_id}:multi_branch_count_mismatch")
        if "interference" in tags:
            evidence = record.get("interference_evidence")
            if not isinstance(evidence, Mapping):
                errors.append(f"{record_id}:interference_evidence_missing")
            else:
                for key in ("occluder", "occlusion_layer"):
                    if key in evidence and not _exists(evidence[key], root=base):
                        errors.append(f"{record_id}:interference_{key}_missing")
        acceptance_method = _acceptance_method(review)
        if label_state != "confirmed_negative" and acceptance_method is None:
            warnings.append(f"{record_id}:human_acceptance_missing")
        if source_kind == "blender_render":
            blender_image_roles.append(str(record.get("image_role") or "raw_render"))
        results.append(
            {
                "id": record_id,
                "image": image,
                "review_image": (
                    record.get("review_image")
                    or record.get("overlay")
                    or image
                ),
                "source_kind": source_kind,
                "generation_method": record.get("generation_method"),
                "content_branches": tags,
                "instance_count": record.get("instance_count"),
                "label_state": label_state,
                "background_state": record.get("background_state"),
                "accepted_by": review.get("accepted_by"),
                "acceptance_method": acceptance_method,
                "parent_roles": sorted(parent_roles),
                "evidence": dict(record.get("evidence") or {}),
            }
        )
        manual_review.append(
            {
                "id": record_id,
                "image": image,
                "source_kind": source_kind,
                "class_names": copy.deepcopy(record.get("class_names")),
                "parent_sources": copy.deepcopy(record.get("parent_sources")),
                "content_branches": tags,
                "evidence": copy.deepcopy(record.get("evidence") or {}),
                "questions": [
                    "全图是否还有未列出的任务目标？",
                    "每个目标的类别和状态是否正确？",
                    "可见框是否只包含可见区域，遮挡关系是否记录？",
                    "来源父图/视频、生成方式与现有证据是否一致？",
                ],
                "status_matrix": _matrix_template(record),
            }
        )

    for source_kind in required_sources:
        if not source_counts[source_kind]:
            errors.append(f"source_branch_missing:{source_kind}")
    for branch in required_content:
        if not content_counts[branch]:
            errors.append(f"content_branch_missing:{branch}")
    if blender_image_roles and all(role == "review_overlay_not_raw_render" for role in blender_image_roles):
        warnings.append("blender_render:raw_render_branch_missing")
    status = "fail" if errors else ("review" if warnings else "pass_candidate")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_local(),
        "kind": "branch_sample_validation",
        "task": manifest.get("task"),
        "batch": manifest.get("batch"),
        "purpose": manifest.get("purpose"),
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "records": results,
        "manual_review": manual_review,
        "summary": {
            "records": len(results),
            "source_branches": dict(sorted(source_counts.items())),
            "content_branches": dict(sorted(content_counts.items())),
            "required_source_branches": required_sources,
            "required_content_branches": required_content,
            "human_accepted": sum(
                1
                for record in records
                if isinstance(record, Mapping)
                and isinstance(record.get("review"), Mapping)
                and _acceptance_method(record["review"]) == "human"
            ),
            "programmatic_accepted": sum(
                1
                for record in records
                if isinstance(record, Mapping)
                and isinstance(record.get("review"), Mapping)
                and _acceptance_method(record["review"]) == "programmatic"
            ),
        },
        "status_layers": sample_status_layers(records, results),
        "boundary": (
            "Branch coverage proves that the toolchain has a bounded evidence path "
            "for each branch. It does not prove image-label truth or model quality."
        ),
    }


def sample_status_layers(
    records: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count source sample layers without treating pending review as rejection."""

    by_id = {
        str(record.get("id")): record
        for record in records
        if isinstance(record, Mapping)
    }
    human = 0
    automatic = 0
    annotated = 0
    excluded = 0
    for result in results:
        record = by_id.get(str(result.get("id")), {})
        label_state = str(record.get("label_state") or "unknown")
        if label_state in {"complete", "partial", "confirmed_negative"}:
            annotated += 1
        if label_state == "excluded":
            excluded += 1
        method = result.get("acceptance_method")
        if method == "human":
            human += 1
        elif method == "programmatic":
            automatic += 1
    total = len(results)
    return {
        "source/available": {"frames": total},
        "annotated": {"frames": annotated},
        "candidate": {"frames": total},
        "auto_validated": {"frames": automatic},
        "human_confirmed": {"frames": human},
        "training_admitted": {
            "frames": 0,
            "status": "not_recorded",
            "reason": "branch coverage validation does not grant training admission",
        },
        "pending": {
            "frames": max(0, total - human - automatic - excluded),
        },
        "rejected_or_excluded": {
            "frames": excluded,
            "rule": "count only explicit label_state=excluded with recorded evidence",
        },
    }


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if any(type(item) not in (int, float) for item in value):
        return False
    x1, y1, x2, y2 = value
    return 0 <= x1 < x2 and 0 <= y1 < y2


def _bbox_contains(outer: Sequence[Any], inner: Sequence[Any]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _effective_accepted_by(matrix: Mapping[str, Any]) -> str | None:
    for key in ("accepted_by", "reviewer"):
        value = matrix.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _automatic_acceptance(value: Any) -> bool:
    return isinstance(value, str) and value in AUTOMATIC_ACCEPTORS


def _check_objects(
    objects: Any,
    *,
    label: str,
    errors: list[str],
    warnings: list[str],
) -> int:
    if not isinstance(objects, list):
        errors.append(f"{label}:objects_not_array")
        return 0
    rows = 0
    ids: set[str] = set()
    task_targets = 0
    for index, item in enumerate(objects):
        where = f"{label}:objects[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{where}:not_object")
            continue
        rows += 1
        row_id = item.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            errors.append(f"{where}:row_id_missing")
        elif row_id in ids:
            errors.append(f"{where}:row_id_duplicate:{row_id}")
        else:
            ids.add(row_id)
        for key, allowed in MANUAL_REVIEW_ENUMS.items():
            if key not in item:
                continue
            value = item[key]
            if value is not None and value not in allowed:
                errors.append(f"{where}:{key}_invalid:{value}")
        for key in ("is_task_target",):
            value = item.get(key)
            if value is not None and type(value) is not bool:
                errors.append(f"{where}:{key}_not_boolean")
            if value is True:
                task_targets += 1
        visible_box = item.get("visible_bbox_xyxy")
        full_box = item.get("full_bbox_xyxy")
        if visible_box is not None and not _valid_bbox(visible_box):
            errors.append(f"{where}:visible_bbox_xyxy_invalid")
        if full_box is not None and not _valid_bbox(full_box):
            errors.append(f"{where}:full_bbox_xyxy_invalid")
        if full_box is not None and visible_box is None:
            errors.append(f"{where}:full_bbox_without_visible_bbox")
        if (
            _valid_bbox(visible_box)
            and _valid_bbox(full_box)
            and not _bbox_contains(full_box, visible_box)
        ):
            errors.append(f"{where}:visible_bbox_exceeds_full_bbox")
        box_accepted_by = item.get("box_accepted_by")
        if box_accepted_by is not None and (
            not isinstance(box_accepted_by, str) or not box_accepted_by.strip()
        ):
            errors.append(f"{where}:box_accepted_by_invalid")
        if item.get("annotation_action") == "label_visible_box" and visible_box is None:
            warnings.append(f"{where}:visible_bbox_pending")
        if (
            item.get("annotation_action") in {"hard_negative_no_box", "do_not_label"}
            and visible_box is not None
        ):
            warnings.append(f"{where}:box_present_without_label_action")
        if box_accepted_by and item.get("annotation_action") != "label_visible_box":
            warnings.append(f"{where}:box_acceptance_without_label_action")
        if item.get("category") in {"person_standee_back", "person_standee_nonfront"}:
            if item.get("is_task_target") is not False:
                warnings.append(f"{where}:nonfront_should_not_be_target")
            if item.get("annotation_action") != "hard_negative_no_box":
                warnings.append(f"{where}:nonfront_action_should_be_hard_negative")
        if item.get("is_task_target") is False and item.get("annotation_action") == "label_visible_box":
            errors.append(f"{where}:non_target_cannot_be_labeled")
        if item.get("visibility") in {"partial", "cropped", "occluded"} and item.get("occluder") == "none":
            warnings.append(f"{where}:occlusion_source_missing")
    return task_targets


def _check_status_matrix(
    matrix: Any,
    *,
    label: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(matrix, Mapping):
        errors.append(f"{label}:status_matrix_not_object")
        return
    legacy = [
        key
        for key in matrix
        if str(key).startswith("objects") and str(key) != "objects"
    ]
    if legacy:
        errors.append(f"{label}:legacy_panel_keys:{','.join(map(str, legacy))}")
    if matrix.get("image_role") == "contact_sheet" or "panels" in matrix:
        unknown = set(matrix) - {
            "image_role",
            "panels",
            "reviewer",
            "accepted_by",
            "review_state",
            "notes",
        }
        if unknown:
            errors.append(f"{label}:contact_sheet_unknown_fields:{','.join(sorted(unknown))}")
        panels = matrix.get("panels")
        if not isinstance(panels, list) or not panels:
            errors.append(f"{label}:panels_missing")
            return
        panel_ids: set[str] = set()
        for index, panel in enumerate(panels):
            panel_label = f"{label}:panels[{index}]"
            if not isinstance(panel, Mapping):
                errors.append(f"{panel_label}:not_object")
                continue
            panel_id = panel.get("panel_id")
            if not isinstance(panel_id, str) or not panel_id:
                errors.append(f"{panel_label}:panel_id_missing")
            elif panel_id in panel_ids:
                errors.append(f"{panel_label}:panel_id_duplicate")
            else:
                panel_ids.add(panel_id)
            _check_single_matrix(panel, label=panel_label, errors=errors, warnings=warnings)
        return
    _check_single_matrix(matrix, label=label, errors=errors, warnings=warnings)


def _check_single_matrix(
    matrix: Mapping[str, Any],
    *,
    label: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    allowed = {
        "image_role",
        "panel_id",
        "source_name",
        "presence",
        "task_target_count",
        "background_state",
        "objects",
        "other_objects",
        "reviewer",
        "accepted_by",
        "full_frame_checked",
        "review_state",
        "notes",
    }
    unknown = set(matrix) - allowed
    if unknown:
        errors.append(f"{label}:unknown_fields:{','.join(sorted(unknown))}")
    presence = matrix.get("presence")
    if presence is not None and presence not in MANUAL_REVIEW_ENUMS["presence"]:
        errors.append(f"{label}:presence_invalid:{presence}")
    background = matrix.get("background_state")
    if background is not None and background not in MANUAL_REVIEW_ENUMS["background_state"]:
        errors.append(f"{label}:background_state_invalid:{background}")
    count = matrix.get("task_target_count")
    if count is not None and (type(count) is not int or count < 0):
        errors.append(f"{label}:task_target_count_invalid")
    review_state = matrix.get("review_state")
    if review_state is not None and review_state not in {
        "pending",
        "confirmed",
        "rejected",
        "excluded",
    }:
        errors.append(f"{label}:review_state_invalid:{review_state}")
    for key in ("reviewer", "accepted_by"):
        value = matrix.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"{label}:{key}_invalid")
    objects = matrix.get("objects")
    actual = _check_objects(objects, label=label, errors=errors, warnings=warnings)
    other = matrix.get("other_objects", [])
    _check_objects(other, label=f"{label}:other", errors=errors, warnings=warnings)
    if count is not None and count != actual:
        errors.append(f"{label}:task_target_count_mismatch:{count}!={actual}")
    expected_presence = "none" if actual == 0 else ("single" if actual == 1 else "multi")
    if presence is not None and presence != expected_presence:
        errors.append(f"{label}:presence_count_mismatch:{presence}!={expected_presence}")
    if background is None:
        warnings.append(f"{label}:background_state_unfilled")
    elif actual and background != "contains_target":
        errors.append(f"{label}:background_state_must_be_contains_target")
    elif not actual and background == "contains_target":
        errors.append(f"{label}:background_state_conflicts_with_zero_targets")
    accepted_by = _effective_accepted_by(matrix)
    if accepted_by is None:
        warnings.append(f"{label}:accepted_by_missing")
    if matrix.get("full_frame_checked") is not True:
        warnings.append(f"{label}:full_frame_checked_missing")
    elif accepted_by is None and review_state not in {"rejected", "excluded"}:
        warnings.append(f"{label}:full_frame_checked_without_accepted_by")
    if review_state in {"rejected", "excluded"} and accepted_by is not None:
        warnings.append(f"{label}:rejected_or_excluded_with_accepted_by")


def _review_matrices(review: Mapping[str, Any]):
    """Yield each single frame or contact-sheet panel with stable identifiers."""

    for item in review.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        item_id = item.get("id")
        matrix = item.get("status_matrix")
        if not isinstance(matrix, Mapping):
            continue
        if matrix.get("image_role") == "contact_sheet" or "panels" in matrix:
            for panel in matrix.get("panels") or []:
                if isinstance(panel, Mapping):
                    yield str(item_id), str(panel.get("panel_id") or ""), panel
        else:
            yield str(item_id), None, matrix


def review_has_manual_content(review: Mapping[str, Any]) -> bool:
    """Return whether a review contains decisions or malformed manual additions."""

    standard_matrix = {
        "image_role",
        "panel_id",
        "source_name",
        "presence",
        "task_target_count",
        "background_state",
        "objects",
        "other_objects",
        "reviewer",
        "accepted_by",
        "full_frame_checked",
        "review_state",
        "notes",
        "panels",
    }
    standard_object = set(OBJECT_MATRIX_COLUMNS)
    for _, _, matrix in _review_matrices(review):
        if set(matrix) - standard_matrix:
            return True
        if any(
            matrix.get(key) not in (None, "", [], {}, False, "pending")
            for key in (
                "presence",
                "task_target_count",
                "background_state",
                "reviewer",
                "accepted_by",
                "notes",
            )
        ):
            return True
        if matrix.get("full_frame_checked") is True:
            return True
        for key in ("objects", "other_objects"):
            for row in matrix.get(key) or []:
                if not isinstance(row, Mapping):
                    return True
                if set(row) - standard_object:
                    return True
                if any(value not in (None, "", [], {}) for value in row.values()):
                    return True
    return False


def _merge_preserving(existing: Any, generated: Any) -> Any:
    """Recursively add missing generated fields without replacing existing data."""

    if isinstance(existing, Mapping):
        result = copy.deepcopy(dict(existing))
        if isinstance(generated, Mapping):
            for key, value in generated.items():
                if key not in result:
                    result[key] = copy.deepcopy(value)
                else:
                    result[key] = _merge_preserving(result[key], value)
        return result
    if isinstance(existing, list):
        result = copy.deepcopy(existing)
        if not isinstance(generated, list):
            return result
        key = next(
            (
                candidate
                for candidate in ("id", "row_id", "panel_id")
                if all(
                    isinstance(item, Mapping) and candidate in item
                    for item in [*result, *generated]
                )
            ),
            None,
        )
        if key is not None:
            positions = {
                item.get(key): index
                for index, item in enumerate(result)
                if isinstance(item, Mapping)
            }
            for item in generated:
                if not isinstance(item, Mapping):
                    continue
                identity = item.get(key)
                if identity in positions:
                    index = positions[identity]
                    result[index] = _merge_preserving(result[index], item)
                else:
                    positions[identity] = len(result)
                    result.append(copy.deepcopy(dict(item)))
        return result
    return copy.deepcopy(existing)


def merge_review_preserving_existing(
    existing: Mapping[str, Any],
    generated: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge a regenerated queue while preserving every existing manual value."""

    return _merge_preserving(existing, generated)


def review_status_layers(review: Mapping[str, Any]) -> dict[str, Any]:
    """Count the review layers without collapsing pending into rejection."""

    matrices = list(_review_matrices(review))
    available = len(matrices)
    annotated = 0
    candidate_frames = 0
    candidate_targets = 0
    candidate_boxes = 0
    auto_frames = 0
    auto_boxes = 0
    human_frames = 0
    human_positive_frames = 0
    human_negative_frames = 0
    human_boxes = 0
    rejected = 0
    for _, _, matrix in matrices:
        objects = [
            row
            for key in ("objects", "other_objects")
            for row in (matrix.get(key) or [])
            if isinstance(row, Mapping)
        ]
        if objects or any(
            matrix.get(key) not in (None, "", [], {})
            for key in ("presence", "task_target_count", "background_state", "notes")
        ):
            annotated += 1
        if objects:
            candidate_frames += 1
        for row in objects:
            candidate = any(
                row.get(key) not in (None, "", [], {})
                for key in (
                    "category",
                    "view",
                    "visibility",
                    "occluder",
                    "is_task_target",
                    "annotation_action",
                    "visible_bbox_xyxy",
                    "full_bbox_xyxy",
                    "notes",
                )
            )
            if candidate:
                candidate_targets += 1
            if row.get("annotation_action") == "label_visible_box":
                candidate_boxes += 1
            if (
                row.get("annotation_action") == "label_visible_box"
                and _valid_bbox(row.get("visible_bbox_xyxy"))
                and _automatic_acceptance(row.get("box_accepted_by"))
            ):
                auto_boxes += 1
            if (
                row.get("annotation_action") == "label_visible_box"
                and _valid_bbox(row.get("visible_bbox_xyxy"))
                and isinstance(row.get("box_accepted_by"), str)
                and row.get("box_accepted_by")
                and not _automatic_acceptance(row.get("box_accepted_by"))
            ):
                human_boxes += 1
        accepted_by = _effective_accepted_by(matrix)
        checked = matrix.get("full_frame_checked") is True
        if checked and _automatic_acceptance(accepted_by):
            auto_frames += 1
        elif checked and accepted_by:
            human_frames += 1
            target_count = matrix.get("task_target_count")
            if type(target_count) is not int:
                target_count = sum(
                    row.get("annotation_action") == "label_visible_box"
                    for row in matrix.get("objects") or []
                    if isinstance(row, Mapping)
                )
            if target_count:
                human_positive_frames += 1
            elif matrix.get("background_state") == "confirmed_empty":
                human_negative_frames += 1
        if matrix.get("review_state") in {"rejected", "excluded"}:
            rejected += 1
    terminal = min(available, human_frames + auto_frames + rejected)
    pending_frames = max(0, available - terminal)
    pending_boxes = max(0, candidate_boxes - human_boxes - auto_boxes)
    return {
        "source/available": {
            "frames": available,
            "definition": "review units declared by the current manual package",
        },
        "annotated": {"frames": annotated},
        "candidate": {
            "frames": candidate_frames,
            "instances": candidate_targets,
            "label_action_boxes": candidate_boxes,
        },
        "auto_validated": {
            "frames": auto_frames,
            "box_instances": auto_boxes,
            "automatic_acceptors": sorted(AUTOMATIC_ACCEPTORS),
        },
        "human_confirmed": {
            "frames": human_frames,
            "positive_frames": human_positive_frames,
            "negative_frames": human_negative_frames,
            "box_instances": human_boxes,
            "semantic_and_box_rule": "accepted_by confirms semantics; box_accepted_by must name a human for a confirmed box",
        },
        "training_admitted": {
            "frames": 0,
            "box_instances": 0,
            "status": "not_recorded",
            "reason": "manual review does not grant training admission",
        },
        "pending": {
            "frames": pending_frames,
            "label_action_boxes": pending_boxes,
        },
        "rejected_or_excluded": {
            "frames": rejected,
            "rule": "count only explicit review_state with failure evidence",
        },
    }


def backup_manual_review(
    path: str | Path,
    *,
    root: str | Path | None = None,
) -> Path | None:
    """Back up a review file when it contains manual decisions."""

    target = resolve_workspace(path, root=root)
    if not target.exists():
        return None
    raw = target.read_text(encoding="utf-8")
    has_content = bool(raw.strip())
    if has_content:
        try:
            existing = json.loads(raw)
            has_content = isinstance(existing, Mapping) and review_has_manual_content(existing)
        except json.JSONDecodeError:
            has_content = True
    if not has_content:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(f"{target.stem}.pre_rebuild_{stamp}{target.suffix}")
    suffix = 1
    while backup.exists():
        backup = target.with_name(
            f"{target.stem}.pre_rebuild_{stamp}_{suffix:03d}{target.suffix}"
        )
        suffix += 1
    shutil.copy2(target, backup)
    return backup


def rebuild_review_template(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Explicitly rebuild a review template, backing up edited content first."""

    target = resolve_workspace(path, root=root)
    backup = backup_manual_review(target, root=root)
    atomic_write_json(target, payload, root=root)
    return {
        "written": workspace_relative(target, root=root),
        "backup": workspace_relative(backup, root=root) if backup else None,
    }


def _coerce_box(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"box table value is not JSON: {value!r}") from exc


def _coerce_bool(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return value


def load_review_answers(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    """Load explicit answers from JSON or a simple CSV decision table."""

    source = resolve_workspace(path, root=root, must_exist=True)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("review answers JSON must be an object")
        return payload
    if source.suffix.lower() != ".csv":
        raise ValueError("review answers must be JSON or CSV")
    import csv

    grouped: dict[tuple[str, str | None], dict[str, Any]] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            item_id = str(raw.get("item_id") or "").strip()
            if not item_id:
                continue
            panel_id = str(raw.get("panel_id") or "").strip() or None
            key = (item_id, panel_id)
            answer = grouped.setdefault(
                key,
                {
                    "id": item_id,
                    **({"panel_id": panel_id} if panel_id else {}),
                    "objects": [],
                },
            )
            for field in (
                "presence",
                "background_state",
                "reviewer",
                "accepted_by",
                "notes",
                "review_state",
            ):
                value = str(raw.get(field) or "").strip()
                if value:
                    answer[field] = value
            count = str(raw.get("task_target_count") or "").strip()
            if count:
                answer["task_target_count"] = int(count)
            checked = _coerce_bool(str(raw.get("full_frame_checked") or ""))
            if checked is not None:
                answer["full_frame_checked"] = checked
            row_id = str(raw.get("row_id") or "").strip()
            if row_id:
                row = {"row_id": row_id}
                for field in (
                    "category",
                    "view",
                    "visibility",
                    "occluder",
                    "annotation_action",
                    "box_accepted_by",
                    "notes",
                ):
                    value = str(raw.get(field) or "").strip()
                    if value:
                        row[field] = value
                target = _coerce_bool(str(raw.get("is_task_target") or ""))
                if target is not None:
                    row["is_task_target"] = target
                for field in ("visible_bbox_xyxy", "full_bbox_xyxy"):
                    value = _coerce_box(raw.get(field))
                    if value is not None:
                        row[field] = value
                answer["objects"].append(row)
    return {
        "items": [value for _, value in sorted(grouped.items())],
        "input_format": "csv",
    }


def normalize_review_answers(
    review: Mapping[str, Any],
    answers: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply explicit decisions to a review while retaining candidate rows."""

    result = copy.deepcopy(dict(review))
    index = {
        (item_id, panel_id): matrix
        for item_id, panel_id, matrix in _review_matrices(result)
    }
    default_reviewer = answers.get("accepted_by") or answers.get("reviewer")
    frame_fields = {
        "presence",
        "task_target_count",
        "background_state",
        "reviewer",
        "accepted_by",
        "full_frame_checked",
        "review_state",
        "notes",
    }
    object_fields = set(OBJECT_MATRIX_COLUMNS) - {"row_id"}
    for answer in answers.get("items") or []:
        if not isinstance(answer, Mapping):
            raise ValueError("review answer items must be objects")
        item_id = str(answer.get("id") or "")
        panel_id = str(answer.get("panel_id") or "") or None
        matrix = index.get((item_id, panel_id))
        if matrix is None:
            raise ValueError(f"review answer target not found: {item_id}/{panel_id}")
        for field in frame_fields:
            if field in answer:
                matrix[field] = copy.deepcopy(answer[field])
        if default_reviewer and not matrix.get("reviewer") and not matrix.get("accepted_by"):
            matrix["reviewer"] = default_reviewer
            matrix["accepted_by"] = default_reviewer
        if matrix.get("reviewer") and not matrix.get("accepted_by"):
            matrix["accepted_by"] = matrix["reviewer"]
        for collection in ("objects", "other_objects"):
            incoming = answer.get(collection)
            if not isinstance(incoming, list):
                continue
            rows = matrix.setdefault(collection, [])
            by_id = {
                row.get("row_id"): row
                for row in rows
                if isinstance(row, dict)
            }
            for raw_row in incoming:
                if not isinstance(raw_row, Mapping):
                    raise ValueError("review answer object rows must be objects")
                row_id = str(raw_row.get("row_id") or "")
                if not row_id:
                    raise ValueError("review answer object row requires row_id")
                row = by_id.get(row_id)
                if row is None:
                    row = {"row_id": row_id}
                    rows.append(row)
                    by_id[row_id] = row
                for field, value in raw_row.items():
                    if field == "row_id":
                        continue
                    if field not in object_fields:
                        raise ValueError(f"unsupported object field: {field}")
                    row[field] = copy.deepcopy(value)
        targets = [
            row
            for row in matrix.get("objects") or []
            if isinstance(row, Mapping) and row.get("annotation_action") == "label_visible_box"
        ]
        target_count = matrix.get("task_target_count")
        if target_count is None and matrix.get("full_frame_checked") is True:
            target_count = len(targets)
            matrix["task_target_count"] = target_count
        if matrix.get("full_frame_checked") is True and type(target_count) is int:
            matrix["presence"] = "none" if target_count == 0 else (
                "single" if target_count == 1 else "multi"
            )
            if matrix.get("background_state") is None:
                matrix["background_state"] = (
                    "confirmed_empty" if target_count == 0 else "contains_target"
                )
        if matrix.get("full_frame_checked") is True and _effective_accepted_by(matrix):
            if matrix.get("review_state") in (None, "pending"):
                matrix["review_state"] = "confirmed"
    return result


def label_manifest_from_review(review: Mapping[str, Any]) -> dict[str, Any]:
    """Translate single-frame review decisions into the explicit label-manifest schema."""

    class_names = review.get("class_names")
    if class_names is None:
        candidates = [
            item.get("class_names")
            for item in review.get("items") or []
            if isinstance(item, Mapping) and isinstance(item.get("class_names"), list)
        ]
        if candidates:
            class_names = candidates[0]
            if any(value != class_names for value in candidates[1:]):
                raise ValueError("manual review items disagree on class_names")
    if (
        not isinstance(class_names, list)
        or not class_names
        or any(not isinstance(name, str) or not name for name in class_names)
    ):
        raise ValueError("manual review requires class_names before label export")
    class_ids = {name: index for index, name in enumerate(class_names)}
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for item in review.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        item_id = str(item.get("id") or "")
        matrix = item.get("status_matrix")
        if not isinstance(matrix, Mapping):
            continue
        if matrix.get("image_role") == "contact_sheet" or "panels" in matrix:
            skipped.append(
                {
                    "id": item_id,
                    "reason": "contact_sheet_requires_panel_geometry",
                }
            )
            continue
        target_rows = [
            row
            for row in matrix.get("objects") or []
            if isinstance(row, Mapping)
            and row.get("annotation_action") == "label_visible_box"
        ]
        valid_rows = [
            row for row in target_rows if _valid_bbox(row.get("visible_bbox_xyxy"))
        ]
        semantic_owner = _effective_accepted_by(matrix)
        full_frame_checked = matrix.get("full_frame_checked") is True
        box_owner = next(
            (
                str(row.get("box_accepted_by"))
                for row in valid_rows
                if isinstance(row.get("box_accepted_by"), str)
                and row.get("box_accepted_by")
                and not _automatic_acceptance(row.get("box_accepted_by"))
            ),
            None,
        )
        all_boxes_human_confirmed = bool(valid_rows) and all(
            isinstance(row.get("box_accepted_by"), str)
            and row.get("box_accepted_by")
            and not _automatic_acceptance(row.get("box_accepted_by"))
            for row in valid_rows
        )
        if matrix.get("review_state") in {"rejected", "excluded"}:
            label_state = "excluded"
        elif (
            full_frame_checked
            and matrix.get("background_state") == "confirmed_empty"
            and not target_rows
        ):
            label_state = "confirmed_negative"
        elif full_frame_checked and valid_rows and len(valid_rows) == len(target_rows):
            label_state = "complete"
        elif valid_rows:
            label_state = "partial"
        else:
            label_state = "unknown"
        instances = []
        if label_state in {"complete", "partial"}:
            for index, row in enumerate(valid_rows, start=1):
                category = str(row.get("category") or "")
                class_name = (
                    category[:-7]
                    if category.endswith("_person")
                    else category
                )
                if class_name not in class_ids:
                    raise ValueError(
                        f"{item_id}/{row.get('row_id')}: category {category!r} "
                        "does not map to class_names"
                    )
                instances.append(
                    {
                        "id": str(row.get("row_id") or f"instance_{index}"),
                        "class_id": class_ids[class_name],
                        "visibility": row.get("visibility") or "unknown",
                        "visible_bbox_xyxy": copy.deepcopy(row["visible_bbox_xyxy"]),
                        "full_bbox_xyxy": copy.deepcopy(row.get("full_bbox_xyxy")),
                        "box_accepted_by": row.get("box_accepted_by"),
                    }
                )
        record = {
            "id": item_id,
            "image": item.get("image"),
            "source_kind": item.get("source_kind"),
            "class_names": copy.deepcopy(class_names),
            "label_state": label_state,
            "parent_sources": copy.deepcopy(item.get("parent_sources") or []),
            "review": {
                "full_frame_checked": full_frame_checked,
                "accepted_by": semantic_owner,
                "box_accepted_by": (
                    box_owner
                    if all_boxes_human_confirmed
                    else None
                ),
                "box_acceptance": "human" if all_boxes_human_confirmed else "candidate",
                "method": "manual_full_frame_review" if semantic_owner and not _automatic_acceptance(semantic_owner)
                else "recorded_full_frame_assessment",
                "evidence": copy.deepcopy(item.get("evidence") or {}),
            },
            "instances": instances,
        }
        if label_state == "partial":
            record["partial_reason"] = "frame_scope_or_box_geometry_incomplete"
        records.append(record)
    return {
        "schema_version": SCHEMA_VERSION,
        "task": review.get("task"),
        "batch": review.get("batch"),
        "records": records,
        "skipped": skipped,
        "boundary": (
            "Frame coverage and box geometry determine label completeness. "
            "Reviewer fields record actual human acceptance."
        ),
    }


def validate_manual_review(review: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a filled ``manual_review.json`` without changing its content."""

    items = review.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("manual review requires a nonempty items list")
    errors: list[str] = []
    warnings: list[str] = []
    ids: set[str] = set()
    summaries = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            errors.append(f"items[{index}]:not_object")
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            errors.append(f"items[{index}]:id_missing")
            label = f"items[{index}]"
        else:
            label = item_id
            if item_id in ids:
                errors.append(f"{label}:id_duplicate")
            ids.add(item_id)
        matrix = item.get("status_matrix")
        _check_status_matrix(matrix, label=label, errors=errors, warnings=warnings)
        if isinstance(matrix, Mapping):
            if matrix.get("image_role") == "contact_sheet" or "panels" in matrix:
                summaries.append(
                    {
                        "id": item_id,
                        "image_role": "contact_sheet",
                        "panels": len(matrix.get("panels") or []),
                    }
                )
            else:
                summaries.append(
                    {
                        "id": item_id,
                        "image_role": "single_frame",
                        "task_targets": sum(
                            1
                            for row in matrix.get("objects") or []
                            if isinstance(row, Mapping) and row.get("is_task_target") is True
                        ),
                    }
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_local(),
        "kind": "manual_review_validation",
        "task": review.get("task"),
        "batch": review.get("batch"),
        "status": "fail" if errors else ("review" if warnings else "pass"),
        "errors": errors,
        "warnings": warnings,
        "items": summaries,
        "status_layers": review_status_layers(review),
    }


def review_acceptance_document(
    review: Mapping[str, Any],
    *,
    source_review: str | Path,
    validation: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Create a tracked decision record from a validated manual review."""

    base = Path(root or Path(__file__).resolve().parents[3]).resolve()
    source = resolve_workspace(source_review, root=base, must_exist=True)
    checked = dict(validation or validate_manual_review(review))
    blocking_warnings = [
        warning
        for warning in checked.get("warnings") or []
        if not str(warning).endswith(":visible_bbox_pending")
    ]
    if checked.get("errors") or blocking_warnings:
        raise ValueError(
            "manual review must pass semantic validation before acceptance"
        )
    items = review.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("manual review contains no items")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_local(),
        "kind": "manual_review_acceptance",
        "task": review.get("task"),
        "batch": review.get("batch"),
        "status": "accepted",
        "source_review": workspace_relative(source, root=base),
        "source_sha256": digest,
        "source_digest_scope": "SHA-256 of the accepted source-review bytes at this acceptance time.",
        "source_normalization": copy.deepcopy(review.get("source_normalization")),
        "recovered_from": copy.deepcopy(review.get("recovered_from")),
        "object_matrix_columns": OBJECT_MATRIX_COLUMNS,
        "items": items,
        "status_layers": review_status_layers(review),
        "boundary": (
            "This records human semantic review only. Detector box coordinates, masks, "
            "raw Blender render status and training eligibility remain separate fields. "
            "accepted_by records semantic confirmation; box_accepted_by records box "
            "confirmation; programmatic values do not populate the human layer."
        ),
    }


def write_review_template_if_missing(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> bool:
    """Create a review template without ever replacing an edited review."""

    target = resolve_workspace(path, root=root)
    if target.exists():
        return False
    atomic_write_json(target, payload, root=root)
    return True
