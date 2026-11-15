"""Discover and summarise background candidates without upgrading uncertainty.

Discovery reads the rebuildable SQLite media cache and returns candidate paths
only. Semantic state comes from the existing cross-task review selection. Split
membership is read from each task's authoritative ``source.yaml`` pairing block.
No image content is changed and no automatic ``confirmed_empty`` decision exists.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from datasets.scripts.common.review.evidence import read_json, read_metrics

from .contracts import (
    BACKGROUND_STATES,
    REVIEW_VERDICTS,
    SCHEMA_VERSION,
    atomic_write_json,
    code_digest,
    now_local,
    require_string,
    resolve_workspace,
    workspace_relative,
)


def _media_item(row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
    raw = row[0]
    item = json.loads(raw)
    if not isinstance(item, dict):
        raise ValueError("media cache item is not an object")
    return item


def discover_media_candidates(
    database: str | Path,
    *,
    contains: str | None = None,
    media_type: str | None = "image",
    limit: int = 200,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Read image candidates from the synchronized media cache.

    The cache is a discovery view, not a semantic source gate. The returned
    snapshot metadata and candidate observations must be carried into review.
    """

    path = resolve_workspace(database, root=root, must_exist=True)
    count = int(limit)
    if count <= 0:
        raise ValueError("limit must be positive")
    uri = path.as_uri() + "?mode=ro"
    db = sqlite3.connect(uri, uri=True, timeout=10)
    try:
        meta_rows = db.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
        meta = {str(key): value for key, value in meta_rows}
        rows = db.execute(
            "SELECT item_json FROM media ORDER BY path_key, position"
        ).fetchall()
    finally:
        db.close()

    needle = contains.casefold() if contains else None
    candidates: list[dict[str, Any]] = []
    for row in rows:
        item = _media_item(row)
        item_path = str(item.get("path") or "")
        if not item_path:
            continue
        if media_type and item.get("media_type") != media_type:
            continue
        if needle and needle not in item_path.casefold():
            continue
        candidates.append(
            {
                "path": item_path,
                "media_type": item.get("media_type"),
                "bytes": item.get("bytes"),
                "image_size": item.get("image_size"),
                "evidence": list(item.get("evidence") or []),
                "recorded_claims": dict(item.get("recorded_claims") or {}),
                "review_state": "unknown",
                "semantic_review": "pending",
            }
        )
        if len(candidates) >= count:
            break
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_local(),
        "kind": "background_candidate_discovery",
        "database": workspace_relative(path, root=root),
        "database_meta": meta,
        "filters": {
            "contains": contains,
            "media_type": media_type,
            "limit": count,
        },
        "candidates": candidates,
        "summary": {
            "returned": len(candidates),
            "media_types": dict(Counter(str(item["media_type"]) for item in candidates)),
            "semantic_state": "all_unknown_until_full_frame_review",
        },
        "boundary": (
            "Cache discovery does not prove target absence, parent identity, split "
            "ownership, duplicate status, or human acceptance."
        ),
    }


def _selection_records(selection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in selection.get("records") or []:
        if not isinstance(record, Mapping):
            continue
        frame = record.get("frame") or {}
        path = frame.get("path") if isinstance(frame, Mapping) else None
        if isinstance(path, str) and path:
            result[path] = dict(record)
    return result


def _metric_records(metrics: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not metrics:
        return {}
    rows = metrics.get("records") or []
    result: dict[str, dict[str, Any]] = {}
    for record in rows:
        if not isinstance(record, Mapping):
            continue
        frame = record.get("frame") or {}
        path = frame.get("path") if isinstance(frame, Mapping) else None
        if isinstance(path, str) and path:
            result[path] = dict(record)
    return result


def _background_state(record: Mapping[str, Any] | None) -> tuple[str, list[str]]:
    """Map one review verdict to a conservative background state."""

    if record is None:
        return "unknown", ["review_missing"]
    verdict = str(record.get("verdict") or "unknown")
    if verdict not in REVIEW_VERDICTS:
        return "unknown", ["review_verdict_invalid"]
    reasons = {str(code) for code in record.get("reason_codes") or []}
    existing = list(record.get("existing_targets") or [])
    if verdict == "excluded":
        if "unlabelled_target_suspected" in reasons or existing:
            return "contains_target", sorted(reasons)
        return "rejected", sorted(reasons)
    if verdict == "unknown":
        return "unknown", sorted(reasons)
    if existing or "unlabelled_target_suspected" in reasons:
        return "contains_target", sorted(reasons)
    if record.get("accepted_by") and "no_unlabelled_target" in reasons:
        return "confirmed_empty", sorted(reasons)
    return "candidate_empty", sorted(reasons or {"human_acceptance_missing"})


def _task_source_groups(root: Path, task: str) -> list[dict[str, Any]]:
    source_path = root / "datasets" / "scripts" / task / "config" / "source.yaml"
    if not source_path.is_file():
        return []
    document = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    pairing = document.get("pairing") if isinstance(document, Mapping) else None
    if not isinstance(pairing, Mapping):
        return []
    groups: list[dict[str, Any]] = []
    for key in ("task_real", "backgrounds"):
        for group in pairing.get(key) or []:
            if isinstance(group, Mapping):
                item = dict(group)
                item["_section"] = key
                groups.append(item)
    return groups


def _declared_split(
    groups: Sequence[Mapping[str, Any]],
    *,
    candidate_paths: Sequence[str],
    role: str | None,
) -> dict[str, Any]:
    """Find the unique containing parent group and report unresolved ambiguity."""

    matches: list[Mapping[str, Any]] = []
    for group in groups:
        roles = [str(value) for value in group.get("roles") or []]
        if role == "background" and group.get("class") != "background":
            continue
        if role and role != "background" and role not in roles:
            continue
        group_paths = [str(value).replace("\\", "/") for value in group.get("paths") or []]
        for candidate in candidate_paths:
            normalized = candidate.replace("\\", "/")
            if any(normalized == root or normalized.startswith(root.rstrip("/") + "/") for root in group_paths):
                matches.append(group)
                break
    if not matches:
        return {"status": "unresolved", "reason": "no_declared_parent_group"}
    if len(matches) > 1:
        return {
            "status": "unresolved",
            "reason": "ambiguous_parent_group",
            "groups": [group.get("id") for group in matches],
        }
    group = matches[0]
    split = group.get("split") if isinstance(group.get("split"), Mapping) else {}
    return {
        "status": "declared",
        "group_id": group.get("id"),
        "class": group.get("class"),
        "kind": group.get("kind"),
        "primary": split.get("primary"),
        "same_source_allowed_in": list(split.get("same_source_allowed_in") or []),
        "reservation_status": split.get("status"),
        "review_state": (group.get("review") or {}).get("state")
        if isinstance(group.get("review"), Mapping)
        else None,
        "labels_status": (group.get("labels") or {}).get("status")
        if isinstance(group.get("labels"), Mapping)
        else None,
    }


def summarize_review(
    *,
    task: str,
    batch: str,
    review_plan: str | Path,
    review_selection: str | Path,
    metrics: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Merge review artifacts with declared parent/split membership."""

    base = Path(root).resolve() if root else None
    task_name = require_string({"task": task}, "task")
    batch_name = require_string({"batch": batch}, "batch")
    plan_path = resolve_workspace(review_plan, root=base, must_exist=True)
    selection_path = resolve_workspace(review_selection, root=base, must_exist=True)
    plan = read_json(plan_path)
    selection = read_json(selection_path)
    metric_path = resolve_workspace(metrics, root=base, must_exist=True) if metrics else None
    if metric_path and metric_path.suffix.lower() in {".jsonl", ".ndjson"}:
        header, rows = read_metrics(metric_path)
        metric_doc = {"header": header, "records": rows} if metric_path else None
    else:
        metric_doc = read_json(metric_path) if metric_path else None
    if plan.get("task") and plan.get("task") != task_name:
        raise ValueError("review plan task does not match --task")
    if selection.get("task") and selection.get("task") != task_name:
        raise ValueError("review selection task does not match --task")

    selected = _selection_records(selection)
    measured = _metric_records(metric_doc)
    groups = _task_source_groups(Path(root).resolve(), task_name) if root else _task_source_groups(
        Path(__file__).resolve().parents[3], task_name
    )
    records: list[dict[str, Any]] = []
    for index, frame in enumerate(plan.get("frames") or []):
        if not isinstance(frame, Mapping):
            continue
        path = str(frame.get("path") or "")
        if not path:
            continue
        review = selected.get(path)
        state, reasons = _background_state(review)
        if state not in BACKGROUND_STATES:
            raise AssertionError(state)
        role = str(frame.get("role") or "background")
        parent_paths = [
            value
            for value in (
                path,
                frame.get("parent_video"),
                frame.get("parent_image"),
                frame.get("source_group"),
            )
            if isinstance(value, str) and value
        ]
        split = _declared_split(groups, candidate_paths=parent_paths, role=role)
        metric = measured.get(path)
        records.append(
            {
                "id": f"{batch_name}:{index:04d}",
                "path": path,
                "parent_video": frame.get("parent_video"),
                "parent_image": frame.get("parent_image"),
                "source_group": frame.get("source_group"),
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "role": role,
                "review_state": state,
                "review_reasons": reasons,
                "review_verdict": None if review is None else review.get("verdict"),
                "accepted_by": None if review is None else review.get("accepted_by"),
                "human_accepted": bool(review and review.get("accepted_by")),
                "near_duplicate_group": None
                if review is None
                else review.get("near_duplicate_group"),
                "existing_targets": [] if review is None else list(review.get("existing_targets") or []),
                "split_reservation": split,
                "evidence": {
                    "review_plan": workspace_relative(plan_path, root=base),
                    "review_selection": workspace_relative(selection_path, root=base),
                    "metrics": workspace_relative(metric_path, root=base) if metric_path else None,
                    "review_record_index": index,
                    "metric_present": metric is not None,
                },
            }
        )

    state_counts = Counter(item["review_state"] for item in records)
    split_counts = Counter(
        str(item["split_reservation"].get("primary") or "unresolved") for item in records
    )
    group_counts = Counter(str(item.get("source_group") or item.get("parent_video") or "unknown") for item in records)
    implementation_files = sorted(
        str(path.relative_to(Path(__file__).resolve().parents[3])).replace("\\", "/")
        for path in (Path(__file__).resolve().parent).glob("*.py")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_local(),
        "kind": "background_pool_review",
        "task": task_name,
        "batch": batch_name,
        "authority": {
            "semantic_review": workspace_relative(selection_path, root=base),
            "parent_and_split": f"datasets/scripts/{task_name}/config/source.yaml#pairing",
            "discovery_cache": "datasets/annotated/media_index/query_cache.sqlite3",
        },
        "records": records,
        "summary": {
            "total": len(records),
            "by_review_state": dict(sorted(state_counts.items())),
            "by_parent_group": dict(sorted(group_counts.items())),
            "by_declared_split": dict(sorted(split_counts.items())),
            "human_confirmed_empty": state_counts["confirmed_empty"],
            "candidate_empty_not_negative": state_counts["candidate_empty"],
            "unknown_not_negative": state_counts["unknown"],
        },
        "code_hash": code_digest(implementation_files),
        "boundary": (
            "Only an explicit accepted_by on a ready full-frame review becomes "
            "confirmed_empty. candidate_empty and unknown are not negatives."
        ),
    }


def write_background_report(
    report: Mapping[str, Any],
    output: str | Path,
    *,
    root: str | Path | None = None,
) -> Path:
    """Write a background review report to its declared observation location."""

    state = (report.get("summary") or {}).get("by_review_state") or {}
    if any(value not in BACKGROUND_STATES for value in state):
        raise ValueError("background report contains an unknown review state")
    return atomic_write_json(output, report, root=root)
