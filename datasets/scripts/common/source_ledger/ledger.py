"""Build and check the cross-task background/source ledger.

The ledger is a derived view: task ``source.yaml`` pairing is the authoring
record for parent groups and split reservations, while batch manifests, the
shared background pool and the Blender registry remain their own authorities.
Checks here cover declaration consistency, evidence reachability, review-state
role gates, parent-group split accounting and cross-task reuse. They do not
prove image semantics, label truth, independent capture or model quality.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..paths import WORKSPACE
from . import config as cfg
from .evidence import manifest_records, path_exists, validate_evidence
from .schema import (
    ACTIONS,
    BLENDER_PRIORITIES,
    CAPTURE_SESSIONS,
    CAPTURE_STATUSES,
    COMPOSITE_BACKGROUND_STATES,
    DATA_SOURCE_KINDS,
    GROUP_CLASSES,
    LABEL_STATES,
    PARENT_CLASSES,
    PARENT_KINDS,
    PAIRING_STATUSES,
    REVIEW_STATES,
    ROLES,
    SPLIT_STATUSES,
    SPLITS,
    DataSource,
    Issue,
    ParentGroup,
)

QUALIFIED_SPLITS = frozenset({"train", "val", "test"})
CAPABLE_TARGET_LABELS = frozenset({"complete", "partial", "ready"})
DEFAULT_TRAIN_CAP = 0.8


def build_ledger(
    root: str | Path | None = None,
    tasks: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build one derived ledger document for the requested tasks."""

    root = Path(root or WORKSPACE).resolve()
    task_names = tuple(tasks) if tasks is not None else cfg.DEFAULT_TASKS
    issues: list[Issue] = []
    entries: dict[str, dict[str, Any]] = {}
    group_objects: dict[str, list[ParentGroup]] = {}
    for task in task_names:
        entry, groups = _task_entry(root, task, issues)
        entries[task] = entry
        group_objects[task] = groups
    shared = _build_shared(root, entries, group_objects, issues)
    _check_cross_task(entries, group_objects, shared, issues)
    summary = _summarize(entries, shared, issues)
    return {
        "schema_version": 1,
        "generated_at": _now(),
        "root": root.as_posix(),
        "tasks": entries,
        "shared": shared,
        "issues": [issue.to_json() for issue in issues],
        "summary": summary,
        "authority": {
            "parent_groups": "datasets/scripts/<task>/config/source.yaml pairing",
            "background_manifests": "batch manifests referenced by task configuration and pairing evidence",
            "shared_background_pool": shared["background_pool"]["manifest"],
            "blender_registry": cfg.BLENDER_REGISTRY,
            "note": "本报告由上述记录重算；不是新的来源权威，也不证明画面语义、标签真值或独立采集。",
        },
    }


def background_candidates(
    ledger: Mapping[str, Any], task: str, role: str, split: str
) -> dict[str, Any]:
    """Return eligible and excluded parent groups for one task/role/split."""

    entry = ledger.get("tasks", {}).get(task)
    if not isinstance(entry, Mapping):
        return {"task": task, "role": role, "split": split, "adoption": "missing", "eligible": [], "excluded": [], "reason": "task_missing"}
    adoption = entry.get("adoption")
    if adoption != "pairing":
        return {
            "task": task,
            "role": role,
            "split": split,
            "adoption": adoption,
            "eligible": [],
            "excluded": [],
            "reason": "pairing_not_adopted",
        }
    eligible, excluded = [], []
    for group in entry.get("groups", []):
        if group.get("class") != "background":
            continue
        ok, reason = background_eligibility(group, role, split)
        item = {
            "id": group.get("id"),
            "paths": list(group.get("paths") or []),
            "review": group.get("review"),
            "split": group.get("split"),
        }
        if ok:
            if reason == "unassigned_source":
                item["reason"] = reason
            eligible.append(item)
        else:
            item["reason"] = reason
            excluded.append(item)
    return {
        "task": task,
        "role": role,
        "split": split,
        "adoption": adoption,
        "eligible": eligible,
        "excluded": excluded,
        "reason": "ok" if eligible else "no_eligible_group",
    }


def background_eligibility(
    group: Mapping[str, Any], role: str, split: str
) -> tuple[bool, str]:
    """Decide whether a declared background may serve one role in one split."""

    if group.get("class") != "background":
        return False, "not_background"
    roles = group.get("roles") or []
    if role not in roles:
        return False, "role_not_declared"
    split_info = group.get("split") or {}
    primary = split_info.get("primary")
    allowed = {primary} if primary else set()
    allowed.update(split_info.get("same_source_allowed_in") or [])
    pending_assignment = primary == "unassigned" and split in QUALIFIED_SPLITS
    if split not in allowed and not pending_assignment:
        return False, "split_not_assigned"
    labels = group.get("labels") or {}
    label_status = labels.get("status")
    if label_status in {"excluded", "blocked"}:
        return False, f"labels_{label_status}"
    review = group.get("review") or {}
    state = review.get("state")
    if state == "rejected":
        return False, "review_rejected"
    if role == "empty_negative":
        if state != "confirmed_empty":
            return False, f"review_state_{state}"
    elif role == "composite_background":
        if state not in COMPOSITE_BACKGROUND_STATES:
            return False, f"review_state_{state}"
    elif state not in REVIEW_STATES or state == "rejected":
        return False, f"review_state_{state}"
    labels = group.get("labels") or {}
    if state == "contains_target" and labels.get("status") not in CAPABLE_TARGET_LABELS:
        return False, "target_labels_unknown"
    return True, "unassigned_source" if pending_assignment else "ok"

def _now() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _issue(
    issues: list[Issue],
    code: str,
    level: str,
    message: str,
    task: str | None = None,
    group: str | None = None,
) -> None:
    issues.append(Issue(code, level, message, task=task, group=group))


def _task_entry(
    root: Path, task: str, issues: list[Issue]
) -> tuple[dict[str, Any], list[ParentGroup]]:
    config_path = cfg.source_config_path(root, task)
    entry: dict[str, Any] = {
        "task": task,
        "source_config": config_path.relative_to(root).as_posix(),
        "adoption": "missing",
        "pairing_status": None,
        "groups": [],
        "data_sources": [],
        "captures": [],
        "observed": [],
        "selections": [],
        "blender": [],
        "shared_pool": [],
        "plan_bindings": {},
        "train_cap": None,
    }
    if not config_path.is_file():
        _issue(issues, "BSL030", "error", "缺少 source.yaml", task=task)
        return entry, []
    try:
        source = cfg.load_source_config(root, task)
    except (OSError, ValueError) as error:
        _issue(issues, "BSL030", "error", f"source.yaml 读取失败: {error}", task=task)
        return entry, []
    observed = cfg.collect_observed_references(source)
    selections = cfg.selected_backgrounds(source)
    pairing = cfg.pairing_block(source)
    groups: list[ParentGroup] = []
    data_sources: list[DataSource] = []
    captures: list[dict[str, Any]] = []
    if pairing is not None:
        entry["adoption"] = "pairing"
        entry["pairing_status"] = pairing.get("status")
        _validate_pairing_structure(task, pairing, issues)
        groups = cfg.build_groups(task, pairing)
        data_sources = cfg.build_data_sources(task, pairing)
        captures = cfg.build_captures(pairing)
        _validate_groups(root, task, groups, issues)
        _validate_data_sources(root, task, groups, data_sources, issues)
        entry["train_cap"] = _validate_cap(task, groups, pairing, issues)
        _check_capture_placeholders(task, groups, captures, issues)
        _check_plan_bindings(root, task, pairing, issues)
        entry["plan_bindings"] = pairing.get("plan_bindings") or {}
    else:
        entry["adoption"] = "legacy"
        _issue(
            issues,
            "BSL018",
            "warning",
            "source.yaml 未提供 pairing；本任务按 legacy 观察记录，不参与父组与分集核对",
            task=task,
        )
    _check_legacy_references(root, task, entry["adoption"], observed, selections, issues)
    entry["observed"] = observed
    entry["selections"] = selections
    entry["groups"] = [group.to_json() for group in groups]
    entry["data_sources"] = [item.to_json() for item in data_sources]
    entry["captures"] = captures
    entry["blender"] = cfg.collect_blender_references(task, groups, observed)
    entry["shared_pool"] = cfg.shared_pool_references(groups, observed, selections)
    _check_shared_pool_usage(task, entry, issues)
    return entry, groups


def _validate_pairing_structure(
    task: str, pairing: Mapping[str, Any], issues: list[Issue]
) -> None:
    if pairing.get("schema_version") != 1:
        _issue(issues, "BSL001", "error", "pairing.schema_version 必须为 1", task=task)
    if pairing.get("status") not in PAIRING_STATUSES:
        _issue(issues, "BSL001", "error", f"pairing.status 无效: {pairing.get('status')!r}", task=task)
    policy = pairing.get("policy")
    if not isinstance(policy, Mapping):
        _issue(issues, "BSL001", "error", "pairing.policy 缺失或不是映射", task=task)
    else:
        parent_classes = policy.get("parent_classes")
        if not isinstance(parent_classes, list) or not parent_classes or any(
            value not in PARENT_CLASSES for value in parent_classes
        ):
            _issue(issues, "BSL001", "error", "policy.parent_classes 必须是非空的 task_real/background 列表", task=task)
        if policy.get("split_unit") != "parent_group":
            _issue(issues, "BSL001", "error", "policy.split_unit 必须为 parent_group", task=task)
        cap = policy.get("train_parent_cap", DEFAULT_TRAIN_CAP)
        if not isinstance(cap, (int, float)) or isinstance(cap, bool) or not 0 < cap <= 1:
            _issue(issues, "BSL001", "error", "policy.train_parent_cap 必须在 (0, 1] 内", task=task)
        if policy.get("val_test_mode") not in ("reserve_only", "assigned_sources_with_split_specific_foregrounds"):
            _issue(issues, "BSL001", "error", f"policy.val_test_mode 无效: {policy.get('val_test_mode')!r}", task=task)
        if policy.get("derived_inherits_parent_split") is not True:
            _issue(issues, "BSL001", "error", "policy.derived_inherits_parent_split 必须为 true", task=task)
        for key in ("cross_task_duplicates", "same_source_splits"):
            value = policy.get(key)
            if not isinstance(value, str) or not value.strip():
                _issue(issues, "BSL001", "error", f"policy.{key} 必须是非空字符串", task=task)
    composition = pairing.get("composition_policy")
    if not isinstance(composition, Mapping):
        _issue(issues, "BSL001", "error", "pairing.composition_policy 缺失或不是映射", task=task)
    else:
        if not isinstance(composition.get("target_source"), str) or not composition["target_source"].strip():
            _issue(issues, "BSL001", "error", "composition_policy.target_source 必须是非空字符串", task=task)
        if composition.get("blender_priority") not in BLENDER_PRIORITIES:
            _issue(issues, "BSL001", "error", "composition_policy.blender_priority 无效", task=task)
        for key in ("target_components", "background_components"):
            values = composition.get(key)
            if not isinstance(values, list) or not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                _issue(issues, "BSL001", "error", f"composition_policy.{key} 必须是非空字符串列表", task=task)
        share = composition.get("blender_share_target")
        if share is not None and (
            not isinstance(share, (int, float)) or isinstance(share, bool) or not 0 <= share <= 1
        ):
            _issue(issues, "BSL001", "error", "composition_policy.blender_share_target 必须在 [0, 1] 内", task=task)
    for key in ("task_real", "backgrounds", "data_sources", "blender", "captures"):
        if not isinstance(pairing.get(key, []), list):
            _issue(issues, "BSL001", "error", f"pairing.{key} 必须是列表", task=task)
    if isinstance(policy, Mapping) and policy.get("val_test_mode") == "reserve_only":
        for path in _count_keys(pairing):
            _issue(
                issues,
                "BSL014",
                "error",
                f"reserve_only 下不得声明数量、预算或配额字段: {path}",
                task=task,
            )
    bindings = pairing.get("plan_bindings")
    if bindings is not None:
        if not isinstance(bindings, Mapping):
            _issue(issues, "BSL001", "error", "pairing.plan_bindings 必须是映射", task=task)
        else:
            for split, values in bindings.items():
                if split not in SPLITS:
                    _issue(issues, "BSL001", "error", f"plan_bindings 的 split 无效: {split}", task=task)
                if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                    _issue(issues, "BSL001", "error", f"plan_bindings.{split} 必须是路径列表", task=task)


def _count_keys(value: Any, path: tuple[str, ...] = ()) -> Iterable[str]:
    """Yield paths of budget-like keys under a ``reserve_only`` pairing block."""

    keys = {"count", "counts", "quota", "quotas", "budget", "target_count"}
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_path = path + (str(key),)
            if str(key).casefold() in keys:
                yield ".".join(key_path)
            yield from _count_keys(item, key_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _count_keys(item, path + (str(index),))


def _validate_groups(
    root: Path, task: str, groups: list[ParentGroup], issues: list[Issue]
) -> None:
    seen: set[str] = set()
    for group in groups:
        planned = group.split.get("action") == "to_capture"
        if not group.group_id or group.group_id in seen:
            _issue(issues, "BSL002", "error", f"父组 ID 缺失或重复: {group.group_id!r}", task=task)
        seen.add(group.group_id)
        if group.group_class not in GROUP_CLASSES:
            _issue(issues, "BSL020", "error", f"{group.group_id}: class 无效", task=task, group=group.group_id)
        valid_kinds = PARENT_KINDS if group.group_class != "blender" else frozenset({"blender_scene"})
        if group.kind not in valid_kinds:
            _issue(issues, "BSL020", "error", f"{group.group_id}: kind 无效: {group.kind!r}", task=task, group=group.group_id)
        if not group.paths:
            _issue(issues, "BSL003", "error", f"{group.group_id}: 未声明 paths", task=task, group=group.group_id)
        for path in group.paths:
            if not cfg.is_workspace_path(path):
                _issue(issues, "BSL003", "error", f"{group.group_id}: 非工作空间相对路径: {path}", task=task, group=group.group_id)
            elif not planned and not path_exists(root, path):
                _issue(issues, "BSL003", "error", f"{group.group_id}: 路径不存在: {path}", task=task, group=group.group_id)
        if group.group_class in {"task_real", "background"}:
            if not group.evidence:
                if not planned:
                    _issue(issues, "BSL004", "error", f"{group.group_id}: 缺少 evidence", task=task, group=group.group_id)
            else:
                validate_evidence(
                    root, task, group.group_id, group.evidence, issues, allow_note_only=planned
                )
        elif group.evidence:
            validate_evidence(root, task, group.group_id, group.evidence, issues)
        if not group.roles or any(role not in ROLES for role in group.roles):
            _issue(issues, "BSL020", "error", f"{group.group_id}: roles 缺失或含未知值", task=task, group=group.group_id)
        split = group.split
        if not isinstance(split, Mapping):
            _issue(issues, "BSL005", "error", f"{group.group_id}: split 不是映射", task=task, group=group.group_id)
        else:
            if split.get("primary") not in SPLITS:
                _issue(issues, "BSL005", "error", f"{group.group_id}: split.primary 无效: {split.get('primary')!r}", task=task, group=group.group_id)
            if split.get("status") not in SPLIT_STATUSES:
                _issue(issues, "BSL005", "error", f"{group.group_id}: split.status 无效: {split.get('status')!r}", task=task, group=group.group_id)
            action = split.get("action")
            if action is not None and action not in ACTIONS:
                _issue(issues, "BSL005", "error", f"{group.group_id}: split.action 无效: {action!r}", task=task, group=group.group_id)
            allowed_in = split.get("same_source_allowed_in", [])
            if allowed_in is not None:
                if not isinstance(allowed_in, list) or any(value not in SPLITS for value in allowed_in):
                    _issue(issues, "BSL005", "error", f"{group.group_id}: split.same_source_allowed_in 无效", task=task, group=group.group_id)
                elif split.get("primary") in allowed_in:
                    _issue(issues, "BSL005", "error", f"{group.group_id}: same_source_allowed_in 不能包含 primary", task=task, group=group.group_id)
                elif "train" in allowed_in and split.get("primary") != "train":
                    _issue(issues, "BSL012", "info", f"{group.group_id}: 父组跨集合复用，评估按同源口径解读", task=task, group=group.group_id)
            if group.group_class == "task_real" and split.get("primary") == "shared":
                _issue(issues, "BSL005", "warning", f"{group.group_id}: task_real 父组通常不应使用 shared split", task=task, group=group.group_id)
        if group.capture_session is not None and group.capture_session not in CAPTURE_SESSIONS:
            _issue(issues, "BSL013", "error", f"{group.group_id}: capture_session 无效: {group.capture_session!r}", task=task, group=group.group_id)
        if group.group_class in {"task_real", "background"}:
            if group.labels is None:
                level = "warning" if planned or group.group_class == "background" else "error"
                _issue(issues, "BSL020", level, f"{group.group_id}: 缺少 labels 状态", task=task, group=group.group_id)
            elif group.labels.get("status") not in LABEL_STATES:
                _issue(issues, "BSL020", "error", f"{group.group_id}: labels.status 无效: {group.labels.get('status')!r}", task=task, group=group.group_id)
        if group.group_class == "background":
            review = group.review
            if not isinstance(review, Mapping) or review.get("state") not in REVIEW_STATES:
                _issue(issues, "BSL006", "error", f"{group.group_id}: 背景缺少有效 review.state", task=task, group=group.group_id)
            else:
                state = review.get("state")
                if state == "rejected":
                    _issue(issues, "BSL021", "warning", f"{group.group_id}: 背景已被拒绝，不应进入生成候选", task=task, group=group.group_id)
                if "empty_negative" in group.roles and state != "confirmed_empty":
                    _issue(issues, "BSL007", "error", f"{group.group_id}: empty_negative 需要 confirmed_empty，当前 {state}", task=task, group=group.group_id)
                if state == "contains_target" and (group.labels or {}).get("status") not in CAPABLE_TARGET_LABELS:
                    _issue(issues, "BSL022", "warning", f"{group.group_id}: contains_target 背景缺少可用的目标标签状态", task=task, group=group.group_id)


def _validate_cap(
    task: str,
    groups: Iterable[ParentGroup],
    pairing: Mapping[str, Any],
    issues: list[Issue],
) -> dict[str, Any]:
    policy = pairing.get("policy") or {}
    cap = policy.get("train_parent_cap", DEFAULT_TRAIN_CAP)
    if not isinstance(cap, (int, float)) or isinstance(cap, bool) or not 0 < cap <= 1:
        cap = DEFAULT_TRAIN_CAP
    summary: dict[str, Any] = {"cap": cap, "roles": []}
    for group_class in ("task_real", "background"):
        members = [group for group in groups if group.group_class == group_class]
        roles = sorted({role for group in members for role in group.roles})
        for role in roles:
            matching = [group for group in members if role in group.roles]
            qualified = [group for group in matching if _qualified(group, role)]
            train = [group for group in qualified if group.split.get("primary") == "train"]
            item = {
                "class": group_class,
                "role": role,
                "qualified": len(qualified),
                "train": len(train),
                "train_fraction": len(train) / len(qualified) if qualified else None,
                "status": "assessable",
            }
            if len(qualified) < 5:
                item["status"] = "insufficient"
                _issue(
                    issues,
                    "BSL011",
                    "info",
                    f"{group_class}/{role}: 已分配的合格父组 {len(qualified)} 个，来源集中度需结合实际分集解读",
                    task=task,
                )
            elif len(train) > int(cap * len(qualified)):
                item["status"] = "exceeded"
                _issue(
                    issues,
                    "BSL011",
                    "info",
                    f"{group_class}/{role}: train 使用 {len(train)}/{len(qualified)} 个合格父组，高于 {cap:.0%} 设计参考值",
                    task=task,
                )
            summary["roles"].append(item)
    return summary


def _qualified(group: ParentGroup, role: str) -> bool:
    if group.split.get("primary") not in QUALIFIED_SPLITS:
        return False
    if group.labels and group.labels.get("status") in {"excluded", "blocked"}:
        return False
    if group.group_class == "background":
        state = (group.review or {}).get("state")
        if role == "empty_negative":
            return state == "confirmed_empty"
        if role == "composite_background":
            if state not in COMPOSITE_BACKGROUND_STATES:
                return False
            return not (
                state == "contains_target"
                and (group.labels or {}).get("status") not in CAPABLE_TARGET_LABELS
            )
        if state not in REVIEW_STATES or state == "rejected":
            return False
    return True


def _validate_data_sources(
    root: Path,
    task: str,
    groups: list[ParentGroup],
    data_sources: list[DataSource],
    issues: list[Issue],
) -> None:
    by_id = {group.group_id: group for group in groups}
    seen: set[str] = set()
    for item in data_sources:
        planned = (item.split or {}).get("action") == "to_capture"
        paths = list(item.paths) or ([item.path] if item.path else [])
        if not item.source_id or item.source_id in seen:
            _issue(issues, "BSL002", "error", f"数据源 ID 缺失或重复: {item.source_id!r}", task=task)
        seen.add(item.source_id)
        if item.kind not in DATA_SOURCE_KINDS:
            _issue(issues, "BSL020", "error", f"{item.source_id}: kind 无效: {item.kind!r}", task=task, group=item.source_id)
        if not item.roles or any(role not in ROLES for role in item.roles):
            _issue(issues, "BSL020", "error", f"{item.source_id}: roles 缺失或含未知值", task=task, group=item.source_id)
        if item.kind == "fixed_artwork" and not paths:
            _issue(issues, "BSL003", "error", f"{item.source_id}: fixed_artwork 需要 path 或 paths", task=task, group=item.source_id)
        for path in paths:
            if not cfg.is_workspace_relative(path):
                _issue(issues, "BSL003", "error", f"{item.source_id}: 非工作空间相对路径: {path}", task=task, group=item.source_id)
            elif not planned and not path_exists(root, path):
                _issue(issues, "BSL003", "error", f"{item.source_id}: 路径不存在: {path}", task=task, group=item.source_id)
        if item.kind in {"real_cutout", "mask", "blender_target"} and not item.parent_group:
            _issue(issues, "BSL008", "error", f"{item.source_id}: {item.kind} 需要 parent_group", task=task, group=item.source_id)
        parent = by_id.get(item.parent_group) if item.parent_group else None
        if item.parent_group and parent is None:
            _issue(issues, "BSL008", "error", f"{item.source_id}: parent_group 未解析: {item.parent_group}", task=task, group=item.source_id)
        primary = (item.split or {}).get("primary")
        if primary not in SPLITS:
            _issue(issues, "BSL005", "error", f"{item.source_id}: split.primary 无效: {primary!r}", task=task, group=item.source_id)
        if parent is not None:
            parent_primary = parent.split.get("primary")
            declared_reuse = parent.split.get("same_source_allowed_in")
            assign_child = parent_primary == "unassigned" and primary in QUALIFIED_SPLITS
            reuse_child = isinstance(declared_reuse, list) and primary in declared_reuse
            if primary not in {"inherit", "unassigned", parent_primary} and not assign_child and not reuse_child:
                _issue(issues, "BSL009", "error", f"{item.source_id}: 派生数据源 split={primary!r} 未继承父组 {parent.group_id} 的 {parent_primary!r}", task=task, group=item.source_id)


def _check_capture_placeholders(
    task: str,
    groups: list[ParentGroup],
    captures: list[dict[str, Any]],
    issues: list[Issue],
) -> None:
    capture_by_id = {}
    for item in captures:
        capture_id = item.get("id")
        if not isinstance(capture_id, str) or not capture_id or capture_id in capture_by_id:
            _issue(issues, "BSL013", "error", f"capture ID 缺失或重复: {capture_id!r}", task=task)
            continue
        capture_by_id[capture_id] = item
        if item.get("status") not in CAPTURE_STATUSES:
            _issue(issues, "BSL013", "error", f"{capture_id}: capture status 无效", task=task)
        if item.get("target_split") not in QUALIFIED_SPLITS:
            _issue(issues, "BSL013", "error", f"{capture_id}: target_split 必须是 train/val/test", task=task)
        save_to = item.get("save_to")
        if not isinstance(save_to, str) or not save_to.strip() or not cfg.is_workspace_path(save_to):
            _issue(issues, "BSL013", "error", f"{capture_id}: save_to 缺失或不是工作空间相对路径", task=task)
    for group in groups:
        if group.split.get("action") != "to_capture":
            continue
        capture_id = group.capture_id
        if not capture_id or capture_id not in capture_by_id:
            _issue(issues, "BSL013", "error", f"{group.group_id}: to_capture 缺少有效 capture_id", task=task, group=group.group_id)
            continue
        capture = capture_by_id[capture_id]
        if capture.get("target_split") != group.split.get("primary"):
            _issue(issues, "BSL013", "error", f"{group.group_id}: capture target_split 与占位父组 split 不一致", task=task, group=group.group_id)
        if group.capture_session != "planned":
            _issue(issues, "BSL013", "error", f"{group.group_id}: 计划补采必须使用 capture_session=planned", task=task, group=group.group_id)
        save_to = capture.get("save_to")
        if isinstance(save_to, str) and cfg.normalize_path(save_to) not in group.paths:
            _issue(issues, "BSL013", "error", f"{group.group_id}: 占位路径未包含 capture save_to", task=task, group=group.group_id)


def _check_plan_bindings(
    root: Path, task: str, pairing: Mapping[str, Any], issues: list[Issue]
) -> None:
    bindings = pairing.get("plan_bindings")
    if not isinstance(bindings, Mapping):
        return
    for split, values in bindings.items():
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str) or not cfg.is_workspace_relative(value):
                _issue(issues, "BSL017", "error", f"plan_bindings.{split} 含无效路径: {value!r}", task=task)
            elif not path_exists(root, value):
                _issue(issues, "BSL017", "warning", f"plan_bindings.{split} 指向不存在的计划: {value}", task=task)


def _check_legacy_references(
    root: Path,
    task: str,
    adoption: str,
    observed: Iterable[Mapping[str, Any]],
    selections: Iterable[Mapping[str, Any]],
    issues: list[Issue],
) -> None:
    for item in observed:
        kind = item.get("kind")
        path = item.get("path")
        if kind not in {"background", "shared_pool"}:
            continue
        if not path_exists(root, path):
            code = "BSL023" if kind == "shared_pool" else "BSL019"
            level = "error" if kind == "shared_pool" else "warning"
            _issue(issues, code, level, f"声明路径不存在: {path}", task=task)
    for item in selections:
        path = item.get("path")
        if path and not path_exists(root, path):
            level = "error" if adoption == "pairing" else "warning"
            _issue(issues, "BSL019", level, f"current_selection 背景不存在: {path}", task=task)


def _check_shared_pool_usage(task: str, entry: Mapping[str, Any], issues: list[Issue]) -> None:
    references = entry.get("shared_pool") or []
    if not references:
        return
    declared = any(item.get("origin") in {"pairing", "legacy"} for item in references)
    if not declared:
        _issue(
            issues,
            "BSL023",
            "warning",
            "已在 current_selection 使用共享背景池，但未在来源配置中声明该池背景组",
            task=task,
        )


def _build_shared(
    root: Path,
    entries: Mapping[str, dict[str, Any]],
    groups: Mapping[str, list[ParentGroup]],
    issues: list[Issue],
) -> dict[str, Any]:
    manifest_name = cfg.SHARED_POOL_MANIFEST
    if not (root / manifest_name).is_file() and (root / cfg.LEGACY_POOL).is_file():
        manifest_name = cfg.LEGACY_POOL
    pool_manifest = root / manifest_name
    records = None
    if pool_manifest.is_file():
        try:
            records = manifest_records(pool_manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            records = None
    pool_parents: dict[str, dict[str, Any]] = {}
    image_parent: dict[str, str] = {}
    allocated: dict[str, dict[str, set[tuple[str, str]]]] = {}
    if records is not None:
        for record in records:
            image = record.get("image")
            parent = record.get("parent_group") or record.get("source_group") or record.get("source_video")
            if not isinstance(parent, str):
                continue
            parent = cfg.normalize_path(parent)
            item = pool_parents.setdefault(
                parent,
                {"source_video": parent, "frames": 0, "sample_images": []},
            )
            item["parent_group"] = parent
            item["frames"] += 1
            if isinstance(image, str):
                image = cfg.normalize_path(image)
                image_parent[image] = parent
                if len(item["sample_images"]) < 3:
                    item["sample_images"].append(image)
                if isinstance(record.get("source_image"), str):
                    image_parent[cfg.normalize_path(record["source_image"])] = parent
                for task, task_use in record.get("tasks", {}).items():
                    if "composite_background" in task_use.get("uses", []):
                        allocated.setdefault(parent, {}).setdefault(task, set()).add(
                            (task_use.get("split") or "unassigned", record.get("capture_identity") or image)
                        )
    usage: dict[str, dict[str, dict[str, Any]]] = {}
    for task, entry in entries.items():
        shared_references = entry.get("shared_pool", [])
        declared = any(reference.get("origin") in {"pairing", "legacy"} for reference in shared_references)
        group_id = next(
            (reference.get("group") for reference in shared_references if reference.get("origin") == "pairing"),
            None,
        )
        by_parent: dict[str, list[dict[str, Any]]] = {}
        for item in entry.get("selections", []):
            parent = image_parent.get(item.get("path"))
            if parent is None:
                continue
            by_parent.setdefault(parent, []).append(
                {
                    "path": item["path"],
                    "parent_video": item.get("parent_video"),
                }
            )
        for parent, items in by_parent.items():
            usage.setdefault(parent, {})[task] = {
                "declared": declared,
                "group": group_id,
                "items": items,
            }
        for parent, tasks in allocated.items():
            if task in tasks:
                usage.setdefault(parent, {}).setdefault(task, {"declared": declared, "group": group_id, "items": []})
    if records is None and any(entry.get("shared_pool") for entry in entries.values()):
        _issue(issues, "BSL023", "error", f"共享背景池清单不可读: {manifest_name}")
    pool = {
        "manifest": manifest_name,
        "records": len(records) if records is not None else None,
        "parent_groups": [
            {
                **pool_parents[parent],
                "tasks": [
                    {
                        "task": task,
                        "declared": record["declared"],
                        "group": record["group"],
                        "selected": len(record["items"]),
                        "allocated_by_split": dict(Counter(split for split, _ in allocated.get(parent, {}).get(task, ()))),
                        "paths": record["items"],
                    }
                    for task, record in sorted(usage.get(parent, {}).items())
                ],
            }
            for parent in sorted(pool_parents)
        ],
    }
    blender: dict[str, dict[str, Any]] = {}
    for task, entry in entries.items():
        for reference in entry.get("blender", []):
            path = reference.get("path")
            item = blender.setdefault(
                path,
                {
                    "path": path,
                    "tasks": [],
                    "exists": path_exists(root, path),
                    "registry": _registry_record(root, path),
                },
            )
            item["tasks"].append(
                {
                    "task": task,
                    "origin": reference.get("origin"),
                    "role": reference.get("role"),
                    "split": reference.get("split") or {},
                }
            )
    for path, item in blender.items():
        for reference in item["tasks"]:
            if reference.get("origin") == "pairing" and not item["exists"]:
                _issue(issues, "BSL016", "error", f"Blender 场景不存在: {path}", task=reference["task"])
            elif reference.get("origin") == "legacy" and not item["exists"]:
                _issue(issues, "BSL016", "warning", f"旧来源引用的 Blender 场景不存在: {path}", task=reference["task"])
        if item["exists"] and item["registry"] is None:
            tasks = ",".join(sorted({reference["task"] for reference in item["tasks"]}))
            _issue(issues, "BSL016", "warning", f"Blender 场景未在登记库中: {path}（任务 {tasks}）")
    return {"background_pool": pool, "blender_scenes": [blender[path] for path in sorted(blender)]}


def _registry_record(root: Path, path: str) -> dict[str, Any] | None:
    database = root / cfg.BLENDER_REGISTRY
    if not database.is_file():
        return None
    try:
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
            row = db.execute(
                "SELECT a.asset_id,a.current_path,a.lifecycle_state,a.classification,a.truth_level "
                "FROM assets a JOIN asset_paths p ON p.asset_id=a.asset_id "
                "WHERE p.path=? OR a.current_path=? LIMIT 1",
                (path, path),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return {
        "asset_id": row[0],
        "current_path": row[1],
        "lifecycle_state": row[2],
        "classification": row[3],
        "truth_level": row[4],
    }


def _check_cross_task(
    entries: Mapping[str, dict[str, Any]],
    groups: Mapping[str, list[ParentGroup]],
    shared: Mapping[str, Any],
    issues: list[Issue],
) -> None:
    paths: dict[str, list[str]] = {}
    for task in sorted(groups):
        for group in groups[task]:
            if group.group_class not in {"task_real", "background"}:
                continue
            for path in group.paths:
                paths.setdefault(path, []).append(task)
    for path, tasks in sorted(paths.items()):
        unique = sorted(set(tasks))
        if len(unique) > 1:
            _issue(issues, "BSL025", "info", f"父来源路径被多个任务声明: {path} -> {','.join(unique)}")
    for item in shared.get("blender_scenes", []):
        tasks = sorted({reference["task"] for reference in item["tasks"]})
        if len(tasks) > 1:
            _issue(issues, "BSL025", "info", f"Blender 场景被多个任务引用: {item['path']} -> {','.join(tasks)}")
        if "/shared/" in item["path"]:
            for reference in item["tasks"]:
                if reference.get("origin") == "pairing" and (reference.get("split") or {}).get("primary") != "shared":
                    _issue(
                        issues,
                        "BSL026",
                        "warning",
                        f"共享母场景未声明 shared split: {item['path']}（任务 {reference['task']}）",
                        task=reference["task"],
                    )
    for item in shared.get("background_pool", {}).get("parent_groups", []):
        tasks = [usage["task"] for usage in item.get("tasks", [])]
        if len(tasks) > 1:
            _issue(
                issues,
                "BSL025",
                "info",
                f"共享背景父来源被多个任务选用: {item['parent_group']} -> {','.join(tasks)}",
            )


def _summarize(
    entries: Mapping[str, dict[str, Any]],
    shared: Mapping[str, Any],
    issues: Iterable[Issue],
) -> dict[str, Any]:
    adoption = Counter(entry.get("adoption") for entry in entries.values())
    review = Counter()
    splits = Counter()
    groups = Counter()
    for entry in entries.values():
        for group in entry.get("groups", []):
            groups[group.get("class")] += 1
            splits[group.get("split", {}).get("primary")] += 1
            if group.get("review"):
                review[group.get("review", {}).get("state")] += 1
    levels = Counter(issue.level for issue in issues)
    return {
        "tasks": len(entries),
        "adoption": dict(adoption),
        "group_classes": dict(groups),
        "splits": dict(splits),
        "review_states": dict(review),
        "shared_pool_parent_groups": len(shared.get("background_pool", {}).get("parent_groups", [])),
        "blender_scenes": len(shared.get("blender_scenes", [])),
        "issues": dict(levels),
    }
