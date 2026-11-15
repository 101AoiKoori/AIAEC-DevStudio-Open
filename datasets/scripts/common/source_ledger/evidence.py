"""Resolve and check evidence files referenced by the source ledger.

Only bounded, read-only inspection happens here: paths are checked, manifests
are counted or parsed up to a fixed size limit, and record indices are range
checked. A manifest is generation-time evidence; a valid path or count does not
prove image semantics or label truth.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import is_workspace_relative, normalize_path
from .schema import Issue

MAX_MANIFEST_BYTES = 64 * 1024 * 1024
COLLECTION_KEYS = (
    "records",
    "images",
    "frames",
    "samples",
    "items",
    "entries",
    "backgrounds",
    "sources",
)


def resolve_path(root: Path, value: str) -> Path:
    """Resolve a declared workspace-relative path without following links.

    The path is normalized lexically and must stay inside ``root``. Existence and
    reads may still pass through a directory link or junction; the ledger never
    writes to a declared evidence path.
    """

    if not is_workspace_relative(value):
        raise ValueError(f"不是工作空间相对路径: {value!r}")
    base = Path(os.path.abspath(root))
    path = Path(os.path.abspath(base / normalize_path(value)))
    if not path.is_relative_to(base) or path == base:
        raise ValueError(f"路径越出工作空间: {value!r}")
    return path


def path_exists(root: Path, value: str) -> bool:
    try:
        return resolve_path(root, value).exists()
    except ValueError:
        return False


def read_json_document(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _collection_count(data: Any) -> int | None:
    if isinstance(data, list):
        return len(data)
    if isinstance(data, Mapping):
        for key in COLLECTION_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
            if isinstance(value, Mapping):
                return len(value)
    return None


def manifest_count(path: Path) -> int | None:
    """Return a declared collection size, or ``None`` when the shape is unknown."""

    path = Path(path)
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        return None
    if path.suffix.casefold() == ".jsonl":
        count = 0
        with open(path, encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    count += 1
        return count
    if path.suffix.casefold() == ".json":
        try:
            data = read_json_document(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        return _collection_count(data)
    return None


def manifest_records(path: Path) -> list[dict[str, Any]] | None:
    """Return record dictionaries for a manifest whose shape supports them."""

    path = Path(path)
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        return None
    if path.suffix.casefold() == ".jsonl":
        records = []
        with open(path, encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, Mapping):
                    records.append(dict(item))
        return records
    if path.suffix.casefold() != ".json":
        return None
    try:
        data = read_json_document(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, Mapping)]
    if isinstance(data, Mapping):
        for key in COLLECTION_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
    return None


def validate_evidence(
    root: Path,
    task: str,
    group_id: str,
    evidence: Iterable[Mapping[str, Any]],
    issues: list[Issue],
    *,
    allow_note_only: bool = False,
) -> None:
    for position, item in enumerate(evidence):
        manifest = item.get("manifest")
        if not isinstance(manifest, str) or not manifest.strip():
            if allow_note_only and isinstance(item.get("note"), str) and item["note"].strip():
                continue
            issues.append(
                Issue(
                    "BSL004",
                    "error",
                    f"evidence[{position}] 缺少 manifest",
                    task=task,
                    group=group_id,
                )
            )
            continue
        manifest = normalize_path(manifest)
        try:
            path = resolve_path(root, manifest)
        except ValueError as error:
            issues.append(
                Issue("BSL004", "error", f"evidence manifest 无效: {error}", task=task, group=group_id)
            )
            continue
        if not path.is_file():
            issues.append(
                Issue(
                    "BSL004",
                    "error",
                    f"evidence manifest 不存在: {manifest}",
                    task=task,
                    group=group_id,
                )
            )
            continue
        relation = item.get("relation")
        if relation is not None and (not isinstance(relation, str) or not relation.strip()):
            issues.append(
                Issue(
                    "BSL004",
                    "error",
                    f"evidence relation 无效: {manifest}",
                    task=task,
                    group=group_id,
                )
            )
        record_index = item.get("record_index")
        if record_index is None:
            continue
        if not isinstance(record_index, int) or isinstance(record_index, bool) or record_index < 0:
            issues.append(
                Issue(
                    "BSL004",
                    "error",
                    f"evidence record_index 必须是非负整数: {manifest}",
                    task=task,
                    group=group_id,
                )
            )
            continue
        count = manifest_count(path)
        if count is None:
            issues.append(
                Issue(
                    "BSL004",
                    "warning",
                    f"manifest 结构不支持记录计数，record_index 未核: {manifest}",
                    task=task,
                    group=group_id,
                )
            )
        elif record_index >= count:
            issues.append(
                Issue(
                    "BSL004",
                    "error",
                    f"record_index={record_index} 超出 {manifest} 的 {count} 条记录",
                    task=task,
                    group=group_id,
                )
            )
