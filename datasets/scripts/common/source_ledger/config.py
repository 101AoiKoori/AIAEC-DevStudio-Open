"""Read task source configurations and normalize background/source declarations.

The module only reads task-owned ``config/source.yaml`` files. Pairing blocks
are optional: a task that has not adopted the parent-group structure keeps its
legacy fields, and the ledger reports that state instead of inventing parent
groups from directory names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ..background_pool import LEGACY_POOL, POOL_MANIFEST, POOL_ROOT
from ..config import read_yaml
from ..paths import TASKS
from .schema import DataSource, ParentGroup

SHARED_POOL_MANIFEST = POOL_MANIFEST
SHARED_POOL_PREFIXES = (POOL_ROOT, str(Path(LEGACY_POOL).parent.as_posix()))
BLENDER_REGISTRY = "datasets/blender_source_registry.sqlite3"
DEFAULT_TASKS = tuple(sorted(TASKS))


def source_config_path(root: Path, task: str) -> Path:
    return Path(root) / "datasets" / "scripts" / task / "config" / "source.yaml"


def load_source_config(root: Path, task: str) -> dict[str, Any]:
    path = source_config_path(root, task)
    data = read_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"{task}: source.yaml 不是映射")
    if data.get("task") != task:
        raise ValueError(f"{task}: source.yaml 的 task 字段不匹配")
    return data


def normalize_path(value: str) -> str:
    return str(value).replace("\\", "/").strip().rstrip("/")


def is_shared_pool_path(path: str) -> bool:
    lower = normalize_path(path).casefold()
    return (lower == SHARED_POOL_MANIFEST.casefold()
            or any(lower == prefix.casefold() or lower.startswith(prefix.casefold() + "/")
                   for prefix in SHARED_POOL_PREFIXES))


def is_workspace_path(value: str) -> bool:
    text = normalize_path(value)
    if not text or "://" in text or text.startswith("/") or text.startswith("~"):
        return False
    if len(text) > 1 and text[1] == ":":
        return False
    return text.startswith("datasets/") or text.startswith("Archive/")


def is_workspace_relative(value: str) -> bool:
    """Accept any lexically workspace-relative path, not only source roots."""

    text = normalize_path(value)
    if not text or "://" in text or text.startswith("/") or text.startswith("~"):
        return False
    if len(text) > 1 and text[1] == ":":
        return False
    return ".." not in Path(text).parts


def iter_path_values(
    value: Any, key_path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], str]]:
    if isinstance(value, str):
        if is_workspace_path(value):
            yield key_path, normalize_path(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from iter_path_values(item, key_path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_path_values(item, key_path + (str(index),))


def classify_reference(key_path: tuple[str, ...], path: str) -> str:
    key = ".".join(part for part in key_path if not part.isdigit()).casefold()
    lower = path.casefold()
    if lower.endswith(".blend1"):
        return "blender_backup"
    if lower.endswith(".blend"):
        return "blender_scene"
    if is_shared_pool_path(path):
        return "shared_pool"
    if "shared_background" in key:
        return "shared_pool"
    if "background" in key or "background" in lower:
        return "background"
    return "other"


def collect_observed_references(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect declared workspace paths without claiming parent-group status."""

    references: dict[tuple[str, str], dict[str, Any]] = {}
    for key_path, path in iter_path_values(source):
        key = ".".join(key_path)
        if key.startswith("pairing."):
            continue
        if key.startswith("databases."):
            continue
        if key.startswith("current_selection.backgrounds"):
            continue
        kind = classify_reference(key_path, path)
        references.setdefault(
            (key, path),
            {"key": key, "path": path, "kind": kind},
        )
    return [references[key] for key in sorted(references)]


def pairing_block(source: Mapping[str, Any]) -> dict[str, Any] | None:
    pairing = source.get("pairing")
    return pairing if isinstance(pairing, dict) else None


def _paths_of(item: Mapping[str, Any]) -> list[str]:
    values = item.get("paths")
    if isinstance(values, list):
        return [normalize_path(value) for value in values if isinstance(value, str)]
    path = item.get("path")
    if isinstance(path, str):
        return [normalize_path(path)]
    return []


def _roles_of(item: Mapping[str, Any]) -> list[str]:
    roles = item.get("roles")
    if isinstance(roles, list):
        return [role for role in roles if isinstance(role, str)]
    role = item.get("role")
    return [role] if isinstance(role, str) else []


def _split_of(item: Mapping[str, Any]) -> dict[str, Any]:
    split = item.get("split")
    return dict(split) if isinstance(split, dict) else {}


def build_groups(task: str, pairing: Mapping[str, Any]) -> list[ParentGroup]:
    groups: list[ParentGroup] = []
    for key, group_class in (("task_real", "task_real"), ("backgrounds", "background")):
        for item in pairing.get(key, []) or []:
            if not isinstance(item, Mapping):
                continue
            groups.append(
                ParentGroup(
                    task=task,
                    group_id=str(item.get("id", "")),
                    group_class=group_class,
                    kind=str(item.get("kind", "")),
                    paths=_paths_of(item),
                    evidence=[
                        dict(evidence)
                        for evidence in (item.get("evidence") or [])
                        if isinstance(evidence, Mapping)
                    ],
                    roles=_roles_of(item),
                    split=_split_of(item),
                    review=dict(item["review"]) if isinstance(item.get("review"), Mapping) else None,
                    labels=dict(item["labels"]) if isinstance(item.get("labels"), Mapping) else None,
                    axes=dict(item.get("axes") or {}),
                    capture_session=item.get("capture_session"),
                    capture_id=item.get("capture_id"),
                    raw_key=key,
                )
            )
    for item in pairing.get("blender", []) or []:
        if not isinstance(item, Mapping):
            continue
        groups.append(
            ParentGroup(
                task=task,
                group_id=str(item.get("id", "")),
                group_class="blender",
                kind=str(item.get("kind", "blender_scene")),
                paths=_paths_of(item),
                evidence=[
                    dict(evidence)
                    for evidence in (item.get("evidence") or [])
                    if isinstance(evidence, Mapping)
                ],
                roles=_roles_of(item),
                split=_split_of(item),
                review=None,
                labels=None,
                axes={},
                raw_key="blender",
            )
        )
    return groups


def build_data_sources(task: str, pairing: Mapping[str, Any]) -> list[DataSource]:
    result: list[DataSource] = []
    for item in pairing.get("data_sources", []) or []:
        if not isinstance(item, Mapping):
            continue
        paths = _paths_of(item)
        result.append(
            DataSource(
                task=task,
                source_id=str(item.get("id", "")),
                kind=str(item.get("kind", "")),
                path=paths[0] if paths else (
                    normalize_path(item["path"]) if isinstance(item.get("path"), str) else None
                ),
                parent_group=item.get("parent_group")
                if isinstance(item.get("parent_group"), str)
                else None,
                roles=_roles_of(item),
                split=_split_of(item),
                paths=paths,
            )
        )
    return result


def build_captures(pairing: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item) for item in (pairing.get("captures", []) or []) if isinstance(item, Mapping)
    ]


def selected_backgrounds(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return explicitly selected background paths from ``current_selection``."""

    selection = source.get("current_selection")
    if not isinstance(selection, Mapping):
        return []
    result: list[dict[str, Any]] = []
    for item in selection.get("backgrounds", []) or []:
        if isinstance(item, str):
            result.append(
                {
                    "path": normalize_path(item),
                    "parent_video": None,
                    "manifest": None,
                    "record_index": None,
                }
            )
        elif isinstance(item, Mapping):
            path = item.get("image", item.get("path"))
            if isinstance(path, str):
                result.append(
                    {
                        "path": normalize_path(path),
                        "parent_video": item.get("parent_video")
                        if isinstance(item.get("parent_video"), str)
                        else None,
                        "manifest": item.get("manifest")
                        if isinstance(item.get("manifest"), str)
                        else None,
                        "record_index": item.get("record_index")
                        if isinstance(item.get("record_index"), int)
                        else None,
                    }
                )
    return result


def collect_blender_references(
    task: str,
    groups: Iterable[ParentGroup],
    observed: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for group in groups:
        if group.group_class != "blender":
            continue
        for path in group.paths:
            references.setdefault(
                path,
                {
                    "task": task,
                    "path": path,
                    "origin": "pairing",
                    "group": group.group_id,
                    "role": group.roles[0] if group.roles else None,
                    "split": dict(group.split),
                },
            )
    for item in observed:
        if item.get("kind") not in {"blender_scene", "blender_backup"}:
            continue
        path = item["path"]
        references.setdefault(
            path,
            {
                "task": task,
                "path": path,
                "origin": "legacy",
                "group": None,
                "role": item.get("key"),
                "split": {},
            },
        )
    return [references[path] for path in sorted(references)]


def shared_pool_references(
    groups: Iterable[ParentGroup],
    observed: Iterable[Mapping[str, Any]],
    selections: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for group in groups:
        for path in group.paths:
            if is_shared_pool_path(path):
                found[(group.group_id, path)] = {
                    "origin": "pairing",
                    "group": group.group_id,
                    "path": path,
                }
        for evidence in group.evidence:
            manifest = evidence.get("manifest")
            if isinstance(manifest, str) and is_shared_pool_path(manifest):
                found[(group.group_id, normalize_path(manifest))] = {
                    "origin": "pairing",
                    "group": group.group_id,
                    "path": normalize_path(manifest),
                }
    for item in observed:
        if item.get("kind") != "shared_pool":
            continue
        found[(item["key"], item["path"])] = {
            "origin": "legacy",
            "group": None,
            "path": item["path"],
            "key": item["key"],
        }
    for item in selections:
        path = item.get("path", "")
        if is_shared_pool_path(path):
            found[("selection", path)] = {
                "origin": "selection",
                "group": None,
                "path": path,
            }
    return [found[key] for key in sorted(found)]

