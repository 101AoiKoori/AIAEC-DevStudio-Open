"""Shared contracts and file helpers for dataset authoring aids.

The helpers in this package use workspace-relative paths in durable records.
They may read local data to measure it, but they never make a negative label,
split reservation, or human acceptance on their own.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[3]

SCHEMA_VERSION = 1

SOURCE_KINDS = frozenset(
    {
        "direct_camera_capture",
        "real_image_enhancement",
        "real_image_composite",
        "blender_render",
        "render_real_mix",
    }
)

LABEL_STATES = frozenset(
    {
        "complete",
        "partial",
        "unknown",
        "confirmed_negative",
        "excluded",
    }
)

BACKGROUND_STATES = frozenset(
    {
        "confirmed_empty",
        "candidate_empty",
        "contains_target",
        "unknown",
        "rejected",
    }
)

REVIEW_VERDICTS = frozenset({"ready", "excluded", "unknown"})


def now_local() -> str:
    """Return a timezone-aware local timestamp for generated records."""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve_workspace(
    value: str | Path,
    *,
    root: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve a workspace-relative path without accepting an escape."""

    base = Path(root or ROOT).resolve()
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"path escapes the workspace: {value}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"path does not exist: {value}")
    return resolved


def workspace_relative(path: str | Path, *, root: str | Path | None = None) -> str:
    """Return a canonical workspace-relative POSIX path."""

    resolved = resolve_workspace(path, root=root)
    return resolved.relative_to(Path(root or ROOT).resolve()).as_posix()


def read_json(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    """Read one JSON object."""

    source = resolve_workspace(path, root=root, must_exist=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document is not an object: {source}")
    return payload


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> Path:
    """Atomically write a stable, indented JSON object."""

    base = Path(root or ROOT).resolve()
    target = resolve_workspace(path, root=base)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def code_digest(paths: Iterable[str | Path], *, root: str | Path | None = None) -> str:
    """Hash sorted workspace-relative files by path and content."""

    base = Path(root or ROOT).resolve()
    entries: list[tuple[str, Path]] = []
    for value in paths:
        path = resolve_workspace(value, root=base, must_exist=True)
        entries.append((path.relative_to(base).as_posix(), path))
    entries.sort()
    if len({name for name, _ in entries}) != len(entries):
        raise ValueError("code_digest received duplicate paths")
    digest = hashlib.sha256()
    for name, path in entries:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


def require_string(record: Mapping[str, Any], key: str) -> str:
    """Return a nonempty string field or raise a field-specific error."""

    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def require_paths(value: Any, *, key: str) -> list[str]:
    """Validate a list of nonempty workspace-relative path strings."""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a nonempty path list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or Path(item).is_absolute():
            raise ValueError(f"{key} entries must be workspace-relative strings")
        result.append(item.replace("\\", "/"))
    if len(result) != len(set(result)):
        raise ValueError(f"{key} contains duplicate paths")
    return result
