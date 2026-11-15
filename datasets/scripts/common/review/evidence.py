"""Evidence files for candidate review: hashes, JSONL metrics and JSON plans.

Every artifact written here is reproducible from media plus code. The module
owns locations and encodings only; it never reads images and never decides a
verdict. `batch_dir` describes a path but does not create it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..paths import WORKSPACE
from .schema import MetricRecord, to_jsonable

_HASH_LENGTH = 16
_JSON_KWARGS = {"ensure_ascii": False, "sort_keys": True}


def _workspace_key(path: str | Path) -> str:
    """Return the workspace-relative POSIX key of a file, or its absolute key."""

    candidate = Path(path).expanduser()
    resolved = candidate if candidate.is_absolute() else WORKSPACE / candidate
    resolved = resolved.resolve()
    try:
        return resolved.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def code_hash(paths: Sequence[str | Path]) -> str:
    """Hash code files by sorted workspace-relative path and content.

    Each file contributes the SHA-256 of its bytes. The per-file digests are
    concatenated in workspace-relative POSIX path order and hashed again. A
    changed file order, a changed byte or a missing file therefore changes the
    result (a missing file raises :class:`FileNotFoundError`).
    """

    pairs = sorted(((_workspace_key(path), path) for path in paths), key=lambda pair: pair[0])
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("code_hash received duplicate workspace paths")
    digests = []
    for _, path in pairs:
        candidate = Path(path).expanduser()
        resolved = candidate if candidate.is_absolute() else WORKSPACE / candidate
        digests.append(hashlib.sha256(resolved.resolve().read_bytes()).hexdigest())
    joined = "".join(digests).encode("ascii")
    return hashlib.sha256(joined).hexdigest()[:_HASH_LENGTH]


def params_hash(params: dict) -> str:
    """Hash measurement parameters independently of their key order."""

    text = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_HASH_LENGTH]


def _jsonable(record: Any) -> dict[str, Any]:
    if isinstance(record, MetricRecord):
        return to_jsonable(record)
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError(f"unsupported metric record: {type(record)!r}")


def _dump(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, **_JSON_KWARGS)


def write_metrics(
    path: str | Path,
    records: Sequence[MetricRecord],
    *,
    header: dict | None = None,
    task: str = "",
) -> int:
    """Write one JSONL metric record per line and return the record count.

    The optional header is written first as ``{"kind": "header", ...}``. Only
    the header carries a ``kind`` key; metric records stay untouched. Parent
    directories are created on demand.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if header is not None or task:
        head: dict[str, Any] = {"kind": "header"}
        if task:
            head["task"] = task
        if header:
            head.update(header)
        lines.append(_dump(head))
    for record in records:
        lines.append(_dump(_jsonable(record)))
    text = "".join(line + "\n" for line in lines)
    target.write_text(text, encoding="utf-8", newline="\n")
    return len(records)


def read_metrics(path: str | Path) -> tuple[dict | None, list[dict]]:
    """Read a metrics JSONL file as ``(header, records)``.

    Blank lines are skipped. A malformed line raises :class:`ValueError` naming
    the original 1-based line number. The header is only recognised on the first
    non-blank line so a record never silently disappears.
    """

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    header: dict | None = None
    records: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_number} of {source}: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"line {line_number} of {source} is not a JSON object")
        if item.get("kind") == "header" and header is None and not records:
            header = item
            continue
        records.append(item)
    return header, records


def write_json(path: str | Path, payload: Any) -> None:
    """Write a JSON document with two-space indent and insertion key order."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)
    target.write_text(text + "\n", encoding="utf-8", newline="\n")


def read_json(path: str | Path) -> dict:
    """Read a JSON document that must contain one object."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return payload


def batch_dir(task: str, batch: str, *, root: str = "datasets/annotated") -> Path:
    """Return ``root/<task>/<batch>`` without creating it."""

    if not task or not batch:
        raise ValueError("batch_dir needs non-empty task and batch names")
    return Path(root) / task / batch
