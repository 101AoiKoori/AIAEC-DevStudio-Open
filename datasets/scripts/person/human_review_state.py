"""Merge-preserving helpers for person review tooling."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

HUMAN_FIELDS = (
    "accepted_by",
    "box_accepted_by",
    "box_acceptance",
    "full_frame_checked",
    "human_decision",
    "human_label",
    "human_box_corrections",
)


def has_human_content(value: Any) -> bool:
    """Return true when a review document contains a nonempty human decision."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in HUMAN_FIELDS and _field_has_value(item):
                if key in {"box_acceptance"} and item in {"pending", "candidate"}:
                    continue
                if (
                    key == "box_accepted_by"
                    and value.get("box_acceptance") in {"pending", "candidate"}
                ):
                    continue
                if key == "full_frame_checked" and item is False:
                    continue
                return True
            if has_human_content(item):
                return True
    elif isinstance(value, list):
        return any(has_human_content(item) for item in value)
    return False


def _field_has_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def merge_cases(
    existing: Mapping[str, Any],
    proposed: Mapping[str, Any],
    *,
    id_key: str = "id",
    fields: tuple[str, ...] = HUMAN_FIELDS,
) -> tuple[dict[str, Any], list[str]]:
    """Merge human fields from existing cases into freshly derived cases."""

    existing_by_id = {
        str(case.get(id_key)): case
        for case in existing.get("cases") or []
        if isinstance(case, Mapping) and case.get(id_key)
    }
    merged = dict(proposed)
    merged_cases = []
    preserved = []
    for case in proposed.get("cases") or []:
        item = dict(case)
        prior = existing_by_id.get(str(item.get(id_key)))
        if prior is not None:
            carried = {}
            for field in fields:
                value = prior.get(field)
                if _field_has_value(value):
                    carried[field] = value
            if carried:
                item.update(carried)
                item["preserved_from_existing_review"] = True
                preserved.append(str(item.get(id_key)))
        merged_cases.append(item)
    merged["cases"] = merged_cases
    return merged, preserved


def backup_files(paths: list[Path]) -> list[Path]:
    """Back up nonempty human review files before an explicit rebuild."""

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    backups = []
    for path in paths:
        if not path.is_file():
            continue
        backup = path.with_name(f"{path.name}.bak.{stamp}")
        shutil.copy2(path, backup)
        backups.append(backup)
    return backups
