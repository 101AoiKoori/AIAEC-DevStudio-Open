"""Write the source ledger report and render a compact text summary."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..paths import WORKSPACE

DEFAULT_OUTPUT = "datasets/annotated/common/sources/ledger.json"
ISSUE_ORDER = {"error": 0, "warning": 1, "info": 2}


def write_ledger(
    ledger: Mapping[str, Any],
    output: str | Path | None = None,
    *,
    root: str | Path | None = None,
) -> Path:
    """Write one ledger document atomically and return its resolved path."""

    base = Path(root or WORKSPACE).resolve()
    target = Path(output) if output else base / DEFAULT_OUTPUT
    if not target.is_absolute():
        target = base / target
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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


def issue_counts(ledger: Mapping[str, Any]) -> dict[str, int]:
    counts = Counter(issue.get("level") for issue in ledger.get("issues", []))
    return {level: counts.get(level, 0) for level in ("error", "warning", "info")}


def format_summary(ledger: Mapping[str, Any], *, max_issues: int = 20) -> str:
    summary = ledger.get("summary") or {}
    adoption = summary.get("adoption") or {}
    levels = issue_counts(ledger)
    lines = [
        "背景与来源总账 {}".format(ledger.get("generated_at", "")),
        "任务 {}：pairing={} legacy={} missing={}".format(
            summary.get("tasks", 0),
            adoption.get("pairing", 0),
            adoption.get("legacy", 0),
            adoption.get("missing", 0),
        ),
        "父组：{}；split：{}；审查状态：{}".format(
            summary.get("group_classes", {}),
            summary.get("splits", {}),
            summary.get("review_states", {}),
        ),
        "共享背景池父视频 {}；Blender 场景 {}；问题 error={} warning={} info={}".format(
            summary.get("shared_pool_parent_groups", 0),
            summary.get("blender_scenes", 0),
            levels["error"],
            levels["warning"],
            levels["info"],
        ),
    ]
    issues = sorted(
        ledger.get("issues", []),
        key=lambda issue: (
            ISSUE_ORDER.get(issue.get("level"), 9),
            issue.get("code", ""),
            str(issue.get("task") or ""),
            str(issue.get("group") or ""),
        ),
    )
    if issues:
        lines.append("")
        lines.append("问题：")
        for issue in issues[:max_issues]:
            where = "/".join(
                value for value in (issue.get("task"), issue.get("group")) if value
            )
            prefix = f"[{issue.get('code')}]"
            if where:
                prefix += f" {where}"
            lines.append(f"  {prefix}: {issue.get('message')}")
        if len(issues) > max_issues:
            lines.append(f"  ... 其余 {len(issues) - max_issues} 条见 ledger.json")
    return "\n".join(lines)
