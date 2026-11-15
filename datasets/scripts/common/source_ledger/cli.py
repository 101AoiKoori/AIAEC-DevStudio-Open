"""Command line entry for the cross-task background/source ledger.

    python -m datasets.scripts.common.source_ledger.cli check [--task T ...]
    python -m datasets.scripts.common.source_ledger.cli write [--output PATH]
    python -m datasets.scripts.common.source_ledger.cli query --task T --role R --split S
    python -m datasets.scripts.common.source_ledger.cli matrix [--task T ...]

``check`` validates declared pairing/source records and prints a summary; it
never writes media. ``write`` additionally writes the derived ledger under
``datasets/annotated/common/sources/ledger.json``. ``query`` answers which
declared backgrounds may serve one task/role/split. ``matrix`` prints the
task x class x role x split parent-group counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from ..paths import WORKSPACE
from . import config as cfg
from .ledger import background_candidates, build_ledger
from .report import DEFAULT_OUTPUT, format_summary, issue_counts, write_ledger
from .schema import ROLES, SPLITS

PROG = "python -m datasets.scripts.common.source_ledger.cli"


def _root(value: str | None) -> Path:
    return Path(value).resolve() if value else WORKSPACE


def _tasks(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(values) if values else cfg.DEFAULT_TASKS


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=None, help="工作空间根目录；默认当前工作空间")
    parser.add_argument(
        "--task",
        action="append",
        choices=cfg.DEFAULT_TASKS,
        default=None,
        help="只处理指定任务；可重复",
    )


def _emit_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _run_check(args: argparse.Namespace) -> int:
    ledger = build_ledger(_root(args.root), _tasks(args.task))
    levels = issue_counts(ledger)
    if args.json:
        _emit_json({"summary": ledger["summary"], "issues": ledger["issues"]})
    else:
        print(format_summary(ledger))
    if levels["error"] or (args.strict_warnings and levels["warning"]):
        return 1
    return 0


def _run_write(args: argparse.Namespace) -> int:
    root = _root(args.root)
    ledger = build_ledger(root, _tasks(args.task))
    target = write_ledger(ledger, args.output, root=root)
    print(f"写入 {target}")
    print(format_summary(ledger))
    levels = issue_counts(ledger)
    if levels["error"] or (args.strict_warnings and levels["warning"]):
        return 1
    return 0


def _run_query(args: argparse.Namespace) -> int:
    ledger = build_ledger(_root(args.root), (args.task,))
    _emit_json(background_candidates(ledger, args.task, args.role, args.split))
    return 0


def _run_matrix(args: argparse.Namespace) -> int:
    ledger = build_ledger(_root(args.root), _tasks(args.task))
    tasks = ledger.get("tasks", {})
    print("任务\t配对状态\t父组(task_real/background/blender)\t数据源\tcaptures")
    for task in sorted(tasks):
        entry = tasks[task]
        groups = Counter(group.get("class") for group in entry.get("groups", []))
        print(
            "{}\t{}\t{}/{}/{}\t{}\t{}".format(
                task,
                entry.get("adoption"),
                groups.get("task_real", 0),
                groups.get("background", 0),
                groups.get("blender", 0),
                len(entry.get("data_sources", [])),
                len(entry.get("captures", [])),
            )
        )
    print()
    print("任务\t类别\t角色\tsplit\t父组数")
    rows = Counter()
    for task in sorted(tasks):
        for group in tasks[task].get("groups", []):
            if group.get("class") not in {"task_real", "background"}:
                continue
            split = (group.get("split") or {}).get("primary") or "unassigned"
            for role in group.get("roles") or ["unknown"]:
                rows[(task, group.get("class"), role, split)] += 1
    for key in sorted(rows):
        print("{}\t{}\t{}\t{}\t{}".format(*key, rows[key]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="校验声明并输出摘要")
    _common(check)
    check.add_argument("--json", action="store_true", help="输出机器可读的摘要与问题列表")
    check.add_argument("--strict-warnings", action="store_true", help="警告也返回非零退出码")
    check.set_defaults(handler=_run_check)

    write = subparsers.add_parser("write", help="写出派生总账 JSON")
    _common(write)
    write.add_argument("--output", default=None, help=f"输出路径；默认 {DEFAULT_OUTPUT}")
    write.add_argument("--strict-warnings", action="store_true", help="警告也返回非零退出码")
    write.set_defaults(handler=_run_write)

    query = subparsers.add_parser("query", help="查询某任务/角色/split 可用的背景父组")
    query.add_argument("--root", default=None, help="工作空间根目录")
    query.add_argument("--task", required=True, choices=cfg.DEFAULT_TASKS)
    query.add_argument("--role", required=True, choices=sorted(ROLES))
    query.add_argument("--split", required=True, choices=sorted(SPLITS - {"inherit"}))
    query.set_defaults(handler=_run_query)

    matrix = subparsers.add_parser("matrix", help="打印任务×类别×角色×split 父组矩阵")
    _common(matrix)
    matrix.set_defaults(handler=_run_matrix)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"{PROG}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
