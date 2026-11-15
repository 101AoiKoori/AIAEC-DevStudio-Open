#!/usr/bin/env python3
"""Check active Markdown links, file targets and section anchors."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[3]
INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
HTML_ID = re.compile(r"<a\s+(?:id|name)=[\"']([^\"']+)[\"']", re.IGNORECASE)
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
EXTERNAL_SCHEMES = {
    "codex",
    "data",
    "http",
    "https",
    "mailto",
    "tel",
}
ACTIVE_GLOBS = (
    "AGENTS.md",
    "README.md",
    "models/**/*.md",
    "ws/**/*.md",
    "docs/**/*.md",
    "datasets/**/*.md",
    "lab/**/*.md",
    "refs/provincial/**/*.md",
    ".agents/skills/**/*.md",
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    target: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "target": self.target,
            "reason": self.reason,
        }


def active_documents(root: Path = ROOT) -> list[Path]:
    """Return tracked and untracked Markdown in documentation scope."""

    paths = []
    for pattern in ACTIVE_GLOBS:
        paths.extend(root.glob(pattern))
    return sorted(
        {
            path.resolve()
            for path in paths
            if path.is_file() and not path.is_symlink() and root.resolve() in path.resolve().parents
        }
    )


def visible_lines(text: str) -> list[tuple[int, str]]:
    """Yield lines outside fenced code blocks."""

    result = []
    fence = None
    for line_number, line in enumerate(text.splitlines(), 1):
        match = FENCE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None
            continue
        if fence is None:
            result.append((line_number, line))
    return result


def _heading_text(value: str) -> str:
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"[*_~]", "", value)
    return value.strip().lower()


def github_slug(value: str) -> str:
    """Return a GitHub-style heading slug for the Markdown used here."""

    value = _heading_text(value)
    value = "".join(
        character
        for character in value
        if character.isalnum() or character in {" ", "-", "_"}
    )
    value = re.sub(r"\s+", "-", value.strip())
    return re.sub(r"-+", "-", value)


def anchors(text: str) -> set[str]:
    """Collect heading and explicit HTML anchors, including duplicate suffixes."""

    result = set(match.group(1) for match in HTML_ID.finditer(text))
    counts: Counter[str] = Counter()
    for _, line in visible_lines(text):
        match = HEADING.match(line)
        if not match:
            continue
        slug = github_slug(match.group(2))
        occurrence = counts[slug]
        result.add(slug if occurrence == 0 else f"{slug}-{occurrence}")
        counts[slug] += 1
    return result


def _link_target(raw: str) -> str | None:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        return value[1 : value.index(">")]
    if not value or any(character.isspace() for character in value):
        value = re.split(r"\s+[\"'(]", value, maxsplit=1)[0].strip("\"'")
    return value or None


def check(root: Path = ROOT, documents: list[Path] | None = None) -> dict[str, object]:
    """Return link and anchor findings for active Markdown documents."""

    root = root.resolve()
    docs = documents or active_documents(root)
    findings: list[Finding] = []
    link_count = 0
    anchor_cache: dict[Path, set[str]] = {}
    for document in docs:
        try:
            text = document.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            findings.append(Finding(str(document.relative_to(root)), 1, "", f"unreadable: {exc}"))
            continue
        for line_number, line in visible_lines(text):
            for match in INLINE_LINK.finditer(line):
                target = _link_target(match.group(1))
                if target is None or target.startswith("#"):
                    fragment = unquote(target[1:]) if target and target.startswith("#") else ""
                    if fragment and fragment not in anchor_cache.setdefault(document, anchors(text)):
                        findings.append(
                            Finding(str(document.relative_to(root)), line_number, target, "missing-anchor")
                        )
                    link_count += 1
                    continue
                if SCHEME.match(target):
                    scheme = target.split(":", 1)[0].lower()
                    if scheme in EXTERNAL_SCHEMES:
                        continue
                    findings.append(
                        Finding(str(document.relative_to(root)), line_number, target, f"unsupported-scheme:{scheme}")
                    )
                    continue
                link_count += 1
                path_text, separator, requested_anchor = target.partition("#")
                path_text = unquote(path_text)
                candidate = (document.parent / path_text).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    findings.append(
                        Finding(str(document.relative_to(root)), line_number, target, "outside-workspace")
                    )
                    continue
                if not candidate.exists():
                    findings.append(
                        Finding(str(document.relative_to(root)), line_number, target, "missing-file")
                    )
                    continue
                if separator and requested_anchor and candidate.suffix.lower() == ".md":
                    requested_anchor = unquote(requested_anchor)
                    if requested_anchor not in anchor_cache.setdefault(
                        candidate, anchors(candidate.read_text(encoding="utf-8"))
                    ):
                        findings.append(
                            Finding(str(document.relative_to(root)), line_number, target, "missing-anchor")
                        )
    return {
        "documents": len(docs),
        "links": link_count,
        "findings": [finding.as_dict() for finding in findings],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", help="print structured results")
    args = parser.parse_args()
    try:
        report = check(args.root)
    except OSError as exc:
        print(f"documentation check failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for finding in report["findings"]:
            print(
                "{path}:{line}: {reason}: {target}".format(**finding)
            )
        print(
            "Checked {documents} Markdown files and {links} local links; "
            "{errors} findings.".format(errors=len(report["findings"]), **report)
        )
    return 1 if report["findings"] else 0


if __name__ == "__main__":
    sys.exit(main())
