#!/usr/bin/env python3
"""Check the whole Blender MCP chain for this workspace (DSH-specific).

Verifies, in order:

1. the Blender add-on socket accepts a TCP connection,
2. one ``execute`` round-trip returns a usable scene report,
3. the DSH profile patch contains the ``mcp-blender`` registration row.

Prints one JSON object on stdout and exits 0 only when every step passes.
Reuses the bridge in ``lab/scripts/blender/blender_mcp_bridge.py`` rather than re-implementing
the framing.
"""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "lab/scripts/blender"))

from blender_mcp_bridge import HOST, PORT, STATUS_CODE, execute  # noqa: E402

REQUIRED_STATUS_KEYS = ("blender", "file", "scene", "frame_start", "frame_end", "engine", "camera")
PATCH_GLOBS = ("cordis.patch.yml",)


def tcp_ok() -> str | None:
    """Return None when the port accepts a connection, else the failure text."""
    try:
        with socket.create_connection((HOST, PORT), timeout=5):
            return None
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"


def registration() -> dict:
    """Scan DSH profile patches for the mcp-blender entry (informational)."""
    profiles = Path.home() / ".dsh" / "profiles"
    found = []
    for pattern in PATCH_GLOBS:
        for patch in sorted(profiles.glob(f"*/{pattern}")):
            try:
                text = patch.read_text(encoding="utf8")
            except OSError as exc:
                found.append({"profile": patch.parent.name, "path": str(patch),
                              "error": f"{type(exc).__name__}: {exc}"})
                continue
            if "mcp-blender" in text or "mcp__blender" in text:
                found.append({
                    "profile": patch.parent.name,
                    "path": str(patch),
                    "has_mcp_client": "@deepseek-ai/dsh-mcp-client" in text,
                    "has_server_name": "serverName: blender" in text,
                })
    return {"profiles_dir": str(profiles), "patches_with_blender": found}


def main() -> int:
    report: dict = {
        "target": f"{HOST}:{PORT}",
        "bridge": str(ROOT / "lab/scripts/blender/blender_mcp_bridge.py"),
        "workspace": str(ROOT),
    }

    report["tcp"] = "ok" if (err := tcp_ok()) is None else err
    if report["tcp"] != "ok":
        report["ok"] = False
        report["hint"] = "Blender 未启动，或 MCP 扩展的 Server 未 running，或端口与 BLENDER_MCP_PORT 不符"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    try:
        response = execute(STATUS_CODE, timeout=30)
    except (OSError, ValueError) as exc:
        report["execute"] = f"{type(exc).__name__}: {exc}"
        report["ok"] = False
        report["hint"] = "TCP 可连但请求无响应：确认请求以 \\0 结束、strict_json 为布尔"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    report["execute_status"] = response.get("status")
    result = response.get("result") or {}
    missing = [key for key in REQUIRED_STATUS_KEYS if key not in result]
    report["scene"] = {key: result.get(key) for key in REQUIRED_STATUS_KEYS}
    report["missing_keys"] = missing
    report["registration"] = registration()

    report["ok"] = response.get("status") == "ok" and not missing
    if not report["ok"]:
        report["hint"] = "代码执行失败或返回字段不全，检查 Blender 内 traceback"
    elif not report["registration"]["patches_with_blender"]:
        report["hint"] = "连接正常但没有找到 mcp-blender 注册行；按 SKILL.md 补 profile patch 后热重载"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
