#!/usr/bin/env python3
"""Bridge the DSH MCP stdio protocol to Blender's bundled MCP socket add-on.

The official Blender add-on (Blender Lab, "MCP") does not speak Model Context
Protocol itself. It listens on a TCP port and accepts null-byte-delimited JSON
requests of the form ``{"type": "execute", "code": ..., "strict_json": true}``,
executing the given Python inside Blender and returning
``{"status": "ok", "result": {...}}``.

This script is a dependency-free stdio MCP server that exposes that socket as
MCP tools. Every tool call is one socket connection; the add-on closes the
connection after writing the response.

Usage as an MCP server (DSH stdio capability)::

    python lab/scripts/blender/blender_mcp_bridge.py

Usage as a one-shot client, useful for checks without an MCP client::

    python lab/scripts/blender/blender_mcp_bridge.py --code "import bpy; result = {...}"
    python lab/scripts/blender/blender_mcp_bridge.py --code-file path/to/snippet.py

Environment: BLENDER_MCP_HOST (default localhost), BLENDER_MCP_PORT (9876).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys

HOST = os.environ.get("BLENDER_MCP_HOST", "localhost")
PORT = int(os.environ.get("BLENDER_MCP_PORT", "9876"))
DEFAULT_TIMEOUT = 300.0
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024

STATUS_CODE = """
import bpy
scene = bpy.context.scene
result = {
    "blender": bpy.app.version_string,
    "file": bpy.data.filepath,
    "scene": scene.name,
    "frame_current": scene.frame_current,
    "frame_start": scene.frame_start,
    "frame_end": scene.frame_end,
    "engine": scene.render.engine,
    "resolution": [scene.render.resolution_x, scene.render.resolution_y,
                   scene.render.resolution_percentage],
    "camera": scene.camera.name if scene.camera else None,
    "objects": len(bpy.data.objects),
}
"""

TOOLS = [
    {
        "name": "blender_execute",
        "description": (
            "Execute Python code inside the running Blender instance and return the JSON "
            "result. The code must assign a JSON-serializable dict to the name `result`; "
            "anything printed to stdout is returned separately. `bpy` is importable. "
            "Blocked by the add-on: sys.exit, bpy.ops.wm.quit_blender, "
            "bpy.ops.wm.read_factory_settings, bpy.ops.wm.read_userpref, "
            "bpy.ops.wm.read_factory_userpref."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute inside Blender."},
                "timeout_seconds": {
                    "type": "number",
                    "description": "Socket timeout in seconds; raise it for renders. Default 300.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "blender_status",
        "description": (
            "Report the running Blender instance: version, open .blend path, active scene, "
            "current frame, frame range, render engine, resolution and camera."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


class BlenderError(RuntimeError):
    """Blender MCP socket replied with an error or an unusable payload."""


def execute(code: str, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Send one execute request to the Blender add-on and return its response."""
    payload = json.dumps({"type": "execute", "code": code, "strict_json": True}).encode("utf-8") + b"\0"
    with socket.create_connection((HOST, PORT), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(payload)
        buffer = bytearray()
        while b"\0" not in buffer:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buffer.extend(chunk)
            if len(buffer) > _MAX_RESPONSE_BYTES:
                raise BlenderError("response exceeded size limit")
    if b"\0" not in buffer:
        raise BlenderError("Blender closed the connection without a complete response")
    return json.loads(bytes(buffer.split(b"\0", 1)[0]).decode("utf-8"))


def call_tool(name: str, arguments: dict) -> dict:
    """Run one MCP tool and return an MCP tools/call result object."""
    if name == "blender_status":
        code = STATUS_CODE
        timeout = 30.0
    elif name == "blender_execute":
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return {"content": [{"type": "text", "text": "`code` must be a non-empty string"}],
                    "isError": True}
        timeout = float(arguments.get("timeout_seconds") or DEFAULT_TIMEOUT)
    else:
        return {"content": [{"type": "text", "text": f"unknown tool: {name}"}], "isError": True}

    try:
        response = execute(code, timeout)
    except (OSError, BlenderError, ValueError) as exc:
        return {"content": [{"type": "text",
                             "text": f"{type(exc).__name__}: {exc} (target {HOST}:{PORT})"}],
                "isError": True}

    text = json.dumps(response, indent=2, ensure_ascii=False)
    return {"content": [{"type": "text", "text": text}],
            "isError": response.get("status") != "ok"}


def _write(message: dict) -> None:
    """Write one JSON-RPC message as a single stdout line."""
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _error(msg_id: object, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def serve() -> int:
    """Serve MCP over stdio until stdin closes."""
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            _write(_error(None, -32700, "parse error"))
            continue
        if not isinstance(message, dict):
            _write(_error(None, -32600, "invalid request"))
            continue

        method = message.get("method")
        msg_id = message.get("id")
        if msg_id is None or method is None:
            # Notifications (e.g. notifications/initialized) need no reply.
            continue

        if method == "initialize":
            client_version = (message.get("params") or {}).get("protocolVersion")
            _write({"jsonrpc": "2.0", "id": msg_id, "result": {
                "protocolVersion": client_version or "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "blender-mcp-bridge", "version": "1.0.0"},
                "instructions": (
                    "Controls the running Blender GUI on "
                    f"{HOST}:{PORT}. Use blender_execute with Python that assigns `result`."
                ),
            }})
        elif method == "tools/list":
            _write({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                _write(_error(msg_id, -32602, "arguments must be an object"))
                continue
            _write({"jsonrpc": "2.0", "id": msg_id, "result": call_tool(name, arguments)})
        elif method == "ping":
            _write({"jsonrpc": "2.0", "id": msg_id, "result": {}})
        else:
            _write(_error(msg_id, -32601, f"method not found: {method}"))
    return 0


def main() -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code", help="one-shot: Python source to run in Blender, then exit")
    parser.add_argument("--code-file", help="one-shot: read Python source from a file, then exit")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="one-shot socket timeout")
    args = parser.parse_args()

    if args.code is not None or args.code_file is not None:
        code = args.code if args.code is not None else open(args.code_file, encoding="utf-8").read()
        print(json.dumps(execute(code, args.timeout), indent=2, ensure_ascii=False))
        return 0

    return serve()


if __name__ == "__main__":
    sys.exit(main())
