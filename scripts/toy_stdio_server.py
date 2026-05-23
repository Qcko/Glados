"""Standalone MCP-shape stdio server exposing the toy tools.

This is the v2 stdio-MCP soak-test: re-implements `echo`, `add`,
`roll_dice` from `glados.servers.toy_server` as a real subprocess that
speaks line-delimited JSON-RPC on stdin/stdout. The stdio client adapter
in `glados.mcp.stdio_client` spawns this and proves the transport
against a server we control before any third-party MCP shows up.

Protocol (subset of MCP, line-delimited JSON-RPC 2.0):
  -> {"jsonrpc":"2.0","id":1,"method":"initialize"}
  <- {"jsonrpc":"2.0","id":1,"result":{"server":"toy_stdio"}}
  -> {"jsonrpc":"2.0","id":2,"method":"tools/list"}
  <- {"jsonrpc":"2.0","id":2,"result":{"tools":[<ToolSpec>...]}}
  -> {"jsonrpc":"2.0","id":3,"method":"tools/call",
      "params":{"name":"echo","arguments":{"text":"hi"}}}
  <- {"jsonrpc":"2.0","id":3,"result":{"ok":true,"content":{"text":"hi"}}}

Errors come back as `{"jsonrpc":"2.0","id":N,"error":{"code":...,"message":...}}`.
Tool-level failure (bad args, etc.) still returns a `result` with
`ok:false` — JSON-RPC-level `error` is reserved for protocol problems
(unknown method, malformed request).
"""

from __future__ import annotations

import json
import random
import sys
from typing import Any


_TOOLS = [
    {
        "server": "toy_stdio",
        "name": "echo",
        "description": "Echo a string back to the caller.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "server": "toy_stdio",
        "name": "add",
        "description": "Return the sum of two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
    {
        "server": "toy_stdio",
        "name": "roll_dice",
        "description": "Roll `count` dice with `sides` faces each.",
        "parameters": {
            "type": "object",
            "properties": {
                "sides": {"type": "integer", "minimum": 2, "maximum": 1000},
                "count": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["sides"],
            "additionalProperties": False,
        },
        # Gated by the v2 permission framework so we have a real
        # confirm-prompt path to exercise from the demo without
        # inventing a side-effecting tool. Rolling dice isn't actually
        # destructive — it's just here as scaffolding.
        "requires_confirmation": True,
    },
]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _call_echo(args: dict) -> dict:
    text = args.get("text")
    if not isinstance(text, str):
        return {"ok": False, "error": "`text` must be a string"}
    return {"ok": True, "content": {"text": text}}


def _call_add(args: dict) -> dict:
    a, b = args.get("a"), args.get("b")
    if not _is_number(a) or not _is_number(b):
        return {"ok": False, "error": "`a` and `b` must be numbers"}
    return {"ok": True, "content": {"sum": a + b}}


def _call_roll_dice(args: dict) -> dict:
    sides = args.get("sides")
    count = args.get("count", 1)
    if not _is_int(sides) or not 2 <= sides <= 1000:
        return {"ok": False, "error": "`sides` must be an integer in [2, 1000]"}
    if not _is_int(count) or not 1 <= count <= 100:
        return {"ok": False, "error": "`count` must be an integer in [1, 100]"}
    rolls = [random.randint(1, sides) for _ in range(count)]
    return {"ok": True, "content": {"rolls": rolls, "total": sum(rolls)}}


_DISPATCH = {
    "echo": _call_echo,
    "add": _call_add,
    "roll_dice": _call_roll_dice,
}


def _handle(req: dict) -> dict:
    rid = req.get("id")
    method = req.get("method")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {"server": "toy_stdio"}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": _TOOLS}}
    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        fn = _DISPATCH.get(name)
        if fn is None:
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": -32601, "message": f"unknown tool: {name}"},
            }
        return {"jsonrpc": "2.0", "id": rid, "result": fn(args)}
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"unknown method: {method}"},
    }


def main() -> None:
    # Line-buffered I/O so the parent process sees each response promptly
    # without explicit flushes scattered through the handler.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"parse error: {e}"},
                    }
                )
                + "\n"
            )
            sys.stdout.flush()
            continue
        resp = _handle(req)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
