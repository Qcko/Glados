"""Standalone real-MCP stdio server exposing the toy tools.

Re-implements `echo`, `add`, `roll_dice` from `glados.servers.toy_server`
as a real subprocess that speaks the Model Context Protocol over stdio.
Used both as a soak-test for GLaDOS's `StdioServer` adapter and as a
worked example of the wire shape any third-party server must use.

Protocol (line-delimited JSON-RPC 2.0):
  -> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}
  <- {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05",
        "capabilities":{"tools":{}},"serverInfo":{"name":"toy_stdio",...}}}
  -> {"jsonrpc":"2.0","method":"notifications/initialized"}     (no resp)
  -> {"jsonrpc":"2.0","id":2,"method":"tools/list"}
  <- {"jsonrpc":"2.0","id":2,"result":{"tools":[
        {"name":"echo","description":"...","inputSchema":{...}}, ...]}}
  -> {"jsonrpc":"2.0","id":3,"method":"tools/call",
        "params":{"name":"echo","arguments":{"text":"hi"}}}
  <- {"jsonrpc":"2.0","id":3,"result":{
        "content":[{"type":"text","text":"{\"text\":\"hi\"}"}],
        "isError":false}}

Structured tool data rides as JSON-encoded strings inside text content
blocks -- the MCP wire schema has no first-class object response.
GLaDOS's StdioServer parses those text blocks back into dicts before
handing them to the LLM.

`requires_confirmation` is NOT on the wire. The corresponding overlay
for `roll_dice` lives in `configs/servers.toml` `[server.tool_overlays]`.
"""

from __future__ import annotations

import json
import random
import sys
from typing import Any


_PROTOCOL_VERSION = "2024-11-05"

_TOOLS = [
    {
        "name": "echo",
        "description": "Echo a string back to the caller.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "add",
        "description": "Return the sum of two numbers.",
        "inputSchema": {
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
        "name": "roll_dice",
        "description": "Roll `count` dice with `sides` faces each.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sides": {"type": "integer", "minimum": 2, "maximum": 1000},
                "count": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["sides"],
            "additionalProperties": False,
        },
    },
]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _call_echo(args: dict) -> tuple[bool, str]:
    text = args.get("text")
    if not isinstance(text, str):
        return False, "`text` must be a string"
    return True, json.dumps({"text": text})


def _call_add(args: dict) -> tuple[bool, str]:
    a, b = args.get("a"), args.get("b")
    if not _is_number(a) or not _is_number(b):
        return False, "`a` and `b` must be numbers"
    return True, json.dumps({"sum": a + b})


def _call_roll_dice(args: dict) -> tuple[bool, str]:
    sides = args.get("sides")
    count = args.get("count", 1)
    if not _is_int(sides) or not 2 <= sides <= 1000:
        return False, "`sides` must be an integer in [2, 1000]"
    if not _is_int(count) or not 1 <= count <= 100:
        return False, "`count` must be an integer in [1, 100]"
    rolls = [random.randint(1, sides) for _ in range(count)]
    return True, json.dumps({"rolls": rolls, "total": sum(rolls)})


_DISPATCH = {
    "echo": _call_echo,
    "add": _call_add,
    "roll_dice": _call_roll_dice,
}


def _initialize_result() -> dict:
    return {
        "protocolVersion": _PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "toy_stdio", "version": "0.2"},
    }


def _tool_call_result(name: str, args: dict) -> dict:
    fn = _DISPATCH.get(name)
    if fn is None:
        return {
            "content": [{"type": "text", "text": f"unknown tool: {name}"}],
            "isError": True,
        }
    ok, text = fn(args)
    return {
        "content": [{"type": "text", "text": text}],
        "isError": not ok,
    }


def _handle(req: dict) -> dict | None:
    """Return a JSON-RPC response dict, or None for notifications (no
    response expected)."""
    rid = req.get("id")
    method = req.get("method")
    # Notifications have no `id`; per JSON-RPC spec we send no response.
    if rid is None:
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": _initialize_result()}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": _TOOLS}}
    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name", "")
        args = params.get("arguments") or {}
        return {"jsonrpc": "2.0", "id": rid, "result": _tool_call_result(name, args)}
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"unknown method: {method}"},
    }


def main() -> None:
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
        if resp is None:
            continue
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
