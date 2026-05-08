"""FastAPI server exposing `/ws/v1`.

Handshake → register client connection → forward `user_text` to the
Organizer. The Organizer owns sessions, tool dispatch, and egress; this file
only handles wire I/O and connection bookkeeping.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, TypeAdapter, ValidationError

from ..brain.llm.fake import FakeLLM
from ..brain.llm.ollama import OllamaLLM
from ..mcp.registry import MCPRegistry
from ..servers.time_server import NowTool
from .adapters import LLM
from .config import (
    GladosConfig,
    LLMConfig,
    RoomsConfig,
    load_glados_config,
    load_rooms_config,
)
from .organizer import Organizer
from .protocols import (
    ClientMessage,
    ErrorMessage,
    Hello,
    UserText,
)
from .sessions import SessionRegistry
from .traces import TraceStore


CONFIG_DIR = Path(os.environ.get("GLADOS_CONFIG_DIR", "configs"))

_glados_cfg: GladosConfig = load_glados_config(CONFIG_DIR / "glados.toml")
_rooms_cfg: RoomsConfig = load_rooms_config(CONFIG_DIR / "rooms.toml")
_client_msg = TypeAdapter(ClientMessage)

def _build_llm(cfg: LLMConfig) -> LLM:
    if cfg.backend == "ollama":
        return OllamaLLM(
            host=cfg.host,
            model=cfg.model,
            temperature=cfg.temperature,
            timeout=cfg.timeout,
        )
    return FakeLLM()


_traces = TraceStore(_glados_cfg.server.traces_dir)
_mcp = MCPRegistry()
_mcp.register(NowTool())
_llm: LLM = _build_llm(_glados_cfg.llm)
_connections: dict[str, WebSocket] = {}


async def _send(client_id: str, msg: BaseModel) -> None:
    ws = _connections.get(client_id)
    if ws is not None:
        await ws.send_json(msg.model_dump())


def _clients_in_room(room_id: str) -> list[str]:
    return [
        c.client_id
        for c in _rooms_cfg.clients
        if c.room_id == room_id and c.client_id in _connections
    ]


_sessions = SessionRegistry()
_organizer = Organizer(
    llm=_llm,
    mcp=_mcp,
    traces=_traces,
    sessions=_sessions,
    send=_send,
    binding_for_client=_rooms_cfg.find,
    clients_in_room=_clients_in_room,
)

app = FastAPI(title="GLaDOS", version="0.1.0")
_CLIENT_INDEX = Path(__file__).resolve().parent.parent / "client_web" / "index.html"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_CLIENT_INDEX)


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "ok": True,
        "rooms": len({c.room_id for c in _rooms_cfg.clients}),
        "tools": [s.qualified for s in _mcp.specs()],
    }


@app.websocket("/ws/v1")
async def ws_v1(ws: WebSocket) -> None:
    await ws.accept()
    client_id: str | None = None
    try:
        binding = await _handshake(ws)
        if binding is None:
            return
        client_id = binding.client_id
        await _replace_connection(client_id, ws)
        await _serve(ws, client_id)
    except WebSocketDisconnect:
        return
    finally:
        if client_id is not None and _connections.get(client_id) is ws:
            del _connections[client_id]


async def _handshake(ws: WebSocket):
    raw = await ws.receive_json()
    try:
        msg = _client_msg.validate_python(raw)
    except ValidationError as e:
        await _send_error(ws, "bad_message", str(e))
        await ws.close()
        return None
    if not isinstance(msg, Hello):
        await _send_error(ws, "expected_hello", "first message must be hello")
        await ws.close()
        return None

    expected = _glados_cfg.auth.tokens.get(msg.client_id)
    if expected is None or expected != msg.token:
        await _send_error(ws, "auth_failed", "unknown client or bad token")
        await ws.close()
        return None

    binding = _rooms_cfg.find(msg.client_id)
    if binding is None:
        await _send_error(ws, "unbound_client", f"{msg.client_id} not in rooms.toml")
        await ws.close()
        return None
    if binding.room_id != msg.room_id or binding.role != msg.role:
        await _send_error(ws, "binding_mismatch", "room/role differs from rooms.toml")
        await ws.close()
        return None

    return binding


async def _serve(ws: WebSocket, client_id: str) -> None:
    while True:
        raw = await ws.receive_json()
        try:
            msg = _client_msg.validate_python(raw)
        except ValidationError as e:
            await _send_error(ws, "bad_message", str(e))
            continue

        if isinstance(msg, UserText):
            await _organizer.handle_user_text(client_id, msg.text)
        # other types are accepted but no-op in v0


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    await ws.send_json(ErrorMessage(code=code, message=message).model_dump())


async def _replace_connection(client_id: str, ws: WebSocket) -> None:
    prev = _connections.get(client_id)
    if prev is not None and prev is not ws:
        try:
            await prev.close()
        except Exception:
            pass
    _connections[client_id] = ws
