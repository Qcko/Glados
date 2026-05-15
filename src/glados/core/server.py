"""FastAPI server exposing `/ws/v1`.

Handshake → register client connection → forward `user_text` to the
Organizer. The Organizer owns sessions, tool dispatch, and egress; this file
only handles wire I/O and connection bookkeeping.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, TypeAdapter, ValidationError

from ..brain.llm.fake import FakeLLM
from ..brain.llm.ollama import OllamaLLM
from ..mcp.registry import MCPRegistry
from ..servers.time_server import NowTool
from ..servers.toy_server import TOY_TOOLS
from .adapters import LLM
from .audio_sink import AudioSink, FrameTooShort
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
for _tool in TOY_TOOLS:
    _mcp.register(_tool)
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
# Source-checkout layout only: src/glados/core/server.py → repo root is parents[3].
# GLaDOS is self-hosted, not pip-distributed; if that ever changes, ship the
# built client as package data and resolve via importlib.resources.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLIENT_DIST = _REPO_ROOT / "client_web" / "dist"
_CLIENT_INDEX = _CLIENT_DIST / "index.html"
_CLIENT_ASSETS = _CLIENT_DIST / "assets"

if _CLIENT_ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=_CLIENT_ASSETS), name="assets")


_NOT_BUILT_HTML = (
    "<!doctype html><meta charset=utf-8><title>GLaDOS</title>"
    "<style>body{font-family:system-ui;background:#0e0f12;color:#e6e9ef;"
    "padding:2rem;max-width:40rem;margin:auto}"
    "code{background:#16181d;padding:.15rem .35rem;border-radius:3px}</style>"
    "<h1>Client not built</h1>"
    "<p>Run <code>cd client_web &amp;&amp; npm install &amp;&amp; npm run build</code>"
    " (or <code>npm run dev</code> for HMR on port 5173).</p>"
)


@app.get("/")
async def index() -> Response:
    if _CLIENT_INDEX.is_file():
        return FileResponse(_CLIENT_INDEX)
    return HTMLResponse(_NOT_BUILT_HTML, status_code=503)


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
    sink: AudioSink | None = None
    try:
        binding = await _handshake(ws)
        if binding is None:
            return
        client_id = binding.client_id
        await _replace_connection(client_id, ws)
        sink = AudioSink(_glados_cfg.server.traces_dir, client_id)
        await _serve(ws, client_id, sink)
    except WebSocketDisconnect:
        return
    finally:
        if sink is not None:
            sink.close()
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


async def _serve(ws: WebSocket, client_id: str, sink: AudioSink) -> None:
    while True:
        event = await ws.receive()
        if event["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(event.get("code", 1000))
        if (data := event.get("bytes")) is not None:
            await _handle_audio(ws, sink, data)
            continue
        if (text := event.get("text")) is None:
            # ASGI permits a websocket.receive with neither payload set;
            # ignore rather than spin or crash.
            continue
        try:
            msg = _client_msg.validate_json(text)
        except ValidationError as e:
            await _send_error(ws, "bad_message", str(e))
            continue
        if isinstance(msg, UserText):
            await _organizer.handle_user_text(client_id, msg.text)
        # other types are accepted but no-op in v0


async def _handle_audio(ws: WebSocket, sink: AudioSink, data: bytes) -> None:
    try:
        sink.write(data)
    except FrameTooShort as e:
        await _send_error(ws, "bad_audio_frame", str(e))


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
