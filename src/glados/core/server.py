"""FastAPI server exposing `/ws/v1`.

v0 scope: hello + auth check, echo `user_text` back as a single
`assistant_delta` then `done`. Organizer and LLM are wired in subsequent
slices; this file is intentionally small.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from .config import GladosConfig, RoomsConfig, load_glados_config, load_rooms_config
from .protocols import (
    AssistantDelta,
    ClientMessage,
    Done,
    ErrorMessage,
    Hello,
    UserText,
    Welcome,
)


CONFIG_DIR = Path(os.environ.get("GLADOS_CONFIG_DIR", "configs"))

_glados_cfg: GladosConfig = load_glados_config(CONFIG_DIR / "glados.toml")
_rooms_cfg: RoomsConfig = load_rooms_config(CONFIG_DIR / "rooms.toml")
_client_msg = TypeAdapter(ClientMessage)

app = FastAPI(title="GLaDOS", version="0.1.0")


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "rooms": len({c.room_id for c in _rooms_cfg.clients})}


@app.websocket("/ws/v1")
async def ws_v1(ws: WebSocket) -> None:
    await ws.accept()
    try:
        binding = await _handshake(ws)
        if binding is None:
            return
        await _serve(ws, binding_room=binding.room_id, binding_user=binding.default_user)
    except WebSocketDisconnect:
        return


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


async def _serve(ws: WebSocket, *, binding_room: str, binding_user: str) -> None:
    while True:
        raw = await ws.receive_json()
        try:
            msg = _client_msg.validate_python(raw)
        except ValidationError as e:
            await _send_error(ws, "bad_message", str(e))
            continue

        if isinstance(msg, UserText):
            await _echo_turn(ws, binding_room, binding_user, msg.text)
        # other types are accepted but no-op in v0


async def _echo_turn(ws: WebSocket, room: str, user: str, text: str) -> None:
    session_id = f"{room}:{user}:{uuid.uuid4().hex[:8]}"
    await ws.send_json(Welcome(session_id=session_id).model_dump())
    await ws.send_json(
        AssistantDelta(session_id=session_id, text=f"echo: {text}").model_dump()
    )
    await ws.send_json(Done(session_id=session_id).model_dump())


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    await ws.send_json(ErrorMessage(code=code, message=message).model_dump())
