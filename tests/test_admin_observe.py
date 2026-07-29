"""Loopback admin room-viewer: observe payload allowlist, the broadcast
fan-out tap, and the admin-channel handshake.

The admin surface lets an operator watch any room's conversation as TEXT on a
loopback-only port (ARCHITECTURE section 9). These tests pin the read-only/allowlist
invariants and the authz; the dual-port wiring + the live forward path are
exercised on hardware (the cross-loop forward is awkward under TestClient).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from glados.core.protocols import (
    AssistantDelta,
    ToolCall,
    ToolConfirmRequest,
    ToolResult,
    TtsChunk,
    UserTranscript,
)
from glados.core.server import (
    _make_notify_observers,
    _observed_payload,
    build_admin_app,
    build_app,
)


class FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)


class DeadWS:
    async def send_json(self, data: dict) -> None:
        raise RuntimeError("socket gone")


# ---- _observed_payload: the forward allowlist ---------------------------


def test_observed_payload_forwards_text_turn_events() -> None:
    assert _observed_payload(AssistantDelta(session_id="s", text="hi"))["text"] == "hi"
    t = _observed_payload(UserTranscript(session_id="s", text="yo", source="voice"))
    assert t["source"] == "voice"


def test_observed_payload_drops_audio_and_confirm() -> None:
    assert (
        _observed_payload(
            TtsChunk(session_id="s", seq=0, sample_rate=22_050, pcm_b64="AA==")
        )
        is None
    )
    assert (
        _observed_payload(
            ToolConfirmRequest(
                session_id="s", request_id="r", tool="t", args_summary={}, ttl_s=1.0
            )
        )
        is None
    )


def test_observed_payload_minimizes_tool_result() -> None:
    # Raw content can hold sensitive tool output + <external> bytes -- drop it.
    p = _observed_payload(
        ToolResult(session_id="s", call_id="c", ok=True, content={"secret": "x"})
    )
    assert "content" not in p
    assert p["ok"] is True and p["call_id"] == "c"


def test_observed_payload_minimizes_tool_call_args() -> None:
    # Args carry the same sensitive material as results -- drop them, keep name.
    p = _observed_payload(
        ToolCall(
            session_id="s",
            call_id="c",
            server="dunnes",
            name="bootstrap_login",
            args={"password": "hunter2"},
        )
    )
    assert "args" not in p
    assert p["server"] == "dunnes" and p["name"] == "bootstrap_login"


# ---- _make_notify_observers: the broadcast tap --------------------------


@pytest.mark.asyncio
async def test_notify_fans_out_only_to_observers_of_that_room() -> None:
    conns: dict = {}
    observed: dict = {}
    notify = _make_notify_observers(conns, observed)
    ws = FakeWS()
    conns["a"] = ws
    observed["a"] = "livingroom"

    await notify("livingroom", AssistantDelta(session_id="s", text="hi"))
    assert len(ws.sent) == 1
    env = ws.sent[0]
    assert env["type"] == "observed_event"
    assert env["room_id"] == "livingroom"
    assert env["event"]["text"] == "hi"

    # An event for a different room is not delivered to this observer.
    await notify("desk", AssistantDelta(session_id="s", text="elsewhere"))
    assert len(ws.sent) == 1

    # A non-allowlisted type (audio) is never forwarded.
    await notify(
        "livingroom", TtsChunk(session_id="s", seq=1, sample_rate=22_050, pcm_b64="AA==")
    )
    assert len(ws.sent) == 1


@pytest.mark.asyncio
async def test_notify_drops_dead_socket_without_raising() -> None:
    conns: dict = {"a": DeadWS()}
    observed: dict = {"a": "r"}
    notify = _make_notify_observers(conns, observed)
    await notify("r", AssistantDelta(session_id="s", text="hi"))  # must not raise
    assert "a" not in conns and "a" not in observed


# ---- admin-channel handshake authz --------------------------------------


def _admin_client(token: str = "adm-secret"):
    os.environ["GLADOS_CONFIG_DIR"] = str(Path(__file__).parent.parent / "configs")
    from glados.core.secrets import InMemorySecrets

    app = build_app()
    secrets = InMemorySecrets()
    secrets.set("admin", "observe-token", token)
    app.state.secrets = secrets
    return app, TestClient(build_admin_app(app))


def test_admin_index_serves_the_viewer_page() -> None:
    _app, client = _admin_client()
    r = client.get("/")
    assert r.status_code == 200
    # The page must speak the live protocol -- assert the message types so a
    # schema rename breaks this test, not the operator's debugging session.
    for token in (
        "admin_hello",
        "hello_ack",
        "observe_room",
        "observed_event",
        "auth_failed",
    ):
        assert token in r.text


def test_admin_rejects_bad_secret() -> None:
    _app, client = _admin_client()
    with client.websocket_connect("/ws/admin") as ws:
        ws.send_json({"type": "admin_hello", "token": "wrong"})
        msg = ws.receive_json()
    assert msg["type"] == "error" and msg["code"] == "auth_failed"


def test_admin_rejects_when_secret_unset() -> None:
    # No admin secret configured -> even a plausible token is refused (the
    # `not expected` branch), never an open-by-default admin surface.
    os.environ["GLADOS_CONFIG_DIR"] = str(Path(__file__).parent.parent / "configs")
    app = build_app()
    # leave app.state.secrets as the default; ensure no admin secret resolves
    from glados.core.secrets import InMemorySecrets

    app.state.secrets = InMemorySecrets()  # empty
    client = TestClient(build_admin_app(app))
    with client.websocket_connect("/ws/admin") as ws:
        ws.send_json({"type": "admin_hello", "token": "anything"})
        msg = ws.receive_json()
    assert msg["type"] == "error" and msg["code"] == "auth_failed"


def test_admin_rejects_observe_before_hello() -> None:
    # The first message MUST be an AdminHello; an ObserveRoom first is refused.
    _app, client = _admin_client()
    with client.websocket_connect("/ws/admin") as ws:
        ws.send_json({"type": "observe_room", "room_id": "livingroom"})
        msg = ws.receive_json()
    assert msg["type"] == "error"


def test_admin_good_secret_acks_with_room_list() -> None:
    _app, client = _admin_client()
    with client.websocket_connect("/ws/admin") as ws:
        ws.send_json({"type": "admin_hello", "token": "adm-secret"})
        ack = ws.receive_json()
    assert ack["type"] == "hello_ack"
    assert "livingroom" in ack["rooms"]


def test_admin_observe_updates_registry_then_cleans_up_on_disconnect() -> None:
    app, client = _admin_client()
    with client.websocket_connect("/ws/admin") as ws:
        ws.send_json({"type": "admin_hello", "token": "adm-secret"})
        ws.receive_json()  # hello_ack
        ws.send_json({"type": "observe_room", "room_id": "livingroom"})
        # Round-trip a bad message to force the server to process the prior
        # observe_room before we inspect shared state.
        ws.send_json({"type": "nonsense"})
        ws.receive_json()  # error for the nonsense message
        assert "livingroom" in app.state.admin_observed.values()
    # After the socket closes, the handler's finally clears the registries.
    assert app.state.admin_conns == {}
    assert app.state.admin_observed == {}
