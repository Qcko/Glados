"""End-to-end WS test: hello + auth + echo turn."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """See test_audio_pipeline.py for the rationale on context-managing
    TestClient — needed so each module's room workers are torn down before
    the next module's TestClient creates its own event loop."""
    os.environ["GLADOS_CONFIG_DIR"] = str(Path(__file__).parent.parent / "configs")
    os.environ["GLADOS_LLM_BACKEND"] = "fake"
    # Import after env vars are set so server picks up correct config.
    from glados.core.server import app

    # /healthz is loopback-gated; present as a loopback caller (the WS
    # handshake test below is unaffected by the peer address).
    with TestClient(app, client=("127.0.0.1", 12345)) as c:
        yield c


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_handshake_and_echo(client: TestClient) -> None:
    with client.websocket_connect("/ws/v1") as ws:
        ws.send_json(
            {
                "type": "hello",
                "client_id": "desk-ui",
                "room_id": "desk",
                "role": "ui",
                "token": "dev-token-desk",
            }
        )
        ws.send_json({"type": "user_text", "text": "ping"})
        welcome = ws.receive_json()
        transcript = ws.receive_json()
        delta = ws.receive_json()
        tts = ws.receive_json()
        outcome = ws.receive_json()
        done = ws.receive_json()

    assert welcome["type"] == "welcome"
    assert transcript["type"] == "user_transcript"
    assert transcript["text"] == "ping"
    assert transcript["source"] == "text"
    assert delta["type"] == "assistant_delta"
    assert delta["text"] == "echo: ping"
    assert tts["type"] == "tts_chunk"
    assert outcome["type"] == "turn_outcome"
    assert done["type"] == "done"
    assert (
        welcome["session_id"]
        == transcript["session_id"]
        == delta["session_id"]
        == tts["session_id"]
        == done["session_id"]
    )


def test_bad_token_rejected(client: TestClient) -> None:
    with client.websocket_connect("/ws/v1") as ws:
        ws.send_json(
            {
                "type": "hello",
                "client_id": "desk-ui",
                "room_id": "desk",
                "role": "ui",
                "token": "wrong",
            }
        )
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "auth_failed"
