"""End-to-end WS test: hello + auth + echo turn."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    os.environ["GLADOS_CONFIG_DIR"] = str(Path(__file__).parent.parent / "configs")
    os.environ["GLADOS_LLM_BACKEND"] = "fake"
    # Import after env vars are set so server picks up correct config.
    from glados.core.server import app

    return TestClient(app)


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
        delta = ws.receive_json()
        tts = ws.receive_json()
        done = ws.receive_json()

    assert welcome["type"] == "welcome"
    assert delta["type"] == "assistant_delta"
    assert delta["text"] == "echo: ping"
    assert tts["type"] == "tts_chunk"
    assert done["type"] == "done"
    assert welcome["session_id"] == delta["session_id"] == tts["session_id"] == done["session_id"]


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
