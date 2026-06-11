"""End-to-end WS test: hello + auth + echo turn."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


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


# ---- handshake admission gate (caps + lockout + timeout) ----------------
#
# These use a fresh app per test (not the module fixture) because they
# deliberately poison the gate's per-IP state, and swap in a gate with a
# fake clock to drive lockout expiry without sleeping.


def _fresh_client() -> TestClient:
    os.environ["GLADOS_CONFIG_DIR"] = str(Path(__file__).parent.parent / "configs")
    from glados.core.server import build_app

    app = build_app()
    return TestClient(app, client=("127.0.0.1", 12345))


def _swap_gate(client: TestClient, **cfg_overrides):
    from glados.core.config import HandshakeConfig
    from glados.core.handshake_gate import HandshakeGate

    class _Clock:
        now = 1000.0

        def __call__(self) -> float:
            return self.now

    clock = _Clock()
    gate = HandshakeGate(HandshakeConfig(**cfg_overrides), clock=clock)
    client.app.state.handshake_gate = gate
    return gate, clock


def _hello(ws, token: str) -> None:
    ws.send_json(
        {
            "type": "hello",
            "client_id": "desk-ui",
            "room_id": "desk",
            "role": "ui",
            "token": token,
        }
    )


def test_lockout_engages_after_repeated_bad_tokens_and_recovers() -> None:
    with _fresh_client() as client:
        gate, clock = _swap_gate(client, fail_threshold=3, lockout_s=30.0)
        for _ in range(3):
            with client.websocket_connect("/ws/v1") as ws:
                _hello(ws, "wrong")
                assert ws.receive_json()["code"] == "auth_failed"
        # Locked out: rejected before the hello is even read, with a
        # distinct retryable code — not a misleading auth_failed.
        with client.websocket_connect("/ws/v1") as ws:
            err = ws.receive_json()
        assert err["code"] == "rate_limited"
        # A correct token from the same IP is still locked out (the
        # lockout is real), but only until it expires...
        clock.now += 31.0
        with client.websocket_connect("/ws/v1") as ws:
            _hello(ws, "dev-token-desk")
            ws.send_json({"type": "user_text", "text": "ping"})
            assert ws.receive_json()["type"] == "welcome"
        assert gate.pending_count == 0  # leak canary


def test_busy_refused_before_accept_when_pending_cap_reached() -> None:
    # The pre-accept peek refuses the upgrade outright under flood — no
    # 101, no server_busy frame — so rejects stay cheaper than the attack.
    with _fresh_client() as client:
        _swap_gate(client, max_pending=0)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/v1"):
                pass


def test_peek_admit_race_still_gets_server_busy_frame() -> None:
    # Capacity can fill between the pre-accept peek and the authoritative
    # admit() (during the ws.accept() suspension). The raced peer must get
    # the explanatory server_busy frame, not a hang or a 500.
    from glados.core.handshake_gate import Verdict

    with _fresh_client() as client:
        gate, _ = _swap_gate(client, max_pending=0)
        gate.peek = lambda ip: Verdict.OK  # simulate losing the race
        with client.websocket_connect("/ws/v1") as ws:
            err = ws.receive_json()
        assert err["code"] == "server_busy"


def test_legitimate_client_unaffected_by_other_ip_lockout() -> None:
    with _fresh_client() as client:
        gate, _ = _swap_gate(client, fail_threshold=3)
        for _ in range(3):
            gate.record_failure("192.168.50.99")
        with client.websocket_connect("/ws/v1") as ws:  # 127.0.0.1
            _hello(ws, "dev-token-desk")
            ws.send_json({"type": "user_text", "text": "ping"})
            assert ws.receive_json()["type"] == "welcome"
        assert gate.pending_count == 0


def test_binding_mismatch_does_not_count_toward_lockout() -> None:
    with _fresh_client() as client:
        gate, _ = _swap_gate(client, fail_threshold=2)
        for _ in range(3):
            with client.websocket_connect("/ws/v1") as ws:
                ws.send_json(
                    {
                        "type": "hello",
                        "client_id": "desk-ui",
                        "room_id": "wrong-room",
                        "role": "ui",
                        "token": "dev-token-desk",
                    }
                )
                assert ws.receive_json()["code"] == "binding_mismatch"
        # Valid-token misconfiguration never locks the device out.
        with client.websocket_connect("/ws/v1") as ws:
            _hello(ws, "dev-token-desk")
            ws.send_json({"type": "user_text", "text": "ping"})
            assert ws.receive_json()["type"] == "welcome"


def test_slow_handshake_times_out() -> None:
    os.environ["GLADOS_HANDSHAKE_TIMEOUT_S"] = "0.2"
    try:
        with _fresh_client() as client:
            with client.websocket_connect("/ws/v1") as ws:
                # Send no hello; the server must close rather than hold the
                # pending slot open indefinitely.
                event = ws.receive()
            assert event["type"] == "websocket.close"
            assert client.app.state.handshake_gate.pending_count == 0
    finally:
        del os.environ["GLADOS_HANDSHAKE_TIMEOUT_S"]


def test_non_ascii_token_rejected_cleanly(client: TestClient) -> None:
    # The constant-time check compares as bytes; a non-ASCII attacker token must
    # reject (auth_failed), not raise from .encode()/compare_digest.
    with client.websocket_connect("/ws/v1") as ws:
        ws.send_json(
            {
                "type": "hello",
                "client_id": "desk-ui",
                "room_id": "desk",
                "role": "ui",
                "token": "wrøng-töken-日本語",
            }
        )
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "auth_failed"
