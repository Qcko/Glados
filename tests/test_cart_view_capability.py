"""Test 3 of DESIGN-checkout-reconcile.md: `cart_view` counts only when
rooms.toml grants it AND the client's hello declares it, and only for a ui."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    os.environ["GLADOS_CONFIG_DIR"] = str(Path(__file__).parent.parent / "configs")
    os.environ["GLADOS_LLM_BACKEND"] = "fake"
    from glados.core.server import app

    with TestClient(app, client=("127.0.0.1", 12345)) as c:
        yield c


def _screens_while_connected(client: TestClient, hello: dict) -> list[str]:
    with client.websocket_connect("/ws/v1") as ws:
        ws.send_json({"type": "hello", **hello})
        ws.send_json({"type": "user_text", "text": "ping"})
        while ws.receive_json()["type"] != "done":
            pass
        organizer = client.app.state.organizer
        return [b.client_id for b in organizer._clients_with_capability("cart_view")]


_DESK = {"client_id": "desk-ui", "room_id": "desk", "role": "ui", "token": "dev-token-desk"}


def test_granted_and_declared_counts(client: TestClient) -> None:
    assert _screens_while_connected(client, {**_DESK, "capabilities": ["cart_view"]}) == ["desk-ui"]


def test_granted_but_not_declared_does_not(client: TestClient) -> None:
    assert _screens_while_connected(client, _DESK) == []


def test_it_is_gone_once_the_client_disconnects(client: TestClient) -> None:
    _screens_while_connected(client, {**_DESK, "capabilities": ["cart_view"]})
    assert client.app.state.organizer._clients_with_capability("cart_view") == []


def test_a_non_ui_client_cannot_be_granted_a_screen_capability() -> None:
    from glados.core.config import ClientBinding

    with pytest.raises(ValueError, match="need role 'ui'"):
        ClientBinding(client_id="m", room_id="r", role="mic", capabilities=["cart_view"])
