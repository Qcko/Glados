"""Server-side surfacing of memory BLOCK notices (ARCH §14).

When a trusted server's lessons fail the LocalGuard gate at load time, nothing
is injected — but the BLOCK is made visible to operators as metadata only: at
`GET /admin/memory` and pushed to `ui`-role clients on connect. These tests
seed `app.state.memory_blocks` directly rather than standing up a real trusted
server + LocalGuard; the gate's own outcomes are covered in test_memory_gate.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from glados.core.protocols import MemoryBlockNotice


@pytest.fixture()
def client():
    os.environ["GLADOS_CONFIG_DIR"] = str(Path(__file__).parent.parent / "configs")
    os.environ["GLADOS_LLM_BACKEND"] = "fake"
    from glados.core.server import build_app

    app = build_app()
    # Present as a loopback caller: /admin/memory is gated to loopback-only
    # now that GLADOS_HOST can expose the server on the LAN.
    with TestClient(app, client=("127.0.0.1", 12345)) as c:
        yield c


_NOTICE = MemoryBlockNotice(
    source="dunnes",
    sha256="a" * 64,
    length=3700,
    reason="not approved (exit 1): unknown content; run localguard memory approve",
)


def test_admin_memory_empty_by_default(client: TestClient) -> None:
    r = client.get("/admin/memory")
    assert r.status_code == 200
    assert r.json() == {"blocks": []}


def test_admin_memory_reports_blocks_metadata_only(client: TestClient) -> None:
    client.app.state.memory_blocks = [_NOTICE]
    r = client.get("/admin/memory")
    assert r.status_code == 200
    blocks = r.json()["blocks"]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "memory_block_notice"
    assert blocks[0]["source"] == "dunnes"
    assert blocks[0]["sha256"] == "a" * 64
    assert blocks[0]["length"] == 3700
    assert "approve" in blocks[0]["reason"]


def test_admin_memory_rejects_non_loopback() -> None:
    os.environ["GLADOS_CONFIG_DIR"] = str(Path(__file__).parent.parent / "configs")
    os.environ["GLADOS_LLM_BACKEND"] = "fake"
    from glados.core.server import build_app

    app = build_app()
    with TestClient(app, client=("192.168.50.99", 5555)) as lan:
        assert lan.get("/admin/memory").status_code == 403
        assert lan.get("/healthz").status_code == 403


def test_ui_client_gets_block_notice_on_connect(client: TestClient) -> None:
    client.app.state.memory_blocks = [_NOTICE]
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
        notice = ws.receive_json()
    assert notice["type"] == "memory_block_notice"
    assert notice["source"] == "dunnes"
    assert notice["sha256"] == "a" * 64
