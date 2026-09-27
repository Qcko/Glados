"""The server-side hooks a production deploy relies on: per-machine config
overlays, the Ollama autostart switch, drain-before-exit, and a healthz that
says which release is running and whether it is ready (DEPLOY.md)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from glados.core.config import load_glados_config, load_rooms_config
from glados.core.ollama_lifecycle import OllamaLifecycle
from glados.core.room_queues import RoomQueueManager

_CONFIGS = Path(__file__).parent.parent / "configs"


# ---- config overlays ---------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_overlay_replaces_scalars_and_merges_tables(tmp_path, monkeypatch):
    monkeypatch.delenv("GLADOS_LOCAL_CONFIG_DIR", raising=False)
    _write(tmp_path / "glados.toml", '[server]\nport = 8765\ntraces_dir = "traces"\n')
    _write(tmp_path / "glados.local.toml", '[server]\ntraces_dir = "elsewhere"\n')

    cfg = load_glados_config(tmp_path / "glados.toml")

    assert cfg.server.traces_dir == Path("elsewhere")
    assert cfg.server.port == 8765


def test_overlay_appends_arrays_so_a_box_can_add_a_client(tmp_path, monkeypatch):
    monkeypatch.delenv("GLADOS_LOCAL_CONFIG_DIR", raising=False)
    tracked = '[[clients]]\nclient_id = "desk-ui"\nroom_id = "desk"\nrole = "ui"\n'
    local = '[[clients]]\nclient_id = "prod-desk-ui"\nroom_id = "prod-desk"\nrole = "ui"\n'
    _write(tmp_path / "rooms.toml", tracked)
    _write(tmp_path / "rooms.local.toml", local)

    rooms = load_rooms_config(tmp_path / "rooms.toml")

    assert [c.client_id for c in rooms.clients] == ["desk-ui", "prod-desk-ui"]


def test_overlay_restating_a_tracked_client_fails_the_load(tmp_path, monkeypatch):
    monkeypatch.delenv("GLADOS_LOCAL_CONFIG_DIR", raising=False)
    row = '[[clients]]\nclient_id = "desk-ui"\nroom_id = "desk"\nrole = "ui"\n'
    _write(tmp_path / "rooms.toml", row)
    _write(tmp_path / "rooms.local.toml", row.replace('"desk"', '"elsewhere"'))

    with pytest.raises(ValueError, match="bound twice"):
        load_rooms_config(tmp_path / "rooms.toml")


def test_overlay_is_read_from_the_local_config_dir(tmp_path, monkeypatch):
    tracked_dir = tmp_path / "release"
    local_dir = tmp_path / "local"
    tracked_dir.mkdir()
    local_dir.mkdir()
    _write(tracked_dir / "glados.toml", '[server]\ntraces_dir = "traces"\n')
    _write(tracked_dir / "glados.local.toml", '[server]\ntraces_dir = "ignored"\n')
    _write(local_dir / "glados.local.toml", '[server]\ntraces_dir = "from-local"\n')
    monkeypatch.setenv("GLADOS_LOCAL_CONFIG_DIR", str(local_dir))

    assert load_glados_config(tracked_dir / "glados.toml").server.traces_dir == Path("from-local")


def test_no_overlay_means_the_tracked_file_alone(tmp_path, monkeypatch):
    monkeypatch.delenv("GLADOS_LOCAL_CONFIG_DIR", raising=False)
    _write(tmp_path / "glados.toml", '[server]\ntraces_dir = "traces"\n')

    assert load_glados_config(tmp_path / "glados.toml").server.traces_dir == Path("traces")


def test_servers_toml_resolves_from_the_local_config_dir(tmp_path, monkeypatch):
    from glados.core.server import _resolve_servers_toml

    _write(tmp_path / "servers.toml", "")
    monkeypatch.setenv("GLADOS_LOCAL_CONFIG_DIR", str(tmp_path))

    assert _resolve_servers_toml(_CONFIGS) == tmp_path / "servers.toml"


# ---- Ollama autostart switch -------------------------------------------


class _Probes:
    def __init__(self, results: list[bool]) -> None:
        self._results = list(results)

    async def __call__(self) -> bool:
        return self._results.pop(0) if self._results else False


@pytest.mark.asyncio
async def test_autostart_off_waits_for_the_daemon_and_never_launches(monkeypatch):
    lc = OllamaLifecycle("http://localhost:11434", autostart=False)
    monkeypatch.setattr(lc, "_probe", _Probes([False, False, True]))
    monkeypatch.setattr("glados.core.ollama_lifecycle._BOOT_POLL_INTERVAL_S", 0.0)
    launched: list = []
    monkeypatch.setattr(
        "glados.core.ollama_lifecycle.subprocess.Popen",
        lambda *a, **k: launched.append(a) or object(),
    )

    await lc.ensure()

    assert launched == []
    assert lc.started_by_us is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, True), ("1", True), ("0", False), ("off", False), ("False", False)],
)
def test_autostart_reads_the_env_switch(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("GLADOS_OLLAMA_AUTOSTART", raising=False)
    else:
        monkeypatch.setenv("GLADOS_OLLAMA_AUTOSTART", raw)

    assert OllamaLifecycle("http://localhost:11434")._autostart is expected


# ---- drain -------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_waits_for_the_running_action_then_refuses_new_ones():
    mgr = RoomQueueManager()
    release = asyncio.Event()
    finished: list[str] = []

    async def turn():
        await release.wait()
        finished.append("turn")

    mgr.enqueue("desk", turn)
    await asyncio.sleep(0)
    drain = asyncio.create_task(mgr.drain(timeout_s=2.0))
    await asyncio.sleep(0.05)
    assert not drain.done()
    release.set()

    assert await drain is True
    assert finished == ["turn"]
    assert mgr.enqueue("desk", turn) is False
    await mgr.close()


@pytest.mark.asyncio
async def test_drain_timeout_cancels_nothing_and_reopens_intake():
    mgr = RoomQueueManager()
    release = asyncio.Event()

    async def turn():
        await release.wait()

    mgr.enqueue("desk", turn)
    await asyncio.sleep(0)

    assert await mgr.drain(timeout_s=0.05) is False
    assert mgr.draining is False
    assert mgr._active_actions["desk"].done() is False
    assert mgr.enqueue("desk", turn) is True
    release.set()
    await mgr.flush()
    await mgr.close()


# ---- healthz + shutdown ------------------------------------------------


@pytest.fixture
def app():
    os.environ["GLADOS_CONFIG_DIR"] = str(_CONFIGS)
    from glados.core.server import build_app

    return build_app()


def test_healthz_reports_release_and_readiness(app):
    with TestClient(app, client=("127.0.0.1", 1)) as c:
        body = c.get("/healthz").json()

    assert set(body["release"]) == {"sha", "tag"}
    assert body["ready"]["draining"] is False
    assert "desk-ui" in body["ready"]["clients_with_token"]
    assert isinstance(body["ready"]["llm_warm"], bool)


def test_shutdown_drains_then_requests_exit(app):
    exits: list[bool] = []
    app.state.request_exit = lambda: exits.append(True)
    with TestClient(app, client=("127.0.0.1", 1)) as c:
        r = c.post("/admin/shutdown", params={"timeout_s": 5})
        c.portal.call(asyncio.sleep, 0.7)
        draining = c.get("/healthz").json()["ready"]["draining"]

    assert r.status_code == 200
    assert exits == [True]
    assert draining is True


def test_shutdown_refuses_while_a_turn_outlives_the_timeout(app):
    exits: list[bool] = []
    app.state.request_exit = lambda: exits.append(True)
    with TestClient(app, client=("127.0.0.1", 1)) as c:
        release = c.portal.call(asyncio.Event)

        async def stuck():
            await release.wait()

        c.portal.call(_enqueue, app, stuck)
        r = c.post("/admin/shutdown", params={"timeout_s": 1})
        c.portal.call(release.set)

    assert r.status_code == 409
    assert exits == []


async def _enqueue(app, action) -> None:
    app.state.organizer._queues.enqueue("desk", action)
    await asyncio.sleep(0)


def test_shutdown_is_loopback_only(app):
    with TestClient(app, client=("192.168.50.99", 1)) as lan:
        assert lan.post("/admin/shutdown").status_code == 403
