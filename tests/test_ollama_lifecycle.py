"""Tests for the Ollama daemon lifecycle helper."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

from glados.core.ollama_lifecycle import OllamaLifecycle


class _FakeProbe:
    """Drives `_probe()` through a scripted sequence of return values."""

    def __init__(self, results: list[bool]) -> None:
        self._results = list(results)
        self.calls = 0

    async def __call__(self) -> bool:
        self.calls += 1
        if not self._results:
            return False
        return self._results.pop(0)


@pytest.mark.asyncio
async def test_ensure_noop_when_already_running(monkeypatch):
    lc = OllamaLifecycle("http://localhost:11434")
    monkeypatch.setattr(lc, "_probe", _FakeProbe([True]))
    popen_calls: list = []
    monkeypatch.setattr(
        "glados.core.ollama_lifecycle.subprocess.Popen",
        lambda *a, **k: popen_calls.append((a, k)) or object(),
    )

    await lc.ensure()

    assert lc.started_by_us is False
    assert popen_calls == []


@pytest.mark.asyncio
async def test_stop_if_started_skips_when_we_didnt_start(monkeypatch):
    lc = OllamaLifecycle("http://localhost:11434")
    taskkill_calls: list = []
    monkeypatch.setattr(
        "glados.core.ollama_lifecycle.subprocess.run",
        lambda *a, **k: taskkill_calls.append((a, k)) or None,
    )

    await lc.stop_if_started()

    assert taskkill_calls == []


@pytest.mark.skipif(sys.platform != "win32", reason="tray launch is Windows-only")
@pytest.mark.asyncio
async def test_ensure_launches_tray_and_marks_started(monkeypatch, tmp_path):
    fake_tray = tmp_path / "ollama app.exe"
    fake_tray.write_bytes(b"")
    monkeypatch.setenv("GLADOS_OLLAMA_TRAY", str(fake_tray))

    lc = OllamaLifecycle("http://localhost:11434")
    # First probe: down. Second probe (inside _wait_until_ready): up.
    monkeypatch.setattr(lc, "_probe", _FakeProbe([False, True]))
    popen_calls: list = []
    monkeypatch.setattr(
        "glados.core.ollama_lifecycle.subprocess.Popen",
        lambda *a, **k: popen_calls.append((a, k)) or object(),
    )

    await lc.ensure()

    assert lc.started_by_us is True
    assert len(popen_calls) == 1
    assert popen_calls[0][0][0] == [str(fake_tray)]
    si = popen_calls[0][1]["startupinfo"]
    import subprocess as _sp
    assert si.dwFlags & _sp.STARTF_USESHOWWINDOW
    assert si.wShowWindow == 7  # SW_SHOWMINNOACTIVE


@pytest.mark.skipif(sys.platform != "win32", reason="taskkill is Windows-only")
@pytest.mark.asyncio
async def test_stop_if_started_kills_both_images(monkeypatch):
    lc = OllamaLifecycle("http://localhost:11434")
    lc.started_by_us = True
    images: list[str] = []

    def _run(cmd, **_kwargs):
        # cmd is ["taskkill", "/F", "/T", "/IM", "<image>"]
        images.append(cmd[-1])
        return None

    monkeypatch.setattr("glados.core.ollama_lifecycle.subprocess.run", _run)

    await lc.stop_if_started()

    assert images == ["ollama app.exe", "ollama.exe"]
    assert lc.started_by_us is False


@pytest.mark.asyncio
async def test_probe_returns_false_on_connect_error(monkeypatch):
    lc = OllamaLifecycle("http://127.0.0.1:1")  # nothing listens here
    # Real _probe — should hit a connect error fast and return False.
    assert await lc._probe() is False


@pytest.mark.asyncio
async def test_probe_returns_true_on_200(monkeypatch):
    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/version"
        return httpx.Response(200, json={"version": "0.23.2"})

    transport = httpx.MockTransport(_handler)

    class _PatchedClient(httpx.AsyncClient):
        def __init__(self, *a, **k):
            k["transport"] = transport
            super().__init__(*a, **k)

    monkeypatch.setattr("glados.core.ollama_lifecycle.httpx.AsyncClient", _PatchedClient)

    lc = OllamaLifecycle("http://localhost:11434")
    assert await lc._probe() is True


def test_resolve_tray_path_uses_env_override(monkeypatch, tmp_path):
    fake = tmp_path / "tray.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("GLADOS_OLLAMA_TRAY", str(fake))
    assert OllamaLifecycle._resolve_tray_path() == fake


def test_resolve_tray_path_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("GLADOS_OLLAMA_TRAY", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))  # no Programs/Ollama under here
    assert OllamaLifecycle._resolve_tray_path() is None
