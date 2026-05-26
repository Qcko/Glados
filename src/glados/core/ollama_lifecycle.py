"""Ollama daemon lifecycle for GLaDOS boot/shutdown.

Probes the configured Ollama host on startup. If the daemon is already
running, leaves it alone. If it isn't running and we're on Windows,
launches the bundled tray app (which in turn spawns `ollama.exe serve`)
and waits for the API to come up. On GLaDOS shutdown, kills the tray +
daemon only if we started them — a daemon that was already up may have
other consumers.

The tray executable is located via `%LOCALAPPDATA%\\Programs\\Ollama\\ollama app.exe`
— the default Ollama Windows install path — or via the `GLADOS_OLLAMA_TRAY`
env var override. No hardcoded user paths.

Non-Windows platforms get a warning and a no-op: GLaDOS deployment targets
Windows today, and `ollama serve` on Linux/macOS is typically managed by
systemd / launchd rather than a desktop tray.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import httpx


log = logging.getLogger(__name__)

_PROBE_TIMEOUT_S = 2.0
_BOOT_TIMEOUT_S = 30.0
_BOOT_POLL_INTERVAL_S = 1.0
_TRAY_ENV = "GLADOS_OLLAMA_TRAY"
_TRAY_RELATIVE = Path("Programs") / "Ollama" / "ollama app.exe"


class OllamaLifecycle:
    def __init__(self, host: str) -> None:
        self._host = host.rstrip("/")
        self.started_by_us = False

    async def ensure(self) -> None:
        if await self._probe():
            log.info("Ollama already running at %s", self._host)
            return
        if sys.platform != "win32":
            log.warning(
                "Ollama not reachable at %s and auto-start is Windows-only; "
                "start it manually before invoking the LLM",
                self._host,
            )
            return
        tray = self._resolve_tray_path()
        if tray is None:
            log.warning(
                "Ollama tray app not found (set %s to its full path); "
                "LLM calls will fail until Ollama is started manually",
                _TRAY_ENV,
            )
            return
        log.info("Ollama not running; launching tray at %s", tray)
        try:
            # Off the event loop — CreateProcess on Windows can spend
            # 100-500 ms doing tray-icon setup and we don't want to stall
            # other lifespan work behind it.
            # DETACHED_PROCESS so a GLaDOS crash leaves Ollama running
            # instead of orphaning a half-dead process tree; next clean
            # boot will find it up and started_by_us stays False.
            # SW_SHOWMINNOACTIVE (7): tray starts minimized without
            # stealing focus — GLaDOS boot shouldn't pop windows.
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 7
            await asyncio.to_thread(
                subprocess.Popen,  # noqa: S603 — path is from env or known-good probe
                [str(tray)],
                close_fds=True,
                creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
                startupinfo=startupinfo,
            )
        except OSError as e:
            log.error("Failed to launch Ollama tray: %s — first LLM call will fail", e)
            return
        if await self._wait_until_ready():
            self.started_by_us = True
            log.info("Ollama is up (started by GLaDOS)")
        else:
            log.error(
                "Ollama did not become ready within %.0fs after tray launch "
                "— first LLM call will fail",
                _BOOT_TIMEOUT_S,
            )

    async def stop_if_started(self) -> None:
        if not self.started_by_us:
            return
        if sys.platform != "win32":
            return
        log.info("Stopping Ollama (started by GLaDOS this session)")
        # Kill the tray app first so it can't respawn the daemon, then the
        # `ollama.exe serve` daemon itself. /T to take child processes too.
        for image in ("ollama app.exe", "ollama.exe"):
            try:
                await asyncio.to_thread(
                    subprocess.run,  # noqa: S603,S607
                    ["taskkill", "/F", "/T", "/IM", image],
                    check=False,
                    capture_output=True,
                )
            except OSError as e:
                log.warning("taskkill for %s failed: %s", image, e)
        self.started_by_us = False

    async def _probe(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S) as c:
                r = await c.get(f"{self._host}/api/version")
                return r.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    async def _wait_until_ready(self) -> bool:
        deadline = asyncio.get_running_loop().time() + _BOOT_TIMEOUT_S
        while asyncio.get_running_loop().time() < deadline:
            if await self._probe():
                return True
            await asyncio.sleep(_BOOT_POLL_INTERVAL_S)
        return False

    @staticmethod
    def _resolve_tray_path() -> Path | None:
        override = os.environ.get(_TRAY_ENV)
        if override:
            p = Path(override)
            return p if p.is_file() else None
        # Per-user install (default for the Windows installer), then
        # machine-wide install fallback. Both probe the standard layout —
        # no hardcoded user paths.
        for root_env in ("LOCALAPPDATA", "ProgramFiles"):
            root = os.environ.get(root_env)
            if not root:
                continue
            p = Path(root) / _TRAY_RELATIVE
            if p.is_file():
                return p
        return None
