"""Shared spine for the room clients (mic, speaker).

Both clients open the same `/ws/v1`, send a role `Hello` first, then run a
role-specific session — mic uploads captured audio, speaker plays received TTS.
The connect + capped-jittered-backoff reconnect loop, the terminal-handshake
error set, and the token/config loading are identical across roles and
security-relevant (the terminal set gates whether a bad token retries forever;
the token precedence decides where a secret is read from), so they live here
once instead of drifting between two copies.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
import random
from pathlib import Path

from . import wire

log = logging.getLogger("client_room.client")

# Terminal handshake errors — the server closes the socket after sending one,
# and retrying with the same credentials/binding would just loop. The server
# handshake (`server.py` `_handshake`) is role-agnostic, so every role shares
# this set; `tests/test_client_room.py` drift-guards it against the server.
TERMINAL_ERROR_CODES = frozenset(
    {"auth_failed", "unbound_client", "binding_mismatch", "expected_hello", "bad_message"}
)


class ReconnectingClient(abc.ABC):
    """Connect, run one session, reconnect with capped+jittered backoff until
    stopped or a terminal handshake error. Subclasses implement `_session`
    (the role-specific body) and dispatch their own non-error text frames;
    the base owns the lifecycle, the hello, and terminal-error detection."""

    def __init__(
        self,
        *,
        server_url: str,
        client_id: str,
        room_id: str,
        role: str,
        token: str,
        connect=None,
        backoff_min_s: float = 0.5,
        backoff_max_s: float = 30.0,
    ) -> None:
        self._url = server_url.rstrip("/") + wire.WS_PATH
        self._client_id = client_id
        self._room_id = room_id
        self._role = role
        self._token = token
        if connect is None:
            import websockets  # lazy: keeps the import cost off test paths

            connect = websockets.connect
        self._connect = connect
        self._backoff_min = backoff_min_s
        self._backoff_max = backoff_max_s
        self._stop = False
        self._terminal = False

    async def run(self) -> None:
        """Connect, run a session, and reconnect with capped+jittered backoff
        until stopped or a terminal handshake error. Cancellation-safe."""
        backoff = self._backoff_min
        while not self._stop:
            self._terminal = False
            try:
                await self._session()
                backoff = self._backoff_min  # clean drop — reconnect promptly
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - any transport error → retry
                log.warning(
                    "%s session ended (%s: %s); will reconnect",
                    self._role, type(e).__name__, e,
                )
            if self._stop or self._terminal:
                break
            await asyncio.sleep(backoff)
            backoff = min(self._backoff_max, backoff * 2) + random.uniform(0, 0.5)

    def stop(self) -> None:
        self._stop = True

    @property
    def terminated(self) -> bool:
        """True once `run` has exited because of a terminal handshake error (bad
        token / wrong room-role binding) rather than a graceful `stop`. A
        supervisor reads this to tell a misconfigured client apart from one it
        shut down deliberately — the two otherwise both just return from `run`."""
        return self._terminal

    async def _send_hello(self, ws) -> None:
        """Send the role Hello as the FIRST message, before any other frame.
        A good handshake is answered with silence (no `welcome` ack)."""
        await ws.send(json.dumps(
            wire.hello(self._client_id, self._room_id, self._role, self._token)
        ))
        log.info(
            "%s %s connected to %s (room %s)",
            self._role, self._client_id, self._url, self._room_id,
        )

    def _check_terminal(self, msg: dict) -> bool:
        """Inspect a parsed server message; if it is a terminal `error`, set the
        terminal flag (so `run` stops instead of reconnecting) and return True.
        Non-error / non-terminal messages return False and are the subclass's
        to dispatch."""
        if msg.get("type") != "error":
            return False
        code = msg.get("code", "")
        log.error("server error: %s — %s", code, msg.get("message", ""))
        if code in TERMINAL_ERROR_CODES:
            self._terminal = True
            return True
        return False

    @abc.abstractmethod
    async def _session(self) -> None:
        """Run one connected session. Must open the socket via
        `self._connect(self._url)`, call `self._send_hello(ws)` first, and tear
        down cleanly on exit (the reconnect loop calls this repeatedly)."""
        ...


# ---- Config + token loading (shared by every role's `main`) ----------------


def load_config(path: str) -> dict:
    import tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)


def _check_token_file_perms(path: Path) -> None:
    """Refuse a token file readable by group/other (SSH-style). POSIX only —
    the mode bits are meaningless on Windows, so skip the check there."""
    if os.name != "posix":
        return
    mode = path.stat().st_mode
    if mode & 0o077:
        raise SystemExit(
            f"token file {path} is group/other-accessible "
            f"(mode {oct(mode & 0o777)}); tighten it: chmod 600 {path}"
        )


def load_token(
    client_id: str,
    *,
    env_var: str | None = None,
    token_file: str | None = None,
) -> str:
    """Resolve the client auth token. Precedence: keyring → env → file; the
    first present source wins and short-circuits. Logs WHICH source won (name
    only, never the value). Raises listing every source tried if none yields a
    token.

    On a shared device a mode-600 file is preferred over an env var: env tokens
    leak via `ps -e` / `/proc/<pid>/environ` to any local user."""
    tried: list[str] = []

    try:
        import keyring

        tok = keyring.get_password("glados.client-tokens", client_id)
        if tok:
            log.info("token for %s loaded from keyring", client_id)
            return tok
        tried.append("keyring (empty)")
    except Exception as e:  # noqa: BLE001 - keyring backend may be absent
        tried.append(f"keyring (unavailable: {type(e).__name__})")

    if env_var:
        tok = os.environ.get(env_var)
        if tok:
            log.info("token for %s loaded from env var %s", client_id, env_var)
            return tok
        tried.append(f"env {env_var} (unset)")

    if token_file:
        path = Path(token_file).expanduser()
        if path.exists():
            _check_token_file_perms(path)
            tok = path.read_text(encoding="utf-8").strip()
            if tok:
                log.info("token for %s loaded from file %s", client_id, path)
                return tok
            tried.append(f"file {path} (empty)")
        else:
            tried.append(f"file {path} (missing)")

    raise SystemExit(
        f"no token for {client_id}; tried: {', '.join(tried) or 'no sources configured'}. "
        f"Set one in the keyring (python -m glados.secrets set client-tokens "
        f"{client_id} <token>), or configure token_env / token_file."
    )
