"""Admission control for the `/ws/v1` handshake.

Two cross-connection controls (the per-handshake timeout lives inline in
the WS handler): concurrent pending-handshake caps (global + per-source-IP)
and a per-source-IP failure lockout. Design and panel adjudications in
client_room/deploy/DESIGN-ws-handshake-rate-limit.md.

The gate is a plain synchronous object — the server runs on one event loop
and every method is atomic between awaits. The caller resolves the peer IP
(no reverse proxy is assumed; see HandshakeConfig) and anchors
`admit`/`release` in its own try/finally so a slot is released exactly once
on every exit path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .config import HandshakeConfig

log = logging.getLogger(__name__)


class Verdict(Enum):
    OK = "ok"
    BUSY = "busy"
    LOCKED_OUT = "locked_out"


@dataclass
class _IPState:
    failure_times: list[float] = field(default_factory=list)
    locked_until: float = 0.0
    last_activity: float = 0.0

    def expired(self, now: float, window_s: float) -> bool:
        return (
            now >= self.locked_until
            and all(now - t > window_s for t in self.failure_times)
        )


class HandshakeGate:
    def __init__(
        self,
        cfg: HandshakeConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._clock = clock
        self._pending_total = 0
        self._pending_by_ip: dict[str, int] = {}
        self._ips: dict[str, _IPState] = {}

    def admit(self, ip: str) -> Verdict:
        """Check lockout + caps and, when OK, take a pending slot. The
        caller must `release(ip)` exactly once for every OK verdict."""
        if self._locked_out(ip):
            return Verdict.LOCKED_OUT
        if self._at_capacity(ip):
            return Verdict.BUSY
        self._pending_total += 1
        self._pending_by_ip[ip] = self._pending_by_ip.get(ip, 0) + 1
        return Verdict.OK

    def release(self, ip: str) -> None:
        # Clamp rather than trust the caller contract (one release per OK
        # admit) — a future second caller must not be able to drive the
        # counter negative and widen the cap.
        self._pending_total = max(0, self._pending_total - 1)
        remaining = self._pending_by_ip.get(ip, 0) - 1
        if remaining <= 0:
            self._pending_by_ip.pop(ip, None)
        else:
            self._pending_by_ip[ip] = remaining

    def record_failure(self, ip: str) -> None:
        """Count a credential failure (unknown client id / bad token).
        Engages a lockout when the threshold is hit inside the window."""
        now = self._clock()
        state = self._touch(ip, now)
        if now < state.locked_until:
            return  # already locked; don't extend or log per-attempt
        state.failure_times = [
            t for t in state.failure_times if now - t <= self._cfg.fail_window_s
        ]
        state.failure_times.append(now)
        log.warning("handshake auth failure from %s", ip)
        if len(state.failure_times) >= self._cfg.fail_threshold:
            state.locked_until = now + self._cfg.lockout_s
            state.failure_times.clear()
            log.warning(
                "handshake lockout engaged for %s (%d failures in %.0fs window; "
                "locked for %.0fs)",
                ip,
                self._cfg.fail_threshold,
                self._cfg.fail_window_s,
                self._cfg.lockout_s,
            )

    def record_success(self, ip: str) -> None:
        self._ips.pop(ip, None)

    @property
    def pending_count(self) -> int:
        return self._pending_total

    def _locked_out(self, ip: str) -> bool:
        state = self._ips.get(ip)
        if state is None:
            return False
        now = self._clock()
        if state.locked_until and now >= state.locked_until:
            log.info("handshake lockout expired for %s", ip)
            state.locked_until = 0.0
        return now < state.locked_until

    def _at_capacity(self, ip: str) -> bool:
        return (
            self._pending_total >= self._cfg.max_pending
            or self._pending_by_ip.get(ip, 0) >= self._cfg.max_pending_per_ip
        )

    def _touch(self, ip: str, now: float) -> _IPState:
        state = self._ips.get(ip)
        if state is None:
            state = _IPState()
            self._ips[ip] = state
        state.last_activity = now
        if len(self._ips) > self._cfg.max_tracked_ips:
            self._evict(now, keep=ip)
        return state

    def _evict(self, now: float, keep: str) -> None:
        """Prune expired entries first; only if still over the cap, drop the
        oldest-activity entries. Never evicts `keep` (the IP being touched)
        so an active attacker cannot launder its own state by overflowing
        the table in the same call."""
        for ip in [
            ip
            for ip, s in self._ips.items()
            if ip != keep and s.expired(now, self._cfg.fail_window_s)
        ]:
            del self._ips[ip]
        overflow = len(self._ips) - self._cfg.max_tracked_ips
        if overflow <= 0:
            return
        oldest_first = sorted(
            (ip for ip in self._ips if ip != keep),
            key=lambda ip: self._ips[ip].last_activity,
        )
        for ip in oldest_first[:overflow]:
            del self._ips[ip]
