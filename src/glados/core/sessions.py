"""Session registry.

A session is keyed by `(room_id, speaker_id)` (ARCH section 3). A follow-up utterance
reuses the live session for that key when it lands within the idle window of the
last activity; after the gap a fresh session opens. The session_id is the stable
handle the organizer hangs per-conversation history on, so reuse is what makes
"add it back" / "do that instead" follow-ups work.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Session:
    session_id: str
    room_id: str
    speaker_id: str


class SessionRegistry:
    def __init__(
        self,
        idle_window_s: float = 180.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._latest: dict[tuple[str, str], Session] = {}
        # session_id -> monotonic timestamp of the last utterance on it.
        self._last_activity: dict[str, float] = {}
        self._idle_window_s = idle_window_s
        self._clock = clock

    def get_or_open(self, room_id: str, speaker_id: str) -> Session:
        key = (room_id, speaker_id)
        now = self._clock()
        existing = self._latest.get(key)
        if existing is not None and self._within_window(existing.session_id, now):
            self._last_activity[existing.session_id] = now
            return existing
        session = Session(
            session_id=f"{room_id}:{speaker_id}:{uuid.uuid4().hex[:8]}",
            room_id=room_id,
            speaker_id=speaker_id,
        )
        if existing is not None:
            self._last_activity.pop(existing.session_id, None)
        self._latest[key] = session
        self._last_activity[session.session_id] = now
        return session

    def latest(self, room_id: str, speaker_id: str) -> Session | None:
        return self._latest.get((room_id, speaker_id))

    def _within_window(self, session_id: str, now: float) -> bool:
        last = self._last_activity.get(session_id)
        return last is not None and now - last <= self._idle_window_s
