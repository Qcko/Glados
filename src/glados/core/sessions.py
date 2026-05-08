"""Session registry.

v0 policy: every utterance opens a fresh session. The `get_or_open` hook is
where v1+ idle-window continuation logic plugs in without touching callers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class Session:
    session_id: str
    room_id: str
    speaker_id: str


class SessionRegistry:
    def __init__(self) -> None:
        self._latest: dict[tuple[str, str], Session] = {}

    def get_or_open(self, room_id: str, speaker_id: str) -> Session:
        session = Session(
            session_id=f"{room_id}:{speaker_id}:{uuid.uuid4().hex[:8]}",
            room_id=room_id,
            speaker_id=speaker_id,
        )
        self._latest[(room_id, speaker_id)] = session
        return session

    def latest(self, room_id: str, speaker_id: str) -> Session | None:
        return self._latest.get((room_id, speaker_id))
