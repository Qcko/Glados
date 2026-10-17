"""Wake-word arbitration and audio admission (DESIGN.md, wake word section).

Pure, clock-free logic: every method takes the loop time it should judge
against, so tests drive it with plain numbers. The Organizer owns one
`WakeGate`, feeds it wake hits from every mic connection, and asks it -- at
each VAD segment end, BEFORE transcription -- whether the segment may become
text at all.

Arbitration: the first hit opens a `dedup_window_s` window; at its close the
loudest hit (energy plus the room's gain offset, score breaking ties) wins.
Only the winner's room is armed, and the winning mic owns that room until its
follow-up window closes, so a command heard by several mics runs once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

WakeMode = Literal["off", "required"]

AdmitReason = Literal[
    "wake_off", "one_breath", "armed", "confirm", "follow_up", "barge_in_only"
]

DropReason = Literal[
    "not_woken", "lost_arbitration", "not_owner", "detector_fault"
]

# Slack after a segment's VAD end within which a hit still counts as inside
# it: openWakeWord fires near the END of the phrase.
_INSIDE_SLACK_S = 0.3


@dataclass(frozen=True)
class WakeHit:
    client_id: str
    room_id: str
    score: float
    energy_dbfs: float
    at: float


@dataclass(frozen=True)
class Admission:
    reason: AdmitReason


@dataclass(frozen=True)
class Drop:
    reason: DropReason


@dataclass
class _Window:
    opened_at: float
    deadline: float
    best: dict[str, WakeHit] = field(default_factory=dict)


@dataclass
class _RoomState:
    wins: list[WakeHit] = field(default_factory=list)
    losses: list[tuple[str, float]] = field(default_factory=list)
    owner: str | None = None
    owner_until: float = float("-inf")
    last_turn_end: float | None = None
    faulted: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class WakeTimings:
    armed_window_s: float = 8.0
    followup_window_s: float = 6.0
    dedup_window_s: float = 0.5


@dataclass(frozen=True)
class SegmentFacts:
    client_id: str
    room_id: str
    start: float
    end: float
    turn_in_flight: bool
    confirm_pending: bool


class WakeGate:
    def __init__(
        self,
        *,
        timings: WakeTimings,
        mode_for_room: Callable[[str], WakeMode],
        gain_for_room: Callable[[str], float] = lambda _room: 0.0,
    ) -> None:
        self._t = timings
        self._mode_for_room = mode_for_room
        self._gain_for_room = gain_for_room
        self._window: _Window | None = None
        self._rooms: dict[str, _RoomState] = {}

    def mode(self, room_id: str) -> WakeMode:
        return self._mode_for_room(room_id)

    def offer(self, hit: WakeHit, now: float) -> float | None:
        """Add a hit. Returns the window deadline when this hit OPENED a
        window (the caller schedules `settle` for it), else None."""
        self.settle(now)
        if self.mode(hit.room_id) == "off":
            return None
        opened = self._window is None
        if opened:
            self._window = _Window(hit.at, hit.at + self._t.dedup_window_s)
        best = self._window.best.get(hit.client_id)
        if best is None or hit.score > best.score:
            self._window.best[hit.client_id] = hit
        return self._window.deadline if opened else None

    def pending_deadline(self, start: float, end: float) -> float | None:
        """Deadline of an open window that could still decide a segment
        spanning [start, end], else None."""
        window = self._window
        if window is None or window.opened_at > end + _INSIDE_SLACK_S:
            return None
        return window.deadline

    def settle(self, now: float) -> WakeHit | None:
        """Close the open window if its deadline has passed. Returns the
        winner when a window closed."""
        window = self._window
        if window is None or now < window.deadline:
            return None
        self._window = None
        winner = max(window.best.values(), key=self._rank)
        self._record(winner, window.best.values())
        return winner

    def fault(self, client_id: str, room_id: str) -> None:
        self._room(room_id).faulted.add(client_id)

    def forget_client(self, client_id: str) -> None:
        if self._window is not None:
            self._window.best.pop(client_id, None)
            if not self._window.best:
                self._window = None
        for state in self._rooms.values():
            state.faulted.discard(client_id)

    def turn_ended(self, room_id: str, at: float, started_at: float | None = None) -> None:
        """Open the follow-up window. The owner mic keeps the room through it
        only if it still owned the room when this turn began."""
        state = self._room(room_id)
        state.last_turn_end = at
        if state.owner is None:
            return
        if state.owner_until >= (at if started_at is None else started_at):
            state.owner_until = at + self._t.followup_window_s
        else:
            state.owner = None

    def decide(self, facts: SegmentFacts, now: float) -> Admission | Drop:
        self.settle(now)
        if self.mode(facts.room_id) == "off":
            return Admission("wake_off")
        state = self._room(facts.room_id)
        self._prune(state, now)
        return self._judge(state, facts)

    def _judge(self, state: _RoomState, facts: SegmentFacts) -> Admission | Drop:
        if facts.client_id in state.faulted:
            return Admission("barge_in_only") if facts.turn_in_flight else Drop("detector_fault")
        if any(c == facts.client_id and self._inside(at, facts) for c, at in state.losses):
            return Drop("lost_arbitration")
        if self._owned_by_other(state, facts):
            return Drop("not_owner")
        return self._admit_by_window(state, facts)

    def _admit_by_window(self, state: _RoomState, facts: SegmentFacts) -> Admission | Drop:
        mine = [w for w in state.wins if w.client_id == facts.client_id]
        if any(self._inside(w.at, facts) for w in mine):
            return Admission("one_breath")
        if any(w.at <= facts.start <= w.at + self._t.armed_window_s for w in mine):
            return Admission("armed")
        if facts.confirm_pending:
            return Admission("confirm")
        if self._in_followup(state, facts.start):
            return Admission("follow_up")
        if facts.turn_in_flight:
            return Admission("barge_in_only")
        return Drop("not_woken")

    def _in_followup(self, state: _RoomState, start: float) -> bool:
        end = state.last_turn_end
        return end is not None and start <= end + self._t.followup_window_s

    def _owned_by_other(self, state: _RoomState, facts: SegmentFacts) -> bool:
        return (
            state.owner is not None
            and state.owner != facts.client_id
            and facts.start <= state.owner_until
        )

    @staticmethod
    def _inside(at: float, facts: SegmentFacts) -> bool:
        return facts.start <= at <= facts.end + _INSIDE_SLACK_S

    def _rank(self, hit: WakeHit) -> tuple[float, float]:
        return (hit.energy_dbfs + self._gain_for_room(hit.room_id), hit.score)

    def _record(self, winner: WakeHit, contenders) -> None:
        state = self._room(winner.room_id)
        state.wins.append(winner)
        state.owner = winner.client_id
        state.owner_until = max(
            winner.at + self._t.armed_window_s,
            self._followup_end(state),
        )
        for hit in contenders:
            if hit is not winner:
                self._room(hit.room_id).losses.append((hit.client_id, hit.at))

    def _followup_end(self, state: _RoomState) -> float:
        if state.last_turn_end is None:
            return float("-inf")
        return state.last_turn_end + self._t.followup_window_s

    def _prune(self, state: _RoomState, now: float) -> None:
        horizon = now - self._t.armed_window_s - self._t.followup_window_s
        state.wins = [w for w in state.wins if w.at >= horizon]
        state.losses = [(c, at) for c, at in state.losses if at >= horizon]

    def _room(self, room_id: str) -> _RoomState:
        return self._rooms.setdefault(room_id, _RoomState())


_NON_WORD = re.compile(r"[^a-z0-9 ]+")


def strip_wake_phrase(text: str, aliases: list[str]) -> str:
    """Remove a leading wake phrase (any alias, punctuation/case-insensitive)
    from a transcript. Returns the remainder with its original casing, or ""
    when the transcript was only the wake phrase."""
    words = text.split()
    normalised = [_NON_WORD.sub("", w.lower()) for w in words]
    for alias in sorted(aliases, key=len, reverse=True):
        n = _alias_word_count(normalised, alias.lower().split())
        if n:
            return " ".join(words[n:]).lstrip(" ,.!?;:-").strip()
    return text.strip()


def _alias_word_count(normalised: list[str], alias: list[str]) -> int:
    """How many leading transcript words spell `alias` (words may be split or
    joined differently, e.g. "glad os" / "heyglados"). 0 if no match."""
    target = "".join(alias)
    joined = ""
    for i, word in enumerate(normalised):
        joined += word
        if joined == target:
            return i + 1
        if not target.startswith(joined):
            return 0
    return 0
