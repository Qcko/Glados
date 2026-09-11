"""Cross-turn memory of additive cart writes that already landed.

The per-turn in-flight ledger on `TurnRecord` refuses a call re-issued while
its first attempt is still outstanding. This one answers the question that
ledger cannot: the write LANDED, the turn ended, and the next utterance asks
for it again. Observed 11-09-2026 on ministral3:8b-instruct: "add tomatoes to
the cart" twice in a row, and on the second the model removed the line and
re-added it with an invented quantity of four. The Dunnes server refuses an
identical additive write inside two minutes, but a model that rewrites the
arguments is not sending an identical write, so the refusal has to live where
the arguments are still the user's -- here, keyed on the call minus the
quantity and minus the server's own `repeat` override.

Every entry is harness-authored: the tool, the arguments the model sent, the
clock, and whether the result was certain. Never the server's result content.
A refusal built from this ledger is spoken to the model OUTSIDE any
`<external>` wrapper, because GLaDOS wrote it -- and that is only true while
nothing in it came off the wire (ARCHITECTURE section 7).

Pure and clock-injected so the window is testable without sleeping.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .adapters import LLMToolCall

# How long an additive write is remembered. Matches the Dunnes server's own
# identical-write window, but the two clocks measure different things (the
# server counts from its write, this counts from the result reaching us), so
# the match is a convenience, not an invariant.
REPEAT_WINDOW_S = 120.0

# The server-side override that disables its identical-write refusal. Dropped
# from the key so a model that adds `repeat: true` on its own cannot turn a
# re-issue into a "different" call; the harness sets the real value from the
# user's words (see `Organizer._align_repeat_flag`).
REPEAT_ARG = "repeat"

WriteKey = tuple[str, str]


@dataclass(frozen=True)
class WriteEntry:
    tool: str
    key: WriteKey
    quantity: int | None
    at: float
    # False when the call timed out after it was sent: it may have landed, so
    # a re-issue is refused with "outcome unknown" rather than "already done".
    certain: bool


def canonical_key(call: LLMToolCall, drop: Iterable[str] = ()) -> WriteKey:
    """Identity of a call for either ledger. Keys are sorted so argument order
    cannot disguise a re-issue; string values are lowercased and stripped so
    "Tomatoes" and "tomatoes " are the same request; `drop` names the
    arguments that must not distinguish two calls."""
    dropped = set(drop)
    args = {
        k: _canonical_value(v) for k, v in call.args.items() if k not in dropped
    }
    return (f"{call.server}.{call.name}", json.dumps(args, sort_keys=True, default=str))


def _canonical_value(value: object) -> object:
    if isinstance(value, str):
        return " ".join(value.lower().split())
    return value


def coerce_quantity(value: object) -> int | None:
    """The model may send 4, "4" or 4.0; the guard compares an integer."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


class WriteLedger:
    """Per-session, LRU-bounded, pruned on every touch."""

    def __init__(
        self,
        *,
        window_s: float = REPEAT_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
        max_sessions: int = 64,
    ) -> None:
        self._window_s = window_s
        self._clock = clock
        self._max_sessions = max_sessions
        self._entries: dict[str, list[WriteEntry]] = {}

    def note(
        self,
        session_id: str,
        call: LLMToolCall,
        key: WriteKey,
        quantity: int | None,
        *,
        certain: bool,
    ) -> WriteEntry:
        entry = WriteEntry(
            tool=f"{call.server}.{call.name}",
            key=key,
            quantity=quantity,
            at=self._clock(),
            certain=certain,
        )
        live = [e for e in self._live(session_id) if e.key != key]
        live.append(entry)
        self._entries.pop(session_id, None)
        self._evict_if_full()
        self._entries[session_id] = live
        return entry

    def recent(self, session_id: str, key: WriteKey) -> WriteEntry | None:
        for entry in self._live(session_id):
            if entry.key == key:
                return entry
        return None

    def seconds_since(self, entry: WriteEntry) -> int:
        return int(self._clock() - entry.at)

    def clear(self, session_id: str, server: str) -> None:
        """A non-additive write landed on `server` (remove, set, adjust):
        whatever the ledger asserted about that server's cart may no longer
        hold. Adds and removes cannot be matched by argument (one is a
        free-text query, the other a product id), so every entry for the
        server is cleared rather than one -- coarse, and it fails open toward
        the server's own refusal. Other servers' entries stay: an intercom
        message or a timer says nothing about the cart."""
        prefix = f"{server}."
        kept = [e for e in self._live(session_id) if not e.tool.startswith(prefix)]
        if kept:
            self._entries[session_id] = kept
        else:
            self._entries.pop(session_id, None)

    def forget(self, session_id: str) -> None:
        self._entries.pop(session_id, None)

    def _live(self, session_id: str) -> list[WriteEntry]:
        cutoff = self._clock() - self._window_s
        live = [e for e in self._entries.get(session_id, []) if e.at >= cutoff]
        if live:
            self._entries[session_id] = live
        else:
            self._entries.pop(session_id, None)
        return live

    def _evict_if_full(self) -> None:
        while len(self._entries) >= self._max_sessions:
            del self._entries[next(iter(self._entries))]
