"""The organizer's whole view of per-speaker memory (DESIGN-speaker-memory.md).

Three narrow calls keep the writer, its queue and its gating out of
`organizer.py`: the block a turn injects, the note a committed turn leaves,
and the push that a turn is starting.
"""

from __future__ import annotations

from collections.abc import Callable

from .memory_facts import MEMORY_PREAMBLE, TurnNote, render_block, wrap_block
from .memory_writer import MemoryWriter
from .prompt_budget import dense_payload
from .speaker_memory import SpeakerMemoryStore


class SpeakerMemory:
    def __init__(
        self,
        *,
        store: SpeakerMemoryStore,
        writer: MemoryWriter,
        room_enabled: Callable[[str], bool],
        inject_chars: int,
    ) -> None:
        self._store = store
        self._writer = writer
        self._room_enabled = room_enabled
        self._inject_chars = inject_chars

    def system_suffix(self, room_id: str, speaker_id: str) -> str:
        """Text to append to this turn's system prompt, or "" for none."""
        if not self._room_enabled(room_id):
            return ""
        block = render_block(
            f"speaker:{speaker_id}",
            self._store.facts(room_id, speaker_id),
            self._inject_chars,
        )
        return f"\n{MEMORY_PREAMBLE}\n{block}" if block else ""

    def worst_case_suffix(self) -> str:
        """The largest suffix `system_suffix` can produce, for the boot budget
        proof: preamble, wrapper, a long speaker id and token-dense filler."""
        filler = dense_payload(self._inject_chars)
        return f"\n{MEMORY_PREAMBLE}\n{wrap_block('speaker:' + 'x' * 64, filler)}"

    def on_turn_committed(
        self,
        *,
        room_id: str,
        speaker_id: str,
        session_id: str,
        user_text: str,
        reply: str,
        untrusted: bool,
    ) -> None:
        if not self._room_enabled(room_id) or not user_text.strip():
            return
        note = TurnNote(user_text=user_text, reply=None if untrusted else reply)
        self._writer.note_turn((room_id, speaker_id), session_id, note)

    def turn_starting(self) -> None:
        self._writer.turn_starting()

    def close(self) -> None:
        self._store.close()
