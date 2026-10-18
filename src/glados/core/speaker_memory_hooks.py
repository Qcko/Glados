"""The organizer's whole view of per-speaker memory (DESIGN-speaker-memory.md).

Three narrow calls keep the writer, its staging and its gating out of
`organizer.py`: the block a turn injects, the note a committed turn leaves,
and the push that a turn is starting.

A committed turn's note is encrypted and inserted by a tracked background
task. The turn never awaits it, and the insert waits at most on the store's
write lock, which no LLM call ever holds.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable

from cryptography.fernet import Fernet

from .memory_facts import MEMORY_PREAMBLE, StagedNote, render_block, wrap_block
from .memory_writer import MemoryWriter
from .prompt_budget import dense_payload
from .speaker import SpeakerIdentity
from .speaker_memory import SpeakerMemoryStore

log = logging.getLogger(__name__)


class SpeakerMemory:
    def __init__(
        self,
        *,
        store: SpeakerMemoryStore,
        writer: MemoryWriter,
        room_enabled: Callable[[str], bool],
        inject_chars: int,
        inject: bool = False,
    ) -> None:
        self._store = store
        self._writer = writer
        self._room_enabled = room_enabled
        self._inject_chars = inject_chars
        self._inject = inject
        self._staging: set[asyncio.Task] = set()
        self.staging_failed = 0

    def unlock(self, cipher: Fernet | None) -> None:
        """Hand the store its notes key. Until then, and for good if the
        keyring gave none, nothing is staged. Reopens a store a previous
        lifespan closed."""
        self._store.open()
        self._store.set_cipher(cipher)

    def system_suffix(self, room_id: str, speaker: SpeakerIdentity | None) -> str:
        """Text to append to this turn's system prompt, or "" for none --
        always "" in shadow mode, where memory is written but never fed back."""
        person = _verified_person(speaker)
        if not self._inject or person is None or not self._room_enabled(room_id):
            return ""
        block = render_block(
            f"speaker:{person}",
            self._store.facts(room_id, person),
            self._inject_chars,
        )
        return f"\n{MEMORY_PREAMBLE}\n{block}" if block else ""

    def worst_case_suffix(self) -> str:
        """The largest suffix `system_suffix` can produce, for the boot budget
        proof: preamble, wrapper, a long speaker id and token-dense filler.
        Nothing in shadow mode, which injects nothing."""
        if not self._inject:
            return ""
        filler = dense_payload(self._inject_chars)
        return f"\n{MEMORY_PREAMBLE}\n{wrap_block('speaker:' + 'x' * 64, filler)}"

    def on_turn_committed(
        self,
        *,
        room_id: str,
        speaker: SpeakerIdentity | None,
        session_id: str,
        user_text: str,
        reply: str,
        untrusted: bool,
    ) -> None:
        person = _verified_person(speaker)
        if person is None or not self._room_enabled(room_id) or not user_text.strip():
            return
        note = StagedNote(
            id=0, user_text=user_text, reply=None if untrusted else reply, untrusted=untrusted
        )
        task = asyncio.create_task(
            asyncio.to_thread(
                self._store.stage_note, (room_id, person, session_id), uuid.uuid4().hex, note
            ),
            name="memory-stage",
        )
        self._staging.add(task)
        task.add_done_callback(self._staged)

    def _staged(self, task: asyncio.Task) -> None:
        self._staging.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self.staging_failed += 1
            log.warning(
                "speaker memory: staging a note failed: %s", type(task.exception()).__name__
            )

    async def drain(self) -> None:
        """Let every staging insert in flight finish. Shutdown calls this
        before `close`, so no thread is left writing to a closed connection."""
        if self._staging:
            await asyncio.wait(set(self._staging))

    def turn_starting(self) -> None:
        self._writer.turn_starting()

    def close(self) -> None:
        self._store.close()


def _verified_person(speaker: SpeakerIdentity | None) -> str | None:
    """Whose memory a turn may touch: only a voice-matched person. A typed
    turn or an unidentified voice falls back to the client's default user,
    who is a real person, so it gets no memory (DESIGN-speaker-id.md)."""
    if speaker is None or not speaker.verified:
        return None
    return speaker.person
