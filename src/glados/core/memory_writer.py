"""Async per-speaker memory writer (DESIGN-speaker-memory.md).

Runs on the SAME resident model as the organizer and must never slow a live
turn. Three mechanisms, in order of how much they carry:

1. Gate: a job starts only when the LLM is warm, no room has a turn active or
   queued, and the session has been quiet for `idle_s`.
2. Preempt: `turn_starting()` is pushed by the organizer the moment a turn is
   enqueued and cancels the in-flight writer call. The call is a streamed chat,
   so cancelling closes the HTTP stream and Ollama stops generating
   (tests/test_ollama_cancel.py). A preempted job is requeued, bounded by
   `max_attempts`.
3. Re-warm: a writer call evicts the organizer's cached prompt prefix from
   Ollama's single slot, so after each job `rewarm` replays an
   organizer-shaped prompt. Preemptible like the job.

`OLLAMA_NUM_PARALLEL` stays 1: a second slot reserves KV for a full context
and concurrent decode slows the live turn, and the gate makes it unnecessary.

The queue lives in RAM. A restart loses turns not yet written; accepted.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field

from .adapters import LLM, LLMMessage, LLMText
from .memory_facts import TurnNote, judge_writer_output, writer_messages
from .speaker_memory import Key, SpeakerMemoryStore, StaleGeneration

log = logging.getLogger(__name__)

MAX_NOTES_PER_JOB = 8
MAX_PENDING_JOBS = 32

# One pending job per (room, speaker, session), so every stored fact names the
# single session it came from and `forget --session` purges exactly that.
JobKey = tuple[str, str, str]


@dataclass
class _Pending:
    purges: int
    notes: list[TurnNote] = field(default_factory=list)
    last_at: float = 0.0
    attempts: int = 0
    committed: bool = False


class MemoryWriter:
    def __init__(
        self,
        *,
        llm: LLM,
        store: SpeakerMemoryStore,
        is_busy: Callable[[], bool],
        is_warm: Callable[[], bool],
        rewarm: Callable[[], Awaitable[None]] | None = None,
        idle_s: float = 60.0,
        poll_s: float = 1.0,
        max_attempts: int = 3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._llm = llm
        self._store = store
        self._is_busy = is_busy
        self._is_warm = is_warm
        self._rewarm = rewarm
        self._idle_s = idle_s
        self._poll_s = poll_s
        self._max_attempts = max_attempts
        self._clock = clock
        self._pending: dict[JobKey, _Pending] = {}
        self._job: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self.jobs_done = 0
        self.jobs_preempted = 0
        self.jobs_dropped = 0

    def note_turn(self, key: Key, session_id: str, note: TurnNote) -> None:
        job_key = (*key, session_id)
        pending = self._pending.pop(job_key, None) or _Pending(
            purges=self._store.purges(*key)
        )
        if len(self._pending) >= MAX_PENDING_JOBS:
            self._drop(next(iter(self._pending)), "queue full")
        pending.notes = (pending.notes + [note])[-MAX_NOTES_PER_JOB:]
        pending.last_at = self._clock()
        self._pending[job_key] = pending

    def turn_starting(self) -> None:
        job = self._job
        if job is not None and not job.done():
            job.cancel()

    def poke(self) -> None:
        """Re-check the gate now instead of at the next poll. Test hook."""
        self._wake.set()

    async def run(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._poll_s)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            job_key = self._due_job()
            if job_key is not None:
                await self._run_job(job_key)

    async def close(self) -> None:
        self.turn_starting()
        if self._job is not None:
            await asyncio.wait({self._job})

    def _due_job(self) -> JobKey | None:
        if not self._is_warm() or self._is_busy():
            return None
        now = self._clock()
        due = [k for k, p in self._pending.items() if now - p.last_at >= self._idle_s]
        return min(due, key=lambda k: self._pending[k].last_at) if due else None

    async def _run_job(self, job_key: JobKey) -> None:
        pending = self._pending.pop(job_key)
        started = self._clock()
        log.info(
            "memory job start room=%s speaker=%s turns=%d",
            *job_key[:2], len(pending.notes),
        )
        self._job = asyncio.create_task(self._write_then_rewarm(job_key, pending))
        await asyncio.wait({self._job})
        job, self._job = self._job, None
        elapsed_ms = int((self._clock() - started) * 1000)
        if job.cancelled() and not pending.committed:
            self._requeue(job_key, pending, elapsed_ms)
        elif job.exception() is not None:
            self._drop(job_key, f"error {job.exception()!r} elapsed_ms={elapsed_ms}")

    async def _write_then_rewarm(self, job_key: JobKey, pending: _Pending) -> None:
        await self._write(job_key, pending)
        pending.committed = True
        if self._rewarm is not None:
            started = self._clock()
            await self._rewarm()
            log.info("memory rewarm done elapsed_ms=%d", int((self._clock() - started) * 1000))

    async def _write(self, job_key: JobKey, pending: _Pending) -> None:
        room_id, speaker_id, session_id = job_key
        started = self._clock()
        generation = self._store.generation(room_id, speaker_id)
        known = self._store.facts(room_id, speaker_id)
        raw = await self._collect(writer_messages(known, pending.notes))
        verdict = judge_writer_output(raw, known)
        apply = asyncio.ensure_future(
            self._store.apply(
                (room_id, speaker_id),
                adds=verdict.adds,
                drop_ids=verdict.drop_ids,
                expected_generation=generation,
                expected_purges=pending.purges,
                source_session=session_id,
                source_utterance=" | ".join(n.user_text for n in pending.notes),
            )
        )
        try:
            await self._committed(apply, pending)
        except StaleGeneration:
            log.info(
                "memory job discarded room=%s speaker=%s: facts changed meanwhile",
                room_id, speaker_id,
            )
            return
        self.jobs_done += 1
        log.info(
            "memory job done room=%s speaker=%s added=%d dropped=%d rejected=%d elapsed_ms=%d",
            room_id, speaker_id, len(verdict.adds), len(verdict.drop_ids),
            verdict.rejected, int((self._clock() - started) * 1000),
        )

    @staticmethod
    async def _committed(apply: asyncio.Future, pending: _Pending) -> None:
        """Await the store write to its end even if preempted meanwhile: the
        thread commits regardless, and a job requeued after its own commit
        would write the same facts twice."""
        try:
            await asyncio.shield(apply)
        except asyncio.CancelledError:
            await asyncio.wait({apply})
            pending.committed = not apply.cancelled() and apply.exception() is None
            raise

    async def _collect(self, messages: list[LLMMessage]) -> str:
        parts: list[str] = []
        async with aclosing(self._llm.chat(messages, [])) as stream:
            async for event in stream:
                if isinstance(event, LLMText):
                    parts.append(event.text)
        return "".join(parts)

    def _requeue(self, job_key: JobKey, pending: _Pending, elapsed_ms: int) -> None:
        self.jobs_preempted += 1
        pending.attempts += 1
        log.info(
            "memory job preempted room=%s speaker=%s attempt=%d elapsed_ms=%d",
            *job_key[:2], pending.attempts, elapsed_ms,
        )
        if pending.attempts >= self._max_attempts:
            self.jobs_dropped += 1
            log.warning(
                "memory job dropped room=%s speaker=%s: preempted %d times",
                *job_key[:2], pending.attempts,
            )
            return
        newer = self._pending.pop(job_key, None)
        if newer is not None:
            pending.notes = (pending.notes + newer.notes)[-MAX_NOTES_PER_JOB:]
        # A fresh idle window, not an immediate retry: the house was just
        # active, and a retry the moment that turn ends would sit in front
        # of the follow-up utterance that usually comes next.
        pending.last_at = self._clock()
        self._pending[job_key] = pending

    def _drop(self, job_key: JobKey, reason: str) -> None:
        self._pending.pop(job_key, None)
        self.jobs_dropped += 1
        log.warning("memory job dropped room=%s speaker=%s: %s", *job_key[:2], reason)
