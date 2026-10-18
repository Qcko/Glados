"""Nightly per-speaker memory batch (DESIGN-speaker-memory.md).

Runs on the SAME resident model as the organizer and must never slow a live
turn. Three mechanisms, in order of how much they carry:

1. Gate: a batch starts only when no run has finished for today's LOCAL date,
   the house has been idle `idle_before_batch_s`, no room has a turn active or
   queued, and the LLM is warm. The date is stored, never derived from a
   monotonic clock, so a clock jump or a DST fall-back cannot run it twice or
   skip it. Prod is often off at night, so the same trigger is the catch-up
   after boot.
2. Preempt: `turn_starting()` is pushed by the organizer the moment a turn is
   enqueued and cancels the in-flight writer call. The batch stops there; the
   preempted chunk's notes are still staged, so the next idle window resumes
   it with no retry bookkeeping.
3. Re-warm: a writer call evicts the organizer's cached prompt prefix from
   Ollama's single slot, so after EACH chunk `rewarm` replays an
   organizer-shaped prompt -- a turn can arrive between chunks. Preemptible
   like the chunk.

One job is one chunk of one (room, person, session), at most
`max_notes_per_batch` notes, run one after another so each sees the facts the
previous one produced. A chunk whose prompt would not fit the window is split
before it is sent, never left for Ollama to truncate from the front.

`OLLAMA_NUM_PARALLEL` stays 1: a second slot reserves KV for a full context
and concurrent decode slows the live turn, and the gate makes it unnecessary.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime
from datetime import time as clock_time

from .adapters import LLM, LLMMessage, LLMText
from .memory_facts import DayBudget, Fact, WriterVerdict, judge_writer_output, writer_messages
from .speaker_memory import (
    Chunk,
    ChunkWrite,
    Group,
    SpeakerMemoryStore,
    StaleGeneration,
)

log = logging.getLogger(__name__)

# Pessimistic for speech (English prose runs about 4), so a chunk is split
# well before the window could truncate it.
_CHARS_PER_TOKEN = 2
_STALE_RETRIES = 3
_DAY_S = 86400.0


class EnrollmentUnknown(Exception):
    """Who is enrolled cannot be read now; the chunk fails and its notes stay."""


def _enrollment_unbound(person: str) -> bool:
    raise EnrollmentUnknown("enrollment check not bound yet")


@dataclass(frozen=True)
class BatchPolicy:
    idle_before_batch_s: float = 600.0
    retention_days: float = 2.0
    max_notes_per_batch: int = 30
    daily_adds: int = 8
    daily_drops: int = 4
    daily_untrusted_adds: int = 1
    context_tokens: int = 12288
    output_tokens: int = 256
    window: tuple[clock_time, clock_time] | None = None


class MemoryWriter:
    def __init__(
        self,
        *,
        llm: LLM,
        store: SpeakerMemoryStore,
        is_busy: Callable[[], bool],
        is_warm: Callable[[], bool],
        rewarm: Callable[[], Awaitable[None]] | None = None,
        is_enrolled: Callable[[str], bool] = _enrollment_unbound,
        policy: BatchPolicy = BatchPolicy(),
        poll_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        local_now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._llm = llm
        self._store = store
        self._is_busy = is_busy
        self._is_warm = is_warm
        self._rewarm = rewarm
        self._is_enrolled = is_enrolled
        self._policy = policy
        self._poll_s = poll_s
        self._clock = clock
        self._local_now = local_now
        self._last_activity = clock()
        self._job: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self.batches_done = 0
        self.chunks_done = 0
        self.chunks_preempted = 0
        self.chunks_failed = 0

    def set_enrollment(self, is_enrolled: Callable[[str], bool]) -> None:
        self._is_enrolled = is_enrolled

    def turn_starting(self) -> None:
        self._last_activity = self._clock()
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
            run_date = self._run_date()
            if self._due(run_date):
                await self._run_batch(run_date)

    async def close(self) -> None:
        """Cancel the chunk in flight and wait for it, which includes any
        threaded apply that had already started: it commits regardless."""
        job = self._job
        self.turn_starting()
        if job is not None:
            await asyncio.wait({job})

    # ---- gate ---------------------------------------------------------------

    def _run_date(self) -> str:
        return self._local_now().date().isoformat()

    def _due(self, run_date: str) -> bool:
        return (
            self._house_idle()
            and self._in_window()
            and self._store.can_read()
            and not self._store.ran_on(run_date)
        )

    def _house_idle(self) -> bool:
        now = self._clock()
        if self._is_busy():
            self._last_activity = now
            return False
        return self._is_warm() and now - self._last_activity >= self._policy.idle_before_batch_s

    def _in_window(self) -> bool:
        window = self._policy.window
        if window is None:
            return True
        start, end = window
        now = self._local_now().time()
        return start <= now < end if start <= end else now >= start or now < end

    # ---- batch --------------------------------------------------------------

    async def _run_batch(self, run_date: str) -> None:
        started = self._clock()
        self.expire()
        failed: set[Group] = set()
        while True:
            if not self._house_idle():
                log.info("memory batch paused: house busy")
                return
            group = next((g for g in self._store.pending_groups() if g not in failed), None)
            if group is None:
                break
            outcome = await self._run_job(group, run_date)
            if outcome == "preempted":
                return
            if outcome == "failed":
                failed.add(group)
        self._store.record_run(run_date)
        self.batches_done += 1
        log.log(
            logging.WARNING if failed else logging.INFO,
            "memory batch done date=%s failed_groups=%d elapsed_ms=%d",
            run_date, len(failed), int((self._clock() - started) * 1000),
        )

    def expire(self) -> int:
        """Drop notes past retention, unprocessed. At boot and at run start."""
        cutoff = time.time() - self._policy.retention_days * _DAY_S
        return self._store.expire_notes(cutoff)

    async def _run_job(self, group: Group, run_date: str) -> str:
        started = self._clock()
        job = self._job = asyncio.create_task(self._chunk_then_rewarm(group, run_date))
        try:
            await asyncio.wait({job})
        except asyncio.CancelledError:
            job.cancel()
            await asyncio.wait({job})
            raise
        finally:
            self._job = None
        elapsed_ms = int((self._clock() - started) * 1000)
        if job.cancelled():
            self.chunks_preempted += 1
            log.info("memory chunk preempted room=%s person=%s elapsed_ms=%d", *group[:2], elapsed_ms)
            return "preempted"
        if job.exception() is not None:
            self.chunks_failed += 1
            log.warning(
                "memory chunk failed room=%s person=%s: %s elapsed_ms=%d",
                *group[:2], type(job.exception()).__name__, elapsed_ms,
            )
            return "failed"
        return job.result()

    async def _chunk_then_rewarm(self, group: Group, run_date: str) -> str:
        outcome, called_llm = await self._write_chunk(group, run_date)
        if called_llm and self._rewarm is not None:
            started = self._clock()
            await self._rewarm()
            log.info("memory rewarm done elapsed_ms=%d", int((self._clock() - started) * 1000))
        return outcome

    async def _write_chunk(self, group: Group, run_date: str) -> tuple[str, bool]:
        """One chunk to its apply. A stale generation (a fact forgotten
        meanwhile) re-reads the facts and the chunk and retries; the notes are
        still staged, so the retry starts from the same oldest note."""
        called_llm = False
        for _ in range(_STALE_RETRIES):
            stamp = self._store.stamp(*group[:2])
            known = self._store.facts(*group[:2])
            chunk = self._fit_chunk(group, known)
            if not chunk.ids:
                return "done", called_llm
            called_llm = called_llm or bool(chunk.notes)
            verdict = await self._judge(chunk, known, run_date)
            write = ChunkWrite(
                group=group, run_date=run_date, note_ids=chunk.ids, expected=stamp,
                adds=tuple(verdict.adds), drop_ids=tuple(verdict.drop_ids),
                reaffirmed=verdict.reaffirmed,
            )
            try:
                outcome = await self._committed(self._store.apply_chunk(write, self._is_enrolled))
            except StaleGeneration:
                log.info("memory chunk stale room=%s person=%s: retrying", *group[:2])
                continue
            self._log_applied(chunk, verdict, outcome)
            return outcome, called_llm
        return "failed", called_llm

    def _log_applied(self, chunk: Chunk, verdict: WriterVerdict, outcome: str) -> None:
        self.chunks_done += 1
        log.info(
            "memory chunk %s room=%s person=%s notes=%d added=%d dropped=%d rejected=%d",
            outcome, *chunk.group[:2], len(chunk.ids), len(verdict.adds),
            len(verdict.drop_ids), verdict.rejected,
        )

    def _fit_chunk(self, group: Group, known: list[Fact]) -> Chunk:
        """The largest leading slice of the group that fits the window. A
        single note too big to fit is consumed with nothing derived."""
        limit = self._policy.max_notes_per_batch
        while True:
            chunk = self._store.read_chunk(group, limit)
            if self._fits(writer_messages(known, chunk.notes)):
                return chunk
            if limit == 1:
                log.warning("memory chunk room=%s person=%s: one note over budget, dropped", *group[:2])
                return Chunk(group=group, notes=[], ids=chunk.ids)
            limit = max(1, min(limit, len(chunk.ids)) // 2)

    def _fits(self, messages: list[LLMMessage]) -> bool:
        chars = sum(len(m.content or "") for m in messages)
        budget = self._policy.context_tokens - self._policy.output_tokens
        return chars // _CHARS_PER_TOKEN <= budget

    async def _judge(self, chunk: Chunk, known: list[Fact], run_date: str) -> WriterVerdict:
        if not chunk.notes:
            return WriterVerdict(adds=[], drop_ids=[], rejected=0)
        raw = await self._collect(writer_messages(known, chunk.notes))
        return judge_writer_output(raw, known, chunk.notes, self._budget(chunk.group, run_date))

    def _budget(self, group: Group, run_date: str) -> DayBudget:
        used = self._store.day_counts(group[1], run_date)
        return DayBudget(
            adds=self._policy.daily_adds - used.adds,
            drops=self._policy.daily_drops - used.drops,
            untrusted_adds=self._policy.daily_untrusted_adds - used.untrusted_adds,
        )

    @staticmethod
    async def _committed(apply: Awaitable[str]) -> str:
        """Await the store write to its end even if preempted meanwhile: the
        thread commits regardless, and the batch must not move on, or shut
        down, while it is still writing."""
        future = asyncio.ensure_future(apply)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            await asyncio.wait({future})
            raise

    async def _collect(self, messages: list[LLMMessage]) -> str:
        parts: list[str] = []
        async with aclosing(self._llm.chat(messages, [])) as stream:
            async for event in stream:
                if isinstance(event, LLMText):
                    parts.append(event.text)
        return "".join(parts)
