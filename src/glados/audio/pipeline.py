"""AudioPipeline: per-connection orchestration of VAD -> STT -> Organizer.

The server hands every inbound binary audio frame (`<BE u32 seq><PCM16-LE>`)
to `feed_frame`. The pipeline parses it once, optionally tees it to a
WAV trace sink for offline replay, and feeds the PCM body through the
VAD. When the VAD emits an utterance end, transcription runs in a
background task so the WS receive loop is never blocked by a multi-
hundred-millisecond Whisper call -- the user can start the next utterance
while the previous one is still being transcribed.

`close()` drains outstanding transcriptions before tearing down the sink
and resetting the VAD so a per-connection `<traces>/audio/<client>/...wav`
is well-formed even if the client disconnects mid-utterance.

Speaker identification (DESIGN-speaker-id.md) runs on the SAME pcm,
concurrently with STT, and its result travels with the text in one
`Utterance` -- never through per-client state. Identification is capped by
its own timeout, so it delays a turn only while STT is still running anyway.
Dispatch is serialised in VadEnd order: each utterance waits for the one
before it, so a slow transcription or identification cannot reorder turns.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol

import numpy as np

log = logging.getLogger(__name__)

from ..core.adapters import STT, VAD, VadEnd, VadStart, WakeWord
from ..core.audio_sink import AudioSink, FrameTooShort
from ..core.protocols import AUDIO_HEADER_LEN
from ..core.speaker import NO_IDENTIFIER, SpeakerMatch, Utterance
from ..core.turn_latency import SpeechTiming

OnUtterance = Callable[[Utterance], Awaitable[None]]

if TYPE_CHECKING:
    from ..voiceid import SpeakerIdentifier

_SAMPLE_RATE = 16000
_ENERGY_WINDOW_S = 1.0
_FULL_SCALE_SQ = 32768.0 ** 2
_SILENCE_DBFS = -120.0
# A segment wholly inside the TTS gate is dropped untranscribed only from this
# length up. The longest barge-in the text check accepts ("hey okay glados stop
# talking please") is about 2.5 s spoken slowly plus the VAD's padding; shorter
# segments still reach STT so a "stop" is heard over a closed gate.
_ECHO_SKIP_MIN_S = 3.0
# Less speech than this after the gate's horizon is reverb or a clipped
# syllable, not a reply: the segment is treated as wholly echo.
_MIN_REPLY_AFTER_ECHO_S = 0.3


class EchoGate(Protocol):
    """The room's TTS feedback gate, as the pipeline sees it before STT."""

    def horizon(self, captured_at: float) -> float | None:
        """Loop time the room's own TTS stops reaching the mic, when a segment
        starting at `captured_at` began under a settled gate; else None."""
        ...

    def dropped(self, duration_s: float) -> None: ...


class WakeHook(Protocol):
    """What the pipeline needs from the Organizer when wake gating is on
    (DESIGN.md, wake word): report hits and faults, and decide -- at each
    VAD segment end, BEFORE transcription -- whether the segment may become
    text. A refused segment is discarded untranscribed."""

    detector: WakeWord
    cooldown_s: float

    def on_hit(self, score: float, energy_dbfs: float, at: float) -> None: ...
    def on_fault(self) -> None: ...
    async def decide(self, start: float, end: float) -> Admission | None: ...


class AudioPipeline:
    def __init__(
        self,
        *,
        sink: AudioSink | None,
        vad: VAD,
        stt: STT,
        on_utterance: OnUtterance,
        identifier: SpeakerIdentifier | None = None,
        wake: WakeHook | None = None,
        echo_gate: EchoGate | None = None,
    ) -> None:
        self._sink = sink
        self._vad = vad
        self._stt = stt
        self._on_utterance = on_utterance
        self._identifier = identifier
        self._tasks: set[asyncio.Task[None]] = set()
        self._last_task: asyncio.Task[None] | None = None
        self._utterance_start_at: float | None = None
        self._wake = wake
        self._echo_gate = echo_gate
        self._wake_exec = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="wake")
            if wake is not None
            else None
        )
        self._wake_faulted = False
        self._closed = False
        self._last_hit_at = float("-inf")
        self._energy: deque[tuple[float, int]] = deque()
        self._energy_samples = 0

    async def feed_frame(self, framed: bytes) -> None:
        pcm = self._parse(framed)
        if self._sink is not None:
            self._sink.write(framed)
        arrived_at = asyncio.get_running_loop().time()
        if self._wake is not None:
            await self._feed_wake(pcm, arrived_at)
        for event in self._vad.feed(pcm):
            if isinstance(event, VadStart):
                # Anchor the utterance to when speech BEGAN, not when the VAD
                # reports it ended. The feedback gate judges suppression against
                # this stamp; using the start defeats the VAD silence-hangover
                # (~200ms) and round-trip latency that would otherwise push a
                # VadEnd stamp past the gate's horizon and leak TTS bleed. A
                # bleed always *begins* while GLaDOS is still playing, so the
                # start lands inside the closed window regardless of how long
                # the utterance is.
                self._utterance_start_at = arrived_at
            elif isinstance(event, VadEnd):
                # Pair with the VadStart stamp (fall back to now for a VadEnd
                # with no recorded start -- a force-cut or a reset mid-utterance).
                captured_at = (
                    self._utterance_start_at
                    if self._utterance_start_at is not None
                    else arrived_at
                )
                self._utterance_start_at = None
                segment = _Segment(
                    event.pcm, captured_at, arrived_at, event.hangover_ms, event.tail_ms
                )
                self._spawn_transcription(segment)

    async def drain(self) -> None:
        """Wait for transcriptions spawned before this call. Tasks spawned
        by `on_utterance` (e.g. Organizer LLM work) are not tracked here --
        they're the Organizer's concern."""
        pending = list(self._tasks)
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)

    async def close(self) -> None:
        await self.drain()
        if self._sink is not None:
            self._sink.close()
        self._vad.reset()
        if self._wake is not None:
            self._closed = True
            self._wake_exec.submit(self._wake.detector.reset)
            self._wake_exec.shutdown(wait=False)

    async def _feed_wake(self, pcm: bytes, arrived_at: float) -> None:
        """Run the detector off the event loop, in frame order. `arrived_at`
        is taken BEFORE inference so executor lag cannot skew arbitration.
        A detector fault is reported once and ends detection on this
        connection -- the room then fails closed (DESIGN.md decision 11)."""
        self._track_energy(pcm)
        if self._wake_faulted or self._closed:
            return
        loop = asyncio.get_running_loop()
        try:
            scores = await loop.run_in_executor(self._wake_exec, self._wake.detector.feed, pcm)
        except Exception:
            log.exception("wake detector failed; this mic stops admitting speech")
            self._wake_faulted = True
            self._wake.on_fault()
            return
        if scores and arrived_at - self._last_hit_at >= self._wake.cooldown_s:
            self._last_hit_at = arrived_at
            self._wake.on_hit(max(scores), self._energy_dbfs(), arrived_at)

    def _track_energy(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
        self._energy.append((float(np.dot(samples, samples)), samples.size))
        self._energy_samples += samples.size
        limit = int(_SAMPLE_RATE * _ENERGY_WINDOW_S)
        while len(self._energy) > 1 and self._energy_samples - self._energy[0][1] >= limit:
            self._energy_samples -= self._energy.popleft()[1]

    def _energy_dbfs(self) -> float:
        total = sum(sq for sq, _n in self._energy)
        if self._energy_samples == 0 or total == 0.0:
            return _SILENCE_DBFS
        return 10.0 * math.log10(total / self._energy_samples / _FULL_SCALE_SQ)

    def _parse(self, framed: bytes) -> bytes:
        if len(framed) < AUDIO_HEADER_LEN:
            raise FrameTooShort(f"frame is {len(framed)} bytes, need >= {AUDIO_HEADER_LEN}")
        pcm = framed[AUDIO_HEADER_LEN:]
        if len(pcm) % 2 != 0:
            raise FrameTooShort(f"PCM payload length {len(pcm)} is not a whole number of int16 samples")
        return pcm

    def _spawn_transcription(self, segment: _Segment) -> None:
        task = asyncio.create_task(self._transcribe_and_dispatch(segment, self._last_task))
        self._last_task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _transcribe_and_dispatch(
        self, segment: _Segment, previous: asyncio.Task[None] | None
    ) -> None:
        admission = None
        if self._wake is not None:
            admission = await self._wake.decide(segment.captured_at, segment.ended_at)
        text, speaker, timing = None, NO_IDENTIFIER, None
        if self._wake is None or admission is not None:
            speech = self._strip_echo(segment)
            if speech is not None:
                (text, timing), speaker = await asyncio.gather(
                    self._timed_transcribe(speech), self._identify(speech.pcm)
                )
                segment = speech
        if previous is not None:
            await asyncio.wait([previous])
            _log_failure(previous)
        if text is not None and text.strip():
            await self._on_utterance(
                Utterance(text, segment.captured_at, speaker, admission, timing)
            )

    def _strip_echo(self, segment: _Segment) -> _Segment | None:
        """Cut the part of a segment the room's own TTS was still playing into,
        so a reply spoken straight after GLaDOS (no pause the VAD can split on)
        is kept instead of dropped with the echo. None = wholly echo, drop.

        Sample times come from frame ARRIVAL, so transport delay maps audio
        slightly late and the cut errs short (a sliver of echo kept), never
        into the reply; on the LAN the gate's drain margin covers it."""
        horizon = None if self._echo_gate is None else self._echo_gate.horizon(segment.captured_at)
        if horizon is None:
            return segment
        duration_s = len(segment.pcm) / 2 / _SAMPLE_RATE
        reply_s = segment.ended_at - segment.tail_ms / 1000 - horizon
        if reply_s < _MIN_REPLY_AFTER_ECHO_S:
            if duration_s < _ECHO_SKIP_MIN_S:
                return segment
            self._echo_gate.dropped(duration_s)
            return None
        cut = int((duration_s - reply_s) * _SAMPLE_RATE) * 2
        if cut <= 0:
            return segment
        log.info("tts gate: cut %.1fs of echo, kept %.1fs after it", cut / 2 / _SAMPLE_RATE, reply_s)
        return replace(segment, pcm=segment.pcm[cut:], captured_at=horizon)

    async def _timed_transcribe(self, segment: _Segment) -> tuple[str | None, SpeechTiming]:
        loop = asyncio.get_running_loop()
        text = await self._transcribe(segment.pcm)
        stt_ms = int((loop.time() - segment.ended_at) * 1000)
        return text, SpeechTiming(segment.ended_at, segment.hangover_ms, stt_ms)

    async def _transcribe(self, pcm: bytes) -> str | None:
        try:
            return await self._stt.transcribe(pcm)
        except Exception:
            # STT failures (model crash, OOM) must not silently disappear
            # into a never-retrieved Task exception; they're rare but
            # debugging-critical when faster-whisper lands.
            log.exception("STT.transcribe failed (utterance %d bytes)", len(pcm))
            return None

    async def _identify(self, pcm: bytes) -> SpeakerMatch:
        if self._identifier is None:
            return NO_IDENTIFIER
        try:
            return await self._identifier.identify(pcm)
        except Exception:
            log.exception("speaker identify failed; utterance goes unidentified")
            return SpeakerMatch.none("error")


@dataclass(frozen=True)
class _Segment:
    pcm: bytes
    captured_at: float
    ended_at: float
    hangover_ms: int
    tail_ms: int = 0


def _log_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("previous utterance dispatch failed", exc_info=task.exception())
