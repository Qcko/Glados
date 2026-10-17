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
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable, Protocol

import numpy as np

log = logging.getLogger(__name__)

from ..core.adapters import STT, VAD, VadEnd, VadStart, WakeWord
from ..core.audio_sink import AudioSink, FrameTooShort
from ..core.protocols import AUDIO_HEADER_LEN
from ..core.wake import Admission

OnUtterance = Callable[..., Awaitable[None]]

_SAMPLE_RATE = 16000
_ENERGY_WINDOW_S = 1.0
_FULL_SCALE_SQ = 32768.0 ** 2
_SILENCE_DBFS = -120.0


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
        wake: WakeHook | None = None,
    ) -> None:
        self._sink = sink
        self._vad = vad
        self._stt = stt
        self._on_utterance = on_utterance
        self._tasks: set[asyncio.Task[None]] = set()
        self._utterance_start_at: float | None = None
        self._wake = wake
        self._wake_exec = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="wake")
            if wake is not None
            else None
        )
        self._wake_faulted = False
        self._last_hit_at = float("-inf")
        self._energy: deque[tuple[float, int]] = deque()
        self._energy_samples = 0

    async def feed_frame(self, framed: bytes) -> None:
        pcm = self._parse(framed)
        if self._sink is not None:
            self._sink.write(framed)
        if self._wake is not None:
            await self._feed_wake(pcm, asyncio.get_running_loop().time())
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
                self._utterance_start_at = asyncio.get_running_loop().time()
            elif isinstance(event, VadEnd):
                # Pair with the VadStart stamp (fall back to now for a VadEnd
                # with no recorded start -- a force-cut or a reset mid-utterance).
                captured_at = (
                    self._utterance_start_at
                    if self._utterance_start_at is not None
                    else asyncio.get_running_loop().time()
                )
                self._utterance_start_at = None
                ended_at = asyncio.get_running_loop().time()
                self._spawn_transcription(event.pcm, captured_at, ended_at)

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
            self._wake.detector.reset()
            self._wake_exec.shutdown(wait=False, cancel_futures=True)

    async def _feed_wake(self, pcm: bytes, arrived_at: float) -> None:
        """Run the detector off the event loop, in frame order. `arrived_at`
        is taken BEFORE inference so executor lag cannot skew arbitration.
        A detector fault is reported once and ends detection on this
        connection -- the room then fails closed (DESIGN.md decision 11)."""
        self._track_energy(pcm)
        if self._wake_faulted:
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

    def _spawn_transcription(self, pcm: bytes, captured_at: float, ended_at: float) -> None:
        task = asyncio.create_task(self._transcribe_and_dispatch(pcm, captured_at, ended_at))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _transcribe_and_dispatch(
        self, pcm: bytes, captured_at: float, ended_at: float
    ) -> None:
        admission = None
        if self._wake is not None:
            admission = await self._wake.decide(captured_at, ended_at)
            if admission is None:
                return
        try:
            text = await self._stt.transcribe(pcm)
        except Exception:
            # STT failures (model crash, OOM) must not silently disappear
            # into a never-retrieved Task exception; they're rare but
            # debugging-critical when faster-whisper lands.
            log.exception("STT.transcribe failed (utterance %d bytes)", len(pcm))
            return
        if not text.strip():
            return
        if admission is None:
            await self._on_utterance(text, captured_at)
        else:
            await self._on_utterance(text, captured_at, admission)
