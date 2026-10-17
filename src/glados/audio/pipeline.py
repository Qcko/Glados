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
from typing import TYPE_CHECKING, Awaitable, Callable

log = logging.getLogger(__name__)

from ..core.adapters import STT, VAD, VadEnd, VadStart
from ..core.audio_sink import AudioSink, FrameTooShort
from ..core.protocols import AUDIO_HEADER_LEN
from ..core.speaker import NO_IDENTIFIER, SpeakerMatch, Utterance

if TYPE_CHECKING:
    from ..voiceid import SpeakerIdentifier

OnUtterance = Callable[[Utterance], Awaitable[None]]


class AudioPipeline:
    def __init__(
        self,
        *,
        sink: AudioSink | None,
        vad: VAD,
        stt: STT,
        on_utterance: OnUtterance,
        identifier: SpeakerIdentifier | None = None,
    ) -> None:
        self._sink = sink
        self._vad = vad
        self._stt = stt
        self._on_utterance = on_utterance
        self._identifier = identifier
        self._tasks: set[asyncio.Task[None]] = set()
        self._last_task: asyncio.Task[None] | None = None
        self._utterance_start_at: float | None = None

    async def feed_frame(self, framed: bytes) -> None:
        pcm = self._parse(framed)
        if self._sink is not None:
            self._sink.write(framed)
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
                self._spawn_transcription(event.pcm, captured_at)

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

    def _parse(self, framed: bytes) -> bytes:
        if len(framed) < AUDIO_HEADER_LEN:
            raise FrameTooShort(f"frame is {len(framed)} bytes, need >= {AUDIO_HEADER_LEN}")
        pcm = framed[AUDIO_HEADER_LEN:]
        if len(pcm) % 2 != 0:
            raise FrameTooShort(f"PCM payload length {len(pcm)} is not a whole number of int16 samples")
        return pcm

    def _spawn_transcription(self, pcm: bytes, captured_at: float) -> None:
        task = asyncio.create_task(
            self._transcribe_and_dispatch(pcm, captured_at, self._last_task)
        )
        self._last_task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _transcribe_and_dispatch(
        self, pcm: bytes, captured_at: float, previous: asyncio.Task[None] | None
    ) -> None:
        text, speaker = await asyncio.gather(
            self._transcribe(pcm), self._identify(pcm)
        )
        if previous is not None:
            await asyncio.wait([previous])
            _log_failure(previous)
        if text is not None and text.strip():
            await self._on_utterance(Utterance(text, captured_at, speaker))

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


def _log_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("previous utterance dispatch failed", exc_info=task.exception())
