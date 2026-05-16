"""AudioPipeline: per-connection orchestration of VAD → STT → Organizer.

The server hands every inbound binary audio frame (`<BE u32 seq><PCM16-LE>`)
to `feed_frame`. The pipeline parses it once, optionally tees it to a
WAV trace sink for offline replay, and feeds the PCM body through the
VAD. When the VAD emits an utterance end, transcription runs in a
background task so the WS receive loop is never blocked by a multi-
hundred-millisecond Whisper call — the user can start the next utterance
while the previous one is still being transcribed.

`close()` drains outstanding transcriptions before tearing down the sink
and resetting the VAD so a per-connection `<traces>/audio/<client>/...wav`
is well-formed even if the client disconnects mid-utterance.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

from ..core.adapters import STT, VAD, VadEnd
from ..core.audio_sink import AudioSink, FrameTooShort
from ..core.protocols import AUDIO_HEADER_LEN

OnUtterance = Callable[[str], Awaitable[None]]


class AudioPipeline:
    def __init__(
        self,
        *,
        sink: AudioSink | None,
        vad: VAD,
        stt: STT,
        on_utterance: OnUtterance,
    ) -> None:
        self._sink = sink
        self._vad = vad
        self._stt = stt
        self._on_utterance = on_utterance
        self._tasks: set[asyncio.Task[None]] = set()

    async def feed_frame(self, framed: bytes) -> None:
        pcm = self._parse(framed)
        if self._sink is not None:
            self._sink.write(framed)
        for event in self._vad.feed(pcm):
            if isinstance(event, VadEnd):
                self._spawn_transcription(event.pcm)

    async def drain(self) -> None:
        """Wait for transcriptions spawned before this call. Tasks spawned
        by `on_utterance` (e.g. Organizer LLM work) are not tracked here —
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

    def _spawn_transcription(self, pcm: bytes) -> None:
        task = asyncio.create_task(self._transcribe_and_dispatch(pcm))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _transcribe_and_dispatch(self, pcm: bytes) -> None:
        try:
            text = await self._stt.transcribe(pcm)
        except Exception:
            # STT failures (model crash, OOM) must not silently disappear
            # into a never-retrieved Task exception; they're rare but
            # debugging-critical when faster-whisper lands.
            log.exception("STT.transcribe failed (utterance %d bytes)", len(pcm))
            return
        if text.strip():
            await self._on_utterance(text)
