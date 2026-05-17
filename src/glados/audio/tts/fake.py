"""FakeTTS: yields one deterministic chunk of silence per non-empty text.

Used by tests so the Organizer + WS path can be exercised end-to-end
without loading a real voice. The `.calls` list captures every text
the backend was asked to speak so tests can assert what reached TTS.

Note: `synthesize` is an async generator; `calls` is appended on first
iteration, not at call time. Tests that want to assert without iterating
should iterate with at least one `async for` step (or use
`anext(tts.synthesize(text))`)."""

from __future__ import annotations

from typing import AsyncIterator

from ...core.adapters import TtsChunkOut


class FakeTTS:
    def __init__(
        self,
        *,
        sample_rate: int = 22_050,
        samples_per_chunk: int = 4_410,
    ) -> None:
        self._sample_rate = sample_rate
        self._samples_per_chunk = samples_per_chunk
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[TtsChunkOut]:
        self.calls.append(text)
        if not text:
            return
        pcm = b"\x00\x00" * self._samples_per_chunk
        yield TtsChunkOut(pcm=pcm, sample_rate=self._sample_rate)
