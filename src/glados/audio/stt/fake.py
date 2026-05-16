"""FakeSTT: returns canned text, records inputs.

Two modes:
- A fixed string (default) — every utterance transcribes to the same text.
- A callable `pcm -> str` — for tests that want per-utterance behaviour.

The `.calls` list captures every PCM payload received so tests can
assert what reached the STT."""

from __future__ import annotations

from typing import Callable


class FakeSTT:
    def __init__(self, text: str | Callable[[bytes], str] = "hello world") -> None:
        self._impl = text
        self.calls: list[bytes] = []

    async def transcribe(self, pcm: bytes) -> str:
        self.calls.append(pcm)
        if callable(self._impl):
            return self._impl(pcm)
        return self._impl
