"""FakeVAD: deterministic utterance segmentation on a fixed sample budget.

Splits the inbound PCM stream into back-to-back utterances of
`utterance_samples` int16 samples each. Emits `VadStart` when an
utterance begins and `VadEnd(pcm=...)` when it ends. Used in tests and
when the server is configured with `vad.backend = "fake"`, so the
pipeline runs without torch."""

from __future__ import annotations

from ...core.adapters import VadEnd, VadEvent, VadStart


class FakeVAD:
    def __init__(self, utterance_samples: int) -> None:
        if utterance_samples <= 0:
            raise ValueError("utterance_samples must be positive")
        self._threshold_bytes = utterance_samples * 2
        self._buf = bytearray()
        self._in_speech = False

    def feed(self, pcm: bytes) -> list[VadEvent]:
        events: list[VadEvent] = []
        if not pcm:
            return events
        self._buf.extend(pcm)
        if not self._in_speech:
            events.append(VadStart())
            self._in_speech = True
        while len(self._buf) >= self._threshold_bytes:
            chunk = bytes(self._buf[: self._threshold_bytes])
            del self._buf[: self._threshold_bytes]
            events.append(VadEnd(pcm=chunk))
            self._in_speech = False
            if self._buf:
                events.append(VadStart())
                self._in_speech = True
        return events

    def reset(self) -> None:
        self._buf.clear()
        self._in_speech = False
