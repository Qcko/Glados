"""FakeWakeWord: scripted detections by feed-call index (tests only)."""

from __future__ import annotations


class FakeWakeWord:
    def __init__(self, hits: dict[int, float] | None = None, fail_at: int | None = None) -> None:
        self._hits = dict(hits or {})
        self._fail_at = fail_at
        self._calls = 0
        self.resets = 0

    def feed(self, pcm: bytes) -> list[float]:
        index = self._calls
        self._calls += 1
        if index == self._fail_at:
            raise RuntimeError("scripted wake detector fault")
        score = self._hits.get(index)
        return [] if score is None else [score]

    def reset(self) -> None:
        self.resets += 1
