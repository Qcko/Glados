"""Speaker-embedding backends behind one small Protocol.

`embed` is synchronous and CPU-bound; the identifier runs it on its own
worker thread. A returned vector is biometric data (DESIGN-speaker-id.md):
it is compared and dropped, never logged or traced.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000


class SpeakerEmbedder(Protocol):
    model_id: str

    def embed(self, pcm: bytes) -> np.ndarray: ...


def unit(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("cannot normalise a zero embedding")
    return v / norm


class FakeEmbedder:
    """Test backend. Known pcm maps to a preset voice; anything else gets a
    stable pseudo-random vector derived from its bytes, so two different
    clips are near-orthogonal strangers."""

    def __init__(
        self,
        voices: Mapping[bytes, Sequence[float]] | None = None,
        *,
        dim: int = 8,
        delay_s: float = 0.0,
        error: Exception | None = None,
        model_id: str = "fake-v1",
    ) -> None:
        self.model_id = model_id
        self._voices = dict(voices or {})
        self._dim = dim
        self._delay_s = delay_s
        self._error = error
        self.calls = 0

    def embed(self, pcm: bytes) -> np.ndarray:
        self.calls += 1
        if self._delay_s:
            time.sleep(self._delay_s)
        if self._error is not None:
            raise self._error
        if pcm in self._voices:
            return unit(self._voices[pcm])
        return unit(self._stranger(pcm))

    def _stranger(self, pcm: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(pcm).digest()[:8], "little")
        return np.random.default_rng(seed).standard_normal(self._dim)
