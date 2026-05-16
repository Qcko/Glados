"""Wiring smoke test for SileroVAD.

Skipped when silero-vad / torch aren't importable. Synthetic tones do
not trip silero by design (it's trained to ignore non-speech), so this
test only verifies the wiring contract: the model loads, `feed` accepts
chunked int16 bytes without crashing, `reset()` returns clean state,
and any emitted `VadEnd` carries a well-formed PCM payload. Accuracy
on real speech is validated manually with the demo client.
"""

from __future__ import annotations

import struct

import pytest

silero_vad = pytest.importorskip("silero_vad")
torch = pytest.importorskip("torch")  # noqa: F401

import numpy as np  # noqa: E402

from glados.audio.vad.silero import SileroVAD  # noqa: E402
from glados.core.adapters import VadEnd  # noqa: E402


_SR = 16_000


def _pcm_from_float(f: np.ndarray) -> bytes:
    clipped = np.clip(f, -1.0, 1.0)
    return b"".join(struct.pack("<h", int(x * 32767)) for x in clipped)


def test_silero_accepts_chunked_feed_without_crash() -> None:
    rng = np.random.default_rng(0)
    audio = rng.uniform(-0.05, 0.05, size=_SR * 2).astype(np.float32)
    pcm = _pcm_from_float(audio)

    vad = SileroVAD()
    step = int(0.05 * _SR) * 2  # 50 ms chunks like the worklet
    events = []
    for i in range(0, len(pcm), step):
        events.extend(vad.feed(pcm[i : i + step]))

    # Any VadEnd that does fire must carry a non-empty int16-aligned slice.
    for e in events:
        if isinstance(e, VadEnd):
            assert e.pcm
            assert len(e.pcm) % 2 == 0


def test_silero_reset_restores_clean_state() -> None:
    vad = SileroVAD()
    rng = np.random.default_rng(1)
    audio = _pcm_from_float(rng.uniform(-0.1, 0.1, size=_SR).astype(np.float32))
    vad.feed(audio)
    vad.reset()
    assert vad.feed(_pcm_from_float(np.zeros(_SR // 2, dtype=np.float32))) == []
