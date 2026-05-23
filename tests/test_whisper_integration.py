"""Wiring smoke test for WhisperSTT.

Skipped when faster-whisper isn't importable. The first run downloads
`distil-small.en` (~250 MB) to HF_HOME; subsequent runs are fast. We
only assert the contract: `transcribe` returns a `str` and runs the
heavy work off the asyncio loop (via `asyncio.to_thread`). Accuracy
is validated with the demo client end-to-end.
"""

from __future__ import annotations

import os
import struct
import time

import pytest

faster_whisper = pytest.importorskip("faster_whisper")

import asyncio  # noqa: E402

import numpy as np  # noqa: E402


# Don't pay the ~150 MB download unless the user opts in. Local dev with
# the demo will warm the cache; once warmed, this env knob can be flipped.
RUN = os.environ.get("GLADOS_WHISPER_SMOKE") == "1"


@pytest.mark.skipif(not RUN, reason="set GLADOS_WHISPER_SMOKE=1 to enable")
def test_whisper_transcribes_silence_to_string() -> None:
    from glados.audio.stt.whisper import WhisperSTT

    stt = WhisperSTT()
    pcm = b"".join(struct.pack("<h", 0) for _ in range(16_000))  # 1 s silence
    result = asyncio.run(stt.transcribe(pcm))
    assert isinstance(result, str)


@pytest.mark.skipif(not RUN, reason="set GLADOS_WHISPER_SMOKE=1 to enable")
def test_whisper_runs_off_event_loop() -> None:
    from glados.audio.stt.whisper import WhisperSTT

    stt = WhisperSTT()
    pcm = b"".join(struct.pack("<h", 0) for _ in range(16_000))

    async def main() -> tuple[str, float]:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        t0 = time.perf_counter()
        text = await stt.transcribe(pcm)
        elapsed = time.perf_counter() - t0
        t.cancel()
        return text, elapsed if ticks > 0 else -1.0

    text, elapsed = asyncio.run(main())
    assert isinstance(text, str)
    # Ticker ran at least once during transcription -> blocking call
    # didn't starve the loop.
    assert elapsed > 0
