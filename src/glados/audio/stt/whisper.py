"""WhisperSTT: faster-whisper behind the `core.adapters.STT` Protocol.

Model is loaded once on construct (defaults: distil-small.en CPU int8 —
~150 MB download to HF_HOME, English-only, runs on consumer CPUs).
`transcribe` runs the blocking inference call in a worker thread via
`asyncio.to_thread`, so the asyncio event loop stays responsive to the
WebSocket and the Organizer while a transcription is in flight.
"""

from __future__ import annotations

import asyncio
import threading


class WhisperSTT:
    def __init__(
        self,
        *,
        model: str = "distil-small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._language = language
        # ctranslate2 state inside WhisperModel is not safe for concurrent
        # `transcribe` calls — serialise. The pipeline is willing to wait
        # because transcriptions run in background tasks anyway.
        self._lock = threading.Lock()

    async def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        return await asyncio.to_thread(self._run, pcm)

    def _run(self, pcm: bytes) -> str:
        import numpy as np

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        with self._lock:
            segments, _info = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=1,
                vad_filter=False,
            )
            return "".join(seg.text for seg in segments).strip()
