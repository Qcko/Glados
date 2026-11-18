"""WhisperSTT: faster-whisper behind the `core.adapters.STT` Protocol.

Model is loaded once on construct (defaults: English-only
`distil-small.en` CPU int8 -- ~250 MB download to HF_HOME). `language`
defaults to `None` (auto-detect); pin to a code (e.g. `"en"`) to skip
detection. For multilingual operation see the recipe in
configs/glados.toml.

`transcribe` runs the blocking inference call in a worker thread via
`asyncio.to_thread`, so the asyncio event loop stays responsive to the
WebSocket and the Organizer while a transcription is in flight.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

_CHECK_SAMPLES = 16_000


class WhisperSTT:
    def __init__(
        self,
        *,
        model: str = "distil-small.en",
        device: str = "cpu",
        device_index: int = 0,
        compute_type: str = "int8",
        language: str | None = None,
        hotwords: str | None = None,
        initial_prompt: str | None = None,
    ) -> None:
        from faster_whisper import WhisperModel

        if device == "cuda":
            _expose_nvidia_dlls()
        self._model = WhisperModel(
            model, device=device, device_index=device_index, compute_type=compute_type
        )
        self._language = language
        # Vocabulary biasing toward domain words/names (no retraining). Passed
        # straight to faster-whisper; None = no biasing.
        self._hotwords = hotwords or None
        self._initial_prompt = initial_prompt or None
        # ctranslate2 state inside WhisperModel is not safe for concurrent
        # `transcribe` calls -- serialise. The pipeline is willing to wait
        # because transcriptions run in background tasks anyway.
        self._lock = threading.Lock()

    def check(self) -> None:
        """Run one blocking transcription of silence. CUDA libraries load
        lazily, so a missing cuBLAS/cuDNN raises here rather than on the
        first real utterance."""
        self._run(bytes(2) * _CHECK_SAMPLES)

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
                hotwords=self._hotwords,
                initial_prompt=self._initial_prompt,
            )
            return "".join(seg.text for seg in segments).strip()


def _expose_nvidia_dlls() -> None:
    """The `cuda` extra installs cuBLAS/cuDNN as pip wheels whose DLLs sit in
    site-packages/nvidia/*/bin, which Windows does not search. No-op where
    the wheels are absent or on other platforms."""
    if os.name != "nt":
        return
    try:
        import nvidia
    except ImportError:
        return
    for root in nvidia.__path__:
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
