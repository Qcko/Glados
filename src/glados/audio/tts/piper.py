"""PiperTTS: piper-tts behind the `core.adapters.TTS` Protocol.

Voice (.onnx + .onnx.json) is loaded once on construct. `synthesize`
runs the blocking piper inference in a worker thread via
`asyncio.to_thread` and yields one chunk per piper-internal segment
(typically one chunk per sentence). The blocking work is collected in
a thread, then awaited; chunks are yielded back to the caller in order.

Voice files are downloaded from the rhasspy/piper-voices HuggingFace
repo on first use into `voices_dir` (default `E:\\dev\\piper\\voices`)
if not already present. Single-speaker voices only for v1 step 3 —
multi-speaker (`libritts_r` etc.) would need a `speaker_id` arg.

License heads-up: piper-tts 1.4.x is GPL-3 (OHF-Voice/piper1-gpl).
Fine for local-only personal use; would infect GLaDOS if ever
redistributed."""

from __future__ import annotations

import asyncio
import logging
import threading
import urllib.request
from pathlib import Path
from typing import AsyncIterator

from ...core.adapters import TtsChunkOut

log = logging.getLogger(__name__)

_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


class PiperTTS:
    def __init__(
        self,
        *,
        voice: str = "en_GB-cori-high",
        voices_dir: Path = Path(r"E:\dev\piper\voices"),
    ) -> None:
        from piper import PiperVoice

        onnx_path = _ensure_voice(voice, voices_dir)
        self._voice = PiperVoice.load(str(onnx_path))
        # piper's onnxruntime session isn't documented as concurrent-safe;
        # serialise to be conservative. Synthesis runs in a worker thread
        # so the asyncio loop is free regardless.
        self._lock = threading.Lock()

    async def synthesize(self, text: str) -> AsyncIterator[TtsChunkOut]:
        if not text.strip():
            return
        # piper's synthesize() is a synchronous generator. Drain it in one
        # thread hop and queue chunks for async consumption. This keeps
        # the GIL pinned on the worker only while inference runs.
        chunks = await asyncio.to_thread(self._collect, text)
        for chunk in chunks:
            yield chunk

    def _collect(self, text: str) -> list[TtsChunkOut]:
        out: list[TtsChunkOut] = []
        with self._lock:
            for chunk in self._voice.synthesize(text):
                out.append(
                    TtsChunkOut(
                        pcm=chunk.audio_int16_bytes,
                        sample_rate=chunk.sample_rate,
                    )
                )
        return out


def _ensure_voice(voice: str, voices_dir: Path) -> Path:
    voices_dir.mkdir(parents=True, exist_ok=True)
    onnx = voices_dir / f"{voice}.onnx"
    config = voices_dir / f"{voice}.onnx.json"
    if not onnx.exists() or not config.exists():
        url_base = f"{_HF_BASE}/{_voice_url_path(voice)}"
        if not onnx.exists():
            _download(f"{url_base}/{voice}.onnx", onnx)
        if not config.exists():
            _download(f"{url_base}/{voice}.onnx.json", config)
    return onnx


def _voice_url_path(voice: str) -> str:
    # rhasspy/piper-voices layout: en/en_GB/cori/high/en_GB-cori-high.onnx.
    # Voice name encodes lang_region, speaker, quality. Speaker slugs may
    # contain `-` themselves (rare in today's catalog but allowed), so
    # split off the first and last segments and rejoin the middle.
    parts = voice.split("-")
    if len(parts) < 3:
        raise ValueError(
            f"voice {voice!r} not in 'lang_region-speaker-quality' form"
        )
    lang_region, quality = parts[0], parts[-1]
    speaker = "-".join(parts[1:-1])
    lang = lang_region.split("_")[0]
    return f"{lang}/{lang_region}/{speaker}/{quality}"


def _download(url: str, dest: Path) -> None:
    log.info("piper: downloading %s -> %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()
