"""PiperTTS: piper-tts behind the `core.adapters.TTS` Protocol.

Construction is cheap: `__init__` only records the voice name and
directory. The blocking work — voice file download (~110 MB on first
run for `en_GB-cori-high`) and onnx model load — is deferred to the
first `synthesize()` call and runs in worker threads so the asyncio
loop is never blocked. The server's lifespan already calls
`_warmup(stt, tts)` in a background task, which triggers that first
load asynchronously while `/healthz` answers immediately. A second
benefit: PiperTTS can be constructed in tests without touching the
disk as long as `synthesize()` is never called.

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
from typing import TYPE_CHECKING, AsyncIterator, Optional

from ...core.adapters import TtsChunkOut

if TYPE_CHECKING:
    from piper import PiperVoice as _PiperVoice  # for type-only reference

log = logging.getLogger(__name__)

_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


class PiperTTS:
    def __init__(
        self,
        *,
        voice: str = "en_GB-cori-high",
        voices_dir: Path = Path(r"E:\dev\piper\voices"),
    ) -> None:
        self._voice_name = voice
        self._voices_dir = voices_dir
        # `_voice` is the loaded PiperVoice (Any to avoid importing piper
        # at module load — see _ensure_loaded).
        self._voice: Optional["_PiperVoice"] = None
        # piper's onnxruntime session isn't documented as concurrent-safe;
        # serialise inference to be conservative. Synthesis runs in a worker
        # thread so the asyncio loop is free regardless.
        self._infer_lock = threading.Lock()
        # Async lock guards the lazy load — constructed lazily inside
        # `_ensure_loaded` so it binds to whichever event loop first
        # calls `synthesize()`. Constructing in `__init__` would bind to
        # whatever loop was running at construct time (or none); a
        # subsequent `synthesize` on a different loop would raise
        # "Lock is bound to a different event loop".
        self._load_lock: Optional[asyncio.Lock] = None

    async def synthesize(self, text: str) -> AsyncIterator[TtsChunkOut]:
        if not text.strip():
            return
        await self._ensure_loaded()
        # piper's synthesize() is a synchronous generator. Drain it in one
        # thread hop and queue chunks for async consumption. This keeps
        # the GIL pinned on the worker only while inference runs.
        chunks = await asyncio.to_thread(self._collect, text)
        for chunk in chunks:
            yield chunk

    async def _ensure_loaded(self) -> None:
        """Lazy-load the voice. First call may take ~30 s on a cold
        machine (download) + ~1–2 s (onnx load); both run in threads so
        the asyncio loop stays responsive. Subsequent calls are no-ops.
        Concurrent first calls serialise on `_load_lock` and the second
        caller short-circuits via the double-check.

        Retry policy: if download or load fails, the exception
        propagates and `_voice` stays None — the next `synthesize` will
        retry. v1 acceptable; consider an exponential-backoff guard if
        the network is flaky enough that retries pile up."""
        if self._voice is not None:
            return
        if self._load_lock is None:
            # First-arrival lazy lock construction. asyncio is
            # single-threaded within a loop, so two concurrent first-
            # synth calls can't both observe `is None` and both create
            # — the second sees the assignment from the first because
            # there's no `await` between the check and the create.
            self._load_lock = asyncio.Lock()
        async with self._load_lock:
            if self._voice is not None:
                return
            # Import deferred: `piper` pulls in onnxruntime which costs
            # ~200 ms on import. Keeping it out of `__init__` means a
            # PiperTTS instance that's never synthesised costs nothing.
            # The import itself runs on the event-loop thread (not in a
            # to_thread hop) — cheap and only happens once because of
            # the import-cache.
            from piper import PiperVoice

            onnx_path = await asyncio.to_thread(
                _ensure_voice, self._voice_name, self._voices_dir
            )
            self._voice = await asyncio.to_thread(
                PiperVoice.load, str(onnx_path)
            )

    def _collect(self, text: str) -> list[TtsChunkOut]:
        out: list[TtsChunkOut] = []
        with self._infer_lock:
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
