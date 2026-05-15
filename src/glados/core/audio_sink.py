"""Per-client WAV writer for inbound mic frames.

Slice goal: receive binary audio frames over WS and stash them on disk so
later slices (Whisper integration) have something to replay against. No
STT, no Organizer involvement — purely a debugging artifact.

Each AudioSink owns one WAV file under
`<traces_dir>/audio/<client_id>/<utc_iso>.wav`. The file is opened on the
first frame and closed when `close()` is called (typically on WS
disconnect). Out-of-order frames are tolerated by writing in arrival
order; the seq prefix is logged but not used to reorder, since dropped
audio is preferable to head-of-line blocking on a real-time stream.
"""

from __future__ import annotations

import struct
import wave
from datetime import datetime, timezone
from pathlib import Path

from .protocols import AUDIO_HEADER_LEN, AUDIO_SAMPLE_RATE


class FrameTooShort(ValueError):
    pass


def _safe_client_id(client_id: str) -> str:
    if not client_id or "/" in client_id or "\\" in client_id or client_id in {".", ".."}:
        raise ValueError(f"unsafe client_id for filesystem path: {client_id!r}")
    return client_id


class AudioSink:
    def __init__(self, root: Path, client_id: str) -> None:
        safe = _safe_client_id(client_id)
        self._root = root / "audio" / safe
        self._client_id = safe
        self._wav: wave.Wave_write | None = None
        self._path: Path | None = None
        self._frames_written = 0
        self._samples_written = 0
        self._last_seq: int | None = None
        self._dropped = 0

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def samples_written(self) -> int:
        return self._samples_written

    @property
    def dropped(self) -> int:
        return self._dropped

    def write(self, data: bytes) -> None:
        if len(data) < AUDIO_HEADER_LEN:
            raise FrameTooShort(f"frame is {len(data)} bytes, need >= {AUDIO_HEADER_LEN}")
        seq = struct.unpack(">I", data[:AUDIO_HEADER_LEN])[0]
        pcm = data[AUDIO_HEADER_LEN:]
        if len(pcm) % 2 != 0:
            raise FrameTooShort(f"PCM payload length {len(pcm)} is not a whole number of int16 samples")

        self._track_seq(seq)
        self._open_if_needed()
        assert self._wav is not None
        self._wav.writeframes(pcm)
        self._frames_written += 1
        self._samples_written += len(pcm) // 2

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None

    def _track_seq(self, seq: int) -> None:
        # A drop in seq below the last value means the client restarted
        # its counter (e.g. user toggled mic off then on within one WS
        # session). Treat as a fresh stream rather than a giant gap.
        if self._last_seq is not None and seq > self._last_seq:
            gap = seq - self._last_seq - 1
            if gap > 0:
                self._dropped += gap
        self._last_seq = seq

    def _open_if_needed(self) -> None:
        if self._wav is not None:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._path = self._root / f"{stamp}.wav"
        wav = wave.open(str(self._path), "wb")
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(AUDIO_SAMPLE_RATE)
        self._wav = wav
