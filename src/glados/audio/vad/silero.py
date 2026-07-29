"""SileroVAD: silero-vad behind the `core.adapters.VAD` Protocol.

The model expects exactly 512-sample chunks of float32 mono at 16 kHz;
we accumulate inbound int16 bytes in `_pending`, feed VADIterator one
chunk at a time, and emit `VadStart` / `VadEnd` events as it reports
speech boundaries. `_history` retains the raw int16 PCM since the
current utterance began so we can carve the exact bytes for `VadEnd.pcm`
using the absolute sample indices VADIterator returns.

VADIterator's `speech_pad_ms` means the reported `start` index can sit
slightly in the past relative to the chunk currently being fed; that's
why we keep history rather than just the in-flight chunk.
"""

from __future__ import annotations

from ...core.adapters import VadEnd, VadEvent, VadStart

_SAMPLE_RATE = 16_000
_CHUNK_SAMPLES = 512
_CHUNK_BYTES = _CHUNK_SAMPLES * 2
# Safety cap to bound memory during long stretches of silence between
# utterances. 60 s at 16 kHz int16 = 1.9 MB.
_HISTORY_CAP_SAMPLES = 60 * _SAMPLE_RATE
# Hard upper bound on a single utterance. If silero never fires `end`
# (stuck speech run, runaway noise) we force-cut at this length so
# `_history` doesn't grow without bound and so Whisper isn't handed an
# unreasonably long slice.
_MAX_UTTERANCE_SAMPLES = 30 * _SAMPLE_RATE


_shared_model = None


def _load_model_once():
    """`load_silero_vad()` is fast but not free, and the model is
    immutable + thread-safe -- share it across VAD instances."""
    global _shared_model
    if _shared_model is None:
        from silero_vad import load_silero_vad

        _shared_model = load_silero_vad()
    return _shared_model


class SileroVAD:
    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_silence_ms: int = 200,
        speech_pad_ms: int = 30,
    ) -> None:
        from silero_vad import VADIterator

        self._model = _load_model_once()
        self._vad = VADIterator(
            self._model,
            threshold=threshold,
            sampling_rate=_SAMPLE_RATE,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self._pending = bytearray()
        self._history = bytearray()
        self._history_base_sample = 0
        self._in_speech = False
        self._speech_start_abs = 0

    def feed(self, pcm: bytes) -> list[VadEvent]:
        import numpy as np
        import torch

        events: list[VadEvent] = []
        if not pcm:
            return events
        self._pending.extend(pcm)
        while len(self._pending) >= _CHUNK_BYTES:
            chunk = bytes(self._pending[:_CHUNK_BYTES])
            del self._pending[:_CHUNK_BYTES]
            self._history.extend(chunk)
            samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
            speech = self._vad(torch.from_numpy(samples), return_seconds=False)
            if speech:
                events.extend(self._emit(speech))
            self._cap_history_if_idle()
            forced = self._force_cut_if_too_long()
            if forced is not None:
                events.append(forced)
        return events

    def reset(self) -> None:
        self._vad.reset_states()
        self._pending.clear()
        self._history.clear()
        self._history_base_sample = 0
        self._in_speech = False
        self._speech_start_abs = 0

    def _emit(self, speech: dict) -> list[VadEvent]:
        events: list[VadEvent] = []
        if "start" in speech:
            self._in_speech = True
            self._speech_start_abs = int(speech["start"])
            events.append(VadStart())
        if "end" in speech and self._in_speech:
            end_abs = int(speech["end"])
            events.append(VadEnd(pcm=self._slice_utterance(end_abs)))
            self._in_speech = False
            self._trim_history(end_abs)
        return events

    def _slice_utterance(self, end_abs: int) -> bytes:
        # speech_pad_ms can place start slightly before history_base only
        # in degenerate cases (e.g. reset() mid-utterance); treat it as
        # the oldest sample we still have rather than asserting.
        start_off = max(0, (self._speech_start_abs - self._history_base_sample) * 2)
        end_off = min(len(self._history), (end_abs - self._history_base_sample) * 2)
        return bytes(self._history[start_off:end_off])

    def _trim_history(self, end_abs: int) -> None:
        # Never let the base move backward -- that would corrupt all
        # subsequent absolute-index arithmetic.
        if end_abs <= self._history_base_sample:
            return
        drop = (end_abs - self._history_base_sample) * 2
        del self._history[:drop]
        self._history_base_sample = end_abs

    def _force_cut_if_too_long(self) -> VadEnd | None:
        if not self._in_speech:
            return None
        utterance_samples = (
            self._history_base_sample + len(self._history) // 2 - self._speech_start_abs
        )
        if utterance_samples < _MAX_UTTERANCE_SAMPLES:
            return None
        end_abs = self._speech_start_abs + _MAX_UTTERANCE_SAMPLES
        chunk = self._slice_utterance(end_abs)
        self._in_speech = False
        self._vad.reset_states()
        self._trim_history(end_abs)
        return VadEnd(pcm=chunk)

    def _cap_history_if_idle(self) -> None:
        if self._in_speech:
            return
        excess_samples = len(self._history) // 2 - _HISTORY_CAP_SAMPLES
        if excess_samples <= 0:
            return
        del self._history[: excess_samples * 2]
        self._history_base_sample += excess_samples
