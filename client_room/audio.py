"""Audio capture seam for the room client.

Three pieces, each independently testable without hardware:

- `Resampler` — pure DSP: native-rate float32 mono → 16 kHz PCM16-LE in
  800-sample frames, mirroring `client_web/src/audio/processor.js` (fractional
  cursor decimation with fractional carry-over across blocks) so the server's
  VAD/STT see exactly what the browser produces.
- `BoundedAudioQueue` — a thread-safe, bounded hand-off from the PortAudio
  callback thread to the asyncio send loop. Drops the OLDEST block when full
  (the newest mic audio is what matters) and counts drops.
- `InputDevice` — a tiny Protocol over a capture device, with a real
  `sounddevice` implementation and a `NullInputDevice` for tests.
"""

from __future__ import annotations

import logging
import queue
from typing import Callable, Protocol

import numpy as np

from .wire import AUDIO_SAMPLE_RATE, BATCH_SAMPLES

log = logging.getLogger("client_room.audio")

# int16 little-endian regardless of host byte order — the wire is LE and the
# deploy target (phone/ARM) must agree with the dev box.
_PCM_DTYPE = np.dtype("<i2")
_INT16_MAX = 0x7FFF

AudioBlock = np.ndarray  # mono float32 samples at the device's native rate


class Resampler:
    """Decimate native-rate float32 mono to 16 kHz PCM16-LE, emitting whole
    800-sample frames. Stateful: the fractional read cursor carries across
    `process` calls so block boundaries don't drop or duplicate samples. This
    is a faithful port of the browser AudioWorklet, not a high-quality
    resampler — matching the browser (which feeds the same VAD) beats adding an
    anti-aliasing filter the server has never seen."""

    def __init__(self, native_rate: int) -> None:
        if native_rate <= 0:
            raise ValueError(f"native_rate must be positive, got {native_rate}")
        self._ratio = native_rate / AUDIO_SAMPLE_RATE
        self._cursor = 0.0
        self._batch = np.zeros(BATCH_SAMPLES, dtype=_PCM_DTYPE)
        self._fill = 0

    def process(self, channel: AudioBlock) -> list[bytes]:
        """Feed one native-rate block; return any completed 800-sample PCM16-LE
        frames. A block shorter than the remaining batch yields nothing yet."""
        out: list[bytes] = []
        n = len(channel)
        while self._cursor < n:
            sample = float(channel[int(self._cursor)])
            clamped = -1.0 if sample < -1.0 else 1.0 if sample > 1.0 else sample
            # int() truncates toward zero, matching JS `(x * 0x7fff) | 0`.
            self._batch[self._fill] = int(clamped * _INT16_MAX)
            self._fill += 1
            self._cursor += self._ratio
            if self._fill >= BATCH_SAMPLES:
                out.append(self._batch.tobytes())
                self._fill = 0
        # Carry the fractional remainder into the next block.
        self._cursor -= n
        return out


class BoundedAudioQueue:
    """Bounded SPSC-ish hand-off from the audio callback thread to asyncio.

    `put` is called from the PortAudio callback and must never block: when the
    queue is full it drops the oldest block (keeping the freshest audio) and
    increments `drops`. `get` blocks and is awaited from the send loop via a
    thread executor."""

    def __init__(self, maxsize: int = 64) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        # GIL-reliant counter: incremented on the callback thread, read once on
        # the async side after the streams have stopped. Fine as a log line, not
        # a live cross-thread readout.
        self.drops = 0

    def put(self, block: AudioBlock) -> None:
        try:
            self._q.put_nowait(block)
        except queue.Full:
            try:
                self._q.get_nowait()
                self.drops += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(block)
            except queue.Full:
                # A concurrent consumer can refill between our get and put;
                # dropping the newest block here is acceptable and still bounded.
                self.drops += 1

    def get(self):
        """Blocking get (run in an executor by the async caller)."""
        return self._q.get()


class InputDevice(Protocol):
    samplerate: int

    def start(self, callback: Callable[[AudioBlock], None]) -> None:
        """Begin capture, delivering native-rate mono float32 blocks to
        `callback` (on the device's own thread for real hardware)."""
        ...

    def stop(self) -> None: ...


class NullInputDevice:
    """Test double: no hardware. `feed` synchronously delivers a canned block to
    the registered callback so tests drive capture deterministically."""

    def __init__(self, samplerate: int = 48_000) -> None:
        self.samplerate = samplerate
        self._callback: Callable[[AudioBlock], None] | None = None
        self.started = False

    def start(self, callback: Callable[[AudioBlock], None]) -> None:
        self._callback = callback
        self.started = True

    def stop(self) -> None:
        self.started = False

    def feed(self, block: AudioBlock) -> None:
        if self._callback is None:
            raise RuntimeError("feed before start")
        self._callback(block)


class SoundDeviceInput:
    """Real capture via PortAudio (`sounddevice`). Imported lazily so the
    package installs and the unit tests run on a box without PortAudio — only
    live capture needs it. Captures at the device's native rate, mono float32;
    the `Resampler` downstream converts to the 16 kHz the wire wants."""

    def __init__(self, device: int | None = None, blocksize: int = 0) -> None:
        import sounddevice as sd  # lazy: optional dependency

        self._sd = sd
        self._device = device
        self._blocksize = blocksize
        info = sd.query_devices(device, "input")
        self.samplerate = int(info["default_samplerate"])
        self._stream: object | None = None

    def start(self, callback: Callable[[AudioBlock], None]) -> None:
        def _cb(indata, _frames, _time, status) -> None:
            if status:
                # Overflow/underflow flags — surface at debug to field-diagnose
                # capture glitches on the phone, but never raise out of the
                # audio callback.
                log.debug("input stream status: %s", status)
            # Copy: PortAudio reuses indata after the callback returns.
            callback(indata[:, 0].copy())

        self._stream = self._sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            blocksize=self._blocksize,
            device=self._device,
            callback=_cb,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
