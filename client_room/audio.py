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
  `sounddevice` implementation, a `SubprocessInput` (external `parec`/PulseAudio
  capture for boxes without PortAudio, e.g. Termux/Android), and a
  `NullInputDevice` for tests.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import subprocess
import threading
from typing import Callable, Optional, Protocol

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

    def put_sentinel(self) -> None:
        """Deliver the `None` shutdown sentinel, evicting a block if the queue
        is full so the signal can never be the item that gets dropped (unlike
        `put`, which drops-oldest for audio). At shutdown the sole consumer is
        blocked on `get`, so evict-then-retry always completes — this is the
        guaranteed unblock for the send loop."""
        while True:
            try:
                self._q.put_nowait(None)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass

    def get(self):
        """Blocking get (run in an executor by the async caller)."""
        return self._q.get()


CloseCallback = Callable[[], None]


class InputDevice(Protocol):
    samplerate: int

    def start(
        self,
        callback: Callable[[AudioBlock], None],
        on_close: Optional[CloseCallback] = None,
    ) -> None:
        """Begin capture, delivering native-rate mono float32 blocks to
        `callback` (on the device's own thread for real hardware).

        `on_close`, if given, is invoked once from the capture thread when the
        capture source ends unexpectedly (e.g. an external process dies). It is
        the device's only error channel: the caller uses it to end the session
        and reconnect rather than block forever on a source that will never
        deliver another block. Devices that can't end unexpectedly may ignore
        it."""
        ...

    def stop(self) -> None: ...


class NullInputDevice:
    """Test double: no hardware. `feed` synchronously delivers a canned block to
    the registered callback so tests drive capture deterministically."""

    def __init__(self, samplerate: int = 48_000) -> None:
        self.samplerate = samplerate
        self._callback: Callable[[AudioBlock], None] | None = None
        self._on_close: Optional[CloseCallback] = None
        self.started = False

    def start(
        self,
        callback: Callable[[AudioBlock], None],
        on_close: Optional[CloseCallback] = None,
    ) -> None:
        self._callback = callback
        self._on_close = on_close
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        """Test hook: simulate the capture source ending unexpectedly."""
        if self._on_close is not None:
            self._on_close()

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

    def start(
        self,
        callback: Callable[[AudioBlock], None],
        on_close: Optional[CloseCallback] = None,
    ) -> None:
        # PortAudio keeps the stream alive across transient glitches, so there
        # is no routine unexpected-EOF to surface; `on_close` is accepted for
        # Protocol conformance and left unwired. (Hardware-loss surfacing via
        # the stream's finished_callback is future work.)
        del on_close

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


# float32 little-endian: parec is told `--format=float32le`, so each sample is
# four bytes regardless of host endianness, matching the wire's LE assumption.
_FLOAT32_LE = np.dtype("<f4")
_BYTES_PER_SAMPLE = _FLOAT32_LE.itemsize


class SubprocessInput:
    """Capture via an external `parec` (PulseAudio) process, for boxes that have
    PulseAudio but no PortAudio (Termux/Android). `parec` writes raw native-rate
    mono float32 to stdout; a daemon reader thread frames whole samples and hands
    native-rate blocks to the callback — byte-for-byte the same downstream path
    as `SoundDeviceInput`, so the existing `Resampler` runs unchanged (the server
    VAD/STT see exactly what the browser produces). PulseAudio is NOT asked to
    resample to 16 kHz; that would feed the VAD a differently anti-aliased signal.

    Reliability: if `parec` dies its stdout EOFs, the reader thread invokes
    `on_close`, which lets the session end and reconnect instead of the send loop
    blocking forever on a queue that will never fill again."""

    def __init__(
        self,
        *,
        samplerate: int = 48_000,
        source: str | None = None,
        extra_args: list[str] | None = None,
        read_size: int = 4096,
        stop_timeout_s: float = 2.0,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        # `samplerate` should match the source's native rate so PulseAudio does
        # no resampling on its side; the Resampler handles native → 16 kHz.
        self.samplerate = samplerate
        self._source = source
        # An argv LIST passed to Popen with shell=False — no shell, no string
        # splitting, so a source name or extra arg can't inject a command.
        self._extra_args = list(extra_args or [])
        self._read_size = read_size
        self._stop_timeout_s = stop_timeout_s
        self._popen = popen
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None

    def _argv(self) -> list[str]:
        argv = [
            "parec",
            "--format=float32le",
            "--channels=1",
            f"--rate={self.samplerate}",
            "--raw",
        ]
        if self._source:
            argv += ["-d", self._source]
        argv += self._extra_args
        return argv

    def start(
        self,
        callback: Callable[[AudioBlock], None],
        on_close: Optional[CloseCallback] = None,
    ) -> None:
        # stderr → DEVNULL so a chatty parec can't fill an unread pipe and block
        # on write, silently stalling stdout (the audio path).
        self._proc = self._popen(
            self._argv(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._reader = threading.Thread(
            target=self._read_loop,
            args=(self._proc, callback, on_close),
            name="parec-reader",
            daemon=True,
        )
        self._reader.start()

    def _read_loop(
        self,
        proc: subprocess.Popen,
        callback: Callable[[AudioBlock], None],
        on_close: Optional[CloseCallback],
    ) -> None:
        stdout = proc.stdout
        assert stdout is not None  # opened with stdout=PIPE
        buf = bytearray()
        try:
            while True:
                chunk = stdout.read(self._read_size)
                if not chunk:  # EOF — parec exited (cleanly or died)
                    break
                buf += chunk
                # Emit only whole float32 samples; carry an odd-byte remainder
                # across read() boundaries so a sample is never split.
                whole = len(buf) - (len(buf) % _BYTES_PER_SAMPLE)
                if whole:
                    block = np.frombuffer(bytes(buf[:whole]), dtype=_FLOAT32_LE).copy()
                    del buf[:whole]
                    callback(block)
        finally:
            # Sole error channel: tell the caller capture ended so it can end
            # the session and reconnect rather than block forever.
            if on_close is not None:
                on_close()

    def stop(self) -> None:
        # Idempotent: a second stop (or stop after the proc already died) is a
        # no-op rather than an error.
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            # SIGTERM closes stdout, which unblocks the reader's blocking read.
            proc.wait(timeout=self._stop_timeout_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=self._stop_timeout_s)
        reader = self._reader
        self._reader = None
        if reader is not None:
            # Abandon (it's a daemon) rather than block teardown if a read is
            # still wedged after kill — never hang the caller.
            reader.join(timeout=self._stop_timeout_s)
