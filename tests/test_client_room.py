"""Tests for the lite room client (`client_room`), slice 3a (mic)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import struct
import subprocess
import threading

import numpy as np
import pytest

from client_room import wire
from client_room.audio import (
    BoundedAudioQueue,
    NullInputDevice,
    Resampler,
    SubprocessInput,
)
from client_room.mic import MicClient, load_token


# ---- wire ---------------------------------------------------------------


def test_frame_layout_and_seq_wrap() -> None:
    assert wire.frame(0, b"\x01\x02") == b"\x00\x00\x00\x00\x01\x02"
    assert wire.frame(1, b"") == b"\x00\x00\x00\x01"
    # uint32 wrap mirrors the browser's `(seq + 1) >>> 0`.
    assert wire.frame(2**32, b"") == b"\x00\x00\x00\x00"
    assert wire.frame(2**32 + 5, b"") == struct.pack(">I", 5)


def test_wire_matches_server_protocol() -> None:
    """Drift guard: the vendored constants + Hello shape must still match the
    server's `protocols.py` (the client can't import it, so this is the seam)."""
    from glados.core import protocols as p

    assert wire.AUDIO_SAMPLE_RATE == p.AUDIO_SAMPLE_RATE
    assert wire.AUDIO_HEADER_LEN == p.AUDIO_HEADER_LEN
    assert wire.WS_PATH == "/ws/v1"
    # The built hello must validate against the real Pydantic model.
    model = p.Hello(**wire.hello("c", "r", "mic", "t"))
    assert model.type == "hello" and model.role == "mic" and model.client_id == "c"


# ---- resampler ----------------------------------------------------------


def test_resampler_48k_to_16k_frame_count() -> None:
    # 1 s at 48 kHz → 16000 samples → exactly 20 frames of 800.
    sig = np.zeros(48_000, dtype=np.float32)
    frames = Resampler(48_000).process(sig)
    assert len(frames) == 20
    assert all(len(f) == wire.BATCH_SAMPLES * 2 for f in frames)


def test_resampler_carries_cursor_across_block_boundary() -> None:
    """The fractional read cursor must carry across `process` calls, not reset
    per block. At ratio 2.0 with an odd 801-sample block, the cursor lands on a
    half-sample offset at the boundary, flipping subsequent picks from even to
    odd source indices. A cursor-reset bug would keep picking even indices."""
    rs = Resampler(32_000)  # 32000/16000 = ratio 2.0
    sig = np.empty(2_000, dtype=np.float32)
    sig[0::2] = 1.0   # even source indices → +0x7fff
    sig[1::2] = -1.0  # odd source indices  → -0x7fff
    out = b""
    for i in range(0, len(sig), 801):
        out += b"".join(rs.process(sig[i : i + 801]))
    samples = np.frombuffer(out, dtype="<i2")
    # Block 1 picks even global indices (0,2,…,800 → +). Block 2 starts at
    # global offset 801, so the carried 1.0 cursor picks its LOCAL-odd =
    # GLOBAL-even indices (802,804,… → +) too. A cursor-RESET bug would pick
    # block 2's local-even = global-odd indices (−). So correct carry ⇒ all +.
    assert len(samples) == wire.BATCH_SAMPLES
    assert (samples == 0x7FFF).all()


def test_resampler_clamps_and_scales() -> None:
    # +1.0 → 0x7fff; out-of-range floats clamp rather than wrap.
    over = np.full(800, 2.0, dtype=np.float32)  # > 1.0, must clamp to +1.0
    samples = np.frombuffer(Resampler(16_000).process(over)[0], dtype="<i2")
    assert samples.min() == 0x7FFF and samples.max() == 0x7FFF


# ---- bounded queue ------------------------------------------------------


def test_bounded_queue_drops_oldest() -> None:
    q = BoundedAudioQueue(maxsize=2)
    q.put("a")
    q.put("b")
    q.put("c")  # full → drop oldest "a"
    assert q.drops == 1
    assert q.get() == "b"
    assert q.get() == "c"


def test_bounded_queue_sentinel_survives_full_queue() -> None:
    """The shutdown sentinel must be delivered even when the queue is full —
    it evicts a block rather than being dropped itself (the send-loop unblock)."""
    q = BoundedAudioQueue(maxsize=2)
    q.put("a")
    q.put("b")  # full
    q.put_sentinel()  # must evict, not drop the sentinel
    drained = [q.get(), q.get()]
    assert None in drained, "the shutdown sentinel must survive a full queue"


# ---- MicClient orchestration -------------------------------------------


class _SpyConn:
    """Stand-in for a `websockets` connection: records sends, blocks recv until
    closed, optionally replays canned inbound text frames first."""

    def __init__(self, inbound: list[str] | None = None) -> None:
        self.sent: list = []
        self._inbound = list(inbound or [])
        self._closed = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self._closed.set()
        return False

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._inbound:
            return self._inbound.pop(0)
        await self._closed.wait()
        raise StopAsyncIteration


class _OneShotDevice:
    """Feeds a single audio block the moment capture starts."""

    samplerate = 48_000

    def __init__(self, block) -> None:
        self._block = block

    def start(self, callback, on_close=None) -> None:
        callback(self._block)

    def stop(self) -> None:
        pass


async def test_mic_client_sends_hello_before_audio() -> None:
    block = np.zeros(2_400, dtype=np.float32)  # 48k/3 → 800 samples → 1 frame
    conns: list[_SpyConn] = []

    def connect(_url):
        c = _SpyConn()
        conns.append(c)
        return c

    client = MicClient(
        server_url="ws://x", client_id="m", room_id="r", token="t",
        device=_OneShotDevice(block), connect=connect,
    )
    task = asyncio.create_task(client.run())
    for _ in range(200):
        if conns and len(conns[0].sent) >= 2:
            break
        await asyncio.sleep(0.01)
    client.stop()
    conns[0]._closed.set()  # end the recv loop → session ends → run() exits
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)

    sent = conns[0].sent
    assert isinstance(sent[0], str), "first message must be the JSON hello"
    assert json.loads(sent[0]) == {
        "type": "hello", "client_id": "m", "room_id": "r", "role": "mic", "token": "t",
    }
    assert isinstance(sent[1], (bytes, bytearray)), "audio must be binary, after hello"
    assert len(sent[1]) == wire.AUDIO_HEADER_LEN + wire.BATCH_SAMPLES * 2
    assert sent[1][:4] == b"\x00\x00\x00\x00"  # first frame seq = 0


async def test_mic_client_terminal_error_does_not_reconnect() -> None:
    conns: list[_SpyConn] = []

    def connect(_url):
        c = _SpyConn(inbound=[json.dumps({"type": "error", "code": "auth_failed", "message": "bad"})])
        conns.append(c)
        return c

    client = MicClient(
        server_url="ws://x", client_id="m", room_id="r", token="bad",
        device=NullInputDevice(48_000), connect=connect,
    )
    await asyncio.wait_for(client.run(), timeout=2.0)
    assert len(conns) == 1, "a terminal auth error must not trigger a reconnect"


async def test_mic_client_reconnects_when_device_closes() -> None:
    """The deadlock fix: when the capture source dies (on_close fires), the
    send loop must unblock, the session end, and run() reconnect — not hang
    forever on a queue that will never fill again."""
    block = np.zeros(2_400, dtype=np.float32)
    conns: list[_SpyConn] = []

    def connect(_url):
        c = _SpyConn()
        conns.append(c)
        return c

    device = NullInputDevice(48_000)
    client = MicClient(
        server_url="ws://x", client_id="m", room_id="r", token="t",
        device=device, connect=connect, backoff_min_s=0.0, backoff_max_s=0.0,
    )
    task = asyncio.create_task(client.run())
    for _ in range(200):
        if device.started:
            break
        await asyncio.sleep(0.01)
    device.feed(block)
    device.close()  # simulate parec death

    for _ in range(200):
        if len(conns) >= 2:
            break
        await asyncio.sleep(0.01)
    client.stop()
    for c in conns:
        c._closed.set()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)

    assert len(conns) >= 2, "device close must end the session and reconnect"


# ---- SubprocessInput (parec) -------------------------------------------


class _FakeStdout:
    """Pipe stand-in: `read` blocks on a FIFO until a chunk is fed or EOF
    (`b''`) is queued, mirroring a real subprocess pipe so the reader thread
    parks rather than busy-spinning on EOF."""

    def __init__(self) -> None:
        self._chunks: queue.Queue = queue.Queue()

    def feed(self, data: bytes) -> None:
        self._chunks.put(data)

    def eof(self) -> None:
        self._chunks.put(b"")

    def read(self, _n: int) -> bytes:
        return self._chunks.get()


class _FakeProc:
    """subprocess.Popen stand-in. `hang=True` makes the first `wait` time out so
    stop() must escalate to kill(). terminate()/kill() close stdout so the reader
    thread's blocking read unblocks, as SIGTERM/SIGKILL would on a real parec."""

    def __init__(self, stdout: _FakeStdout, *, hang: bool = False) -> None:
        self.stdout = stdout
        self._hang = hang
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True
        if not self._hang:
            self.stdout.eof()

    def kill(self) -> None:
        self.killed = True
        self.stdout.eof()

    def wait(self, timeout=None) -> int:
        if self._hang and not self.killed:
            raise subprocess.TimeoutExpired(cmd="parec", timeout=timeout)
        return 0


def test_subprocess_input_argv_is_list_no_shell() -> None:
    dev = SubprocessInput(samplerate=44_100, source="mysrc", extra_args=["--latency-msec=20"])
    argv = dev._argv()
    assert argv[0] == "parec"
    assert "--format=float32le" in argv and "--channels=1" in argv
    assert "--rate=44100" in argv
    assert argv[argv.index("-d") + 1] == "mysrc"
    assert "--latency-msec=20" in argv
    assert all(isinstance(a, str) for a in argv), "argv must be a list of strings"


def test_subprocess_input_frames_whole_samples_across_reads() -> None:
    """Odd-length reads must frame on whole float32 boundaries, carrying the
    remainder so a sample is never split across blocks."""
    out: list[np.ndarray] = []
    closed = threading.Event()
    stdout = _FakeStdout()
    dev = SubprocessInput(popen=lambda *a, **k: _FakeProc(stdout))
    dev.start(out.append, on_close=closed.set)

    s = np.array([0.5, -0.5, 1.0], dtype="<f4").tobytes()  # 12 bytes, 3 samples
    stdout.feed(s[:5])   # 1 whole sample + 1 byte carried
    stdout.feed(s[5:])   # remainder completes samples 2 and 3
    stdout.eof()

    assert closed.wait(2.0)
    samples = np.concatenate(out)
    assert len(samples) == 3
    np.testing.assert_allclose(samples, [0.5, -0.5, 1.0])


def test_subprocess_input_eof_surfaces_on_close() -> None:
    """parec death (stdout EOF) must invoke on_close — the session's only signal
    that capture ended."""
    closed = threading.Event()
    stdout = _FakeStdout()
    dev = SubprocessInput(popen=lambda *a, **k: _FakeProc(stdout))
    dev.start(lambda _b: None, on_close=closed.set)
    stdout.eof()
    assert closed.wait(2.0), "stdout EOF must invoke on_close"


def test_subprocess_input_stop_escalates_to_kill() -> None:
    stdout = _FakeStdout()
    proc = _FakeProc(stdout, hang=True)
    dev = SubprocessInput(popen=lambda *a, **k: proc, stop_timeout_s=0.2)
    dev.start(lambda _b: None)
    dev.stop()
    assert proc.terminated and proc.killed, "a hung terminate must escalate to kill"


def test_subprocess_input_double_stop_is_noop() -> None:
    stdout = _FakeStdout()
    proc = _FakeProc(stdout)
    dev = SubprocessInput(popen=lambda *a, **k: proc, stop_timeout_s=0.5)
    dev.start(lambda _b: None)
    dev.stop()
    dev.stop()  # proc already cleared → no error


# ---- load_token precedence ---------------------------------------------


def _stub_keyring(monkeypatch, value):
    import keyring

    monkeypatch.setattr(keyring, "get_password", lambda *_a: value)


def test_load_token_keyring_wins(monkeypatch) -> None:
    _stub_keyring(monkeypatch, "keytok")
    monkeypatch.setenv("MIC_TOK", "envtok")
    assert load_token("c", env_var="MIC_TOK") == "keytok"


def test_load_token_falls_back_to_env(monkeypatch) -> None:
    _stub_keyring(monkeypatch, None)
    monkeypatch.setenv("MIC_TOK", "envtok")
    assert load_token("c", env_var="MIC_TOK") == "envtok"


def test_load_token_falls_back_to_file(monkeypatch, tmp_path) -> None:
    _stub_keyring(monkeypatch, None)
    monkeypatch.delenv("MIC_TOK", raising=False)
    f = tmp_path / "t.token"
    f.write_text("filetok")
    if os.name == "posix":
        os.chmod(f, 0o600)
    assert load_token("c", env_var="MIC_TOK", token_file=str(f)) == "filetok"


def test_load_token_none_raises_listing_sources(monkeypatch) -> None:
    _stub_keyring(monkeypatch, None)
    monkeypatch.delenv("MIC_TOK", raising=False)
    with pytest.raises(SystemExit):
        load_token("c", env_var="MIC_TOK")


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits only")
def test_load_token_rejects_group_readable_file(monkeypatch, tmp_path) -> None:
    _stub_keyring(monkeypatch, None)
    f = tmp_path / "t.token"
    f.write_text("filetok")
    os.chmod(f, 0o644)  # group/other readable
    with pytest.raises(SystemExit):
        load_token("c", token_file=str(f))
