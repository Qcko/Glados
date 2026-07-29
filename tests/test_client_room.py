"""Tests for the lite room client (`client_room`): slice 3a (mic) + 3b (speaker)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import queue
import struct
import subprocess
import threading
import time

import numpy as np
import pytest

from client_room import wire
from client_room.audio import (
    BoundedAudioQueue,
    JitterBuffer,
    NullInputDevice,
    NullOutputDevice,
    Resampler,
    SubprocessInput,
    SubprocessOutput,
)
from client_room.mic import MicClient, load_token
from client_room.room import RoomSupervisor
from client_room.speaker import SpeakerClient
from client_room._client import _with_tls


# ---- TLS pinning (_with_tls) --------------------------------------------
# The successful wss:// pin path (cert verifies, token authenticates) is
# covered by manual integration against a live TLS server; these lock the
# guard branches that must never silently fall back to cleartext.


def test_with_tls_plain_ws_passes_connect_through() -> None:
    sentinel = object()
    assert _with_tls(sentinel, "ws://host:8765/ws/v1", None) is sentinel


def test_with_tls_ca_on_plain_ws_is_an_error() -> None:
    # tls_ca with a ws:// URL is a misconfig that would silently not encrypt.
    with pytest.raises(SystemExit, match="not wss"):
        _with_tls(object(), "ws://host:8765/ws/v1", "some-ca.pem")


def test_with_tls_missing_ca_file_is_an_error(tmp_path) -> None:
    missing = tmp_path / "nope.pem"
    with pytest.raises(SystemExit, match="not found"):
        _with_tls(object(), "wss://host:8765/ws/v1", str(missing))


def test_with_tls_blank_ca_on_wss_is_an_error() -> None:
    # Present-but-blank tls_ca must not silently unpin to public CAs.
    with pytest.raises(SystemExit, match="empty string"):
        _with_tls(object(), "wss://host:8765/ws/v1", "")


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
    # The built playback_done must validate against the real Pydantic model.
    done = p.PlaybackDone(**wire.playback_done("sess-1"))
    assert done.type == "playback_done" and done.session_id == "sess-1"


# ---- resampler ----------------------------------------------------------


def test_resampler_48k_to_16k_frame_count() -> None:
    # 1 s at 48 kHz -> 16000 samples -> exactly 20 frames of 800.
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
    sig[0::2] = 1.0   # even source indices -> +0x7fff
    sig[1::2] = -1.0  # odd source indices  -> -0x7fff
    out = b""
    for i in range(0, len(sig), 801):
        out += b"".join(rs.process(sig[i : i + 801]))
    samples = np.frombuffer(out, dtype="<i2")
    # Block 1 picks even global indices (0,2,...,800 -> +). Block 2 starts at
    # global offset 801, so the carried 1.0 cursor picks its LOCAL-odd =
    # GLOBAL-even indices (802,804,... -> +) too. A cursor-RESET bug would pick
    # block 2's local-even = global-odd indices (-). So correct carry => all +.
    assert len(samples) == wire.BATCH_SAMPLES
    assert (samples == 0x7FFF).all()


def test_resampler_clamps_and_scales() -> None:
    # +1.0 -> 0x7fff; out-of-range floats clamp rather than wrap.
    over = np.full(800, 2.0, dtype=np.float32)  # > 1.0, must clamp to +1.0
    samples = np.frombuffer(Resampler(16_000).process(over)[0], dtype="<i2")
    assert samples.min() == 0x7FFF and samples.max() == 0x7FFF


# ---- bounded queue ------------------------------------------------------


def test_bounded_queue_drops_oldest() -> None:
    q = BoundedAudioQueue(maxsize=2)
    q.put("a")
    q.put("b")
    q.put("c")  # full -> drop oldest "a"
    assert q.drops == 1
    assert q.get() == "b"
    assert q.get() == "c"


def test_bounded_queue_sentinel_survives_full_queue() -> None:
    """The shutdown sentinel must be delivered even when the queue is full --
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
    block = np.zeros(2_400, dtype=np.float32)  # 48k/3 -> 800 samples -> 1 frame
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
    conns[0]._closed.set()  # end the recv loop -> session ends -> run() exits
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
    send loop must unblock, the session end, and run() reconnect -- not hang
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
    """parec death (stdout EOF) must invoke on_close -- the session's only signal
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
    dev.stop()  # proc already cleared -> no error


# ---- SubprocessOutput (pacat) ------------------------------------------


class _FakeStdin:
    """pacat stdin stand-in: records writes. `die=True` makes `write` raise
    BrokenPipeError, as a real pipe would once pacat has exited."""

    def __init__(self, *, die: bool = False) -> None:
        self._die = die
        self.writes: queue.Queue = queue.Queue()

    def write(self, data: bytes) -> None:
        if self._die:
            raise BrokenPipeError("pacat gone")
        self.writes.put(bytes(data))

    def flush(self) -> None:
        pass


class _FakeOutProc:
    """subprocess.Popen stand-in for pacat. `hang=True` makes the first `wait`
    time out so stop() must escalate to kill()."""

    def __init__(self, stdin: _FakeStdin, *, hang: bool = False) -> None:
        self.stdin = stdin
        self._hang = hang
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None) -> int:
        if self._hang and not self.killed:
            raise subprocess.TimeoutExpired(cmd="pacat", timeout=timeout)
        return 0


def test_subprocess_output_argv_is_list_no_shell() -> None:
    dev = SubprocessOutput(
        22_050, sink="mysink", latency_msec=40, extra_args=["--client-name=glados"]
    )
    argv = dev._argv()
    assert argv[0] == "pacat"
    assert "--format=s16le" in argv and "--channels=1" in argv
    assert "--rate=22050" in argv and "--raw" in argv
    assert "--latency-msec=40" in argv
    assert argv[argv.index("-d") + 1] == "mysink"
    assert "--client-name=glados" in argv
    assert all(isinstance(a, str) for a in argv), "argv must be a list of strings"


def test_subprocess_output_writes_buffer_pcm_to_stdin() -> None:
    """The writer thread must drain the JitterBuffer to pacat's stdin; with no
    prebuffer the first read returns the real audio, so the first write is it."""
    stdin = _FakeStdin()
    buffer = JitterBuffer(prebuffer_bytes=0)
    pcm = struct.pack("<4h", 1000, -2000, 3000, -4000)  # 4 frames, distinctive
    buffer.write(pcm)
    dev = SubprocessOutput(
        2,  # 2 Hz -> chunk_ms=20 rounds to a 1-frame chunk; pcm spans 4 chunks
        chunk_ms=20.0,
        popen=lambda *a, **k: _FakeOutProc(stdin),
        sleep=lambda _d: None,
    )
    closed = threading.Event()
    dev.start(buffer, on_close=closed.set)
    first = stdin.writes.get(timeout=2.0)
    assert first == pcm[:2], "first write must be the buffer's real audio, not silence"
    dev.stop()
    assert closed.wait(2.0), "stop must surface on_close (gen-filtered upstream)"


def test_subprocess_output_self_paces_one_chunk_per_chunk_duration() -> None:
    """Locks the must-fix: the writer is self-clocked (one chunk per chunk_s of
    wall time), so it can't flood pacat's stdin pipe with silence ahead of real
    audio. A fake clock advanced only by sleeping proves each iteration waits
    exactly one chunk_s."""
    clock = [0.0]
    sleeps: queue.Queue = queue.Queue()

    def fake_monotonic() -> float:
        return clock[0]

    def fake_sleep(d: float) -> None:
        clock[0] += d  # sleeping is the only thing that advances the clock
        sleeps.put(d)

    stdin = _FakeStdin()
    dev = SubprocessOutput(
        100,
        chunk_ms=20.0,  # 2-frame chunk -> chunk_s = 0.02
        popen=lambda *a, **k: _FakeOutProc(stdin),
        sleep=fake_sleep,
        monotonic=fake_monotonic,
    )
    dev.start(JitterBuffer(prebuffer_bytes=0), on_close=None)
    observed = [sleeps.get(timeout=2.0) for _ in range(5)]
    dev.stop()
    assert all(abs(d - 0.02) < 1e-9 for d in observed), observed


def test_subprocess_output_resync_prevents_burst_when_write_blocks() -> None:
    """When pacat's stdin write blocks longer than chunk_s (the steady state once
    its sink is draining), `delay` goes negative and the loop resyncs `next_write`
    to now instead of accumulating a schedule it would later burst to catch up.
    A write that advances the clock by > chunk_s must therefore drive the loop
    (no sleeps), and never produce a negative/skipped-ahead sleep."""
    clock = [0.0]
    sleeps: queue.Queue = queue.Queue()
    writes: queue.Queue = queue.Queue()

    class _SlowStdin:
        def write(self, data: bytes) -> None:
            clock[0] += 0.05  # blocking write costs 0.05 s > chunk_s (0.02)
            writes.put(bytes(data))

        def flush(self) -> None:
            pass

    dev = SubprocessOutput(
        100,
        chunk_ms=20.0,  # chunk_s = 0.02
        popen=lambda *a, **k: _FakeOutProc(_SlowStdin()),
        sleep=lambda d: sleeps.put(d),
        monotonic=lambda: clock[0],
    )
    dev.start(JitterBuffer(prebuffer_bytes=0), on_close=None)
    for _ in range(5):  # let several iterations run
        writes.get(timeout=2.0)
    dev.stop()
    assert sleeps.empty(), "a write slower than chunk_s must resync, never sleep"


def test_subprocess_output_broken_pipe_surfaces_on_close() -> None:
    """pacat death (write raises BrokenPipeError) must invoke on_close -- the
    session's only signal that playback ended."""
    closed = threading.Event()
    stdin = _FakeStdin(die=True)
    dev = SubprocessOutput(
        22_050,
        popen=lambda *a, **k: _FakeOutProc(stdin),
        sleep=lambda _d: None,
    )
    dev.start(JitterBuffer(prebuffer_bytes=0), on_close=closed.set)
    assert closed.wait(2.0), "a broken pipe must invoke on_close"


def test_subprocess_output_stop_escalates_to_kill() -> None:
    stdin = _FakeStdin()
    proc = _FakeOutProc(stdin, hang=True)
    dev = SubprocessOutput(
        22_050,
        popen=lambda *a, **k: proc,
        stop_timeout_s=0.2,
        sleep=lambda _d: None,
    )
    dev.start(JitterBuffer(prebuffer_bytes=0))
    dev.stop()
    assert proc.terminated and proc.killed, "a hung terminate must escalate to kill"


def test_subprocess_output_double_stop_is_noop() -> None:
    stdin = _FakeStdin()
    proc = _FakeOutProc(stdin)
    dev = SubprocessOutput(
        22_050,
        popen=lambda *a, **k: proc,
        stop_timeout_s=0.5,
        sleep=lambda _d: None,
    )
    dev.start(JitterBuffer(prebuffer_bytes=0))
    dev.stop()
    dev.stop()  # proc already cleared -> no error


def test_subprocess_output_stop_halts_writer() -> None:
    """stop() must actually end the writer loop (via `_stop_evt`), not just tear
    down the process. With a stdin that never raises, the loop's stop-event guard
    is the only thing that ends it; a regression that dropped it would keep
    writing after stop() returns."""
    stdin = _FakeStdin()
    dev = SubprocessOutput(
        22_050,
        popen=lambda *a, **k: _FakeOutProc(stdin),
        sleep=lambda _d: None,
        stop_timeout_s=0.5,
    )
    dev.start(JitterBuffer(prebuffer_bytes=0))
    stdin.writes.get(timeout=2.0)  # ensure the loop is running
    dev.stop()  # joins the writer; once it returns the thread must be gone
    drained = stdin.writes.qsize()
    time.sleep(0.05)
    assert stdin.writes.qsize() == drained, "writer kept running after stop()"


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


# ---- shared client spine ------------------------------------------------


def test_terminal_error_codes_match_server() -> None:
    """Drift guard: the client's terminal-error set must be a subset of the
    error codes the server actually emits (so a server rename can't leave the
    client retrying a fatal handshake forever)."""
    import inspect
    import re

    from client_room._client import TERMINAL_ERROR_CODES
    import glados.core.server as srv

    emitted = set(re.findall(r'_send_error\(\s*ws,\s*"([^"]+)"', inspect.getsource(srv)))
    missing = TERMINAL_ERROR_CODES - emitted
    assert not missing, f"client terminal codes not emitted by server: {missing}"


def test_mic_reexports_load_token() -> None:
    from client_room._client import load_token as canonical
    from client_room.mic import load_token as reexported

    assert canonical is reexported


# ---- JitterBuffer -------------------------------------------------------


def test_jitter_buffer_prebuffer_gates_then_drains() -> None:
    jb = JitterBuffer(prebuffer_bytes=8)
    assert jb.read(4) == b"\x00\x00\x00\x00", "not armed -> silence"
    jb.write(b"abcd")  # 4 < 8, still not armed
    assert jb.read(2) == b"\x00\x00"
    jb.write(b"efgh")  # now 8 bytes buffered -> armed
    assert jb.read(4) == b"abcd"
    assert jb.read(6) == b"efgh\x00\x00", "underrun zero-pads to exactly n"
    assert jb.underrun_bytes > 0


def test_jitter_buffer_flush_rearms_prebuffer() -> None:
    jb = JitterBuffer(prebuffer_bytes=4)
    jb.write(b"wxyz")
    assert jb.read(2) == b"wx"
    jb.flush()
    assert jb.read(2) == b"\x00\x00", "flush re-arms: silence until prebuffer met again"
    jb.write(b"ABCD")
    assert jb.read(2) == b"AB"


def test_jitter_buffer_caps_memory_dropping_oldest() -> None:
    jb = JitterBuffer(prebuffer_bytes=0, max_bytes=4)
    jb.write(b"1234")
    jb.write(b"5678")  # over cap -> drop oldest 4 bytes
    assert jb.dropped_bytes == 4
    assert jb.read(4) == b"5678"


def test_jitter_buffer_head_cursor_preserves_stream_and_compacts() -> None:
    """The head-cursor read must hand back the exact byte stream in order across
    many interleaved writes/reads, and lazy compaction must keep the backing
    array far below total throughput (proving the consumed prefix is reclaimed,
    not retained -- the whole point of the refactor)."""
    jb = JitterBuffer(prebuffer_bytes=0, max_bytes=1 << 30)  # huge cap -> no drops
    src = bytes((i * 7) % 256 for i in range(20_000))  # >> the 4096 compact floor
    out = bytearray()
    pos = 0
    max_backing = 0
    while pos < len(src) or jb._available() > 0:
        if pos < len(src):
            jb.write(src[pos : pos + 100])
            pos += 100
        avail = jb._available()
        if avail:
            out += jb.read(min(73, avail))  # read size != write size -> crosses chunks
        max_backing = max(max_backing, len(jb._buf))
    assert bytes(out) == src, "head-cursor reads must preserve the byte stream in order"
    assert max_backing < 10_000, (
        f"compaction must reclaim the consumed prefix (backing peaked at "
        f"{max_backing}, would be ~{len(src)} if never compacted)"
    )


def test_jitter_buffer_drop_oldest_after_partial_read() -> None:
    """Overflow drop advances the head; it must measure the live backlog from the
    cursor, not the buffer start, so a partially-consumed prefix isn't counted as
    droppable audio."""
    jb = JitterBuffer(prebuffer_bytes=0, max_bytes=4)
    jb.write(b"AB")
    assert jb.read(1) == b"A"  # head now past 'A'; 'B' is the live backlog
    jb.write(b"CDEF")  # live = B,C,D,E,F = 5 > cap 4 -> drop oldest 1 ('B')
    assert jb.dropped_bytes == 1
    assert jb.read(4) == b"CDEF", "the consumed 'A' must not be re-read or mis-dropped"


def test_jitter_buffer_flush_resets_nonzero_head() -> None:
    """Barge-in path: flush must clear the buffer AND zero a non-trivial `_head`
    cursor, so the next reply refills from a clean slate (no stale consumed-prefix
    bytes leaking into the rearmed buffer)."""
    jb = JitterBuffer(prebuffer_bytes=0, max_bytes=1 << 30)
    jb.write(bytes(2000))
    jb.read(1500)  # head now well past zero, backing non-trivial
    assert jb._head > 0
    jb.flush()
    assert jb._head == 0 and jb._available() == 0
    jb.write(b"NEXT")
    assert jb.read(4) == b"NEXT", "post-flush reads start clean from the new audio"


def test_null_output_device_pulls_from_buffer() -> None:
    jb = JitterBuffer(prebuffer_bytes=0)
    jb.write(np.array([1, 2, 3], dtype="<i2").tobytes())
    dev = NullOutputDevice(22_050)
    dev.start(jb)
    samples = np.frombuffer(dev.pull(3), dtype="<i2")
    assert list(samples) == [1, 2, 3]


# ---- SpeakerClient state machine ---------------------------------------


def _tts_chunk(seq: int, rate: int, samples: list[int]) -> str:
    pcm = np.array(samples, dtype="<i2").tobytes()
    return json.dumps({
        "type": "tts_chunk", "session_id": "sess", "seq": seq,
        "sample_rate": rate, "pcm_b64": base64.b64encode(pcm).decode("ascii"),
    })


def _msg(kind: str) -> str:
    return json.dumps({"type": kind, "session_id": "sess"})


async def _run_speaker(inbound, device_factory):
    """Start a SpeakerClient against a SpyConn replaying `inbound` and return
    (client, conns, task) for mid-session inspection BEFORE teardown flushes the
    buffer. Deterministic settle: the recv loop only yields control once it has
    drained every inbound frame (SpyConn.__anext__ never awaits between frames),
    so when `_inbound` is empty the last frame is already dispatched -- no
    time-based sleep needed."""
    conns: list[_SpyConn] = []

    def connect(_url):
        c = _SpyConn(inbound=list(inbound))
        conns.append(c)
        return c

    client = SpeakerClient(
        server_url="ws://x", client_id="s", room_id="r", token="t",
        device_factory=device_factory, connect=connect,
        prebuffer_ms=0.0, backoff_min_s=0.0, backoff_max_s=0.0,
    )
    task = asyncio.create_task(client.run())
    for _ in range(1000):
        if conns and not conns[0]._inbound:
            break
        await asyncio.sleep(0)
    return client, conns, task


async def _stop_speaker(client, conns, task) -> None:
    client.stop()
    for c in conns:
        c._closed.set()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)


async def test_speaker_plays_tts_chunk() -> None:
    dev = NullOutputDevice(22_050)
    inbound = [_msg("welcome"), _tts_chunk(0, 22_050, [100, 200, 300])]
    client, conns, task = await _run_speaker(inbound, lambda _r: dev)
    assert dev.started
    samples = np.frombuffer(dev.pull(3), dtype="<i2")
    assert list(samples) == [100, 200, 300]
    await _stop_speaker(client, conns, task)


async def test_speaker_enqueues_without_welcome() -> None:
    """A speaker that connects mid-turn gets tts_chunk with no preceding welcome;
    it must still play (the browser never welcome-gates enqueue)."""
    dev = NullOutputDevice(22_050)
    client, conns, task = await _run_speaker([_tts_chunk(0, 22_050, [5, 5, 5])], lambda _r: dev)
    samples = np.frombuffer(dev.pull(3), dtype="<i2")
    assert list(samples) == [5, 5, 5]
    await _stop_speaker(client, conns, task)


async def test_speaker_cancelled_flushes_and_suppresses() -> None:
    dev = NullOutputDevice(22_050)
    inbound = [
        _msg("welcome"), _tts_chunk(0, 22_050, [1, 2, 3]),
        _msg("cancelled"), _tts_chunk(1, 22_050, [4, 5, 6]),
    ]
    client, conns, task = await _run_speaker(inbound, lambda _r: dev)
    # First turn flushed by cancelled; the post-cancel chunk dropped -> silence.
    assert np.frombuffer(dev.pull(3), dtype="<i2").tolist() == [0, 0, 0]
    await _stop_speaker(client, conns, task)


async def test_speaker_done_does_not_flush() -> None:
    dev = NullOutputDevice(22_050)
    inbound = [_msg("welcome"), _tts_chunk(0, 22_050, [7, 8, 9]), _msg("done")]
    client, conns, task = await _run_speaker(inbound, lambda _r: dev)
    assert np.frombuffer(dev.pull(3), dtype="<i2").tolist() == [7, 8, 9]
    await _stop_speaker(client, conns, task)


# ---- Slice B: playback_done drain signal --------------------------------


def _playback_dones(conn) -> list:
    """Session ids of every `playback_done` frame the speaker sent on `conn`."""
    out = []
    for data in conn.sent:
        try:
            m = json.loads(data)
        except (ValueError, TypeError):
            continue
        if isinstance(m, dict) and m.get("type") == "playback_done":
            out.append(m.get("session_id"))
    return out


async def _wait_until(predicate, *, timeout: float = 2.0) -> bool:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


async def test_speaker_signals_playback_done_after_drain() -> None:
    dev = NullOutputDevice(22_050)  # tail_s = 0.0
    inbound = [_msg("welcome"), _tts_chunk(0, 22_050, [7, 8, 9]), _msg("done")]
    client, conns, task = await _run_speaker(inbound, lambda _r: dev)
    # Reply still buffered -> no signal yet (an early one would reopen the mic
    # mid-TTS, the feedback bug Slice A fixed).
    assert _playback_dones(conns[0]) == []
    dev.pull(3)  # the device pulls the last bytes -> buffer drains
    assert await _wait_until(lambda: _playback_dones(conns[0]) == ["sess"])
    await _stop_speaker(client, conns, task)


async def test_speaker_signals_playback_done_for_silent_reply() -> None:
    """A turn that produced no audio (no device ever built) releases the gate at
    once -- nothing is playing locally to feed back."""
    dev = NullOutputDevice(22_050)
    inbound = [_msg("welcome"), _msg("done")]
    client, conns, task = await _run_speaker(inbound, lambda _r: dev)
    assert await _wait_until(lambda: _playback_dones(conns[0]) == ["sess"])
    await _stop_speaker(client, conns, task)


async def test_speaker_cancel_after_done_drops_playback_done() -> None:
    """`cancelled` flushes the buffer (-> instantly empty) AND cancels the drain
    watch: the flush-induced empty must NOT be mistaken for a clean drain and
    fire a stale signal for the cancelled turn."""
    dev = NullOutputDevice(22_050)
    inbound = [
        _msg("welcome"), _tts_chunk(0, 22_050, [1, 2, 3]),
        _msg("done"), _msg("cancelled"),
    ]
    client, conns, task = await _run_speaker(inbound, lambda _r: dev)
    dev.pull(3)  # buffer already flushed empty by cancelled
    assert not await _wait_until(
        lambda: _playback_dones(conns[0]) != [], timeout=0.2
    )
    await _stop_speaker(client, conns, task)


async def test_speaker_suppressed_during_tail_skips_signal() -> None:
    """White-box: a barge-in that lands while the watch is in its post-drain tail
    sleep must be caught by the pre-send recheck, not signalled. Drives the race
    deterministically -- tail (0.2s) >> the 0.05s we wait before flipping
    suppression -- so the flip always lands inside the tail sleep."""
    dev = NullOutputDevice(22_050, tail_s=0.2)
    inbound = [_msg("welcome"), _tts_chunk(0, 22_050, [1, 2, 3]), _msg("done")]
    client, conns, task = await _run_speaker(inbound, lambda _r: dev)
    dev.pull(3)  # drain -> watch passes _await_buffer_empty, enters the tail sleep
    await asyncio.sleep(0.05)
    client._suppressed = True  # a cancel's effect, landing mid-tail
    assert not await _wait_until(
        lambda: _playback_dones(conns[0]) != [], timeout=0.4
    )
    await _stop_speaker(client, conns, task)


async def test_speaker_welcome_reallows_after_cancel() -> None:
    dev = NullOutputDevice(22_050)
    inbound = [
        _msg("welcome"), _tts_chunk(0, 22_050, [1, 2, 3]),
        _msg("cancelled"), _msg("welcome"), _tts_chunk(1, 22_050, [4, 5, 6]),
    ]
    client, conns, task = await _run_speaker(inbound, lambda _r: dev)
    # First turn flushed; second welcome cleared suppression -> second turn plays.
    assert np.frombuffer(dev.pull(3), dtype="<i2").tolist() == [4, 5, 6]
    await _stop_speaker(client, conns, task)


async def test_speaker_rebuilds_on_sample_rate_change() -> None:
    devs: list[NullOutputDevice] = []

    def factory(rate):
        d = NullOutputDevice(rate)
        devs.append(d)
        return d

    inbound = [_msg("welcome"), _tts_chunk(0, 22_050, [1, 2, 3]), _tts_chunk(1, 16_000, [4, 5, 6])]
    client, conns, task = await _run_speaker(inbound, factory)
    assert len(devs) == 2, "a sample-rate change must rebuild the device"
    assert devs[0].stopped, "old device stopped before rebuild"
    assert devs[1].samplerate == 16_000
    assert np.frombuffer(devs[1].pull(3), dtype="<i2").tolist() == [4, 5, 6]
    await _stop_speaker(client, conns, task)


async def test_speaker_reconnects_on_device_death() -> None:
    dev = NullOutputDevice(22_050)
    conns: list[_SpyConn] = []

    def connect(_url):
        c = _SpyConn(inbound=[_tts_chunk(0, 22_050, [1, 2, 3])])
        conns.append(c)
        return c

    client = SpeakerClient(
        server_url="ws://x", client_id="s", room_id="r", token="t",
        device_factory=lambda _r: dev, connect=connect,
        prebuffer_ms=0.0, backoff_min_s=0.0, backoff_max_s=0.0,
    )
    task = asyncio.create_task(client.run())
    for _ in range(200):
        if dev.started:
            break
        await asyncio.sleep(0.01)
    dev.fail()  # simulate the output stream dying
    for _ in range(200):
        if len(conns) >= 2:
            break
        await asyncio.sleep(0.01)
    await _stop_speaker(client, conns, task)
    assert len(conns) >= 2, "output-device death must end the session and reconnect"


async def test_speaker_terminal_error_does_not_reconnect() -> None:
    conns: list[_SpyConn] = []

    def connect(_url):
        c = _SpyConn(inbound=[json.dumps({"type": "error", "code": "binding_mismatch", "message": "x"})])
        conns.append(c)
        return c

    client = SpeakerClient(
        server_url="ws://x", client_id="s", room_id="r", token="bad",
        device_factory=lambda _r: NullOutputDevice(), connect=connect,
    )
    await asyncio.wait_for(client.run(), timeout=2.0)
    assert len(conns) == 1


# ---- RoomSupervisor -----------------------------------------------------


class _FakeRoleClient:
    """Stand-in for a Mic/Speaker run(): hangs until cancelled, or exits
    immediately to simulate a graceful/terminal return."""

    def __init__(self, *, exit_immediately: bool = False, terminal: bool = False) -> None:
        self._exit = exit_immediately
        self._terminal = terminal
        self.run_count = 0
        self.cancelled = False

    @property
    def terminated(self) -> bool:
        return self._terminal

    async def run(self) -> None:
        self.run_count += 1
        if self._exit:
            return
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


async def _await_started(*clients) -> None:
    for _ in range(1000):
        if all(c.run_count for c in clients):
            return
        await asyncio.sleep(0)
    raise AssertionError("clients did not start within the poll budget")


async def test_room_supervisor_shutdown_cancels_both() -> None:
    mic, spk = _FakeRoleClient(), _FakeRoleClient()
    sup = RoomSupervisor(mic, spk)
    task = asyncio.create_task(sup.run())
    await _await_started(mic, spk)
    sup.request_shutdown()
    await asyncio.wait_for(task, timeout=2.0)
    assert mic.cancelled and spk.cancelled


async def test_room_supervisor_keeps_other_role_on_terminal_exit() -> None:
    mic = _FakeRoleClient(exit_immediately=True, terminal=True)
    spk = _FakeRoleClient()  # hangs (stays alive)
    sup = RoomSupervisor(mic, spk)
    task = asyncio.create_task(sup.run())
    await _await_started(spk)
    await asyncio.sleep(0.02)  # let the supervisor observe mic's terminal exit
    assert not task.done(), "a one-role terminal exit must NOT take the device down"
    assert not spk.cancelled
    sup.request_shutdown()
    await asyncio.wait_for(task, timeout=2.0)
    assert spk.cancelled


async def test_room_supervisor_exits_when_both_roles_exit() -> None:
    mic = _FakeRoleClient(exit_immediately=True, terminal=True)
    spk = _FakeRoleClient(exit_immediately=True)
    sup = RoomSupervisor(mic, spk)
    await asyncio.wait_for(sup.run(), timeout=2.0)  # returns on its own


async def test_room_supervisor_signal_wins_over_terminal_exit() -> None:
    mic = _FakeRoleClient(exit_immediately=True, terminal=True)
    spk = _FakeRoleClient()
    sup = RoomSupervisor(mic, spk)
    sup.request_shutdown()  # signal already pending before run
    # Signal wins: run() returns cleanly (no SystemExit) even though mic also
    # exited terminally in the same step. Reaching this line is the assertion;
    # wait_for would propagate a SystemExit if the terminal branch had fired.
    await asyncio.wait_for(sup.run(), timeout=2.0)
    assert sup._shutdown.is_set()


async def test_room_supervisor_cancel_tears_down_children() -> None:
    mic, spk = _FakeRoleClient(), _FakeRoleClient()
    sup = RoomSupervisor(mic, spk)
    task = asyncio.create_task(sup.run())
    await _await_started(mic, spk)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)
    assert mic.cancelled and spk.cancelled


def test_room_from_config_resolves_per_role_tokens(monkeypatch) -> None:
    tokens = {"bedroom-mic": "mtok", "bedroom-speaker": "stok"}
    import keyring

    monkeypatch.setattr(keyring, "get_password", lambda _service, cid: tokens.get(cid))
    cfg = {
        "server_url": "ws://x", "room_id": "bedroom",
        # parec capture builds without touching hardware; speaker factory is a
        # lambda not invoked at construction -- so from_config stays hardware-free.
        "mic": {"client_id": "bedroom-mic", "capture_backend": "parec"},
        "speaker": {"client_id": "bedroom-speaker"},
    }
    sup = RoomSupervisor.from_config(cfg)
    assert sup._mic._client_id == "bedroom-mic" and sup._mic._token == "mtok"
    assert sup._speaker._client_id == "bedroom-speaker" and sup._speaker._token == "stok"
    assert sup._mic._room_id == "bedroom" and sup._speaker._room_id == "bedroom"


def test_room_from_config_fails_before_start_on_missing_token(monkeypatch) -> None:
    import keyring

    # mic token present, speaker token absent -> must raise before constructing
    # either client (no half-started device).
    monkeypatch.setattr(
        keyring, "get_password",
        lambda _service, cid: "mtok" if cid == "bedroom-mic" else None,
    )
    cfg = {
        "server_url": "ws://x", "room_id": "bedroom",
        "mic": {"client_id": "bedroom-mic", "capture_backend": "parec"},
        "speaker": {"client_id": "bedroom-speaker"},
    }
    with pytest.raises(SystemExit):
        RoomSupervisor.from_config(cfg)


def test_room_from_config_missing_key_is_friendly_exit() -> None:
    # Missing [speaker] subtable -> SystemExit (not a bare KeyError traceback).
    with pytest.raises(SystemExit):
        RoomSupervisor.from_config({"server_url": "ws://x", "room_id": "r",
                                    "mic": {"client_id": "m"}})
