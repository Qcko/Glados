"""Tests for the lite room client (`client_room`), slice 3a (mic)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import struct

import numpy as np

from client_room import wire
from client_room.audio import BoundedAudioQueue, NullInputDevice, Resampler
from client_room.mic import MicClient


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

    def start(self, callback) -> None:
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
