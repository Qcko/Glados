"""Speaker room client: play GLaDOS TTS audio received over `/ws/v1`.

Lifecycle of one session:
  1. open the socket, send `hello` (role="speaker") as the first message; a good
     handshake is silent (no `welcome` ack). The speaker is recv-only after that
     — it never uploads audio.
  2. a recv loop dispatches server JSON: `tts_chunk` (base64 PCM16-LE + seq +
     sample_rate) is decoded and written into a `JitterBuffer`; the output
     device's real-time callback pulls from that buffer to feed the speaker.
  3. `cancelled` (barge-in) flushes the buffer and suppresses playback until the
     next `welcome`; `done` lets the buffer drain naturally (no flush). An
     `error` frame is terminal (bad token / binding) — stop, don't reconnect.

Turn state machine (mirrors client_web/src/audio/tts.ts):
  welcome   → clear suppression (allow playback for the new turn)
  tts_chunk → if not suppressed: decode + enqueue (build/rebuild the device on
              the first chunk or a sample-rate change)
  cancelled → flush + suppress (drop the cancelled turn's late chunks)
  done      → nothing; the tail drains

Why "suppress until welcome" is race-free without a turn id on the wire: the
server streams over a single ordered socket and processes a room's turns
sequentially, so every turn-N chunk precedes `Cancelled N`, which precedes
`Welcome N+1`. A stale post-cancel chunk therefore cannot arrive after the next
welcome, so a boolean flag cleared on `welcome` is sufficient.

Requires the client_id to be bound as role "speaker" in the server's
`rooms.toml`, exactly like a mic (the handshake is role-agnostic; a mismatch
yields a terminal `binding_mismatch`).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Callable

from ._client import ReconnectingClient, load_config, load_token
from .audio import JitterBuffer, OutputDevice

log = logging.getLogger("client_room.speaker")

DeviceFactory = Callable[[int], OutputDevice]

_BYTES_PER_SAMPLE = 2  # PCM16-LE mono


class SpeakerClient(ReconnectingClient):
    def __init__(
        self,
        *,
        server_url: str,
        client_id: str,
        room_id: str,
        token: str,
        device_factory: DeviceFactory,
        connect=None,
        prebuffer_ms: float = 120.0,
        max_buffer_s: float = 10.0,
        backoff_min_s: float = 0.5,
        backoff_max_s: float = 30.0,
    ) -> None:
        super().__init__(
            server_url=server_url,
            client_id=client_id,
            room_id=room_id,
            role="speaker",
            token=token,
            connect=connect,
            backoff_min_s=backoff_min_s,
            backoff_max_s=backoff_max_s,
        )
        # Built when the first chunk's sample_rate is known (PortAudio needs the
        # rate at construction), so the device is a factory, not an instance.
        self._device_factory = device_factory
        self._prebuffer_ms = prebuffer_ms
        self._max_buffer_s = max_buffer_s
        self._reset_turn_state()

    def _reset_turn_state(self) -> None:
        self._buffer: JitterBuffer | None = None
        self._device: OutputDevice | None = None
        self._sample_rate: int | None = None
        self._suppressed = False
        self._last_seq: int | None = None
        # Generation token: bumped on every (re)build and on teardown. Each
        # device's on_close captures the gen it was started with and is ignored
        # unless it still matches — so an intentional stop (rate-change rebuild,
        # teardown) can never be mistaken for device death, with NO dependence on
        # host-specific finished_callback timing.
        self._device_gen = 0
        self._wake_closed: Callable[[], None] = lambda: None

    async def _session(self) -> None:
        # MUST run first: clears the device generation a prior teardown left
        # bumped, so the next session's first device starts on a clean gen.
        self._reset_turn_state()
        loop = asyncio.get_running_loop()
        closed = asyncio.Event()
        # Device death (callback thread) wakes the session (asyncio thread).
        self._wake_closed = lambda: loop.call_soon_threadsafe(closed.set)
        async with self._connect(self._url) as ws:
            await self._send_hello(ws)
            recv_task = asyncio.create_task(self._recv_loop(ws), name="speaker-recv")
            closed_task = asyncio.create_task(closed.wait(), name="speaker-closed")
            try:
                await asyncio.wait(
                    {recv_task, closed_task}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                self._teardown_device()
                for t in (recv_task, closed_task):
                    t.cancel()
                await asyncio.gather(recv_task, closed_task, return_exceptions=True)

    def _teardown_device(self) -> None:
        # Invalidate the current device's on_close (our own stop is not a death),
        # then stop the stream first (joins the callback, so the buffer is
        # provably idle) before flushing — never free a buffer a callback reads.
        self._device_gen += 1
        if self._device is not None:
            self._device.stop()
        if self._buffer is not None:
            self._buffer.flush()
        self._device = None
        self._buffer = None

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            if isinstance(raw, (bytes, bytearray)):
                continue  # a speaker receives no binary; ignore
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue  # garbage frame — ignore, don't crash a dumb client
            if not isinstance(msg, dict):
                continue
            if self._check_terminal(msg):
                return
            self._dispatch(msg)

    def _dispatch(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "tts_chunk":
            self._on_tts_chunk(msg)
        elif kind == "cancelled":
            self._on_cancelled()
        elif kind == "welcome":
            # New turn: allow playback again (clears a prior barge-in suppress).
            self._suppressed = False
        # `done` and every other type: ignore. `done` deliberately does NOT
        # flush — the synthesized tail finishes playing.

    def _on_cancelled(self) -> None:
        if self._buffer is not None:
            self._buffer.flush()
        # Stays suppressed until the next `welcome`, so the cancelled turn's
        # in-flight chunks (still arriving on the socket) are dropped, not played.
        self._suppressed = True

    def _on_tts_chunk(self, msg: dict) -> None:
        if self._suppressed:
            return
        rate = msg.get("sample_rate")
        b64 = msg.get("pcm_b64", "")
        if not isinstance(rate, int) or rate <= 0 or not b64:
            return
        # Decode OFF the buffer lock (the real-time callback must never wait on a
        # base64 decode); only the resulting bytes go under the lock in `write`.
        try:
            pcm = base64.b64decode(b64)
        except (ValueError, TypeError):
            log.warning("undecodable tts_chunk pcm_b64; dropping")
            return
        if not pcm:
            return
        self._note_seq(msg.get("seq"))
        self._ensure_device(rate)
        assert self._buffer is not None  # set by _ensure_device
        self._buffer.write(pcm)

    def _note_seq(self, seq) -> None:
        """seq is not used for ordering (the socket is ordered); it's a free
        drop-detector for a headless client with no operator watching. seq
        resets to 0 each turn."""
        if not isinstance(seq, int):
            return
        if self._last_seq is not None and seq not in (0, self._last_seq + 1):
            log.info(
                "tts seq gap: expected %d, got %d (lost frame → likely audible glitch)",
                self._last_seq + 1, seq,
            )
        self._last_seq = seq

    def _ensure_device(self, rate: int) -> None:
        """Build the output device on the first chunk, or tear it down and
        rebuild on a sample-rate change. Runs on the recv thread; the rebuild
        stop()+rebuild is serialized here, never an in-place swap under the
        callback."""
        if self._device is not None and self._sample_rate == rate:
            return
        if self._device is not None:
            # Invalidate the old device's on_close BEFORE stopping it, so a
            # finished_callback fired during stop (whenever the host delivers it)
            # carries a stale gen and is ignored — not seen as a death.
            self._device_gen += 1
            self._device.stop()  # joins the old callback
            if self._buffer is not None:
                self._buffer.flush()
        self._sample_rate = rate
        prebuffer_bytes = int(self._prebuffer_ms / 1000.0 * rate) * _BYTES_PER_SAMPLE
        max_bytes = int(self._max_buffer_s * rate) * _BYTES_PER_SAMPLE
        self._buffer = JitterBuffer(prebuffer_bytes=prebuffer_bytes, max_bytes=max_bytes)
        self._device = self._device_factory(rate)
        self._device.start(self._buffer, self._make_on_close(self._device_gen))

    def _make_on_close(self, gen: int) -> Callable[[], None]:
        """Build the output-device error channel for a specific device
        generation. A real death of the *current* device ends the session so
        `run` reconnects; a callback from a superseded/torn-down device (stale
        gen) is ignored, so an intentional stop never kills a live session."""

        def on_close() -> None:
            if gen != self._device_gen:
                return
            self._wake_closed()

        return on_close


def _make_device_factory(cfg: dict) -> DeviceFactory:
    """Build the playback-device factory from config. `playback_backend` selects
    the backend; only `sounddevice` (PortAudio) ships today (a `pacat` analog is
    deferred to the on-device deploy slice)."""
    backend = cfg.get("playback_backend", "sounddevice")
    if backend == "sounddevice":
        from .audio import SoundDeviceOutput

        device = cfg.get("output_device")
        return lambda rate: SoundDeviceOutput(rate, device=device)
    raise SystemExit(f"unknown playback_backend {backend!r} (want 'sounddevice')")


def main(config_path: str = "client_room/config.toml") -> None:
    """CLI entry: `python -m client_room.speaker`. Reads config + token, opens
    the configured playback device, and runs until Ctrl-C."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    client_id = cfg["client_id"]
    token = load_token(
        client_id,
        env_var=cfg.get("token_env"),
        token_file=cfg.get("token_file"),
    )
    client = SpeakerClient(
        server_url=cfg["server_url"],
        client_id=client_id,
        room_id=cfg["room_id"],
        token=token,
        device_factory=_make_device_factory(cfg),
        prebuffer_ms=cfg.get("prebuffer_ms", 120.0),
    )
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
