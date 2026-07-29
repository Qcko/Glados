"""Speaker room client: play GLaDOS TTS audio received over `/ws/v1`.

Lifecycle of one session:
  1. open the socket, send `hello` (role="speaker") as the first message; a good
     handshake is silent (no `welcome` ack). The speaker never uploads audio; it
     sends only one advisory control frame -- `playback_done` (see below).
  2. a recv loop dispatches server JSON: `tts_chunk` (base64 PCM16-LE + seq +
     sample_rate) is decoded and written into a `JitterBuffer`; the output
     device's real-time callback pulls from that buffer to feed the speaker.
  3. `cancelled` (barge-in) flushes the buffer and suppresses playback until the
     next `welcome`; `done` lets the buffer drain naturally (no flush). An
     `error` frame is terminal (bad token / binding) -- stop, don't reconnect.

Playback-done signal: on `done` the speaker starts a drain watch -- once the
`JitterBuffer` has emptied AND the device's `tail_s` (hardware/sink residue) has
elapsed, it sends `playback_done{session_id}` so the server can early-release its
mic feedback gate from the duration estimate to the short tail cooldown. It is
purely advisory: a dropped/late signal just falls back to the server's estimate
(see Organizer.handle_playback_done). The signal must never be EARLY -- that would
reopen the mic while TTS is still audible -- so the tail is deliberately generous
and the watch is cancelled the instant a `cancelled`/`welcome`/teardown
invalidates the turn it was started for.

Turn state machine (extends client_web/src/audio/tts.ts -- the browser does not
yet send `playback_done`; see the signal note above):
  welcome   -> clear suppression (allow playback for the new turn)
  tts_chunk -> if not suppressed: decode + enqueue (build/rebuild the device on
              the first chunk or a sample-rate change)
  cancelled -> flush + suppress + cancel the drain watch
  done      -> no flush (the tail drains); start the drain watch

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

from . import wire
from ._client import ReconnectingClient, load_config, load_token
from .audio import JitterBuffer, OutputDevice

log = logging.getLogger("client_room.speaker")

DeviceFactory = Callable[[int], OutputDevice]

_BYTES_PER_SAMPLE = 2  # PCM16-LE mono

# How often the drain watch re-checks the JitterBuffer for empty. Coarse on
# purpose: the signal only shortens the gate, so a few tens of ms of detection
# lag is invisible -- and a tight loop would burn a core polling a phone.
_DRAIN_POLL_S = 0.03


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
        tls_ca: str | None = None,
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
            tls_ca=tls_ca,
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
        # unless it still matches -- so an intentional stop (rate-change rebuild,
        # teardown) can never be mistaken for device death, with NO dependence on
        # host-specific finished_callback timing.
        self._device_gen = 0
        self._wake_closed: Callable[[], None] = lambda: None
        # Live socket for this session -- the drain watch reads it (as a spawn
        # argument) to send `playback_done`. Re-set per session in `_session`.
        self._ws = None
        # The in-flight drain watch (one per turn, spawned on `done`). Cancelling
        # it -- on cancelled/welcome/teardown -- is the ONLY thing that keeps it
        # from firing for a turn the client has moved past; `_device_gen` resets
        # to 0 each session, so the task identity, not the gen alone, is the
        # cross-session guard. Teardown cancels it before the next session's
        # `_reset_turn_state` runs, closing the gen-reuse hole.
        self._drain_task: asyncio.Task | None = None

    async def _session(self) -> None:
        # MUST run first: clears the device generation a prior teardown left
        # bumped, so the next session's first device starts on a clean gen.
        self._reset_turn_state()
        loop = asyncio.get_running_loop()
        closed = asyncio.Event()
        # Device death (callback thread) wakes the session (asyncio thread).
        self._wake_closed = lambda: loop.call_soon_threadsafe(closed.set)
        async with self._connect(self._url) as ws:
            self._ws = ws
            await self._send_hello(ws)
            recv_task = asyncio.create_task(self._recv_loop(ws), name="speaker-recv")
            closed_task = asyncio.create_task(closed.wait(), name="speaker-closed")
            try:
                await asyncio.wait(
                    {recv_task, closed_task}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                self._cancel_drain_watch()
                self._ws = None
                self._teardown_device()
                for t in (recv_task, closed_task):
                    t.cancel()
                await asyncio.gather(recv_task, closed_task, return_exceptions=True)

    def _teardown_device(self) -> None:
        # Invalidate the current device's on_close (our own stop is not a death),
        # then stop the stream first (joins the callback, so the buffer is
        # provably idle) before flushing -- never free a buffer a callback reads.
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
                continue  # garbage frame -- ignore, don't crash a dumb client
            if not isinstance(msg, dict):
                continue
            if self._check_terminal(msg):
                return
            self._dispatch(msg)

    def _dispatch(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "tts_chunk":
            self._on_tts_chunk(msg)
        elif kind == "done":
            self._on_done(msg)
        elif kind == "cancelled":
            self._on_cancelled()
        elif kind == "welcome":
            self._on_welcome()
        # every other type: ignore.

    def _on_welcome(self) -> None:
        # New turn: allow playback again (clears a prior barge-in suppress). A
        # drain watch from the previous turn is superseded -- drop it so it can't
        # fire against this turn's audio (the server serializes turns, so this is
        # belt-and-suspenders over the per-turn cancels).
        self._cancel_drain_watch()
        self._suppressed = False

    def _on_cancelled(self) -> None:
        if self._buffer is not None:
            self._buffer.flush()
        # Stays suppressed until the next `welcome`, so the cancelled turn's
        # in-flight chunks (still arriving on the socket) are dropped, not played.
        # Order matters: set suppression BEFORE cancelling the watch, so even if
        # the watch is mid-send it sees `_suppressed` on its pre-send recheck.
        self._suppressed = True
        self._cancel_drain_watch()

    def _on_done(self, msg: dict) -> None:
        # `done` deliberately does NOT flush -- the synthesized tail finishes
        # playing. It DOES start the drain watch: the reply is now complete
        # (ordered socket -> every tts_chunk already enqueued), so an empty buffer
        # from here on means drained, not network-starved.
        session_id = msg.get("session_id")
        if not isinstance(session_id, str) or self._suppressed:
            return
        self._cancel_drain_watch()  # at most one watch in flight
        tail_s = self._device.tail_s if self._device is not None else 0.0
        self._drain_task = asyncio.create_task(
            self._watch_drain(self._ws, session_id, self._buffer, tail_s, self._device_gen),
            name="speaker-drain",
        )

    def _cancel_drain_watch(self) -> None:
        task = self._drain_task
        self._drain_task = None
        if task is not None:
            task.cancel()

    async def _watch_drain(
        self, ws, session_id: str, buffer: JitterBuffer | None, tail_s: float, gen: int
    ) -> None:
        """Wait until this turn's audio has fully played out, then send
        `playback_done`. Captured `(buffer, tail_s, gen)` pin the turn: a
        rate-change rebuild or teardown bumps `_device_gen`, so a stale watch
        bails instead of signalling against the next turn's audio."""
        if buffer is None:
            # No device built (empty or fully-suppressed reply): nothing played
            # locally, so the gate can release now. Recheck first -- a `cancelled`
            # could have landed in the tick between spawn and this task running.
            if gen != self._device_gen or self._suppressed:
                return
            await self._send_playback_done(ws, session_id)
            return
        # A reply shorter than the prebuffer would never arm (so never drain);
        # `done` means no more audio is coming, so drain what we have.
        buffer.force_arm()
        if not await self._await_buffer_empty(buffer, gen):
            return  # invalidated or timed out -- the server estimate covers it.
        # The buffer is empty but the device/sink still holds `tail_s` of audio;
        # wait it out so the signal lands at silence, never before it.
        await asyncio.sleep(tail_s)
        if gen != self._device_gen or self._suppressed:
            return  # a cancelled / next turn landed during the tail wait.
        await self._send_playback_done(ws, session_id)

    async def _await_buffer_empty(self, buffer: JitterBuffer, gen: int) -> bool:
        """Poll until the buffer drains (True) or the watch is invalidated /
        times out (False). The cap is a safety net: at `done` the buffer holds at
        most `max_buffer_s` of audio (its own backlog cap) and drains at realtime,
        so a longer wait means something wedged -- fall back to the estimate."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._max_buffer_s + 2.0
        while True:
            if gen != self._device_gen or self._suppressed:
                return False
            if buffer.bytes_remaining() == 0:
                return True
            if loop.time() >= deadline:
                log.warning(
                    "drain watch exceeded %.1fs without emptying; "
                    "falling back to the server's gate estimate",
                    self._max_buffer_s + 2.0,
                )
                return False
            await asyncio.sleep(_DRAIN_POLL_S)

    async def _send_playback_done(self, ws, session_id: str) -> None:
        if ws is None:
            return
        try:
            await ws.send(json.dumps(wire.playback_done(session_id)))
        except Exception as e:  # noqa: BLE001 - transport/close; estimate covers it
            log.debug(
                "playback_done send failed (%s); server falls back to its estimate",
                type(e).__name__,
            )

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
                "tts seq gap: expected %d, got %d (lost frame -> likely audible glitch)",
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
            # carries a stale gen and is ignored -- not seen as a death.
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
    between PortAudio (`sounddevice`, the dev-box default) and external `pacat`
    (`pacat`, for Termux/Android phones with no PortAudio) -- the mirror of the
    mic's `capture_backend`."""
    backend = cfg.get("playback_backend", "sounddevice")
    if backend == "sounddevice":
        from .audio import SoundDeviceOutput

        device = cfg.get("output_device")
        return lambda rate: SoundDeviceOutput(rate, device=device)
    if backend == "pacat":
        from .audio import SubprocessOutput

        sink = cfg.get("pacat_sink")
        latency_msec = cfg.get("pacat_latency_msec", 80)
        extra_args = cfg.get("pacat_args")
        return lambda rate: SubprocessOutput(
            rate, sink=sink, latency_msec=latency_msec, extra_args=extra_args
        )
    raise SystemExit(
        f"unknown playback_backend {backend!r} (want 'sounddevice' or 'pacat')"
    )


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
        tls_ca=cfg.get("tls_ca"),
    )
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
