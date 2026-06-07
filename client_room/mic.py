"""Mic room client: stream microphone audio up to GLaDOS over `/ws/v1`.

Lifecycle of one session:
  1. open the socket, send `hello` as the first (JSON-text) message, then begin
     streaming immediately — a good handshake is silent (no `welcome` ack).
  2. the capture device delivers native-rate blocks on its own thread; they
     land in a bounded queue (drop-oldest on overflow). An async send loop
     drains the queue, resamples to 16 kHz, frames, and sends binary.
  3. a recv loop reads server text leniently (dispatch on `type`, default
     ignore); an `error` frame is terminal (bad token / binding) — stop, don't
     reconnect.

Shutdown order on session end: stop the device → unblock the send loop →
cancel tasks → close the socket (the `async with` does the last step).

The connect/reconnect loop, hello, terminal-error set, and token/config loading
are shared with the speaker client in `_client.py`.
"""

from __future__ import annotations

import asyncio
import json
import logging

from . import wire
from ._client import ReconnectingClient, load_config, load_token
from .audio import BoundedAudioQueue, InputDevice, Resampler

# Re-exported for back-compat: `load_token` historically lived here.
__all__ = ["MicClient", "load_token", "main"]

log = logging.getLogger("client_room.mic")


class MicClient(ReconnectingClient):
    def __init__(
        self,
        *,
        server_url: str,
        client_id: str,
        room_id: str,
        token: str,
        device: InputDevice,
        connect=None,
        queue_max: int = 64,
        backoff_min_s: float = 0.5,
        backoff_max_s: float = 30.0,
    ) -> None:
        super().__init__(
            server_url=server_url,
            client_id=client_id,
            room_id=room_id,
            role="mic",
            token=token,
            connect=connect,
            backoff_min_s=backoff_min_s,
            backoff_max_s=backoff_max_s,
        )
        self._device = device
        self._queue_max = queue_max
        # Reset per session in `_session`.
        self._audio_q = BoundedAudioQueue(queue_max)
        self._resampler = Resampler(device.samplerate)
        self._seq = 0

    async def _session(self) -> None:
        # Fresh per-session state so a reconnect can't replay stale audio or a
        # stale sequence number.
        self._audio_q = BoundedAudioQueue(self._queue_max)
        self._resampler = Resampler(self._device.samplerate)
        self._seq = 0
        async with self._connect(self._url) as ws:
            await self._send_hello(ws)
            self._device.start(self._on_audio, self._on_device_closed)
            send_task = asyncio.create_task(self._send_loop(ws), name="mic-send")
            recv_task = asyncio.create_task(self._recv_loop(ws), name="mic-recv")
            try:
                await asyncio.wait(
                    {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                self._device.stop()
                # Unblock the send loop's executor `get` so its thread can exit.
                self._audio_q.put_sentinel()
                for t in (send_task, recv_task):
                    t.cancel()
                await asyncio.gather(send_task, recv_task, return_exceptions=True)
            if self._audio_q.drops:
                log.warning("mic %s dropped %d audio blocks (send fell behind)",
                            self._client_id, self._audio_q.drops)

    def _on_audio(self, block) -> None:
        """Capture-thread callback. Cheap and non-blocking: just hand the raw
        native-rate block to the bounded queue (resample/frame happen async).
        Only ever a real audio block — `None` is reserved as the send-loop
        shutdown sentinel that `_session` puts after the device is stopped."""
        self._audio_q.put(block)

    def _on_device_closed(self) -> None:
        """Capture-thread error channel. The device calls this when its source
        ends unexpectedly (e.g. `parec` dies). Push the shutdown sentinel so the
        blocked send loop returns, the `asyncio.wait` in `_session` completes,
        and `run` reconnects — instead of the send loop waiting forever on a
        queue that will never fill again. Thread-safe via the bounded queue;
        `put_sentinel` guarantees the signal is delivered, never drop-oldest'd."""
        self._audio_q.put_sentinel()

    async def _send_loop(self, ws) -> None:
        loop = asyncio.get_running_loop()
        while True:
            block = await loop.run_in_executor(None, self._audio_q.get)
            if block is None:  # shutdown sentinel
                return
            for pcm in self._resampler.process(block):
                await ws.send(wire.frame(self._seq, pcm))
                self._seq = (self._seq + 1) & 0xFFFFFFFF

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            if isinstance(raw, (bytes, bytearray)):
                continue  # a mic client receives no binary; ignore
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue  # garbage frame — ignore, don't crash a dumb client
            if not isinstance(msg, dict):
                continue
            # An `error` may be terminal; every other type is irrelevant to a
            # mic client, so ignore it (an unknown future type never crashes us).
            if self._check_terminal(msg):
                return


def _make_device(cfg: dict):
    """Build the capture device from config. `capture_backend` selects between
    PortAudio (`sounddevice`, the default) and external `parec` (`parec`)."""
    backend = cfg.get("capture_backend", "sounddevice")
    if backend == "parec":
        from .audio import SubprocessInput

        return SubprocessInput(
            samplerate=cfg.get("parec_rate", 48_000),
            source=cfg.get("parec_source"),
            extra_args=cfg.get("parec_args"),
        )
    if backend == "sounddevice":
        from .audio import SoundDeviceInput

        return SoundDeviceInput(device=cfg.get("input_device"))
    raise SystemExit(f"unknown capture_backend {backend!r} (want 'sounddevice' or 'parec')")


def main(config_path: str = "client_room/config.toml") -> None:
    """CLI entry: `python -m client_room.mic`. Reads config + token, opens the
    configured capture device, and runs until Ctrl-C."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    client_id = cfg["client_id"]
    token = load_token(
        client_id,
        env_var=cfg.get("token_env"),
        token_file=cfg.get("token_file"),
    )
    device = _make_device(cfg)
    client = MicClient(
        server_url=cfg["server_url"],
        client_id=client_id,
        room_id=cfg["room_id"],
        token=token,
        device=device,
    )
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
