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
"""

from __future__ import annotations

import asyncio
import json
import logging
import random

from . import wire
from .audio import BoundedAudioQueue, InputDevice, Resampler

log = logging.getLogger("client_room.mic")

# Terminal handshake errors — the server closes the socket after sending one,
# and retrying with the same credentials/binding would just loop.
_TERMINAL_ERROR_CODES = frozenset(
    {"auth_failed", "unbound_client", "binding_mismatch", "expected_hello", "bad_message"}
)


class MicClient:
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
        self._url = server_url.rstrip("/") + wire.WS_PATH
        self._client_id = client_id
        self._room_id = room_id
        self._token = token
        self._device = device
        if connect is None:
            import websockets  # lazy: keeps the import cost off test paths

            connect = websockets.connect
        self._connect = connect
        self._queue_max = queue_max
        self._backoff_min = backoff_min_s
        self._backoff_max = backoff_max_s
        self._stop = False
        self._terminal = False
        # Reset per session in `_session`.
        self._audio_q = BoundedAudioQueue(queue_max)
        self._resampler = Resampler(device.samplerate)
        self._seq = 0

    async def run(self) -> None:
        """Connect, stream, and reconnect with capped+jittered backoff until
        stopped or a terminal handshake error. Cancellation-safe."""
        backoff = self._backoff_min
        while not self._stop:
            self._terminal = False
            try:
                await self._session()
                backoff = self._backoff_min  # clean drop — reconnect promptly
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - any transport error → retry
                log.warning("mic session ended (%s: %s); will reconnect", type(e).__name__, e)
            if self._stop or self._terminal:
                break
            await asyncio.sleep(backoff)
            backoff = min(self._backoff_max, backoff * 2) + random.uniform(0, 0.5)

    def stop(self) -> None:
        self._stop = True

    async def _session(self) -> None:
        # Fresh per-session state so a reconnect can't replay stale audio or a
        # stale sequence number.
        self._audio_q = BoundedAudioQueue(self._queue_max)
        self._resampler = Resampler(self._device.samplerate)
        self._seq = 0
        async with self._connect(self._url) as ws:
            await ws.send(json.dumps(wire.hello(
                self._client_id, self._room_id, "mic", self._token,
            )))
            log.info("mic %s connected to %s (room %s)", self._client_id, self._url, self._room_id)
            self._device.start(self._on_audio)
            send_task = asyncio.create_task(self._send_loop(ws), name="mic-send")
            recv_task = asyncio.create_task(self._recv_loop(ws), name="mic-recv")
            try:
                await asyncio.wait(
                    {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                self._device.stop()
                # Unblock the send loop's executor `get` so its thread can exit.
                self._audio_q.put(None)
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
            self._handle_text(raw)
            if self._terminal:
                return

    def _handle_text(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return  # garbage frame — ignore, don't crash a dumb client
        if not isinstance(msg, dict):
            return
        if msg.get("type") == "error":
            code = msg.get("code", "")
            log.error("server error: %s — %s", code, msg.get("message", ""))
            if code in _TERMINAL_ERROR_CODES:
                self._terminal = True
        # Every other message type is irrelevant to a mic client — ignore so an
        # unknown future `type` never crashes the client.


def _load_config(path: str) -> dict:
    import tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)


def main(config_path: str = "client_room/config.toml") -> None:
    """CLI entry: `python -m client_room.mic`. Reads config + keyring token,
    opens the real capture device, and runs until Ctrl-C."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = _load_config(config_path)
    client_id = cfg["client_id"]

    import keyring

    token = keyring.get_password("glados.client-tokens", client_id)
    if not token:
        raise SystemExit(
            f"no token for {client_id} in keyring; run: "
            f"python -m glados.secrets set client-tokens {client_id} <token>"
        )

    from .audio import SoundDeviceInput

    device = SoundDeviceInput(device=cfg.get("input_device"))
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
