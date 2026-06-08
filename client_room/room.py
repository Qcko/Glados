"""Room supervisor: run a mic client AND a speaker client in one process.

This is the set-and-forget room device (a phone on a shelf): one process, one
config, that captures audio up to GLaDOS and plays its replies back. A device
running both roles needs TWO server identities — a `client_id` binds to exactly
one (room, role), so the mic and speaker each have their own `client_id` (same
`room_id`) and their own token. Both tokens are resolved before either client
starts, so a missing secret fails before anything connects.

Resilience is layered:
  - Each client already self-reconnects with capped+jittered backoff inside
    `ReconnectingClient.run` — the supervisor adds NO second reconnect layer.
  - The supervisor owns the *process* lifecycle: it shuts both clients down
    cleanly on SIGINT/SIGTERM, and if one role hits a terminal handshake error
    (bad token / wrong binding) it logs loudly and keeps the *other* role
    running — a degraded device survives a one-role misconfig rather than going
    fully dark. It exits only on a signal or once both roles have exited.

For a device that must also survive reboots or a both-roles-dead state, run this
under an OS supervisor. On the Android/Termux target that's Termux:Boot +
termux-services (runit) — Android has no systemd/init — wired up in
`client_room/deploy/termux/` (the runit `Restart=on-failure` analog). That is a
phone-deploy concern, separate from this in-process supervision.

The single-role entry points (`python -m client_room.mic` / `.speaker`) remain
for devices that only capture or only play.
"""

from __future__ import annotations

import asyncio
import logging

from ._client import load_config, load_token
from .mic import MicClient, _make_device
from .speaker import SpeakerClient, _make_device_factory

log = logging.getLogger("client_room.room")


class RoomSupervisor:
    """Owns a mic + speaker client and runs both in one event loop."""

    def __init__(self, mic: MicClient, speaker: SpeakerClient) -> None:
        self._mic = mic
        self._speaker = speaker
        # Created here (not bound to a loop until awaited) so a test or a signal
        # handler can request shutdown without reaching into `run`'s locals.
        self._shutdown = asyncio.Event()

    @classmethod
    def from_config(cls, cfg: dict, *, connect=None) -> "RoomSupervisor":
        """Build both clients from a room config: top-level `server_url`/`room_id`
        plus `[mic]` and `[speaker]` subtables (each with `client_id`, its device
        keys, and optional per-role `token_env`/`token_file`). Per-role token
        keys are mandatory-by-subtable: a single shared env var would hand both
        roles the same secret and one would fail `binding_mismatch`."""
        try:
            server_url = cfg["server_url"]
            room_id = cfg["room_id"]
            mic_cfg = cfg["mic"]
            spk_cfg = cfg["speaker"]
            mic_client_id = mic_cfg["client_id"]
            spk_client_id = spk_cfg["client_id"]
        except (KeyError, TypeError) as e:
            # Friendly exit (matches load_token / _make_device), not a bare
            # traceback, for the operator configuring a set-and-forget device.
            raise SystemExit(f"room config missing required key: {e}") from e
        # Resolve BOTH tokens up front so a missing secret fails before either
        # client opens a socket (no half-started device).
        mic_token = load_token(
            mic_client_id,
            env_var=mic_cfg.get("token_env"),
            token_file=mic_cfg.get("token_file"),
        )
        spk_token = load_token(
            spk_client_id,
            env_var=spk_cfg.get("token_env"),
            token_file=spk_cfg.get("token_file"),
        )
        mic = MicClient(
            server_url=server_url,
            client_id=mic_client_id,
            room_id=room_id,
            token=mic_token,
            device=_make_device(mic_cfg),
            connect=connect,
        )
        speaker = SpeakerClient(
            server_url=server_url,
            client_id=spk_client_id,
            room_id=room_id,
            token=spk_token,
            device_factory=_make_device_factory(spk_cfg),
            prebuffer_ms=spk_cfg.get("prebuffer_ms", 120.0),
            connect=connect,
        )
        return cls(mic, speaker)

    def request_shutdown(self) -> None:
        """Ask `run` to tear both clients down. Idempotent; safe from a signal
        handler (it only sets an event, which `add_signal_handler` runs in-loop)."""
        self._shutdown.set()

    def _install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        import signal

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_shutdown)
            except (NotImplementedError, AttributeError, ValueError):
                # Windows (no add_signal_handler for SIGTERM) or a non-main
                # thread: fall back to KeyboardInterrupt cancelling `run`, whose
                # finally still tears both clients down.
                pass

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._install_signal_handlers(loop)
        mic_task = asyncio.create_task(self._mic.run(), name="room-mic")
        spk_task = asyncio.create_task(self._speaker.run(), name="room-speaker")
        sd_task = asyncio.create_task(self._shutdown.wait(), name="room-shutdown")
        meta = {mic_task: ("mic", self._mic), spk_task: ("speaker", self._speaker)}
        live = {mic_task, spk_task}
        try:
            # Keep going while at least one role is alive and no shutdown was
            # requested. A role exiting on its own (terminal misconfig) is logged
            # loudly but does NOT take the other role down.
            while live and not self._shutdown.is_set():
                done, _ = await asyncio.wait(
                    live | {sd_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if sd_task in done:
                    break
                for task in done & live:
                    name, client = meta[task]
                    if client.terminated:
                        log.error(
                            "room %s client exited on a terminal handshake error "
                            "(check its token + room/role binding in rooms.toml); "
                            "the other role keeps running", name,
                        )
                    else:
                        log.warning(
                            "room %s client exited unexpectedly; the other role "
                            "keeps running", name,
                        )
                    live.discard(task)
        finally:
            # Signal won, both roles died, or `run` was cancelled (Ctrl-C): bring
            # everything down. Cancellation reaches each client's `_session`
            # finally, which tears down its device + socket.
            for task in (mic_task, spk_task, sd_task):
                task.cancel()
            await asyncio.gather(mic_task, spk_task, sd_task, return_exceptions=True)
        if self._shutdown.is_set():
            log.info("room client shut down on signal")
        elif not live:
            log.warning("room client stopped: both sub-clients have exited")


def main(config_path: str = "client_room/config.room.toml") -> None:
    """CLI entry: `python -m client_room.room`. Reads the room config + both
    tokens, opens both devices, and runs until Ctrl-C / SIGTERM."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    supervisor = RoomSupervisor.from_config(cfg)
    try:
        asyncio.run(supervisor.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
