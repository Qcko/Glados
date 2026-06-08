# GLaDOS room client — Termux/Android deploy

Run the room client (mic + speaker, `python -m client_room.room`) as a
set-and-forget appliance on an Android phone: started on boot, restarted on
crash, surviving Android's doze. This is the Android replacement for a systemd
`Restart=on-failure` unit — Android has no systemd/init, so supervision is done
with **termux-services** (runit) plus **Termux:Boot**.

The Python `RoomSupervisor` already handles everything *inside* a live process
(per-client reconnect with backoff, clean `SIGTERM` shutdown, keeping one role
alive if the other hits a terminal handshake error). This wrapper only adds what
a process can't do for itself: start on boot, restart after the process exits,
and keep the CPU awake.

## What's here

```
deploy/termux/
  boot/start-glados-room.sh        Termux:Boot entry — wake-lock + start runit
  service/
    glados-pulse/run               supervise PulseAudio (foreground, stock config)
    glados-pulse/log/run           svlogd logger for pulse
    glados-room/run                supervise the room client; gate on pulse, load mic source
    glados-room/log/run            svlogd logger for the client
  glados-room.env.example          operator paths template → ~/.config/glados-room/env
  install.sh                       idempotent dependency installer (Termux)
  requirements-phone.txt           Python deps (numpy, websockets)
  make-phone-bundle.sh             build the one-file .tar.gz (run on the dev box)
  selftest.sh                      on-hardware diagnostic → report.md (see below)
```

PulseAudio is its **own** runit service (not started from inside the client's
`run`), which decouples its lifecycle from client restarts. It runs pulse with
its **stock shipped config** (no `-n`/`--file`) — the same config the confirmed
`pulseaudio --start` mic-capture setup used — to avoid depending on a
`default.pa` existing at a fixed path (some Termux pulse builds compile their
config in, so an `.include` of a missing file would abort startup). The Android
mic source `module-sles-source` is loaded by the **room** service's `run`, once
the daemon is up, with a `grep` guard so a client restart can never stack
duplicate source modules and exhaust the mic.

## Prerequisites (on the phone)

1. Install **Termux** and the **Termux:Boot** and **Termux:API** addon apps.
   Install all three from the same source (F-Droid *or* Play, not mixed) or
   signature checks reject the addons.
2. **Open Termux:Boot once** after installing — Termux:Boot runs nothing until
   its app has been launched at least once.
3. Exempt Termux from battery optimization: Android Settings → Apps → Termux →
   Battery → **Unrestricted**. Otherwise Android kills the runit tree under doze.
4. Packages:
   ```sh
   pkg install python pulseaudio termux-services termux-api
   ```

## Install the client

Two ways to get `client_room/` onto the phone — pick one.

**A. One-file bundle (no git on the phone).** On the dev box, build a tarball of
the tracked client files (this never includes your gitignored config/tokens):

```sh
sh client_room/deploy/termux/make-phone-bundle.sh    # → dist/glados-phone-bundle.tar.gz
```

Copy that one file to the phone, then in Termux:

```sh
tar xzf glados-phone-bundle.tar.gz -C $HOME          # → ~/glados/
sh ~/glados/client_room/deploy/termux/install.sh     # installs all deps (idempotent)
```

**B. Clone the repo** (if the phone has git/network to your remote):

```sh
git clone <your-glados-remote> ~/glados
sh ~/glados/client_room/deploy/termux/install.sh
```

`install.sh` probes each dependency and installs only what's missing
(`python pulseaudio termux-services termux-api`, plus `numpy` from the prebuilt
**`python-numpy`** package — never pip-compiled — and `websockets` via pip). It's
idempotent, so re-running it is safe. The client is pure portable Python and
never imports the server package; `keyring` is optional (token loading falls back
to an env var / mode-600 file). If you'd rather install by hand, the deps are in
`requirements-phone.txt`.

Provide the room config and tokens:

```sh
cp client_room/config.room.example.toml client_room/config.room.toml
$EDITOR client_room/config.room.toml          # set server_url, room_id, both client_ids
# Set capture_backend = "parec" and playback_backend = "pacat" (no PortAudio on
# the phone). Point pacat_sink at your output sink (e.g. a bluez_output.* sink).

# Token files must be mode 600 (the client rejects group/other-readable tokens).
install -m 600 /dev/stdin ~/.secrets/bedroom-mic.token     <<<'…mic token…'
install -m 600 /dev/stdin ~/.secrets/bedroom-speaker.token <<<'…speaker token…'
# …and set token_file for each role in config.room.toml accordingly.
```

`config.room.toml` and the token files are gitignored / never committed.

## Operator env

```sh
mkdir -p ~/.config/glados-room
cp client_room/deploy/termux/glados-room.env.example ~/.config/glados-room/env
$EDITOR ~/.config/glados-room/env     # set GLADOS_ROOM_DIR / GLADOS_ROOM_PYTHON if not default
```

Defaults assume `GLADOS_ROOM_DIR=$HOME/glados` and the system `python`; an empty
file is fine if that matches your layout.

## Provision the services + boot script

The runit service directory is `$PREFIX/var/service`. Symlink the two services
in (symlink, so a `git pull` updates them), make every script executable, and
provision the Termux:Boot script:

```sh
PREFIX=/data/data/com.termux/files/usr
D=~/glados/client_room/deploy/termux

chmod +x "$D"/service/*/run "$D"/service/*/log/run "$D"/boot/start-glados-room.sh

ln -sf "$D/service/glados-pulse" "$PREFIX/var/service/glados-pulse"
ln -sf "$D/service/glados-room"  "$PREFIX/var/service/glados-room"

mkdir -p ~/.termux/boot
ln -sf "$D/boot/start-glados-room.sh" ~/.termux/boot/start-glados-room.sh
```

> Termux:Boot executes `~/.termux/boot/*` directly, so that file must be
> executable and carry the `#!/data/data/com.termux/files/usr/bin/sh` shebang
> (both already true in the repo copy).

## Enable and start

```sh
sv-enable glados-pulse     # auto-start on boot + bring up now
sv-enable glados-room
```

`sv-enable` both marks the service to start at boot and starts it immediately, so
you don't have to reboot to test.

## Verify

```sh
sv status glados-pulse glados-room          # both should read "run"
pactl info                                  # pulse reachable
pactl list short sources | grep sles        # the mic source loaded
tail -f ~/.local/var/log/glados-room/current # client logs (svlogd)
```

A clean room-client handshake is silent; you should see it connect and, when you
speak, audio flow up and TTS play back.

## Self-test — collect a diagnostic report

`selftest.sh` runs every on-hardware check in one pass and writes a single
`report.md` you can hand to whoever is helping with the bring-up. It probes the
risky assumptions directly: PulseAudio starting on its stock config, the OpenSL
ES mic source loading, real mic capture levels, an output sink + a `pacat` test
tone, the Python client importing, and the runit service status. It is
read-mostly — it starts pulse / loads the mic source only if they aren't already
up (and reports what it changed), and never touches your config or services.

```sh
sh client_room/deploy/termux/selftest.sh            # interactive (mic + tone checks)
sh client_room/deploy/termux/selftest.sh --client   # also run the client ~12s
sh client_room/deploy/termux/selftest.sh --non-interactive   # no prompts (headless)
```

It prints a `N PASS · N FAIL · N WARN · N SKIP` tally and the path to
`~/glados-selftest-<timestamp>/report.md` (plus a `.tar.gz` bundle with the raw
captures, including the recorded mic audio). Run it interactively the first time
so the mic ("make noise") and tone ("did you hear it?") checks can be confirmed.
Upload `report.md`.

## Control

```sh
sv down glados-room      # stop the client (pulse keeps running)
sv up glados-room        # start it again
sv restart glados-room   # bounce
sv-disable glados-room   # stop + don't start on next boot
```

## Wake-lock note (a design choice you may want to change)

The boot script acquires `termux-wake-lock` **once and never releases it** — an
always-on room appliance wants the CPU up regardless of which service is
momentarily down, and Termux's wake-lock is a single global (not ref-counted)
flag, so having one owner avoids one service's release dropping another's lock.
The trade-off: if you `sv-disable` everything, the phone still won't sleep until
you run `termux-wake-unlock`. If you'd rather the lock track the room service's
lifetime, move the `termux-wake-lock` into `service/glados-room/run` and add a
`service/glados-room/finish` that runs `termux-wake-unlock` — at the cost of the
CPU being free to sleep during a client crash-loop's restart gaps.

## Troubleshooting

- **Nothing starts on boot.** Termux:Boot app not opened once; or Termux not
  battery-optimization-exempt; or `~/.termux/boot/start-glados-room.sh` not
  executable.
- **Client crash-loops.** `tail` the log. Usual causes: wrong `server_url`,
  bad/missing token file (or not mode 600), or a `client_id`/role binding that
  doesn't match the server's `rooms.toml`. The `run` script's `sleep 1` floor
  keeps a crash-loop from pinning the CPU while you debug.
- **Pulse exits immediately (glados-pulse log shows a startup error).** Usually
  a bad stock config or a runtime-dir permission issue. Try `pulseaudio
  --daemonize=no --exit-idle-time=-1` by hand to see the error; the `sleep 1`
  floor keeps the respawn from pinning a core meanwhile.
- **No mic.** `pactl list short sources` shows no `sles` source → the room
  service's `load-module module-sles-source` failed; run it by hand to see why
  (e.g. the OpenSL ES module isn't available on this build).
- **`pactl` in the service can't reach pulse** (different socket than the
  daemon) → confirm both services see the same `HOME`; they fall back to the
  fixed Termux home, but a custom `HOME` must be consistent across them.
- **No playback.** `pacat_sink` in `config.room.toml` doesn't name a live sink;
  `pactl list short sinks` to find the right one (e.g. a `bluez_output.*` after
  pairing a Bluetooth speaker).
