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
  deps.sh                          idempotent dependency installer (Termux)
  dispatch.sh                      bundle second stage: doctor (default) | install
  provision.sh                     symlink services + boot script (left DOWN by default)
  enroll.sh                        give the device its identity (config + tokens + cert)
  requirements-phone.txt           Python deps (numpy, websockets)
  make-phone-bundle.sh / .ps1      build the self-extracting go.sh (dev box; sh or PowerShell)
  selftest.sh                      on-hardware diagnostic → report.md (see below)
```

## Three paths: doctor, install, enroll

The deploy splits by **secret posture** (see `../ROADMAP.md`):

- **doctor** — *"is this device healthy?"* `sh go.sh` (default) runs deps + the
  self-test, then deletes `~/glados`. Secret-free, re-runnable.
- **install** — *"make this device a GLaDOS appliance."* `sh go.sh install` runs
  deps + provisions the runit services (left **DOWN**) and leaves `~/glados` in
  place. Secret-free.
- **enroll** — *"give this device its identity."* `sh enroll.sh` installs the
  config + tokens + cert (mode-600) and starts the services. The only
  secret-bearing step. Today the secrets are side-loaded via Downloads; the
  ROADMAP target mints them server-side at pairing so no secret file ever
  touches the device.

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
4. **Set up shared storage** so you can move the bundle in and pull the self-test
   report out via Android's file picker (Termux's private `$HOME` is invisible to
   it):
   ```sh
   termux-setup-storage      # grant the permission prompt; creates ~/storage/*
   ```
   With this done, `selftest.sh` publishes its report to `~/storage/downloads`
   (Android **Downloads**), and you can drop the install bundle there too.
5. Packages: handled by `install.sh` below, or by hand —
   ```sh
   pkg install python pulseaudio termux-services termux-api
   ```

## Install the client

On the dev box, build the self-extracting bundle (only TRACKED files — never your
gitignored config/tokens). It just wraps `git archive`, so no `sh` is needed:

```sh
sh client_room/deploy/termux/make-phone-bundle.sh        # Linux/macOS/git-bash
pwsh client_room\deploy\termux\make-phone-bundle.ps1     # Windows PowerShell
# → dist/go.sh   (and dist/glados-phone-bundle.tar.gz alongside)
```

Copy `dist/go.sh` to the phone's **Downloads** (so the file picker / `cp` can
reach it), then in Termux:

```sh
sh ~/storage/downloads/go.sh            # doctor: deps + self-test, then self-cleans
sh ~/storage/downloads/go.sh install    # install: deps + provision, leaves ~/glados
```

- **doctor** (no arg) is the health check — see *Self-test* below. It deletes
  `~/glados` and `go.sh` when done, so it's safe to re-run.
- **install** extracts to `~/glados`, installs deps, wires the runit services +
  Termux:Boot, and leaves everything in place with the services **DOWN** (no
  identity yet). Give it one with `enroll.sh` (next section).

`deps.sh` (run by both paths) probes each dependency and installs only what's
missing (`python pulseaudio termux-services termux-api`, plus `numpy` from the
prebuilt **`python-numpy`** package — never pip-compiled — and `websockets` via
pip). The client is pure portable Python and never imports the server package;
`keyring` is optional (token loading falls back to an env var / mode-600 file).
Deps are also listed in `requirements-phone.txt`.

> **Clone instead of bundle?** If the phone has git, `git clone <remote> ~/glados`
> then `sh ~/glados/client_room/deploy/termux/dispatch.sh install` is equivalent
> (`dispatch.sh` is what `go.sh` runs after extracting).

## Enroll — give the device its identity

`enroll.sh` is the only secret-bearing step. It installs the room config, the
per-role token files (mode 600), and the server's pinned cert, then enables and
starts the services. **A mic token grants live room-audio capture** — treat the
token files as the sensitive material they are.

On the **dev box**, author `config.room.toml` from the example, using the
canonical on-phone paths so nothing needs editing on the device:

```toml
server_url = "wss://<server-LAN-ip>:8765"
tls_ca     = "~/.config/glados-room/glados-server-cert.pem"
[mic]
client_id  = "livingroom-mic"
token_file = "~/.config/glados-room/secrets/livingroom-mic.token"
capture_backend = "parec"
[speaker]
client_id  = "livingroom-speaker"
token_file = "~/.config/glados-room/secrets/livingroom-speaker.token"
playback_backend = "pacat"
# pacat_sink = "bluez_output.XX_XX_XX_XX_XX_XX.1"   # your output sink
```

Stage these four files into the phone's **Downloads** (named exactly so):

| file | what | secret? |
|------|------|---------|
| `config.room.toml` | the config above | no (paths + LAN addr) |
| `<client_id>.token` | one per role, basename = the `token_file` basename | **yes** |
| `glados-server-cert.pem` | the server's PUBLIC pinned cert (`configs/tls/cert.pem`) | no |

Then on the phone:

```sh
sh ~/glados/client_room/deploy/termux/enroll.sh
```

It moves the files out of shared storage, sets the secrets dir to mode 700 and
each token to 600, validates the config parses, removes the Downloads copies
(even on failure), then starts the services. **Re-running enroll** replaces the
full token set and bounces the client, so rotating a token or moving rooms needs
no reinstall and no reboot. After rotating, revoke the old token server-side
(`rooms.toml` / keyring).

`config.room.toml` and the token files are gitignored / never committed; the
token files are also staged out-of-repo on the dev box.

## Operator env

```sh
mkdir -p ~/.config/glados-room
cp client_room/deploy/termux/glados-room.env.example ~/.config/glados-room/env
$EDITOR ~/.config/glados-room/env     # set GLADOS_ROOM_DIR / GLADOS_ROOM_PYTHON if not default
```

Defaults assume `GLADOS_ROOM_DIR=$HOME/glados` and the system `python`; an empty
file is fine if that matches your layout.

## Provision the services + boot script

`provision.sh` (run for you by `install`) makes every script executable and
symlinks the two services into the runit service dir (`$PREFIX/var/service`) and
the boot script into `~/.termux/boot` — symlinks, so a re-extracted bundle is
picked up. It's idempotent. From the deploy dir:

```sh
sh provision.sh            # wire the services but leave them DOWN (no identity yet)
sh provision.sh --enable   # wire + sv-enable both (start now + on every boot)
```

Why DOWN by default: `runsvdir` auto-starts any service it finds symlinked into
`$PREFIX/var/service` **unless a `down` marker exists** in the service dir. An
identity-less room client would crash-loop, so provision drops a `down` marker
and `enroll.sh` removes it (via `--enable`) once the device has a config + tokens.
This makes "enabled but no config" unreachable by construction. (Re-running plain
`provision.sh` re-disables the services — after a reinstall, re-run `enroll.sh`.)

> Termux:Boot executes `~/.termux/boot/*` directly, so that file must be
> executable and carry the `#!/data/data/com.termux/files/usr/bin/sh` shebang
> — `provision.sh` handles both.

`enroll.sh` enables and starts the services for you, so you normally never call
`sv-enable` by hand. To do it manually after a bare `provision.sh`:

```sh
sv-enable glados-pulse     # removes the down marker, starts now + on boot
sv-enable glados-room
```

## Verify

```sh
sv status glados-pulse glados-room          # both should read "run"
pactl info                                  # pulse reachable
pactl list short sources | grep module-sles-source   # the mic source loaded
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

From the deploy dir (`cd ~/glados/client_room/deploy/termux`):

```sh
sh selftest.sh            # interactive (mic + tone checks)
sh selftest.sh --client   # also run the client ~12s
sh selftest.sh --non-interactive   # no prompts (headless)
```

It prints a `N PASS · N FAIL · N WARN · N SKIP` tally and **publishes two files to
`~/storage/downloads`** (Android Downloads, if you ran `termux-setup-storage` —
otherwise `$HOME`): `glados-report-<stamp>.md` (the human-readable report) and
`glados-selftest-<stamp>.tar.gz` (the raw captures, incl. the recorded mic audio).
The scratch build dir is removed afterward, so each run leaves exactly those two
files. Run it interactively the first time: it **prompts you to get ready before**
recording the mic and before playing the tone, and lets you retry just that check
if you miss the cue. Upload the report.

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
- **No mic.** `pactl list short sources` shows no `module-sles-source` → the room
  service's `load-module module-sles-source` failed; run it by hand to see why
  (e.g. the OpenSL ES module isn't available on this build).
- **`pactl` in the service can't reach pulse** (different socket than the
  daemon) → confirm both services see the same `HOME`; they fall back to the
  fixed Termux home, but a custom `HOME` must be consistent across them.
- **No playback.** `pacat_sink` in `config.room.toml` doesn't name a live sink;
  `pactl list short sinks` to find the right one (e.g. a `bluez_output.*` after
  pairing a Bluetooth speaker).
