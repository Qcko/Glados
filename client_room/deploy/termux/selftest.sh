#!/data/data/com.termux/files/usr/bin/sh
# GLaDOS room-client phone self-test (Termux/Android).
#
# Runs the on-hardware checks that can't be done on the dev box and collects the
# results into ONE markdown report you upload back to the assistant. It probes
# exactly the assumptions the deploy wrapper depends on: PulseAudio starting on
# its stock config, the OpenSL ES mic source loading, real mic capture levels,
# an output sink + pacat playback, the Python client backends importing, and
# (optionally) a live client run.
#
# Safe to run before OR after provisioning the runit services. It is read-mostly:
# it will START a PulseAudio daemon and LOAD module-sles-source if they aren't
# already up (both are what the appliance wants anyway) and reports what it
# changed; it does not stop services, delete anything, or touch your config.
#
# Usage:
#   sh client_room/deploy/termux/selftest.sh [--non-interactive] [--client]
#     --non-interactive   skip the "did you hear the tone?" prompts (mark WARN)
#     --client            also run the real client for ~12s and capture its log
#                         (needs a reachable server + valid tokens)

set -u

# ---- options ---------------------------------------------------------------
INTERACTIVE=1
RUN_CLIENT=0
for arg in "$@"; do
  case "$arg" in
    --non-interactive) INTERACTIVE=0 ;;
    --client) RUN_CLIENT=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done
[ -t 0 ] || INTERACTIVE=0   # no tty → can't prompt

# ---- env (mirror the service run scripts) ----------------------------------
: "${HOME:=/data/data/com.termux/files/home}"
: "${PREFIX:=/data/data/com.termux/files/usr}"
ENV_FILE="$HOME/.config/glados-room/env"
[ -r "$ENV_FILE" ] && . "$ENV_FILE"
# Infer the client dir from this script's own location (…/client_room/deploy/
# termux/selftest.sh → three levels up = the dir CONTAINING client_room), so an
# extracted phone bundle runs wherever it lands; fall back to ~/glados.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INFERRED_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." 2>/dev/null && pwd || echo "")
if [ -z "${GLADOS_ROOM_DIR:-}" ]; then
  if [ -n "$INFERRED_DIR" ] && [ -d "$INFERRED_DIR/client_room" ]; then
    GLADOS_ROOM_DIR="$INFERRED_DIR"
  else
    GLADOS_ROOM_DIR="$HOME/glados"
  fi
fi
: "${GLADOS_ROOM_PYTHON:=python}"

# ---- output bundle ---------------------------------------------------------
STAMP=$(date +%Y%m%d-%H%M%S)
OUTDIR="$HOME/glados-selftest-$STAMP"
mkdir -p "$OUTDIR"
REPORT="$OUTDIR/report.md"
SUMMARY="$OUTDIR/.summary"   # tab-separated STATUS\tname, assembled into report
: > "$SUMMARY"

RATE=48000

# ---- helpers ---------------------------------------------------------------
# Append a line to the report (and nothing to the terminal).
r() { printf '%s\n' "$*" >> "$REPORT"; }
# Fence a file's contents into the report under a label; truncate huge captures.
rfile() {
  _label=$1; _path=$2
  r ""; r "**$_label**"; r '```'
  if [ -s "$_path" ]; then head -c 8000 "$_path" >> "$REPORT"; r ""; else r "(empty)"; fi
  r '```'
}
# Record a check result: status name. Echoes to terminal + records for summary.
res() {
  _status=$1; shift; _name=$*
  printf '%s  %s\n' "$_status" "$_name"
  printf '%s\t%s\n' "$_status" "$_name" >> "$SUMMARY"
  r "- **$_status** — $_name"
}
pass() { res PASS "$*"; }
fail() { res FAIL "$*"; }
warn() { res WARN "$*"; }
skip() { res SKIP "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
section() { printf '\n=== %s ===\n' "$1"; r ""; r "## $1"; }

# ---- report header ---------------------------------------------------------
r "# GLaDOS room-client phone self-test"
r ""
r "- generated: \`$STAMP\`"
r "- host HOME: \`$HOME\`  PREFIX: \`$PREFIX\`"
r "- GLADOS_ROOM_DIR: \`$GLADOS_ROOM_DIR\`  python: \`$GLADOS_ROOM_PYTHON\`"
r "- interactive: $INTERACTIVE  run-client: $RUN_CLIENT"
r ""
r "_(Summary table is appended at the end.)_"
r ""
r "> ⚠ Before sharing: this report embeds your \`\$HOME\` path and device model,"
r "> and the \`--client\` log could contain a token if the client logs one."
r "> Skim \`client-run.log\` and redact anything sensitive first."

printf 'GLaDOS phone self-test → %s\n' "$OUTDIR"

# ---- 1. environment --------------------------------------------------------
section "Environment"
{
  echo "model:    $(getprop ro.product.model 2>/dev/null) ($(getprop ro.product.manufacturer 2>/dev/null))"
  echo "android:  $(getprop ro.build.version.release 2>/dev/null) (sdk $(getprop ro.build.version.sdk 2>/dev/null))"
  echo "kernel:   $(uname -srm)"
  echo "abi:      $(getprop ro.product.cpu.abi 2>/dev/null)"
  echo "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-<unset>}"
  echo "PULSE_RUNTIME_PATH=${PULSE_RUNTIME_PATH:-<unset>}"
  echo "PULSE_SERVER=${PULSE_SERVER:-<unset>}"
} > "$OUTDIR/env.txt" 2>&1
rfile "device / env" "$OUTDIR/env.txt"

MISSING=""
for t in "$GLADOS_ROOM_PYTHON" pulseaudio pactl parec pacat; do
  have "$t" || MISSING="$MISSING $t"
done
{
  for t in "$GLADOS_ROOM_PYTHON" pulseaudio pactl parec pacat sv svlogd sv-enable termux-wake-lock; do
    if have "$t"; then printf '%-16s %s\n' "$t" "$(command -v "$t")"; else printf '%-16s MISSING\n' "$t"; fi
  done
  echo "---"
  pulseaudio --version 2>&1
  "$GLADOS_ROOM_PYTHON" --version 2>&1
} > "$OUTDIR/tools.txt" 2>&1
rfile "tools" "$OUTDIR/tools.txt"

if [ -n "$MISSING" ]; then
  fail "required tools present (missing:$MISSING — run install.sh to fix)"
else
  pass "required tools present (python/pulseaudio/pactl/parec/pacat)"
fi

have sv && have svlogd && pass "termux-services present" || warn "termux-services not installed (pkg install termux-services) — supervision won't work"
have termux-wake-lock && pass "termux-api present (wake-lock available)" || warn "termux-api not installed — boot wake-lock will be skipped"
[ -d "$HOME/.termux/boot" ] && pass "~/.termux/boot exists (Termux:Boot provisioned)" || warn "~/.termux/boot missing — Termux:Boot app not installed/opened, or boot script not linked"

if [ -z "$MISSING" ]; then : ; else
  r ""; r "_Core tools missing — audio checks below will be skipped._"
fi

# ---- 2. PulseAudio ---------------------------------------------------------
section "PulseAudio"
STARTED_PULSE=0
if have pactl pulseaudio; then
  if pactl info >/dev/null 2>&1; then
    pass "PulseAudio already running (using it; not starting a second daemon)"
  else
    printf '... starting PulseAudio (stock config, the way the deploy does)\n'
    pulseaudio --start --exit-idle-time=-1 >"$OUTDIR/pulse-start.txt" 2>&1
    sleep 2
    if pactl info >/dev/null 2>&1; then
      STARTED_PULSE=1
      pass "PulseAudio starts on its stock config (the duck's #1 risk — OK here)"
    else
      fail "PulseAudio failed to start on stock config (see pulse-start.txt — this is the crash-loop risk)"
      rfile "pulseaudio --start output" "$OUTDIR/pulse-start.txt"
    fi
  fi

  if pactl info >/dev/null 2>&1; then
    pactl info > "$OUTDIR/pactl-info.txt" 2>&1
    rfile "pactl info" "$OUTDIR/pactl-info.txt"
    # Socket/runtime-dir consistency (duck should-fix): where does the server live?
    grep -i 'Server String\|Runtime Path' "$OUTDIR/pactl-info.txt" >/dev/null 2>&1 \
      && pass "pactl reaches the daemon (HOME/socket consistent)" \
      || warn "could not read server runtime path from pactl info"
  fi
  STOCK_PA="$PREFIX/etc/pulse/default.pa"
  r ""
  if [ -f "$STOCK_PA" ]; then
    r "_Note: stock \`default.pa\` present at \`$STOCK_PA\`._"
  else
    r "_Note: no stock \`default.pa\` at \`$STOCK_PA\` (config likely compiled in — confirms why the deploy uses stock startup, not \`.include\`)._"
  fi
else
  skip "PulseAudio checks (tools missing)"
fi

# ---- 3. mic source (module-sles-source) ------------------------------------
section "Mic source (module-sles-source)"
SRC=""
LOADED_MODULE=0
if pactl info >/dev/null 2>&1; then
  if pactl list short sources 2>/dev/null | grep -q sles; then
    pass "module-sles-source already loaded"
  else
    printf '... loading module-sles-source (grep-guarded, as the deploy does)\n'
    if pactl load-module module-sles-source >"$OUTDIR/load-sles.txt" 2>&1; then
      LOADED_MODULE=1
      pass "module-sles-source loads"
    else
      fail "module-sles-source failed to load (see load-sles.txt — no mic on this build)"
      rfile "load-module output" "$OUTDIR/load-sles.txt"
    fi
  fi
  pactl list short sources > "$OUTDIR/sources.txt" 2>&1
  rfile "sources" "$OUTDIR/sources.txt"
  SRC=$(awk '/sles/ {print $2; exit}' "$OUTDIR/sources.txt")
  [ -n "$SRC" ] && r "" && r "_Using source: \`$SRC\`_"
else
  skip "mic-source checks (no PulseAudio)"
fi

# ---- 4. mic capture (objective level check) --------------------------------
section "Mic capture (parec)"
if [ -n "$SRC" ] && have parec; then
  if [ "$INTERACTIVE" = 1 ]; then
    printf '\n>>> Make some NOISE near the phone for 5 seconds (speak, clap)...\n'
    sleep 1
  fi
  SRC_ARG=""
  [ -n "$SRC" ] && SRC_ARG="-d $SRC"
  # shellcheck disable=SC2086
  timeout 5 parec --format=s16le --rate=$RATE --channels=1 $SRC_ARG --raw \
    > "$OUTDIR/mic.raw" 2>"$OUTDIR/mic.err"
  if "$GLADOS_ROOM_PYTHON" - "$OUTDIR/mic.raw" >"$OUTDIR/mic-level.txt" 2>&1 <<'PY'
import sys, math, array
data = open(sys.argv[1], "rb").read()
n = len(data) // 2
if n == 0:
    print("captured 0 samples (parec produced no audio)"); sys.exit(1)
a = array.array("h"); a.frombytes(data[: n * 2])
peak = max(abs(x) for x in a) or 0
rms = math.sqrt(sum(x * x for x in a) / len(a))
dbfs = lambda v: 20 * math.log10(v / 32768.0) if v > 0 else -120.0
print(f"samples={len(a)}  peak={peak} ({dbfs(peak):.1f} dBFS)  rms={rms:.1f} ({dbfs(rms):.1f} dBFS)")
# Pass on sustained energy (RMS), not a single glitch/DC sample: a dead mic with
# one popped sample can clear a peak-only gate. Require RMS above the floor AND a
# real peak.
sys.exit(0 if dbfs(rms) > -55.0 and dbfs(peak) > -45.0 else 1)
PY
  then
    pass "mic captures non-silent audio ($(cat "$OUTDIR/mic-level.txt"))"
  else
    warn "mic capture silent/low ($(cat "$OUTDIR/mic-level.txt")) — check RECORD permission / source"
    rfile "parec stderr" "$OUTDIR/mic.err"
  fi
  rfile "mic level" "$OUTDIR/mic-level.txt"
else
  skip "mic capture (no sles source or parec)"
fi

# ---- 5. output sink + playback ---------------------------------------------
section "Output sink + playback (pacat)"
if pactl info >/dev/null 2>&1; then
  pactl list short sinks > "$OUTDIR/sinks.txt" 2>&1
  rfile "sinks" "$OUTDIR/sinks.txt"
  if [ -s "$OUTDIR/sinks.txt" ]; then
    BT=$(awk '/bluez_output/ {print $2; exit}' "$OUTDIR/sinks.txt")
    r ""
    if [ -n "$BT" ]; then
      r "_Bluetooth sink detected: \`$BT\` (set this as \`pacat_sink\`)._"
    else
      r "_No \`bluez_output.*\` sink — pair a Bluetooth speaker to get one, else playback uses the default sink._"
    fi
    if have pacat; then
      "$GLADOS_ROOM_PYTHON" - "$OUTDIR/tone.raw" "$RATE" <<'PY'
import sys, math, array
path, rate = sys.argv[1], int(sys.argv[2])
a = array.array("h")
for i in range(int(rate * 1.2)):
    a.append(int(0.3 * 32767 * math.sin(2 * math.pi * 440.0 * i / rate)))
open(path, "wb").write(a.tobytes())
PY
      # Play to the bluez sink the appliance will actually use, if one exists —
      # otherwise the tone proves only the (often silent) default sink works.
      SINK_ARG=""
      [ -n "$BT" ] && SINK_ARG="-d $BT"
      printf '... playing a 440 Hz test tone to %s\n' "${BT:-the default sink}"
      r "_Tone target sink: \`${BT:-<default>}\`_"
      # shellcheck disable=SC2086
      pacat --format=s16le --rate=$RATE --channels=1 $SINK_ARG --raw \
        < "$OUTDIR/tone.raw" >"$OUTDIR/pacat.err" 2>&1
      if [ "$INTERACTIVE" = 1 ]; then
        printf '>>> Did you hear a beep? [y/N] '
        ans=""               # set -u + EOF on read would otherwise abort the run
        read -r ans || ans=""
        case "$ans" in
          y|Y|yes) pass "pacat playback audible (operator confirmed)" ;;
          *) fail "pacat playback NOT heard (operator) — check sink / volume / pacat_sink" ;;
        esac
      else
        warn "tone played but not human-confirmed (--non-interactive) — re-run interactively to verify"
      fi
      [ -s "$OUTDIR/pacat.err" ] && rfile "pacat stderr" "$OUTDIR/pacat.err"
    else
      skip "playback (pacat missing)"
    fi
  else
    fail "no output sinks at all — pacat has nowhere to play"
  fi
else
  skip "playback (no PulseAudio)"
fi

# ---- 6. Python client backends ---------------------------------------------
section "Python client"
if [ -d "$GLADOS_ROOM_DIR/client_room" ]; then
  ( cd "$GLADOS_ROOM_DIR" && "$GLADOS_ROOM_PYTHON" -c \
      "import client_room.audio, client_room.room, client_room.mic, client_room.speaker; print('imports OK')" ) \
      >"$OUTDIR/import.txt" 2>&1 \
    && pass "client_room imports ($(cat "$OUTDIR/import.txt"))" \
    || { fail "client_room failed to import (deps missing? see import.txt)"; rfile "import error" "$OUTDIR/import.txt"; }
  if [ -f "$GLADOS_ROOM_DIR/client_room/config.room.toml" ]; then
    pass "config.room.toml present"
  else
    warn "no client_room/config.room.toml — copy config.room.example.toml and fill it in"
  fi
else
  warn "GLADOS_ROOM_DIR=$GLADOS_ROOM_DIR has no client_room/ — set it in ~/.config/glados-room/env"
fi

# ---- 7. runit services (if provisioned) ------------------------------------
section "runit services"
if have sv; then
  { sv status glados-pulse glados-room 2>&1; } > "$OUTDIR/sv-status.txt"
  rfile "sv status" "$OUTDIR/sv-status.txt"
  grep -q '^run:' "$OUTDIR/sv-status.txt" 2>/dev/null \
    && pass "at least one glados service is supervised (run)" \
    || warn "glados services not running — sv-enable glados-pulse glados-room (or not provisioned yet)"
else
  skip "runit service status (sv missing)"
fi

# ---- 8. live client run (opt-in) -------------------------------------------
section "Live client run"
if [ "$RUN_CLIENT" = 1 ]; then
  if [ -d "$GLADOS_ROOM_DIR/client_room" ]; then
    printf '... running the client for ~12s (Ctrl-C-safe)\n'
    ( cd "$GLADOS_ROOM_DIR" && timeout 12 "$GLADOS_ROOM_PYTHON" -m client_room.room ) \
      >"$OUTDIR/client-run.log" 2>&1
    rc=$?
    rfile "client run log (first 12s)" "$OUTDIR/client-run.log"
    # timeout exits 124 when IT killed the still-running process → the client
    # stayed up the whole window (the healthy case). Any other code means the
    # client exited ON ITS OWN within 12s — a terminal handshake error (bad
    # token/binding) or a crash, both of which the appliance would respawn-loop.
    has_err=0
    grep -qi 'error\|traceback\|refused\|mismatch\|no such\|closed\|401\|403' \
      "$OUTDIR/client-run.log" && has_err=1
    r ""; r "_client exit code: $rc (124 = ran the full window)._"
    if [ "$rc" = 124 ] && [ "$has_err" = 0 ]; then
      pass "client stayed up the full 12s with no error lines"
    elif [ "$rc" = 124 ]; then
      warn "client stayed up 12s but logged error-ish lines — inspect client-run.log"
    else
      warn "client EXITED on its own within 12s (rc=$rc) — likely terminal error (token/binding) or crash; see client-run.log"
    fi
  else
    skip "live client run (GLADOS_ROOM_DIR has no client_room/)"
  fi
else
  skip "live client run (pass --client to enable; needs server + tokens)"
fi

# ---- summary ---------------------------------------------------------------
section "Summary"
r ""
r "| status | check |"
r "|--------|-------|"
TAB=$(printf '\t')
while IFS="$TAB" read -r st nm; do r "| $st | $nm |"; done < "$SUMMARY"
# grep -c always prints a count (0 when none), so no fallback — and its exit
# status is irrelevant here (no `set -e`).
np=$(grep -c '^PASS' "$SUMMARY" 2>/dev/null)
nf=$(grep -c '^FAIL' "$SUMMARY" 2>/dev/null)
nw=$(grep -c '^WARN' "$SUMMARY" 2>/dev/null)
ns=$(grep -c '^SKIP' "$SUMMARY" 2>/dev/null)
r ""
r "**$np PASS · $nf FAIL · $nw WARN · $ns SKIP**"
r ""
r "_State changed by this run: started PulseAudio=$STARTED_PULSE, loaded module-sles-source=$LOADED_MODULE._"
rm -f "$SUMMARY"

# ---- bundle + final instructions -------------------------------------------
TARBALL="$OUTDIR.tar.gz"
( cd "$(dirname "$OUTDIR")" && tar czf "$TARBALL" "$(basename "$OUTDIR")" ) 2>/dev/null

printf '\n========================================\n'
printf '%s PASS · %s FAIL · %s WARN · %s SKIP\n' "$np" "$nf" "$nw" "$ns"
printf 'Report:  %s\n' "$REPORT"
[ -f "$TARBALL" ] && printf 'Bundle:  %s  (raw captures incl. mic.raw)\n' "$TARBALL"
printf '\nUpload report.md back to the assistant. To read it now:\n  cat %s\n' "$REPORT"
