#!/data/data/com.termux/files/usr/bin/sh
# One-shot dependency installer for the GLaDOS room client on Termux/Android.
# Idempotent: probes each dependency and installs ONLY what's missing, so it's
# safe to re-run. Run it after getting the files onto the phone (bundle or clone)
# and before selftest.sh.
#
#   sh client_room/deploy/termux/install.sh
#
# numpy is installed from the prebuilt Termux package (`python-numpy`), NOT pip —
# pip would try to COMPILE numpy on the phone, which is slow and fragile.
# websockets is pure Python, so pip is fine.
set -u

PREFIX=${PREFIX:-/data/data/com.termux/files/usr}
case "$PREFIX" in
  *com.termux*) : ;;
  *) echo "This doesn't look like Termux (PREFIX=$PREFIX); nothing to install. Aborting." >&2
     exit 1 ;;
esac

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
FAILED=""

have() { command -v "$1" >/dev/null 2>&1; }

# Install Termux package $1 if the probe command $2 is absent.
ensure_pkg() {
  _pkg=$1; _probe=$2
  if have "$_probe"; then
    echo "[ok]    $_pkg ($_probe present)"
    return
  fi
  echo "[get]   $_pkg ..."
  if pkg install -y "$_pkg"; then echo "[done]  $_pkg"; else
    echo "[FAIL]  $_pkg"; FAILED="$FAILED $_pkg"; fi
}

# Install Python module $1 via the command(s) in $2 if it can't already import.
ensure_pymod() {
  _mod=$1; _install=$2
  if python -c "import $_mod" >/dev/null 2>&1; then
    echo "[ok]    python:$_mod"
    return
  fi
  echo "[get]   python:$_mod ($_install) ..."
  if sh -c "$_install"; then echo "[done]  python:$_mod"; else
    echo "[FAIL]  python:$_mod"; FAILED="$FAILED $_mod"; fi
}

echo "=== Termux packages ==="
ensure_pkg python          python      # must come first: the pymod probes need it
ensure_pkg pulseaudio      pulseaudio
ensure_pkg termux-services sv
ensure_pkg termux-api      termux-wake-lock

echo
echo "=== Python deps ==="
if have python; then
  ensure_pymod numpy      "pkg install -y python-numpy"
  ensure_pymod websockets "pip install websockets"
else
  echo "[skip]  python deps — python itself isn't installed"
  FAILED="$FAILED python"
fi

echo
if [ -n "$FAILED" ]; then
  echo "FAILED:$FAILED"
  echo "Check network / 'pkg update', then re-run this script."
  exit 1
fi

echo "All dependencies present."
echo
echo "Next:"
echo "  cp $SCRIPT_DIR/../../config.room.example.toml $SCRIPT_DIR/../../config.room.toml"
echo "  \$EDITOR $SCRIPT_DIR/../../config.room.toml      # server_url, room_id, client_ids, tokens"
echo "  sh $SCRIPT_DIR/selftest.sh                       # verify the phone's audio plumbing"
