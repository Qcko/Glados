#!/data/data/com.termux/files/usr/bin/sh
# One-shot dependency installer for the GLaDOS room client on Termux/Android.
# Idempotent: probes each dependency and installs ONLY what's missing (after one
# full `pkg upgrade` if anything is), so it's safe to re-run. Called by dispatch.sh (both the doctor and install paths); can
# also be run on its own after getting the files onto the phone (bundle or clone).
#
#   sh client_room/deploy/termux/deps.sh
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

UPGRADED=""

have() { command -v "$1" >/dev/null 2>&1; }

# A fresh F-Droid Termux ships an old package base; a new package installed on
# top of it (python-numpy) links against libraries the base doesn't have yet and
# fails at import with "cannot locate symbol". Upgrade once, before the first
# install, so a re-run with nothing missing stays offline.
upgrade_base_once() {
  [ -n "$UPGRADED" ] && return
  UPGRADED=1
  echo "[get]   full package upgrade (pkg upgrade; may move python to a new minor version) ..."
  DEBIAN_FRONTEND=noninteractive pkg upgrade -y \
    -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold \
    || echo "[warn]  pkg upgrade failed; new installs may not load"
}

# Install Termux package $1 if the probe command $2 is absent.
ensure_pkg() {
  _pkg=$1; _probe=$2
  if have "$_probe"; then
    echo "[ok]    $_pkg ($_probe present)"
    return
  fi
  upgrade_base_once
  echo "[get]   $_pkg ..."
  if pkg install -y "$_pkg" && have "$_probe"; then echo "[done]  $_pkg"; else
    echo "[FAIL]  $_pkg"; FAILED="$FAILED $_pkg"; fi
}

# Install Python module $1 via the command(s) in $2 if it can't already import.
ensure_pymod() {
  _mod=$1; _install=$2
  if python -c "import $_mod" >/dev/null 2>&1; then
    echo "[ok]    python:$_mod"
    return
  fi
  upgrade_base_once
  echo "[get]   python:$_mod ($_install) ..."
  if sh -c "$_install" && python -c "import $_mod" >/dev/null 2>&1; then echo "[done]  python:$_mod"; else
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
echo "Next (usually driven for you by dispatch.sh / go.sh):"
echo "  sh $SCRIPT_DIR/selftest.sh         # doctor: verify the phone's audio plumbing"
echo "  sh $SCRIPT_DIR/provision.sh        # install: wire the runit services (left DOWN)"
echo "  sh $SCRIPT_DIR/enroll.sh           # give the device its identity, then start it"
