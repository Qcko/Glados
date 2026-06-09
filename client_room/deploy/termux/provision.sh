#!/data/data/com.termux/files/usr/bin/sh
# Provision the runit services + Termux:Boot script for the GLaDOS room client.
# Replaces the by-hand chmod/symlink recipe: makes every script executable and
# symlinks the two services + the boot script into place. Idempotent — re-running
# just refreshes the symlinks (so a `git pull` / re-extracted bundle is picked up).
# Run it after install.sh and after filling in config.room.toml.
#
#   sh provision.sh            # provision only; prints the two sv-enable commands
#   sh provision.sh --enable   # also sv-enable both services (start now + on boot)
set -u

ENABLE=0
for arg in "$@"; do
  case "$arg" in
    --enable) ENABLE=1 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

: "${PREFIX:=/data/data/com.termux/files/usr}"
D=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

chmod +x "$D"/service/*/run "$D"/service/*/log/run "$D"/boot/start-glados-room.sh
mkdir -p "$PREFIX/var/service" "$HOME/.termux/boot"

ln -sf "$D/service/glados-pulse" "$PREFIX/var/service/glados-pulse"
ln -sf "$D/service/glados-room"  "$PREFIX/var/service/glados-room"
ln -sf "$D/boot/start-glados-room.sh" "$HOME/.termux/boot/start-glados-room.sh"
echo "Provisioned: services symlinked into $PREFIX/var/service, boot script into ~/.termux/boot."

if [ "$ENABLE" = 1 ]; then
  echo "Enabling + starting services..."
  sv-enable glados-pulse
  sv-enable glados-room
  echo "Done. Check status with: sv status glados-pulse glados-room"
else
  echo "Enable + start (now and on every boot) with:"
  echo "  sv-enable glados-pulse"
  echo "  sv-enable glados-room"
fi
