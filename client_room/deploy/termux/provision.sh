#!/data/data/com.termux/files/usr/bin/sh
# Provision the runit services + Termux:Boot script for the GLaDOS room client.
# Makes every script executable and symlinks the two services + the boot script
# into place. Idempotent — re-running just refreshes the symlinks (so a re-
# extracted bundle is picked up). Normally called by dispatch.sh's `install`
# path (default, leaves services DOWN) and by enroll.sh (--enable, starts them).
#
#   sh provision.sh            # wire services but leave them DOWN (no identity yet)
#   sh provision.sh --enable   # wire + sv-enable both (start now + on every boot)
#
# Why "wired but down" is the default: runsvdir auto-starts every service
# symlinked into $PREFIX/var/service the moment it sees it — UNLESS a `down`
# file exists in the service dir. Without identity (config.room.toml + tokens,
# installed by enroll.sh) the room client would crash-loop. So provision drops a
# `down` marker by default and enroll removes it (via --enable) once the device
# has an identity. Re-running plain provision re-disables the services — after a
# reinstall, re-run enroll to bring them back up.
set -u

ENABLE=0
for arg in "$@"; do
  case "$arg" in
    --enable) ENABLE=1 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "provision: unknown option: $arg" >&2; exit 2 ;;
  esac
done

: "${PREFIX:=/data/data/com.termux/files/usr}"
D=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

chmod +x "$D"/service/*/run "$D"/service/*/log/run "$D"/boot/start-glados-room.sh
mkdir -p "$PREFIX/var/service" "$HOME/.termux/boot"

# Drop the `down` markers BEFORE the symlinks go live, so runsvdir never sees a
# service without one and never starts an identity-less client. The marker lives
# in the service dir (the symlink target = this repo dir); it's gitignored.
for svc in glados-pulse glados-room; do
  if [ "$ENABLE" = 1 ]; then
    rm -f "$D/service/$svc/down"   # don't rely on sv-enable to clear a stale marker
  else
    : > "$D/service/$svc/down"
  fi
done

ln -sf "$D/service/glados-pulse" "$PREFIX/var/service/glados-pulse"
ln -sf "$D/service/glados-room"  "$PREFIX/var/service/glados-room"
ln -sf "$D/boot/start-glados-room.sh" "$HOME/.termux/boot/start-glados-room.sh"
echo "Provisioned: services symlinked into $PREFIX/var/service, boot script into ~/.termux/boot."

if [ "$ENABLE" = 1 ]; then
  echo "Enabling + starting services..."
  sv-enable glados-pulse   # removes the down marker + starts now and on boot
  sv-enable glados-room
  echo "Done. Check status with: sv status glados-pulse glados-room"
else
  echo "Services wired but left DOWN (no identity yet). Give the device its"
  echo "identity with: sh enroll.sh   (it installs the config + tokens, then enables)"
fi
