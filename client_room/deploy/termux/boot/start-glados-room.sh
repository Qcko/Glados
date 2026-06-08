#!/data/data/com.termux/files/usr/bin/sh
# Termux:Boot entry point. Termux:Boot runs every script in ~/.termux/boot/ on
# device boot; this is NOT that location — provision a copy/symlink of this file
# into ~/.termux/boot/ (see README). The repo copy is the source of truth.
#
# Job: keep the CPU awake and bring up the runit supervision tree so the enabled
# glados-pulse + glados-room services start. Service lifecycle (restart, pulse
# readiness, logging) lives in the runit services, NOT here — keep this minimal.

# Hold a wake-lock so Android's doze doesn't suspend the supervision tree. This
# is an always-on appliance, so the lock is acquired once at boot and left held;
# run `termux-wake-unlock` by hand to release. (Requires the termux-api package +
# the Termux:API app; degrade gracefully if absent so boot still proceeds.)
command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock

# Start the termux-services runit tree (idempotent). This script ships with the
# termux-services package; sourcing it launches runsvdir, which then starts every
# `sv-enable`d service.
START_SERVICES=/data/data/com.termux/files/usr/etc/profile.d/start-services.sh
[ -r "$START_SERVICES" ] && . "$START_SERVICES"
