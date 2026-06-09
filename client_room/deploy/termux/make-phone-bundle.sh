#!/usr/bin/env sh
# Build the phone self-test bundle: ONE .tar.gz with the client_room package +
# deploy scripts + self-test, to copy to the phone for all the local checks
# (1-7), provisioning, and a real run.
#
# Run on the DEV BOX (needs git). On Windows use make-phone-bundle.ps1 instead
# (no `sh` required). Uses `git archive`, so the bundle contains only
# TRACKED files — never your gitignored config.room.toml, tokens, or env file —
# with the committed LF line endings and exec bits intact. It bundles COMMITTED
# state: commit local changes first if you want them included.
#
# Usage:
#   sh client_room/deploy/termux/make-phone-bundle.sh [output.tar.gz]
set -e

ROOT=$(git rev-parse --show-toplevel)
OUT=${1:-"$ROOT/dist/glados-phone-bundle.tar.gz"}
mkdir -p "$(dirname "$OUT")"

# --prefix=glados/ so it extracts to ./glados/ (→ ~/glados when run in $HOME,
# the GLADOS_ROOM_DIR default).
git -C "$ROOT" archive --format=tar.gz --prefix=glados/ HEAD client_room > "$OUT"

echo "wrote $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)"
echo
echo "Copy it to the phone, then in Termux:"
echo "  termux-setup-storage                                  # once, if not done"
echo "  tar xzf ~/storage/downloads/$(basename "$OUT") -C \$HOME"
echo "  sh ~/glados/client_room/deploy/termux/run.sh         # installs + tests + cleans up"
