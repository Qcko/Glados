#!/usr/bin/env sh
# Build the GLaDOS phone bundle: a self-extracting go.sh (and a raw .tar.gz
# alongside it) with the client_room package + deploy scripts + self-test.
#
# Run on the DEV BOX (needs git). On Windows use make-phone-bundle.ps1 instead
# (no `sh` required). Uses `git archive`, so the bundle contains only
# TRACKED files — never your gitignored config.room.toml, tokens, or env file —
# with the committed LF line endings and exec bits intact. It bundles COMMITTED
# state: commit local changes first if you want them included.
#
# Usage:
#   sh client_room/deploy/termux/make-phone-bundle.sh [out-dir]
set -e

ROOT=$(git rev-parse --show-toplevel)
OUTDIR=${1:-"$ROOT/dist"}
mkdir -p "$OUTDIR"
TAR="$OUTDIR/glados-phone-bundle.tar.gz"
GO="$OUTDIR/go.sh"

# --prefix=glados/ so it extracts to ./glados/ (→ ~/glados when run in $HOME).
git -C "$ROOT" archive --format=tar.gz --prefix=glados/ HEAD client_room > "$TAR"

# Build the self-extracting go.sh: LF-only header + raw tar appended.
# tail -n +N skips the N-1 header lines then streams raw bytes (binary-safe).
{
  cat <<'HEADER'
#!/data/data/com.termux/files/usr/bin/sh
# GLaDOS phone bundle — self-extracting.
# Copy this file to the phone, then in Termux (after termux-setup-storage):
#   sh ~/storage/downloads/go.sh
SKIP=$(awk '/^#__BUNDLE__$/{print NR+1; exit}' "$0")
tail -n +$SKIP "$0" | tar xzf - -C "$HOME"
sh "$HOME/glados/client_room/deploy/termux/run.sh"
rm -f "$0"
exit
#__BUNDLE__
HEADER
  cat "$TAR"
} > "$GO"

TAR_BYTES=$(wc -c < "$TAR" | tr -d ' ')
GO_BYTES=$(wc -c < "$GO" | tr -d ' ')
echo "wrote $TAR ($TAR_BYTES bytes)"
echo "wrote $GO ($GO_BYTES bytes)"
echo
echo "Copy go.sh to the phone, then in Termux:"
echo "  termux-setup-storage                        # once, if not done"
echo "  sh ~/storage/downloads/go.sh                # installs + tests + cleans up"
