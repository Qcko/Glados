#!/data/data/com.termux/files/usr/bin/sh
# Second stage of the GLaDOS phone bundle. Normally called by go.sh (the
# self-extracting launcher), which handles extraction and its own cleanup.
# Can also be run directly after a manual tar extraction.
#
# Runs install.sh then selftest.sh, then removes ~/glados/ so nothing lingers.
#
# Usage:
#   sh ~/glados/client_room/deploy/termux/run.sh [selftest options...]
#
# Any flags are forwarded to selftest.sh (e.g. --non-interactive, --client).
set -u

DEPLOY_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
GLADOS_DIR=$(CDPATH= cd -- "$DEPLOY_DIR/../../.." && pwd)
BUNDLE_TAR="$HOME/storage/downloads/glados-phone-bundle.tar.gz"

echo "=== GLaDOS phone bundle — entry point ==="
echo "deploy dir : $DEPLOY_DIR"
echo "glados dir : $GLADOS_DIR"
echo ""

# ---- install deps -----------------------------------------------------------
echo ">>> install.sh"
sh "$DEPLOY_DIR/install.sh"
INSTALL_RC=$?
echo ""
if [ $INSTALL_RC -ne 0 ]; then
  echo "install.sh failed (exit $INSTALL_RC) — aborting." >&2
  exit $INSTALL_RC
fi

# ---- run self-test ----------------------------------------------------------
echo ">>> selftest.sh $*"
sh "$DEPLOY_DIR/selftest.sh" "$@"
SELFTEST_RC=$?
echo ""

# ---- clean up ---------------------------------------------------------------
# selftest is done (pass or fail) — we have the report in Downloads.
# Remove the working tree and the original tar so nothing lingers on the phone.
echo ">>> cleanup"
cd "$HOME"   # leave ~/glados before removing it
rm -rf "$GLADOS_DIR"
echo "removed $GLADOS_DIR"

if [ -f "$BUNDLE_TAR" ]; then
  rm -f "$BUNDLE_TAR"
  echo "removed $BUNDLE_TAR"
else
  echo "(bundle tar not found at $BUNDLE_TAR — skipping)"
fi

echo ""
echo "Done. Collect your report from Android Downloads."
exit $SELFTEST_RC
