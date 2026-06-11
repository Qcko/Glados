#!/data/data/com.termux/files/usr/bin/sh
# Second stage of the GLaDOS phone bundle. Normally called by the self-extracting
# launcher (go.sh), which handles extraction and removes itself afterwards. Can
# also be run directly after a manual tar extraction.
#
# Dispatches one of the two secret-free bring-up paths (the third, enroll, is
# secret-bearing and runs separately — see enroll.sh):
#
#   doctor   (default)  deps + selftest, then delete ~/glados so nothing lingers.
#                       "Is this device healthy?" Read-mostly, re-runnable.
#   install             deps + provision (wire runit services + Termux:Boot,
#                       left DOWN), then LEAVE ~/glados in place. "Make this
#                       device a GLaDOS appliance." Give it an identity next
#                       with enroll.sh.
#
# Usage:
#   sh ~/glados/client_room/deploy/termux/dispatch.sh [doctor|install] [extra args]
#   # doctor forwards extra args to selftest.sh (e.g. --non-interactive, --client)
set -u

MODE=doctor
case "${1:-}" in
  doctor)  MODE=doctor;  shift ;;
  install) MODE=install; shift ;;
  -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
  -*) MODE=doctor ;;  # leading flag (e.g. --non-interactive) → doctor, forwarded to selftest
  "") : ;;            # no arg → default doctor
  *) echo "dispatch: unknown mode '$1' (expected doctor|install)" >&2; exit 2 ;;
esac

DEPLOY_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
GLADOS_DIR=$(CDPATH= cd -- "$DEPLOY_DIR/../../.." && pwd)

echo "=== GLaDOS phone bundle — $MODE ==="
echo "deploy dir : $DEPLOY_DIR"
echo "glados dir : $GLADOS_DIR"
echo ""

# ---- deps (shared by both paths) -------------------------------------------
echo ">>> deps.sh"
sh "$DEPLOY_DIR/deps.sh"
DEPS_RC=$?
echo ""
if [ $DEPS_RC -ne 0 ]; then
  echo "deps.sh failed (exit $DEPS_RC) — aborting $MODE." >&2
  exit $DEPS_RC
fi

if [ "$MODE" = install ]; then
  # ---- install: wire services, keep the tree --------------------------------
  echo ">>> provision.sh (wire services, leave them DOWN until enroll)"
  sh "$DEPLOY_DIR/provision.sh"
  PROV_RC=$?
  echo ""
  if [ $PROV_RC -ne 0 ]; then
    echo "provision.sh failed (exit $PROV_RC) — aborting install." >&2
    exit $PROV_RC
  fi
  echo "Installed. The client lives at $GLADOS_DIR and services are wired but DOWN."
  echo "Give this device its identity (config + tokens) and start it with:"
  echo "  sh $DEPLOY_DIR/enroll.sh"
  exit 0
fi

# ---- doctor: selftest, then remove the tree --------------------------------
echo ">>> selftest.sh $*"
sh "$DEPLOY_DIR/selftest.sh" "$@"
SELFTEST_RC=$?
echo ""

# selftest is done (pass or fail) — the report is already in Downloads. Remove
# the working tree so nothing lingers on the phone. (go.sh removed itself from
# Downloads in its own header, so there's nothing else to clean up here.)
echo ">>> cleanup"
cd "$HOME"   # leave ~/glados before removing it
rm -rf "$GLADOS_DIR"
echo "removed $GLADOS_DIR"

echo ""
echo "Done. Collect your report from Android Downloads."
exit $SELFTEST_RC
