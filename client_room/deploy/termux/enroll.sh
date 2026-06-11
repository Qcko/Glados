#!/data/data/com.termux/files/usr/bin/sh
# Give an installed device its identity: install the room config + token files +
# server cert, then enable and start the runit services. This is the only
# secret-bearing step (see deploy/ROADMAP.md — doctor and install stay
# secret-free; enroll is per-device and, in the target, server-minted via
# pairing). Run it AFTER `sh go.sh install` (or `sh dispatch.sh install`).
#
# Interim transfer (pre-pairing): stage these files into Android Downloads
# (~/storage/downloads), then run this script. It moves them out of shared
# storage, tightens permissions, and wipes the Downloads copies on the way out.
#   - config.room.toml           the room config (NON-secret: paths + LAN addr)
#   - <client_id>.token          one mode-600 token file per role, named exactly
#                                as config.room.toml's token_file references them
#                                (e.g. livingroom-mic.token, livingroom-speaker.token)
#   - glados-server-cert.pem     the server's PUBLIC pinned cert (never the key)
#
# A mic token grants live room-audio capture; treat the token files accordingly.
# Re-running enroll replaces the FULL token set and bounces the client so a
# rotated token takes effect immediately (no reinstall, no reboot).
#
#   sh enroll.sh
set -u

DOWNLOADS="$HOME/storage/downloads"
CFG_SRC="$DOWNLOADS/config.room.toml"
CERT_SRC="$DOWNLOADS/glados-server-cert.pem"
SECRETS_DIR="$HOME/.config/glados-room/secrets"
CERT_DEST="$HOME/.config/glados-room/glados-server-cert.pem"
: "${PREFIX:=/data/data/com.termux/files/usr}"

DEPLOY_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
GLADOS_DIR=$(CDPATH= cd -- "$DEPLOY_DIR/../../.." && pwd)
CFG_DEST="$GLADOS_DIR/client_room/config.room.toml"

die() { echo "enroll: $1" >&2; exit "${2:-1}"; }

# Wipe the secret-bearing inputs from shared storage on EVERY exit path —
# success, failure, or interrupt — so tokens never linger world-app-readable in
# Downloads. Plain rm: Android storage is file-based-encrypted and tokens are
# revocable, so shred buys nothing here.
cleanup_downloads() {
  for f in "$CFG_SRC" "$CERT_SRC" "$DOWNLOADS"/*.token; do
    [ -e "$f" ] || continue
    rm -f "$f" || echo "enroll: WARNING could not remove $f — delete it by hand (it may hold a token)" >&2
  done
}

# ---- preflight: install must have run --------------------------------------
[ -L "$PREFIX/var/service/glados-room" ] || \
  die "services not provisioned — run install first:  sh go.sh install   (or sh dispatch.sh install)"
[ -d "$GLADOS_DIR/client_room" ] || \
  die "client tree missing at $GLADOS_DIR/client_room — run install first"

# ---- preflight: required inputs present in Downloads ------------------------
MISSING=""
[ -f "$CFG_SRC" ]  || MISSING="$MISSING config.room.toml"
[ -f "$CERT_SRC" ] || MISSING="$MISSING glados-server-cert.pem"
set -- "$DOWNLOADS"/*.token
[ -f "$1" ] || MISSING="$MISSING <client_id>.token"
[ -n "$MISSING" ] && die "missing from $DOWNLOADS:$MISSING — stage all enroll inputs there first"

# Inputs are present and install is confirmed — from here we start consuming the
# staged secrets, so arm the cleanup that wipes the Downloads copies on any exit
# (success, failure, or interrupt). Before this point a preflight failure leaves
# them staged so the operator can fix a typo and retry without re-transferring.
# EXIT fires the wipe once; INT/TERM must `exit` (a bare signal handler that
# returns would let the script resume mid-consume on Ctrl-C).
trap 'cleanup_downloads' EXIT
trap 'exit 130' INT TERM

# ---- install tokens (mode 600, full-set replace) ---------------------------
# Replace the whole token set so a rotation/room move never strands an orphan
# token (which would stay valid server-side until revoked).
mkdir -p "$SECRETS_DIR" || die "could not create $SECRETS_DIR"
chmod 700 "$SECRETS_DIR" || die "could not chmod 700 $SECRETS_DIR"
rm -f "$SECRETS_DIR"/*.token
for src in "$DOWNLOADS"/*.token; do
  base=$(basename "$src")
  # `install -m 600` sets the mode AT CREATE — no umask window where the token
  # is briefly world-readable (the move out of sdcardfs is copy+unlink, so the
  # destination mode would otherwise follow the umask, typically 644).
  install -m 600 "$src" "$SECRETS_DIR/$base" || die "could not install token $base"
done

# ---- install cert (public; 644) --------------------------------------------
install -m 644 "$CERT_SRC" "$CERT_DEST" || die "could not install server cert"

# ---- install config (validate it parses before enabling) -------------------
TMP_CFG="$CFG_DEST.tmp.$$"
install -m 600 "$CFG_SRC" "$TMP_CFG" || die "could not stage config"
if ! cfg_err=$(python -c 'import tomllib,sys; tomllib.load(open(sys.argv[1],"rb"))' "$TMP_CFG" 2>&1); then
  rm -f "$TMP_CFG"
  die "config.room.toml is not valid TOML ($cfg_err) — fix it on the dev box and re-stage"
fi
mv "$TMP_CFG" "$CFG_DEST" || { rm -f "$TMP_CFG"; die "could not install config to $CFG_DEST"; }

echo "Identity installed:"
echo "  config : $CFG_DEST"
echo "  tokens : $SECRETS_DIR/ (mode 600)"
echo "  cert   : $CERT_DEST"
echo ""

# ---- enable + start (and bounce, so a rotated token applies now) -----------
echo ">>> provision.sh --enable"
sh "$DEPLOY_DIR/provision.sh" --enable || die "provision --enable failed"
# sv-enable starts a freshly-enabled service; for a re-enroll the service was
# already up with the OLD token, so bounce it to pick up the new one. Harmless
# on first enroll (just (re)starts the fresh process).
sv restart glados-pulse glados-room 2>/dev/null || true

echo ""
echo "Enrolled. Check status:  sv status glados-pulse glados-room"
echo "Client logs:             tail -f ~/.local/var/log/glados-room/current"
