#!/usr/bin/env sh
# Generate a self-signed TLS cert for the GLaDOS server (wss:// / https://).
#
# The server is a self-hosted home appliance with no public CA. Clients PIN this
# exact cert (trust the cert file directly), so there is no CA to guard — see
# client_room/deploy/ROADMAP.md and ARCHITECTURE.md §9. Re-run with --force to
# rotate; every client that pins the cert must then get the new cert file.
#
# Usage:
#   sh scripts/gen-tls-cert.sh [--force] <san> [<san> ...]
#
# Each <san> is a hostname or IP the server is reached at; localhost + 127.0.0.1
# are always included. Example (this machine's LAN IP):
#   sh scripts/gen-tls-cert.sh 192.168.50.176
#
# Writes configs/tls/key.pem (mode 600) + configs/tls/cert.pem (both gitignored).
set -eu

FORCE=0
SANS=""
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) SANS="$SANS $arg" ;;
  esac
done

ROOT=$(git rev-parse --show-toplevel)
OUT="$ROOT/configs/tls"
KEY="$OUT/key.pem"
CRT="$OUT/cert.pem"

if [ "$FORCE" -eq 0 ] && [ -f "$KEY" ]; then
  echo "refusing to overwrite existing $KEY (clients pin it); pass --force to rotate." >&2
  exit 1
fi

mkdir -p "$OUT"

# Build the subjectAltName list: always localhost + loopback, plus each arg.
# IPs go in as IP:, everything else as DNS:.
alt="DNS:localhost,IP:127.0.0.1"
for s in $SANS; do
  case "$s" in
    *[!0-9.]*) alt="$alt,DNS:$s" ;;   # has a non-digit/dot char -> hostname
    *)         alt="$alt,IP:$s"  ;;   # all digits and dots -> IPv4
  esac
done

echo "Generating self-signed cert"
echo "  subjectAltName = $alt"

umask 077  # key.pem must not be group/other-readable
# MSYS_NO_PATHCONV=1 stops Git-Bash on Windows from rewriting the leading-slash
# -subj ("/CN=glados") into a Windows path; harmless on real POSIX shells.
MSYS_NO_PATHCONV=1 openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "$KEY" -out "$CRT" \
  -days 825 -subj "/CN=glados" \
  -addext "subjectAltName=$alt" >/dev/null 2>&1

chmod 600 "$KEY"
chmod 644 "$CRT"

echo "wrote $KEY (mode 600)"
echo "wrote $CRT"
echo
echo "Server: point GLADOS_TLS_CERT/GLADOS_TLS_KEY (or [server] tls_* in"
echo "glados.toml) at these, then launch. Clients: distribute cert.pem and"
echo "set tls_ca to it (config.room.toml) / import it into the browser trust store."
