# Build the phone self-test bundle (Windows / PowerShell). Mirror of
# make-phone-bundle.sh — both just wrap `git archive`, so no `sh` is needed on
# the dev box. Produces ONE .tar.gz with the client_room package + deploy scripts
# + self-test, containing only TRACKED files (never your gitignored
# config.room.toml / tokens / env), with the committed LF + exec bits intact.
#
#   pwsh client_room\deploy\termux\make-phone-bundle.ps1 [-Out path.tar.gz]
param([string]$Out)
$ErrorActionPreference = 'Stop'

$root = (git rev-parse --show-toplevel).Trim()
if (-not $Out) { $Out = Join-Path $root 'dist\glados-phone-bundle.tar.gz' }
New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null

# --prefix=glados/ so it extracts to ./glados/ (-> ~/glados when run in $HOME,
# the GLADOS_ROOM_DIR default the self-test infers).
git -C $root archive --format=tar.gz --prefix=glados/ HEAD client_room -o $Out

Write-Host "wrote $Out ($((Get-Item $Out).Length) bytes)"
Write-Host ""
Write-Host "Copy it to the phone, then in Termux:"
Write-Host "  tar xzf $(Split-Path $Out -Leaf) -C `$HOME       # -> ~/glados/"
Write-Host "  sh ~/glados/client_room/deploy/termux/install.sh    # installs all deps"
Write-Host "  sh ~/glados/client_room/deploy/termux/selftest.sh   # run the checks"
