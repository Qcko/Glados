# Build the GLaDOS phone bundle (Windows / PowerShell). Mirror of
# make-phone-bundle.sh -- both wrap `git archive`, so no `sh` is needed on
# the dev box. Produces a self-extracting go.sh (and a raw .tar.gz alongside
# it) containing only TRACKED files (never your gitignored config.room.toml /
# tokens / env), with the committed LF + exec bits intact.
#
#   powershell -File client_room\deploy\termux\make-phone-bundle.ps1 [-OutDir path]
param([string]$OutDir)
$ErrorActionPreference = 'Stop'

$root = (git rev-parse --show-toplevel).Trim()
if (-not $OutDir) { $OutDir = Join-Path $root 'dist' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$tar = Join-Path $OutDir 'glados-phone-bundle.tar.gz'
$go  = Join-Path $OutDir 'go.sh'

# --prefix=glados/ so it extracts to ./glados/ (-> ~/glados when run in $HOME).
git -C $root archive --format=tar.gz --prefix=glados/ HEAD client_room -o $tar

# Build the self-extracting go.sh: LF-only header bytes + raw tar appended.
# tail -n +N skips the N-1 header lines then streams raw bytes (binary-safe).
$header = "#!/data/data/com.termux/files/usr/bin/sh`n" +
          "# GLaDOS phone bundle -- self-extracting. Copy to the phone, then in Termux`n" +
          "# (after termux-setup-storage):`n" +
          "#   sh ~/storage/downloads/go.sh            # doctor: health-check, then self-clean`n" +
          "#   sh ~/storage/downloads/go.sh install    # install as a persistent appliance`n" +
          "# After install, enroll:  sh ~/glados/client_room/deploy/termux/enroll.sh`n" +
          'SKIP=$(awk ''/^#__BUNDLE__$/{print NR+1; exit}'' "$0")' + "`n" +
          '[ -n "$SKIP" ] || { echo "go.sh: bundle marker not found -- the file is truncated/corrupted; re-copy it." >&2; exit 1; }' + "`n" +
          'tail -n +$SKIP "$0" | tar xzf - -C "$HOME"' + "`n" +
          'sh "$HOME/glados/client_room/deploy/termux/dispatch.sh" "$@"' + "`n" +
          'rc=$?' + "`n" +
          'rm -f "$0"' + "`n" +
          'exit $rc' + "`n" +
          "#__BUNDLE__`n"

$headerBytes = [System.Text.Encoding]::UTF8.GetBytes($header)
$tarBytes    = [System.IO.File]::ReadAllBytes($tar)

$stream = [System.IO.File]::Open($go, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
try {
    $stream.Write($headerBytes, 0, $headerBytes.Length)
    $stream.Write($tarBytes,    0, $tarBytes.Length)
} finally {
    $stream.Close()
}

Write-Host "wrote $tar ($((Get-Item $tar).Length) bytes)"
Write-Host "wrote $go ($((Get-Item $go).Length) bytes)"
Write-Host ""
Write-Host "Copy go.sh to the phone, then in Termux:"
Write-Host "  termux-setup-storage                        # once, if not done"
Write-Host "  sh ~/storage/downloads/go.sh                # doctor: health-check + self-clean"
Write-Host "  sh ~/storage/downloads/go.sh install        # install persistent appliance, then enroll.sh"
