<#
.SYNOPSIS
  The prod scheduled task's action: run the release behind <root>\current
  with this box's settings. Every machine-specific value derives from the
  prod root (this script lives in <root>\bin), so nothing here names a path.

  Runs the release's own venv directly rather than `uv run`: the service
  account only reads the release, and `uv run` may try to write the venv.

  Windows PowerShell 5.1 compatible.
#>
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$Current = Join-Path $Root 'current'

$env:GLADOS_CONFIG_DIR = Join-Path $Current 'configs'
$env:GLADOS_LOCAL_CONFIG_DIR = Join-Path $Root 'local'
$env:GLADOS_LOG_DIR = Join-Path $Root 'logs'
$env:GLADOS_HOST = '0.0.0.0'
$env:GLADOS_TLS_CERT = Join-Path $Root 'tls\cert.pem'
$env:GLADOS_TLS_KEY = Join-Path $Root 'tls\key.pem'
$env:GLADOS_OLLAMA_AUTOSTART = '0'
$env:HF_HOME = Join-Path $Root 'cache\hf'
$env:GLADOS_PIPER_VOICES_DIR = Join-Path $Root 'cache\piper\voices'

$Inbox = Join-Path $Root 'secrets\inbox'
$IngestLog = Join-Path $Root 'logs\secrets-ingest.log'
$IngestTimeoutMs = 30000

# Client tokens dropped in the inbox are stored in THIS account's keyring
# (DESIGN.md, "registering client tokens without the service password").
# Never blocks the server: any failure is logged and the server starts anyway.
function Invoke-SecretsIngest {
    try {
        if (-not (Get-ChildItem -Path $Inbox -Filter '*.token' -File -ErrorAction SilentlyContinue)) { return }
        $refusal = Get-InboxRefusal
        if ($refusal) { Write-IngestLog "refused, nothing ingested: $refusal"; return }
        Start-Ingest
    } catch {
        Write-IngestLog "ingest failed: $($_.Exception.Message)"
    }
}

function Get-InboxRefusal {
    $item = Get-Item -LiteralPath $Inbox -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { return 'inbox is a reparse point' }
    $acl = Get-Acl -LiteralPath $Inbox
    if (-not $acl.AreAccessRulesProtected) { return 'inbox inherits its ACL' }
    $self = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $trusted = @('S-1-5-32-544', 'S-1-5-18')
    $creating = [Security.AccessControl.FileSystemRights]'CreateFiles, CreateDirectories, ChangePermissions, TakeOwnership'
    foreach ($rule in $acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) {
        if ($rule.AccessControlType -ne 'Allow') { continue }
        $sid = $rule.IdentityReference.Value
        if ($trusted -contains $sid) { continue }
        if ($sid -eq $self -and -not ($rule.FileSystemRights -band $creating)) { continue }
        return "inbox grants $($rule.FileSystemRights) to $sid"
    }
    return ''
}

function Start-Ingest {
    $python = Join-Path $Current '.venv\Scripts\python.exe'
    $summary = Join-Path $Root 'state\secrets-ingest.json'
    $line = "/d /c `"`"$python`" -m glados.secrets ingest `"$Inbox`" --summary `"$summary`" >> `"$IngestLog`" 2>&1`""
    Write-IngestLog 'ingest started'
    $process = Start-Process -FilePath 'cmd.exe' -ArgumentList $line -NoNewWindow -PassThru
    $null = $process.Handle
    if ($process.WaitForExit($IngestTimeoutMs)) {
        Write-IngestLog "ingest exited $($process.ExitCode)"
        return
    }
    & cmd.exe /d /c "taskkill /T /F /PID $($process.Id) >nul 2>&1"
    Write-IngestLog "ingest killed after $($IngestTimeoutMs / 1000)s"
}

function Write-IngestLog([string]$message) {
    Add-Content -Path $IngestLog -Value "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') $message" -Encoding ascii -ErrorAction SilentlyContinue
}

$stdout = Join-Path $Root 'logs\stdout.log'
$exe = Join-Path $Current '.venv\Scripts\glados.exe'
Set-Location $Current
Invoke-SecretsIngest
# Through cmd so the server's stderr is appended as text: Windows PowerShell
# turns a native stderr line into an error record under redirection.
& cmd.exe /d /c "`"$exe`" >> `"$stdout`" 2>&1"
exit $LASTEXITCODE
