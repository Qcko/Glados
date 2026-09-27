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

$stdout = Join-Path $Root 'logs\stdout.log'
$exe = Join-Path $Current '.venv\Scripts\glados.exe'
Set-Location $Current
# Through cmd so the server's stderr is appended as text: Windows PowerShell
# turns a native stderr line into an error record under redirection.
& cmd.exe /d /c "`"$exe`" >> `"$stdout`" 2>&1"
exit $LASTEXITCODE
