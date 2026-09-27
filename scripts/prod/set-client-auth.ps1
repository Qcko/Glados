<#
.SYNOPSIS
  Store a client's auth token in the service account's keyring from a file,
  and read it back to prove it. Run it AS the service account:

    runas /user:glados-svc "powershell -NoProfile -ExecutionPolicy Bypass -File <root>\bin\set-client-auth.ps1 -ClientId prod-desk-ui"

  Reads <root>\secrets\<ClientId>.token. From a file rather than a prompt
  because the prompt hides input and does not confirm it: a paste into a
  console can arrive as a control character and be stored without any sign.

  Windows PowerShell 5.1 compatible.
#>
param([Parameter(Mandatory = $true)][ValidatePattern('^[a-z0-9-]+$')][string]$ClientId)
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $Root 'current\.venv\Scripts\python.exe'
$source = Join-Path $Root "secrets\$ClientId.token"

$code = @'
import sys, keyring
client_id, path = sys.argv[1], sys.argv[2]
value = open(path, encoding="utf-8").read().strip()
if not value:
    sys.exit("the file is empty")
keyring.set_password("glados.client-tokens", client_id, value)
stored = keyring.get_password("glados.client-tokens", client_id)
print("stored and verified" if stored == value else "MISMATCH after store")
sys.exit(0 if stored == value else 1)
'@

# Through a file, not `python -c`: Windows PowerShell strips the double
# quotes inside a native command's argument.
$script = Join-Path $env:TEMP "glados-set-client-auth-$PID.py"
Set-Content -Path $script -Value $code -Encoding ascii
try { & $python $script $ClientId $source } finally { Remove-Item $script -ErrorAction SilentlyContinue }
Write-Host "exit $LASTEXITCODE -- press Enter to close"
[void](Read-Host)
