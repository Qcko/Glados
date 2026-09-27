<#
.SYNOPSIS
  Open Edge on the Dunnes login page with the prod GLaDOS browser profile, so
  a human can log in once for the service account. Run it AS that account
  from an interactive desktop:

    runas /user:glados-svc "powershell -NoProfile -ExecutionPolicy Bypass -File <root>\bin\dunnes-login.ps1"

  Log in, then close Edge completely: GLaDOS opens its own Edge on the same
  profile, and two Edge processes cannot share one. The profile's cookies are
  encrypted for the account that wrote them, which is why this cannot be done
  as any other user or copied from another machine.

  Windows PowerShell 5.1 compatible.
#>
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$LoginUrl = 'https://www.dunnesstoresgrocery.com/sm/delivery/rsid/255/login'

function Find-Edge {
    foreach ($base in $env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:LOCALAPPDATA) {
        if (-not $base) { continue }
        $exe = Join-Path $base 'Microsoft\Edge\Application\msedge.exe'
        if (Test-Path $exe) { return $exe }
    }
    throw 'msedge.exe not found in the standard install locations'
}

$profileDir = Join-Path $Root 'secrets\dunnes-edge-profile'
Start-Process -FilePath (Find-Edge) -ArgumentList @(
    "--user-data-dir=$profileDir", '--disable-sync', '--disable-features=msImplicitSignin',
    '--no-first-run', '--no-default-browser-check', $LoginUrl)
