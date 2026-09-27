<#
.SYNOPSIS
  One-off, idempotent setup of a GLaDOS production box. Run as an
  administrator ON the prod box. Safe to re-run: every step skips what
  already exists.

.EXAMPLE
  .\bootstrap.ps1 -Root <root> -HostNames <name>,<ip>

  Then, in an interactive session on the box (it prompts for the service
  account's password, which only you type):
    <root>\bin\bootstrap.ps1 -Root <root> -HostNames ... -RegisterTask

  Windows PowerShell 5.1 compatible. See DEPLOY.md.
#>
param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string[]]$HostNames,
    [string]$RepoUrl = 'https://github.com/Qcko/Glados.git',
    [string]$ServiceUser = 'glados-svc',
    [string]$DeployKeyFile = '',
    [switch]$RegisterTask
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3

$TaskName = 'GLaDOS'
$Port = 8765

function Main {
    Assert-Admin
    Assert-ServiceUser
    New-Layout
    Get-Repo
    Install-Bin
    Grant-Access
    Set-GitSafeDirectory
    Set-UvLocations
    Write-LocalConfig
    New-TlsCert
    Open-Firewall
    if ($DeployKeyFile) { Add-DeployKey }
    if ($RegisterTask) { Register-GladosTask }
    Write-NextSteps
}

function Assert-Admin {
    $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'run this as an administrator'
    }
}

function Assert-ServiceUser {
    if (-not (Get-LocalUser -Name $ServiceUser -ErrorAction SilentlyContinue)) {
        throw "local user '$ServiceUser' does not exist. Create it yourself (a standard, non-admin user with a password), then re-run."
    }
}

function New-Layout {
    foreach ($dir in 'bin', 'releases', 'local', 'tls', 'state\traces', 'logs', 'cache\hf',
                     'cache\piper\voices', 'cache\uv', 'cache\uv-python', 'dunnes', 'secrets') {
        New-Item -ItemType Directory -Force -Path (Join-Path $Root $dir) | Out-Null
    }
}

function Get-Repo {
    $repo = Join-Path $Root 'repo'
    if (Test-Path (Join-Path $repo '.git')) { return }
    & git clone $RepoUrl $repo
    if ($LASTEXITCODE -ne 0) { throw "git clone $RepoUrl failed" }
}

function Install-Bin {
    Copy-Item -Path (Join-Path $Root 'repo\scripts\prod\*.ps1') -Destination (Join-Path $Root 'bin') -Force
}

function Grant-Access {
    # Only admins change code; the service account reads it and writes only state.
    Invoke-Icacls $Root @('/inheritance:r', '/grant:r', 'Administrators:(OI)(CI)F', 'SYSTEM:(OI)(CI)F',
                          "${ServiceUser}:(OI)(CI)RX")
    foreach ($dir in 'state', 'logs', 'cache\hf', 'cache\piper') {
        Invoke-Icacls (Join-Path $Root $dir) @('/grant', "${ServiceUser}:(OI)(CI)M")
    }
    Invoke-Icacls (Join-Path $Root 'secrets') @('/grant', "${ServiceUser}:(OI)(CI)M")
}

function Invoke-Icacls([string]$path, [string[]]$icaclsArgs) {
    & icacls $path @icaclsArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "icacls $path failed" }
}

function Set-GitSafeDirectory {
    # The service account reads git metadata (healthz reports the release) from
    # a tree an admin owns; git refuses that unless the path is marked safe.
    $pattern = ($Root -replace '\\', '/') + '/*'
    $existing = @(& git config --system --get-all safe.directory 2>$null)
    if ($existing -notcontains $pattern) { & git config --system --add safe.directory $pattern }
}

function Set-UvLocations {
    [Environment]::SetEnvironmentVariable('UV_CACHE_DIR', (Join-Path $Root 'cache\uv'), 'Machine')
    [Environment]::SetEnvironmentVariable('UV_PYTHON_INSTALL_DIR', (Join-Path $Root 'cache\uv-python'), 'Machine')
}

function Write-LocalConfig {
    $local = Join-Path $Root 'local'
    Write-IfMissing (Join-Path $local 'glados.local.toml') (Get-GladosOverlay)
    Write-IfMissing (Join-Path $local 'rooms.local.toml') (Get-RoomsOverlay)
    Write-IfMissing (Join-Path $local 'servers.toml') (Get-ServersToml)
}

function Get-GladosOverlay {
    $traces = Join-Path $Root 'state\traces'
    return @"
# This box's overrides, merged over the release's configs/glados.toml.
# Tables merge, arrays append, other values replace.
[server]
traces_dir = '$traces'

[auth]
clients = ["prod-desk-ui"]

[audio]
wav_traces = false
"@
}

function Get-RoomsOverlay {
    return @'
# Clients only this box serves, appended to the release's configs/rooms.toml.
[[clients]]
client_id = "prod-desk-ui"
room_id = "prod-desk"
role = "ui"
default_user = "qcko"
'@
}

function Get-ServersToml {
    $dll = Join-Path $Root 'dunnes\current\McpServer.dll'
    $profileDir = Join-Path $Root 'secrets\dunnes-edge-profile'
    $example = Get-Content -Raw (Join-Path $Root 'repo\configs\servers.example.toml')
    $example = $example -replace '"<path-to>/DunnesStoresMCP/[^"]*McpServer\.dll"', "'$dll'"
    return $example -replace '"<your-secrets-dir>/dunnes-edge-profile"', "'$profileDir'"
}

function Write-IfMissing([string]$path, [string]$content) {
    if (Test-Path $path) { return }
    Set-Content -Path $path -Value $content -Encoding ascii
}

function New-TlsCert {
    $tls = Join-Path $Root 'tls'
    if (Test-Path (Join-Path $tls 'key.pem')) { return }
    $openssl = Join-Path (Split-Path -Parent (Split-Path -Parent (Get-Command git).Source)) 'usr\bin\openssl.exe'
    $san = (Get-HostNameList | ForEach-Object { if ($_ -match '^[\d.]+$') { "IP:$_" } else { "DNS:$_" } }) -join ','
    & $openssl req -x509 -newkey rsa:2048 -nodes -days 825 -subj "/CN=$((Get-HostNameList)[0])" `
        -addext "subjectAltName=$san" -keyout (Join-Path $tls 'key.pem') -out (Join-Path $tls 'cert.pem')
    if ($LASTEXITCODE -ne 0) { throw 'openssl failed to create the TLS certificate' }
    Invoke-Icacls (Join-Path $tls 'key.pem') @('/inheritance:r', '/grant:r', 'Administrators:F', "${ServiceUser}:R")
}

function Get-HostNameList {
    # `powershell -File` passes "a,b" as ONE string, not an array.
    return @($HostNames | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Open-Firewall {
    if (Get-NetFirewallRule -DisplayName $TaskName -ErrorAction SilentlyContinue) { return }
    New-NetFirewallRule -DisplayName $TaskName -Direction Inbound -Protocol TCP -LocalPort $Port `
        -Profile Private -RemoteAddress LocalSubnet -Action Allow | Out-Null
}

function Add-DeployKey {
    $keys = Join-Path $env:ProgramData 'ssh\administrators_authorized_keys'
    $pub = (Get-Content -Raw $DeployKeyFile).Trim()
    if ((Test-Path $keys) -and (Select-String -Path $keys -SimpleMatch $pub -Quiet)) { return }
    $entry = Join-Path $Root 'bin\deploy-release.ps1'
    $forced = "command=`"powershell.exe -NoProfile -ExecutionPolicy Bypass -File $entry`"," +
              'no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-pty'
    Add-Content -Path $keys -Value "$forced $pub" -Encoding ascii
}

function Register-GladosTask {
    $cred = Get-Credential -UserName $ServiceUser -Message 'Password of the GLaDOS service account'
    $run = Join-Path $Root 'bin\run-glados.ps1'
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$run`""
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $trigger.Delay = 'PT1M'
    $settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
        -User $cred.UserName -Password $cred.GetNetworkCredential().Password -Force | Out-Null
}

function Write-NextSteps {
    Write-Output "Bootstrap done under $Root."
    Write-Output 'Remaining by hand (DEPLOY.md, "First deploy"): create the model, set client tokens'
    Write-Output 'as the service account, install the Dunnes build, then deploy the first tag.'
}

Main
