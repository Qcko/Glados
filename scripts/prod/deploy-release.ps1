<#
.SYNOPSIS
  Production-side deploy entrypoint. Runs ON the prod box, invoked only by
  the deploy SSH key's forced command; the verb and its arguments arrive in
  SSH_ORIGINAL_COMMAND and are validated here, never executed as a string.

  Verbs:
    deploy <tag> <sha256>   install release <tag>; the built desk client zip
                            arrives on stdin and must hash to <sha256>
    dunnes <sha256>         install a Dunnes MCP server build from stdin
    rollback                return to the previous good release
    restart                 drain and restart the current release
    status                  print the current release and healthz

  Layout under the prod root (this script lives in <root>\bin):
    repo\  releases\<tag>\  current (junction)  local\  tls\  state\  logs\
    dunnes\<sha>\  dunnes\current (junction)  last-good.txt  deploy.lock

  Windows PowerShell 5.1 compatible.
#>
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3

$Root = Split-Path -Parent $PSScriptRoot
$TaskName = 'GLaDOS'
$HealthUrl = 'https://127.0.0.1:8765/healthz'
$ShutdownUrl = 'https://127.0.0.1:8765/admin/shutdown?timeout_s=120'
$TagPattern = '^prod-\d{4}-\d{2}-\d{2}\.\d{2}$'
$ShaPattern = '^[0-9a-f]{64}$'
$ReadyTimeoutS = 300
$StopTimeoutS = 90
$KeepReleases = 5

# Machine-level env set by bootstrap is not seen by an sshd started before it,
# and a python uv installs under this admin's profile is unreadable to the
# service account. Pin both to the prod root for every run.
$env:UV_CACHE_DIR = Join-Path $Root 'cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root 'cache\uv-python'

function Main {
    $words = @(Split-Words $env:SSH_ORIGINAL_COMMAND)
    if ($words.Count -eq 0) { Fail 'no command; expected deploy|dunnes|rollback|restart|status' }
    $lock = Enter-DeployLock
    try {
        switch ($words[0]) {
            'deploy'   { Assert-Arity $words 3; Invoke-Deploy (Assert-Tag $words[1]) (Assert-Sha $words[2]) }
            'dunnes'   { Assert-Arity $words 2; Invoke-DunnesInstall (Assert-Sha $words[1]) }
            'rollback' { Assert-Arity $words 1; Invoke-Rollback }
            'restart'  { Assert-Arity $words 1; Restart-Current }
            'status'   { Assert-Arity $words 1; Show-Status }
            default    { Fail "unknown verb '$($words[0])'" }
        }
    } finally {
        $lock.Dispose()
    }
}

function Split-Words([string]$line) {
    if (-not $line) { return @() }
    return @($line.Trim() -split '\s+')
}

function Assert-Arity($words, [int]$count) {
    if ($words.Count -ne $count) { Fail "'$($words[0])' takes $($count - 1) argument(s)" }
}

function Assert-Tag([string]$tag) {
    if ($tag -notmatch $TagPattern) { Fail "tag '$tag' does not match $TagPattern" }
    return $tag
}

function Assert-Sha([string]$sha) {
    $sha = $sha.ToLowerInvariant()
    if ($sha -notmatch $ShaPattern) { Fail 'sha256 must be 64 hex characters' }
    return $sha
}

function Enter-DeployLock {
    $path = Join-Path $Root 'deploy.lock'
    try {
        return [System.IO.File]::Open($path, 'OpenOrCreate', 'ReadWrite', 'None')
    } catch {
        Fail 'another deploy is running (deploy.lock is held)'
    }
}

# ---- deploy ------------------------------------------------------------

function Invoke-Deploy([string]$tag, [string]$sha) {
    $release = Install-Release $tag $sha
    $previous = Get-CurrentRelease
    Stop-Glados
    Set-Junction (Join-Path $Root 'current') $release
    Start-ScheduledTask -TaskName $TaskName
    if (Wait-Ready $tag) {
        Set-Content -Path (Join-Path $Root 'last-good.txt') -Value $tag -Encoding ascii
        Update-Bin $release
        Remove-OldReleases
        Write-Output "DEPLOYED $tag"
        return
    }
    Write-Output "release $tag did not become ready; rolling back"
    New-Item -ItemType File -Force -Path (Join-Path $release '.failed') | Out-Null
    Restore-Release $previous
    Fail "deploy of $tag failed; see $(Join-Path $Root 'logs')"
}

function Install-Release([string]$tag, [string]$sha) {
    $release = Join-Path $Root "releases\$tag"
    $zip = Receive-Stdin $sha
    if (Test-Path (Join-Path $release '.ready')) {
        Write-Host "release $tag already built; reusing it"
        Remove-Item $zip
        Remove-Item (Join-Path $release '.failed') -ErrorAction SilentlyContinue
        return $release
    }
    $repo = Join-Path $Root 'repo'
    Invoke-Git $repo @('fetch', '--prune', '--tags', '--force', 'origin')
    Assert-TagOnMain $repo $tag
    if (Test-Path $release) { Remove-Worktree $repo $release }
    Invoke-Git $repo @('worktree', 'add', '--detach', $release, "refs/tags/$tag")
    Invoke-Native $release 'uv' @('sync', '--frozen', '--no-dev')
    Expand-Archive -Path $zip -DestinationPath (Join-Path $release 'client_web\dist') -Force
    Remove-Item $zip
    New-Item -ItemType File -Path (Join-Path $release '.ready') | Out-Null
    return $release
}

function Receive-Stdin([string]$expectedSha) {
    $zip = Join-Path $Root "state\incoming-$PID.zip"
    $in = [Console]::OpenStandardInput()
    $out = [System.IO.File]::Create($zip)
    try { $in.CopyTo($out) } finally { $out.Dispose() }
    $actual = (Get-FileHash -Algorithm SHA256 -Path $zip).Hash.ToLowerInvariant()
    if ($actual -ne $expectedSha) {
        Remove-Item $zip
        Fail "stdin payload sha256 $actual does not match the expected $expectedSha"
    }
    return $zip
}

function Assert-TagOnMain([string]$repo, [string]$tag) {
    & git -C $repo merge-base --is-ancestor "refs/tags/$tag^{commit}" 'refs/remotes/origin/main'
    if ($LASTEXITCODE -ne 0) { Fail "tag $tag is not on origin/main; only commits already on main ship" }
}

function Remove-Worktree([string]$repo, [string]$path) {
    # Native stderr (git progress) must not become a terminating error.
    $ErrorActionPreference = 'Continue'
    & git -C $repo worktree remove --force $path 2>&1 | Out-Null
    if (Test-Path $path) { Remove-Item -Recurse -Force $path }
    & git -C $repo worktree prune 2>&1 | Out-Null
}

# ---- stop / start ------------------------------------------------------

function Stop-Glados {
    if ((Get-ScheduledTask -TaskName $TaskName).State -ne 'Running') { return }
    $code = Invoke-Shutdown
    if ($code -eq '409') { Fail 'prod is mid-turn and did not drain within 120s; nothing changed, try again' }
    if ($code -ne '200') { Fail "shutdown request answered $code; not killing a server that may be mid-turn" }
    if (Wait-TaskStopped) { return }
    Write-Output "GLaDOS drained but did not exit within ${StopTimeoutS}s; stopping the idle task"
    Stop-ScheduledTask -TaskName $TaskName
}

function Invoke-Shutdown {
    return (& curl.exe -sk -o NUL -w '%{http_code}' -X POST --max-time 150 $ShutdownUrl)
}

function Wait-TaskStopped {
    $deadline = (Get-Date).AddSeconds($StopTimeoutS)
    while ((Get-Date) -lt $deadline) {
        if ((Get-ScheduledTask -TaskName $TaskName).State -ne 'Running') { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Wait-Ready([string]$tag) {
    $deadline = (Get-Date).AddSeconds($ReadyTimeoutS)
    while ((Get-Date) -lt $deadline) {
        $health = Get-Health
        if ($health -and (Test-Ready $health $tag)) { return $true }
        Start-Sleep -Seconds 2
    }
    Write-Host "not ready after ${ReadyTimeoutS}s; last healthz: $(Get-Health | ConvertTo-Json -Compress -Depth 5)"
    return $false
}

function Get-Health {
    $raw = & curl.exe -sk --max-time 5 $HealthUrl
    if ($LASTEXITCODE -ne 0 -or -not $raw) { return $null }
    try { return ($raw | ConvertFrom-Json) } catch { return $null }
}

function Test-Ready($health, [string]$tag) {
    return ($health.release.tag -eq $tag) -and $health.ready.llm_warm -and `
        (@($health.ready.clients_with_token).Count -gt 0)
}

function Restore-Release([string]$previous) {
    Stop-Unready
    if (-not $previous) {
        [System.IO.Directory]::Delete((Join-Path $Root 'current'))
        Write-Output 'no previous release to return to; prod is stopped'
        return
    }
    Set-Junction (Join-Path $Root 'current') $previous
    Start-ScheduledTask -TaskName $TaskName
    $tag = Split-Path -Leaf $previous
    if (-not (Wait-Ready $tag)) {
        Write-Output "rollback to $tag is not ready either"
        return
    }
    Set-Content -Path (Join-Path $Root 'last-good.txt') -Value $tag -Encoding ascii
    Write-Output "rolled back to $tag"
}

function Stop-Unready {
    if ((Get-ScheduledTask -TaskName $TaskName).State -ne 'Running') { return }
    $code = Invoke-Shutdown
    if ($code -eq '200' -and (Wait-TaskStopped)) { return }
    Stop-ScheduledTask -TaskName $TaskName
}

function Restart-Current {
    Stop-Glados
    Start-ScheduledTask -TaskName $TaskName
    $tag = Get-CurrentName
    if (-not (Wait-Ready $tag)) { Fail "restarted $tag but it did not become ready" }
    Write-Output "RESTARTED $tag"
}

# ---- rollback / status -------------------------------------------------

function Invoke-Rollback {
    $current = Get-CurrentName
    $target = Get-ChildItem (Join-Path $Root 'releases') -Directory |
        Where-Object { $_.Name -lt $current -and (Test-Healthy $_.FullName) } |
        Sort-Object Name | Select-Object -Last 1
    if (-not $target) { Fail "no built release older than $current" }
    Stop-Glados
    Set-Junction (Join-Path $Root 'current') $target.FullName
    Start-ScheduledTask -TaskName $TaskName
    if (-not (Wait-Ready $target.Name)) { Fail "rolled back to $($target.Name) but it did not become ready" }
    Set-Content -Path (Join-Path $Root 'last-good.txt') -Value $target.Name -Encoding ascii
    Write-Output "ROLLED BACK to $($target.Name)"
}

function Test-Healthy([string]$release) {
    return (Test-Path (Join-Path $release '.ready')) -and -not (Test-Path (Join-Path $release '.failed'))
}

function Show-Status {
    $current = Get-CurrentRelease
    if ($current) { $current = Split-Path -Leaf $current } else { $current = '(none)' }
    Write-Output "current:   $current"
    Write-Output "last good: $(Get-Content (Join-Path $Root 'last-good.txt') -ErrorAction SilentlyContinue)"
    Write-Output "task:      $((Get-ScheduledTask -TaskName $TaskName).State)"
    Write-Output "healthz:   $(Get-Health | ConvertTo-Json -Compress -Depth 5)"
}

# ---- Dunnes server -----------------------------------------------------

function Invoke-DunnesInstall([string]$sha) {
    $zip = Receive-Stdin $sha
    $target = Join-Path $Root "dunnes\$($sha.Substring(0, 12))"
    if (-not (Test-Path $target)) { Expand-Archive -Path $zip -DestinationPath $target }
    Remove-Item $zip
    if (-not (Get-CurrentRelease)) {
        Set-Junction (Join-Path $Root 'dunnes\current') $target
        Write-Output "DUNNES $($sha.Substring(0, 12)) installed; no release is live yet"
        return
    }
    Stop-Glados
    Set-Junction (Join-Path $Root 'dunnes\current') $target
    Start-ScheduledTask -TaskName $TaskName
    $tag = Get-CurrentName
    if (-not (Wait-Ready $tag)) { Fail 'new Dunnes build installed but GLaDOS did not become ready' }
    Write-Output "DUNNES $($sha.Substring(0, 12)) live"
}

# ---- helpers -----------------------------------------------------------

function Get-CurrentRelease {
    $link = Get-Item (Join-Path $Root 'current') -ErrorAction SilentlyContinue
    if (-not $link) { return $null }
    return [string]$link.Target
}

function Get-CurrentName {
    $current = Get-CurrentRelease
    if (-not $current) { Fail 'no release is live yet; deploy a tag first' }
    return (Split-Path -Leaf $current)
}

function Set-Junction([string]$link, [string]$target) {
    if (Test-Path $link) { [System.IO.Directory]::Delete($link) }
    New-Item -ItemType Junction -Path $link -Target $target | Out-Null
}

function Update-Bin([string]$release) {
    Copy-Item -Path (Join-Path $release 'scripts\prod\*.ps1') -Destination $PSScriptRoot -Force
}

function Remove-OldReleases {
    $keep = @(Get-ChildItem (Join-Path $Root 'releases') -Directory | Sort-Object Name |
        Select-Object -Last $KeepReleases | ForEach-Object { $_.FullName })
    $current = Get-CurrentRelease
    Get-ChildItem (Join-Path $Root 'releases') -Directory |
        Where-Object { ($keep -notcontains $_.FullName) -and ($_.FullName -ne $current) } |
        ForEach-Object { Remove-Worktree (Join-Path $Root 'repo') $_.FullName }
}

function Invoke-Git([string]$repo, [string[]]$gitArgs) {
    # Native stderr (git progress) must not become a terminating error.
    $ErrorActionPreference = 'Continue'
    # Out-Host: a function's stray output becomes part of its caller's return value.
    & git -C $repo @gitArgs 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail "git $($gitArgs[0]) failed ($LASTEXITCODE)" }
}

function Invoke-Native([string]$dir, [string]$exe, [string[]]$exeArgs) {
    # Native stderr (git progress) must not become a terminating error.
    $ErrorActionPreference = 'Continue'
    Push-Location $dir
    try {
        & $exe @exeArgs 2>&1 | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail "$exe $($exeArgs[0]) failed ($LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
}

function Fail([string]$message) {
    [Console]::Error.WriteLine("deploy-release: $message")
    exit 1
}

Main
