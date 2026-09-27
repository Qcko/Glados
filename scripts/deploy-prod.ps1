<#
.SYNOPSIS
  Ship a tagged release from the devbox to prod.

  deploy-prod.ps1 -Tag prod-2026-09-27.01     build + deploy that tag
  deploy-prod.ps1 -Status | -Rollback | -Restart
  deploy-prod.ps1 -Dunnes <DunnesStoresMCP checkout>   ship a Dunnes build

  The tag must already be pushed and on origin/main. The desk client is built
  HERE, from a clean checkout of the tag, and piped to prod over the deploy
  key, which can run nothing but prod's deploy-release.ps1. The SSH host
  comes from -SshHost or GLADOS_PROD_SSH_HOST (an ~/.ssh/config alias that
  uses the deploy key).

  Windows PowerShell 5.1 compatible. See DEPLOY.md.
#>
param(
    [string]$Tag = '',
    [switch]$Status,
    [switch]$Rollback,
    [switch]$Restart,
    [string]$Dunnes = '',
    [string]$SshHost = $env:GLADOS_PROD_SSH_HOST
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3

$TagPattern = '^prod-\d{4}-\d{2}-\d{2}\.\d{2}$'
$Ssh = Join-Path $env:SystemRoot 'System32\OpenSSH\ssh.exe'
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Main {
    if (-not $SshHost) { throw 'set GLADOS_PROD_SSH_HOST or pass -SshHost' }
    if ($Status) { Invoke-Remote 'status' $null; return }
    if ($Rollback) { Invoke-Remote 'rollback' $null; return }
    if ($Restart) { Invoke-Remote 'restart' $null; return }
    if ($Dunnes) { Publish-Dunnes $Dunnes; return }
    if (-not $Tag) { throw 'pass -Tag, -Status, -Rollback, -Restart or -Dunnes' }
    Publish-Release $Tag
}

function Publish-Release([string]$tag) {
    if ($tag -notmatch $TagPattern) { throw "tag '$tag' does not match $TagPattern" }
    Assert-TagShippable $tag
    $zip = Build-DeskClient $tag
    try {
        Invoke-Remote "deploy $tag $(Get-Sha $zip)" $zip
    } finally {
        Remove-Item $zip -ErrorAction SilentlyContinue
    }
}

function Assert-TagShippable([string]$tag) {
    Invoke-Git @('fetch', '--tags', 'origin')
    & git -C $RepoRoot merge-base --is-ancestor "refs/tags/$tag^{commit}" 'refs/remotes/origin/main'
    if ($LASTEXITCODE -ne 0) { throw "tag $tag is missing or not on origin/main; push main and the tag first" }
    $remote = & git -C $RepoRoot ls-remote --tags origin "refs/tags/$tag"
    if (-not $remote) { throw "tag $tag is not pushed to origin" }
}

function Build-DeskClient([string]$tag) {
    $work = Join-Path $env:TEMP "glados-release-$tag"
    if (Test-Path $work) { Remove-Worktree $work }
    Invoke-Git @('worktree', 'add', '--detach', $work, "refs/tags/$tag")
    try {
        $web = Join-Path $work 'client_web'
        Invoke-Npm $web @('ci', '--ignore-scripts')
        Invoke-Npm $web @('run', 'build')
        $zip = Join-Path $env:TEMP "glados-desk-$tag.zip"
        if (Test-Path $zip) { Remove-Item $zip }
        Compress-Archive -Path (Join-Path $web 'dist\*') -DestinationPath $zip
        return $zip
    } finally {
        Remove-Worktree $work
    }
}

function Publish-Dunnes([string]$checkout) {
    $publish = Join-Path $env:TEMP 'glados-dunnes-publish'
    if (Test-Path $publish) { Remove-Item -Recurse -Force $publish }
    $project = Join-Path $checkout 'DunnesStoresMCP\McpServer'
    & dotnet publish $project -c Release -r win-x64 --self-contained false -o $publish
    if ($LASTEXITCODE -ne 0) { throw 'dotnet publish of the Dunnes server failed' }
    $zip = Join-Path $env:TEMP 'glados-dunnes.zip'
    if (Test-Path $zip) { Remove-Item $zip }
    Compress-Archive -Path (Join-Path $publish '*') -DestinationPath $zip
    try {
        Invoke-Remote "dunnes $(Get-Sha $zip)" $zip
    } finally {
        Remove-Item $zip -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force $publish -ErrorAction SilentlyContinue
    }
}

function Invoke-Remote([string]$command, [string]$payload) {
    if ($payload) {
        & cmd.exe /d /c "`"$Ssh`" -T $SshHost $command < `"$payload`""
    } else {
        & $Ssh -T $SshHost $command
    }
    if ($LASTEXITCODE -ne 0) { throw "prod '$($command.Split(' ')[0])' failed ($LASTEXITCODE)" }
}

function Get-Sha([string]$path) {
    return (Get-FileHash -Algorithm SHA256 -Path $path).Hash.ToLowerInvariant()
}

function Remove-Worktree([string]$path) {
    & git -C $RepoRoot worktree remove --force $path 2>$null
    if (Test-Path $path) { Remove-Item -Recurse -Force $path }
    & git -C $RepoRoot worktree prune
}

function Invoke-Git([string[]]$gitArgs) {
    & git -C $RepoRoot @gitArgs
    if ($LASTEXITCODE -ne 0) { throw "git $($gitArgs[0]) failed ($LASTEXITCODE)" }
}

function Invoke-Npm([string]$dir, [string[]]$npmArgs) {
    Push-Location $dir
    try {
        & npm.cmd @npmArgs
        if ($LASTEXITCODE -ne 0) { throw "npm $($npmArgs[0]) failed ($LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
}

Main
