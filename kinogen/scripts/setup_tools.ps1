param([Parameter(Mandatory = $true)][string]$Python)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $projectRoot '.venv'
if (-not (Test-Path -LiteralPath (Join-Path $environmentPath 'Scripts\python.exe'))) {
    & $Python -m venv --without-pip $environmentPath
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the tooling environment.' }
}
& (Join-Path $environmentPath 'Scripts\python.exe') -c 'import sys; print(sys.executable); print(sys.version)'
if ($LASTEXITCODE -ne 0) { throw 'The tooling environment could not start.' }
$toolLock = Join-Path $projectRoot 'requirements-tools.lock.txt'
if (Test-Path -LiteralPath $toolLock) {
    $uvExecutable = Get-Command uv -ErrorAction Stop
    & $uvExecutable.Source pip sync --python (Join-Path $environmentPath 'Scripts\python.exe') --cache-dir (Join-Path $projectRoot '.cache\uv') $toolLock
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install the locked download tools.' }
}
Write-Output 'Tooling environment ready. This does not install the unpublished KinoGen algorithm.'
