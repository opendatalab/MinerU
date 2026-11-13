param(
    [Parameter(Mandatory = $true)][string]$Sample,
    [ValidateSet("turbomind")][string]$Backend = "turbomind",
    [string]$Python = "python",
    [string]$ModelsDir = "",
    [string]$IncidentLog = "",
    [switch]$InstallCandidates
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$bundleRoot = $PSScriptRoot
if ($InstallCandidates) {
    $candidateWheels = @(Get-ChildItem -Path (Join-Path $bundleRoot "wheels") -Filter "*.whl" | ForEach-Object { $_.FullName })
    if ($candidateWheels.Count -ne 2) { throw "Expected the two candidate wheels in wheels/." }
    & $Python -m pip install --no-deps --force-reinstall @candidateWheels
    if ($LASTEXITCODE -ne 0) { throw "Candidate installation failed." }
}

$acceptanceArgs = @((Join-Path $bundleRoot "lmdeploy_lifecycle.py"), "--sample", $Sample, "--backend", $Backend)
if ($ModelsDir) { $acceptanceArgs += @("--models-dir", $ModelsDir) }
if ($IncidentLog) { $acceptanceArgs += @("--incident-log", $IncidentLog) }
& $Python @acceptanceArgs
exit $LASTEXITCODE
