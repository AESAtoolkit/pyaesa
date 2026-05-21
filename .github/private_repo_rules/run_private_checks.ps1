param(
    [switch]$Install,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "==> $Name"
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
}

if ($Install) {
    Invoke-ExternalCommand -Name "Upgrade pip" -Executable "python" -Arguments @(
        "-m", "pip", "install", "--upgrade", "pip"
    )
    Invoke-ExternalCommand -Name "Install test and quality dependencies" -Executable "python" -Arguments @(
        "-m", "pip", "install", "-e", ".[test,quality]", "build", "twine"
    )
}

Invoke-ExternalCommand -Name "Run ruff check" -Executable "ruff" -Arguments @(
    "check", "pyaesa", "tests/package"
)

Invoke-ExternalCommand -Name "Run ruff format check" -Executable "ruff" -Arguments @(
    "format", "--check", "pyaesa", "tests/package"
)

Invoke-ExternalCommand -Name "Run pyright" -Executable "pyright" -Arguments @(
    "pyaesa"
)

Invoke-ExternalCommand -Name "Run package tests with branch coverage" -Executable "python" -Arguments @(
    "-m", "pytest", "tests/package", "--cov=pyaesa", "--cov-branch", "--cov-report=term"
)

if (-not $SkipBuild) {
    if (Test-Path "dist") {
        Remove-Item -Recurse -Force "dist"
    }
    if (Test-Path "build") {
        Remove-Item -Recurse -Force "build"
    }

    Invoke-ExternalCommand -Name "Build package artifacts" -Executable "python" -Arguments @(
        "-m", "build"
    )
    Invoke-ExternalCommand -Name "Run twine metadata checks" -Executable "twine" -Arguments @(
        "check", "--strict", "dist/*"
    )
}

Write-Host ""
Write-Host "All private repository checks passed."
