<#
Local Mendix Agent - user-level setup. No administrator rights, nothing installed system-wide,
PATH is not modified. Everything lives in this folder (.venv, data\).

  .\setup.ps1                          create .venv and install dependencies from PyPI
  .\setup.ps1 -Wheelhouse .\wheelhouse offline install from a folder of wheels
  .\setup.ps1 -IndexUrl https://nexus.company/pypi/simple   use a corporate PyPI mirror

If PowerShell blocks scripts, run:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
(that flag applies to this one process only and changes no policy).
Proxy: set HTTPS_PROXY before running if your network requires it.
#>
param(
    [string]$Wheelhouse = "",
    [string]$IndexUrl = "",
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Python {
    if ($Python) { return ,@($Python) }
    foreach ($v in "3.12", "3.13", "3.11") {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py "-$v" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) { return ,@("py", "-$v") }
        }
    }
    foreach ($name in "python", "python3") {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            $ok = & $cmd.Source -c "import sys; print(int(sys.version_info >= (3, 11)))" 2>$null
            if ($ok -eq "1") { return ,@($cmd.Source) }
        }
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Write-Host "Python 3.11+ was not found." -ForegroundColor Red
    Write-Host "Install it for the current user only (no admin): python.org installer with 'Install for me only',"
    Write-Host "or the Microsoft Store 'Python 3.12' package, or ask IT for the approved Python distribution."
    exit 1
}
$pyExe = $py[0]
$pyArgs = @()
if ($py.Count -gt 1) { $pyArgs = $py[1..($py.Count - 1)] }
Write-Host "Using Python: $($py -join ' ')"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment in .venv ..."
    & $pyExe @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Host "venv creation failed" -ForegroundColor Red; exit 1 }
}
$venvPy = ".venv\Scripts\python.exe"
& $venvPy -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pip') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip is missing in .venv, bootstrapping it (ensurepip) ..."
    & $venvPy -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { Write-Host "could not bootstrap pip; delete .venv and run setup again" -ForegroundColor Red; exit 1 }
}

$pipArgs = @("-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements.txt")
if ($Wheelhouse) {
    $pipArgs += @("--no-index", "--find-links", (Resolve-Path $Wheelhouse).Path)
    Write-Host "Installing dependencies offline from $Wheelhouse ..."
} elseif ($IndexUrl) {
    $pipArgs += @("--index-url", $IndexUrl)
    Write-Host "Installing dependencies from $IndexUrl ..."
} else {
    Write-Host "Installing dependencies from PyPI ..."
}
& $venvPy @pipArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Dependency installation failed. Typical causes on a corporate laptop:" -ForegroundColor Red
    Write-Host " - pypi.org is blocked: use -IndexUrl <corporate mirror> or -Wheelhouse <folder>"
    Write-Host "   (build the folder elsewhere:  pip download -r requirements.txt -d wheelhouse --only-binary=:all: --python-version 3.12 --platform win_amd64)"
    Write-Host " - proxy required: `$env:HTTPS_PROXY = 'http://proxy.company:8080'"
    Write-Host " - TLS inspection: `$env:PIP_CERT = 'C:\path\to\company-root-ca.pem'  (or: pip install pip-system-certs)"
    exit 1
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Start:        .\run.bat            (or .venv\Scripts\python.exe main.py)"
Write-Host "Diagnostics:  .\run.bat -c /doctor"
Write-Host "First steps inside the app:  /models   /download runtime   /download <model-id>   /mendix connect"
