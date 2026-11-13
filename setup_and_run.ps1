param(
    [string]$Root = $PSScriptRoot,
    [switch]$NoRun,
    [switch]$SkipInstall,
    [switch]$NoTrain
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([string]$What, [scriptblock]$Command)
    Write-Host "-- $What"
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

$Root = (Resolve-Path -LiteralPath $Root).Path
if (-not (Test-Path -LiteralPath (Join-Path $Root "setup_and_run.ps1") -PathType Leaf)) {
    throw "Root does not contain setup_and_run.ps1: $Root"
}

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) { throw "python is not on PATH" }

foreach ($dir in "vendor", "data", "models", "src") {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $dir) | Out-Null
}
Set-Location -LiteralPath $Root

$repos = [ordered]@{
    "QuixBugs"              = "https://github.com/jkoppel/QuixBugs.git"
    "jm1-defect-prediction" = "https://github.com/messuuu/jm1-software-defect-prediction-.git"
    "DeepLineDP"            = "https://github.com/awsm-research/DeepLineDP.git"
    "auto-code-rover"       = "https://github.com/AutoCodeRoverSG/auto-code-rover.git"
}
$gitCommand = Get-Command git -ErrorAction SilentlyContinue
foreach ($name in $repos.Keys) {
    $dest = Join-Path (Join-Path $Root "vendor") $name
    if (Test-Path -LiteralPath $dest) {
        if (-not (Get-Item -LiteralPath $dest).PSIsContainer) {
            throw "Expected a vendor directory but found a file: $dest"
        }
        Write-Host "-- vendor/$name already exists; skipping clone"
        continue
    }
    if (-not $gitCommand) { throw "git is required to clone missing vendor/$name" }
    Invoke-Checked "clone $name" { git clone --quiet --depth 1 $repos[$name] $dest }
}

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not $SkipInstall) {
    if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
        Invoke-Checked "create .venv" { & $pythonCommand.Source -m venv .venv }
    }
    Invoke-Checked "install requirements" { & $py -m pip install --quiet --disable-pip-version-check --no-cache-dir -r requirements.txt }
} elseif (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    $py = $pythonCommand.Source
}

if (-not $NoTrain) {
    Invoke-Checked "train models" { & $py -m src.train_pipeline }
}

if (-not $NoRun) {
    Write-Host "-- starting streamlit, ctrl+c to stop"
    & $py -m streamlit run app.py
    if ($LASTEXITCODE -ne 0) { throw "streamlit exited with code $LASTEXITCODE" }
}
