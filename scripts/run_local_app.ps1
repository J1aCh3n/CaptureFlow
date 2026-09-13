<#
.SYNOPSIS
    Start the local CaptureFlow Streamlit app from this checkout.

.DESCRIPTION
    Resolves the repository root from this script's own location, puts this
    checkout's src on PYTHONPATH ahead of any editable install of another
    checkout, verifies that captureflow actually imports from here, and then
    runs Streamlit from the repository root so the app uses this checkout's
    data/captureflow.sqlite3.

    It refuses to start if the interpreter would import captureflow from a
    different directory, because that failure is silent: the app runs, but on
    the other checkout's code.

.PARAMETER Python
    Interpreter to use. Defaults to the llm-course Conda environment.

.PARAMETER Model
    Value for CAPTUREFLOW_MODEL. Defaults to the existing variable, then to
    qwen3.6-plus. For a Qwen model the key comes from QWEN_API_KEY, the endpoint
    is DashScope international, and the timeout is 120 seconds, unless those
    variables are already set.

.PARAMETER CheckOnly
    Run every check and print the resolved configuration without starting the
    app.

.EXAMPLE
    .\scripts\run_local_app.ps1

.EXAMPLE
    .\scripts\run_local_app.ps1 -CheckOnly
#>

[CmdletBinding()]
param(
    [string] $Python = 'C:\Users\frank\anaconda3\envs\llm-course\python.exe',
    [string] $Model,
    [switch] $CheckOnly
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$source = Join-Path $root 'src'

if (-not (Test-Path $Python)) {
    throw "Interpreter not found: $Python. Pass -Python with the path to the llm-course python.exe."
}
if (-not (Test-Path (Join-Path $source 'captureflow'))) {
    throw "No captureflow package under $source. Run this script from its own checkout."
}

if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$source;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $source
}

$imported = & $Python -c "import captureflow, os; print(os.path.dirname(os.path.dirname(captureflow.__file__)))"
if ($LASTEXITCODE -ne 0) {
    throw "python could not import captureflow with PYTHONPATH=$env:PYTHONPATH"
}
if ($imported.Trim() -ne $source) {
    throw "captureflow imports from $($imported.Trim()), not $source. Refusing to run another checkout's code."
}

if ($Model) {
    $env:CAPTUREFLOW_MODEL = $Model
} elseif (-not $env:CAPTUREFLOW_MODEL) {
    $env:CAPTUREFLOW_MODEL = 'qwen3.6-plus'
}
if ($env:CAPTUREFLOW_MODEL -like 'qwen*') {
    if (-not $env:CAPTUREFLOW_API_KEY -and $env:QWEN_API_KEY) {
        $env:CAPTUREFLOW_API_KEY = $env:QWEN_API_KEY
    }
    if (-not $env:CAPTUREFLOW_BASE_URL) {
        # A key issued for the international region is rejected by the Beijing endpoint.
        $env:CAPTUREFLOW_BASE_URL = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
    }
    if (-not $env:CAPTUREFLOW_MODEL_TIMEOUT_SECONDS) {
        # Qwen took 18-60 seconds per routing call; the 30-second default would cut it off.
        $env:CAPTUREFLOW_MODEL_TIMEOUT_SECONDS = '120'
    }
} elseif (-not $env:CAPTUREFLOW_API_KEY -and $env:REPOMEDIC_OPENAI_API_KEY) {
    $env:CAPTUREFLOW_API_KEY = $env:REPOMEDIC_OPENAI_API_KEY
}

Write-Host "checkout: $root"
Write-Host "package:  $($imported.Trim())"
Write-Host "model:    $env:CAPTUREFLOW_MODEL"
Write-Host "endpoint: $(if ($env:CAPTUREFLOW_BASE_URL) { $env:CAPTUREFLOW_BASE_URL } else { 'OpenAI default' })"
Write-Host "database: $(if ($env:CAPTUREFLOW_DATABASE_PATH) { $env:CAPTUREFLOW_DATABASE_PATH } else { Join-Path $root 'data\captureflow.sqlite3' })"
if ($env:CAPTUREFLOW_API_KEY) {
    Write-Host 'api key:  configured'
} else {
    Write-Warning 'CAPTUREFLOW_API_KEY is not set; chat will save a visible configuration failure and write no records.'
}

if ($CheckOnly) {
    Write-Host 'checks passed; not starting Streamlit (-CheckOnly).'
    return
}

Push-Location $root
try {
    & $Python -m streamlit run app.py
} finally {
    Pop-Location
}
