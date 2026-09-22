# Arranca la GUI del agente local. Crea el entorno virtual la primera vez.
#   .\start.ps1            ventana nativa
#   .\start.ps1 -Browser   en el navegador (si la ventana nativa da problemas)
param([switch]$Browser)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creando entorno virtual e instalando dependencias (solo la primera vez)..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv .venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv .venv
    } else {
        Write-Host "No encontré Python. Instala Python 3.12 (https://www.python.org/downloads/windows/) y vuelve a ejecutar este script." -ForegroundColor Red
        exit 1
    }
    & .venv\Scripts\python.exe -m pip install --upgrade pip
    & .venv\Scripts\python.exe -m pip install -e ".[dev]"
}

$guiArgs = @()
if ($Browser) { $guiArgs += "--browser" }
& .venv\Scripts\python.exe -m agent.gui @guiArgs
