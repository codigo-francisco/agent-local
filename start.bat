@echo off
rem Arranca la GUI sin cambiar la política de ejecución del sistema (Bypass solo para este proceso).
rem Uso: start.bat            ventana nativa
rem      start.bat -Browser   en el navegador
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if errorlevel 1 pause
