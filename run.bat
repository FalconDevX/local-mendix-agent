@echo off
rem Local Mendix Agent launcher (uses the local .venv, no PATH changes)
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo .venv not found. Run: powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0main.py" %*
