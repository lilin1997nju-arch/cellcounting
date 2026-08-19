@echo off
setlocal
title Install Cell Vision Review Platform
powershell.exe -STA -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_ui.ps1" -Mode review
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Installation failed with exit code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%
