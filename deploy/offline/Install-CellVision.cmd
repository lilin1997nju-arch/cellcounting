@echo off
setlocal
title Cell Vision Offline Installer
powershell.exe -STA -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_ui.ps1" -Mode production
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Cell Vision installation failed with exit code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%
