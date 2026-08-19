@echo off
setlocal
title Cell Vision Offline Installer
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_installer.ps1"
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Cell Vision installation failed with exit code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%
