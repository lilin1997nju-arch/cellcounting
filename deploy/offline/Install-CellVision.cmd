@echo off
setlocal
title Cell Vision Offline Installer
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_offline.ps1" -StartAfterInstall
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Cell Vision installation failed with exit code %EXIT_CODE%.
) else (
  echo.
  echo Cell Vision installation completed successfully.
)
pause
exit /b %EXIT_CODE%
