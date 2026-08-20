@echo off
setlocal
title Configure Cell Vision Service
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0configure_service_launcher.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
  echo Cell Vision service configuration failed with exit code %EXIT_CODE%.
) else (
  echo Cell Vision service configuration completed successfully.
)
pause
exit /b %EXIT_CODE%
