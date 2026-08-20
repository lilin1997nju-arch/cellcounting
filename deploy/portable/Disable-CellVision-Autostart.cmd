@echo off
setlocal
title Disable Cell Vision Automatic Startup
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0configure_service_launcher.ps1" -ScriptName disable_service_autostart.ps1
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
  echo Unable to disable Cell Vision automatic startup. Exit code: %EXIT_CODE%.
) else (
  echo Cell Vision will no longer start automatically when Windows starts.
)
pause
exit /b %EXIT_CODE%
