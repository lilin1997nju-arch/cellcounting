@echo off
setlocal EnableExtensions
title Install Cell Vision Desktop
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_cellvision_zip.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
  echo Cell Vision installation failed with exit code %EXIT_CODE%.
  echo See Workspace\Logs\cellvision-zip-install-*.log for details.
) else (
  echo Cell Vision installation completed successfully.
)
pause
exit /b %EXIT_CODE%
