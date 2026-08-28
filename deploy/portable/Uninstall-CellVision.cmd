@echo off
setlocal EnableExtensions
title Uninstall Cell Vision Desktop
echo Cell Vision program files and the desktop service will be removed.
echo Workspace and all historical project data will be preserved.
echo.
choice /C YN /N /M "Continue uninstall? [Y/N]: "
if errorlevel 2 exit /b 0
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall_cellvision.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
  echo Cell Vision uninstall failed with exit code %EXIT_CODE%.
  echo See Workspace\Logs\cellvision-uninstall-*.log for details.
  pause
) else (
  echo Cell Vision was uninstalled. Workspace was preserved.
  timeout /t 2 /nobreak >nul
)
exit /b %EXIT_CODE%
