@echo off
setlocal EnableExtensions
title Start Cell Vision
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_cellvision_platform.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Cell Vision could not be started. Review the message and log above.
  pause
)
exit /b %EXIT_CODE%
