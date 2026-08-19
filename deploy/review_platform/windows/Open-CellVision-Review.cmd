@echo off
setlocal
title Cell Vision Review Platform
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0open_review_platform.ps1" %*
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Cell Vision Review Platform failed with exit code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%
