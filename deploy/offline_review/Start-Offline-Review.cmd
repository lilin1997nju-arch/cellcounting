@echo off
setlocal
title Cell Vision Offline Review
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_offline_review.ps1"
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Cell Vision offline review failed with exit code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%
