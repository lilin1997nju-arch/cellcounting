@echo off
setlocal EnableExtensions
title Recover Cell Vision Project List
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0recover_cellvision_projects.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
  echo Project list recovery failed with exit code %EXIT_CODE%.
) else (
  echo Project list recovery completed. Refresh the Cell Vision homepage.
)
pause
exit /b %EXIT_CODE%
