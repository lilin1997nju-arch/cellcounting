@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0update.ps1"
set "exit_code=%ERRORLEVEL%"
echo.
if not "%exit_code%"=="0" (
  echo Cell Vision update failed. Review the error above.
) else (
  echo Cell Vision update completed successfully.
)
pause
exit /b %exit_code%
