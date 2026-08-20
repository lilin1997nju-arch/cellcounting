@echo off
setlocal
title Cell Vision
set "PORT=8777"
powershell.exe -NoProfile -Command "try { $r = Invoke-RestMethod -Uri 'http://127.0.0.1:%PORT%/api/ready' -TimeoutSec 2; if ($r.status -eq 'ready') { exit 0 }; exit 1 } catch { exit 1 }"
if errorlevel 1 (
  echo.
  echo Cell Vision service is not ready.
  echo Run Configure-CellVision-Service.cmd once as an administrator.
  pause
  exit /b 1
)
"%~dp0Application\Python312\python.exe" -m cellvision.desktop_bridge --production-url "http://127.0.0.1:%PORT%/"
exit /b %ERRORLEVEL%
