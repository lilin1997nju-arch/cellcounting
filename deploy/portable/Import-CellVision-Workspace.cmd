@echo off
setlocal EnableExtensions
title Import Cell Vision Workspace
echo Select the historical Workspace folder in the next dialog.
powershell.exe -NoLogo -NoProfile -STA -ExecutionPolicy Bypass -Command ^
  "Add-Type -AssemblyName System.Windows.Forms; $dialog=New-Object Windows.Forms.FolderBrowserDialog; $dialog.Description='Select historical Cell Vision Workspace'; if($dialog.ShowDialog() -eq 'OK'){ & '%~dp0import_cellvision_workspace_launcher.ps1' -SourceWorkspace $dialog.SelectedPath; exit $LASTEXITCODE }; exit 0"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo Workspace import failed with exit code %EXIT_CODE%.
if "%EXIT_CODE%"=="0" echo Workspace import completed or was cancelled.
pause
exit /b %EXIT_CODE%
