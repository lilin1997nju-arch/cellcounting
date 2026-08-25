!include "LogicLib.nsh"

!macro customInstall
  CreateDirectory "$INSTDIR\Workspace"
  CreateDirectory "$INSTDIR\Workspace\Projects"
  CreateDirectory "$INSTDIR\Workspace\Logs"

  DetailPrint "Granting the local interactive user access to Workspace ..."
  nsExec::ExecToLog '"$SYSDIR\icacls.exe" "$INSTDIR\Workspace" /inheritance:e /grant "*S-1-5-32-545:(OI)(CI)M" /C /Q'
  Pop $0

  ${If} ${FileExists} "$INSTDIR\vc_redist.x64.exe"
    DetailPrint "Installing the Microsoft Visual C++ x64 runtime ..."
    ExecWait '"$INSTDIR\vc_redist.x64.exe" /install /quiet /norestart' $0
    ${If} $0 != 0
    ${AndIf} $0 != 1638
    ${AndIf} $0 != 3010
      MessageBox MB_ICONSTOP "Microsoft Visual C++ x64 runtime installation failed with exit code $0."
      Abort
    ${EndIf}
  ${EndIf}

  DetailPrint "Configuring the Cell Vision production service ..."
  ExecWait '"powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\configure_service.ps1"' $0
  ${If} $0 != 0
    MessageBox MB_ICONSTOP "Cell Vision service configuration failed with exit code $0. Workspace data was preserved."
    Abort
  ${EndIf}
!macroend

!macro customUnInstall
  ${If} ${FileExists} "$INSTDIR\Application\scripts\install_production_service.ps1"
    DetailPrint "Stopping and unregistering the Cell Vision production service ..."
    ExecWait '"powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\Application\scripts\install_production_service.ps1" -InstallRoot "$INSTDIR\Application" -Action uninstall' $0
  ${EndIf}
!macroend

!macro customRemoveFiles
  # Workspace is deliberately excluded. Upgrades and uninstall preserve all
  # historical projects unless an administrator removes that directory later.
  SetOutPath "$TEMP"
  RMDir /r "$INSTDIR\Application"
  RMDir /r "$INSTDIR\locales"
  RMDir /r "$INSTDIR\resources"
  RMDir /r "$INSTDIR\update-backups"
  Delete "$INSTDIR\*.*"
  RMDir "$INSTDIR"
!macroend
