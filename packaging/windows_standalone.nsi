Unicode true
RequestExecutionLevel user
SilentInstall silent
AutoCloseWindow true
InstallDir "$TEMP\MFQStudio"
SetCompressor /SOLID zlib
Name "MFQ Studio"
Caption "MFQ Studio"
OutFile "@OUTPUT@"
Icon "@ICON@"
VIProductVersion "@VERSION@.0"
VIAddVersionKey /LANG=1033 "ProductName" "MFQ Studio"
VIAddVersionKey /LANG=1033 "FileDescription" "MFQ Studio"
VIAddVersionKey /LANG=1033 "FileVersion" "@VERSION@"
VIAddVersionKey /LANG=1033 "LegalCopyright" "MFQ contributors"

!include "FileFunc.nsh"
!include "LogicLib.nsh"

Var Parameters
Var CacheRoot
Var CachePath
Var Stage
Var Mutex
Var WaitPid

Section
  ${GetParameters} $Parameters
  ReadEnvStr $CacheRoot "MFQ_STUDIO_RUNTIME_CACHE"
  ${If} $CacheRoot == ""
    StrCpy $CacheRoot "$LOCALAPPDATA\com.tylogi.mfq-studio\runtime"
  ${EndIf}
  ${GetOptions} $Parameters "/CACHE_DIR=" $0
  ${IfNot} ${Errors}
    GetFullPathName $CacheRoot "$0"
  ${EndIf}
  StrCpy $CachePath "$CacheRoot\@PAYLOAD_ID@"
  StrCpy $Stage "$CachePath.partial"

  System::Call 'kernel32::CreateMutexW(p 0, i 0, w "Local\MFQStudio-@PAYLOAD_ID@") p .r0'
  StrCpy $Mutex $0
  ${If} $Mutex == 0
    Goto failed
  ${EndIf}
  System::Call 'kernel32::WaitForSingleObject(p $Mutex, i 120000) i .r0'
  ${If} $0 != 0
  ${AndIf} $0 != 128
    Goto failed
  ${EndIf}

  IfFileExists "$CachePath\.ready" 0 extract
  IfFileExists "$CachePath\mfq-studio.exe" 0 extract
  IfFileExists "$CachePath\mfq-runtime.exe" 0 extract
  IfFileExists "$CachePath\mfq-cli\mfq-cli.exe" 0 extract
  IfFileExists "$CachePath\web-ui\index.html" cached extract

extract:
  SetOutPath "$CacheRoot"
  RMDir /r "$Stage"
  ClearErrors
@PAYLOAD_FILES@
  FileOpen $0 "$Stage\.ready" w
  IfErrors failed
  FileWrite $0 "@PAYLOAD_ID@"
  FileClose $0
  IfErrors failed
  SetOutPath "$CacheRoot"
  RMDir /r "$CachePath"
  Rename "$Stage" "$CachePath"
  IfErrors failed

cached:
  System::Call 'kernel32::ReleaseMutex(p $Mutex)'
  System::Call 'kernel32::CloseHandle(p $Mutex)'
  StrCpy $Mutex 0
  ClearErrors
  ${GetOptions} $Parameters "/EXTRACT_ONLY" $0
  ${IfNot} ${Errors}
    Goto done
  ${EndIf}

  ClearErrors
  ${GetOptions} $Parameters "/WAITPID=" $WaitPid
  ${IfNot} ${Errors}
    System::Call 'kernel32::OpenProcess(i 0x100000, i 0, i $WaitPid) p .r0'
    ${If} $0 != 0
      StrCpy $1 $0
      System::Call 'kernel32::WaitForSingleObject(p r1, i 30000) i .r0'
      System::Call 'kernel32::CloseHandle(p r1)'
      ${If} $0 != 0
        Goto failed
      ${EndIf}
    ${EndIf}
  ${EndIf}

  System::Call 'kernel32::SetEnvironmentVariableW(w "MFQ_STUDIO_LAUNCHER", w "$EXEPATH")'
  SetOutPath "$CachePath"
  ClearErrors
  Exec '"$CachePath\mfq-studio.exe"'
  IfErrors failed

done:
  SetErrorLevel 0
  Quit

failed:
  ${If} $Mutex != 0
    System::Call 'kernel32::ReleaseMutex(p $Mutex)'
    System::Call 'kernel32::CloseHandle(p $Mutex)'
  ${EndIf}
  MessageBox MB_OK|MB_ICONSTOP "MFQ Studio could not start. Check available disk space and write access to:$\r$\n$CacheRoot"
  SetErrorLevel 1
SectionEnd
