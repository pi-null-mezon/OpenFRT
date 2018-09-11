; This script generates setup file for the MSVC2015 version of the target application

#define MyAppName "FACE"
#define MyAppVersion "1.0.0.0"
#define MyAppPublisher "DiacareSoft"
#define MyAppExeName "fv2db.exe"
#define MyAppURL "https://repo.nefrosovet.ru/a.a.taranov/OFRT"

;Define if you want to build setup for x64 version
;#define AMD64 

#define Opencv "C:\Programming\3rdParties\opencv310"
#ifndef AMD64  
  #define Qt "C:\Qt\5.7\msvc2015\bin"
  #define ARCH "x86"
#else
  #define Qt "C:\Qt\5.7\msvc2015_64\bin"
  #define ARCH "x64"
#endif

[Setup]
; Do not change GUID in future version because GUID is used to determine if app is installed in system or not
AppId={{DD98B046-1268-426F-A356-7CAB50FC0739}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={pf}\{#MyAppName}
DefaultGroupName={#MyAppPublisher}\{#MyAppName}
;InfoAfterFile=
OutputDir=C:\Programming\Releases
OutputBaseFilename=SETUP_{#MyAppName}_v{#MyAppVersion}_{#ARCH}
Compression=lzma
SolidCompression=yes
#ifdef AMD64
; "ArchitecturesAllowed=x64" specifies that Setup cannot run on
; anything but x64.
ArchitecturesAllowed=x64
; "ArchitecturesInstallIn64BitMode=x64" requests that the install be
; done in "64-bit mode" on x64, meaning it should use the native
; 64-bit Program Files directory and the 64-bit view of the registry.
ArchitecturesInstallIn64BitMode=x64
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
#ifndef AMD64
  Source: "C:\Programming\ofrt\FV2DB\build\build-fv2db-Desktop_Qt_5_7_0_MSVC2015_32bit-Release\release\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
#else
  Source: "C:\Programming\ofrt\FV2DB\build\build-fv2db-Desktop_Qt_5_7_0_MSVC2015_64bit-Release\release\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
#endif
Source: "{#Opencv}\build\{#ARCH}\vc14\bin\opencv_core310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Opencv}\build\{#ARCH}\vc14\bin\opencv_highgui310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Opencv}\build\{#ARCH}\vc14\bin\opencv_imgproc310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Opencv}\build\{#ARCH}\vc14\bin\opencv_objdetect310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Opencv}\build\{#ARCH}\vc14\bin\opencv_video310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Opencv}\build\{#ARCH}\vc14\bin\opencv_videoio310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Opencv}\build\{#ARCH}\vc14\bin\opencv_imgcodecs310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Opencv}\build\{#ARCH}\vc14\bin\opencv_ml310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Opencv}\build\{#ARCH}\vc14\bin\opencv_face310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Opencv}\build\{#ARCH}\vc14\bin\opencv_dnn310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Qt}\Qt5Core.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Qt}\Qt5Concurrent.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Qt}\Qt5Network.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Qt}\Qt5Widgets.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Qt}\Qt5WinExtras.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Qt}\Qt5Sql.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Qt}\..\plugins\platforms\qminimal.dll"; DestDir: "{app}\platforms"; Flags: ignoreversion
Source: "{#Qt}\..\plugins\platforms\qoffscreen.dll"; DestDir: "{app}\platforms"; Flags: ignoreversion
Source: "{#Qt}\..\plugins\platforms\qwindows.dll"; DestDir: "{app}\platforms"; Flags: ignoreversion
Source: "{#Qt}\..\plugins\sqldrivers\qsqlodbc.dll"; DestDir: "{app}\sqldrivers"; Flags: ignoreversion
;Microsoft Visual Studio redistributable packages, also read this http://stackoverflow.com/questions/11137424/how-to-make-vcredist-x86-reinstall-only-if-not-yet-installed
#ifndef AMD64
  Source: "C:\Programming\3rdParties\MSVCredist\x86\vc14\vc_redist.x86.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
#else
  Source: "C:\Programming\3rdParties\MSVCredist\x64\vc14\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
#endif
;Data base connection configuration
Source: "C:\Programming\ofrt\FV2DB\fv2db\Database.ini"; DestDir: "{app}"; Flags: ignoreversion
;And application configuration for automatic run
Source: "C:\Programming\ofrt\FV2DB\fv2db\Config.ini"; DestDir: "{app}"; Flags: ignoreversion 
;VGG Face Descriptor CNN
Source: "C:\Programming\3rdParties\Caffe\models\vgg_face_caffe\VGG_FACE_deploy.prototxt"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Programming\3rdParties\Caffe\models\vgg_face_caffe\VGG_FACE.caffemodel"; DestDir: "{app}"; Flags: ignoreversion
;Let's add utility for the face recognizers training (it should be prebuild before this file compilation)
#ifndef AMD64
  Source: "C:\Programming\ofrt\build\build-trainer-Desktop_Qt_5_7_0_MSVC2015_32bit-Release\release\rectrainer.exe"; DestDir: "{app}"; Flags: ignoreversion
#else
  Source: "C:\Programming\ofrt\build\build-trainer-Desktop_Qt_5_7_0_MSVC2015_64bit-Release\release\rectrainer.exe"; DestDir: "{app}"; Flags: ignoreversion
#endif

[Icons]
;Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:ProgramOnTheWeb,{#MyAppName}}"; Filename: "{#MyAppURL}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
;Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Install msvc redist if it has not been installed yet
#ifndef AMD64
  Filename: "{tmp}\vc_redist.x86.exe"; Check: VCRedistNeedsInstall
#else
  Filename: "{tmp}\vc_redist.x64.exe"; Check: VCRedistNeedsInstall
#endif

[Code]
//-------------------------------------------------------------------------------------
// Delete application if it had been installed before
//-------------------------------------------------------------------------------------
function GetUninstallString: string;
var
  sUnInstPath: string;
  sUnInstallString: String;
begin
  Result := '';
  sUnInstPath := ExpandConstant('Software\Microsoft\Windows\CurrentVersion\Uninstall\{{DD98B046-1268-426F-A356-7CAB50FC0739}_is1'); //Your App GUID/ID
  sUnInstallString := '';
  if not RegQueryStringValue(HKLM, sUnInstPath, 'UninstallString', sUnInstallString) then
    RegQueryStringValue(HKCU, sUnInstPath, 'UninstallString', sUnInstallString);
  Result := sUnInstallString;
end;

function IsUpgrade: Boolean;
begin
  Result := (GetUninstallString() <> '');
end;

function InitializeSetup: Boolean;
var
  V: Integer;
  iResultCode: Integer;
  sUnInstallString: string;
begin
  Result := True; // in case when no previous version is found
  if RegValueExists(HKEY_LOCAL_MACHINE,'Software\Microsoft\Windows\CurrentVersion\Uninstall\{DD98B046-1268-426F-A356-7CAB50FC0739}_is1', 'UninstallString') then  //Your App GUID/ID
  begin
    V := MsgBox(ExpandConstant('An old version of app was detected. You should uninstall it before. Uninstall now?'), mbInformation, MB_YESNO); //Custom Message if App installed
    if V = IDYES then
    begin
      sUnInstallString := GetUninstallString();
      sUnInstallString :=  RemoveQuotes(sUnInstallString);
      Exec(ExpandConstant(sUnInstallString), '', '', SW_SHOW, ewWaitUntilTerminated, iResultCode);
      Result := True; //if you want to proceed after uninstall
      //Exit; //if you want to quit after uninstall
    end
    else
      Result := False; //when older version present and not uninstalled
  end;
end;
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
//Check MSVC redistributable installation
//-------------------------------------------------------------------------------------
#IFDEF UNICODE
  #DEFINE AW "W"
#ELSE
  #DEFINE AW "A"
#ENDIF
type
  INSTALLSTATE = Longint;
const
  INSTALLSTATE_INVALIDARG = -2;  { An invalid parameter was passed to the function. }
  INSTALLSTATE_UNKNOWN = -1;     { The product is neither advertised or installed. }
  INSTALLSTATE_ADVERTISED = 1;   { The product is advertised but not installed. }
  INSTALLSTATE_ABSENT = 2;       { The product is installed for a different user. }
  INSTALLSTATE_DEFAULT = 5;      { The product is installed for the current user. }

  { Visual C++ 2015 Redistributable 14.0.23026 }
  VC_2015_REDIST_X86_MIN = '{A2563E55-3BEC-3828-8D67-E5E8B9E8B675}';
  VC_2015_REDIST_X64_MIN = '{0D3E9E15-DE7A-300B-96F1-B4AF12B96488}';

  VC_2015_REDIST_X86_ADD = '{BE960C1C-7BAD-3DE6-8B1A-2616FE532845}';
  VC_2015_REDIST_X64_ADD = '{BC958BD2-5DAC-3862-BB1A-C1BE0790438D}';

  { Visual C++ 2015 Redistributable 14.0.24210 }
  VC_2015_REDIST_X86 = '{8FD71E98-EE44-3844-9DAD-9CB0BBBC603C}';
  VC_2015_REDIST_X64 = '{C0B2C673-ECAA-372D-94E5-E89440D087AD}';

function MsiQueryProductState(szProduct: string): INSTALLSTATE; 
  external 'MsiQueryProductState{#AW}@msi.dll stdcall';

function VCVersionInstalled(const ProductID: string): Boolean;
begin
  Result := MsiQueryProductState(ProductID) = INSTALLSTATE_DEFAULT;
end;

function VCRedistNeedsInstall: Boolean;
begin
  { here the Result must be True when you need to install your VCRedist }
  { or False when you don't need to, so now it's upon you how you build }
  { this statement, the following won't install your VC redist only when }
  { the Visual C++ 2015 Redist (x86) and Visual C++ 2010 SP1 Redist(x86) }
  { are installed for the current user }
  {Result := not (VCVersionInstalled(VC_2010_REDIST_X86) and VCVersionInstalled(VC_2010_SP1_REDIST_X86));}
  #ifndef AMD64
    Result := not ( VCVersionInstalled(VC_2015_REDIST_X86_MIN) or VCVersionInstalled(VC_2015_REDIST_X86_ADD) or VCVersionInstalled(VC_2015_REDIST_X86) );
  #else 
    Result := not ( VCVersionInstalled(VC_2015_REDIST_X64_MIN) or VCVersionInstalled(VC_2015_REDIST_X64_ADD) or VCVersionInstalled(VC_2015_REDIST_X64) );
  #endif
end;
//-------------------------------------------------------------------------------------