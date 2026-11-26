; Inno Setup script for the napari-storm Windows installer.
;
; installer/build.py compiles it after PyInstaller has frozen the app into
; dist\napari-storm\, passing the version and paths with /D:
;
;   iscc /DAppVersion=3.3.0 /DSourceDir=...\dist\napari-storm
;        /DOutputDir=...\dist /DOutputBaseFilename=napari-storm-3.3.0-windows-x64-setup
;        /DIconFile=...\napari-storm.ico installer\napari-storm.iss
;
; Installs per user by default (no administrator rights needed, into
; %LOCALAPPDATA%\Programs), with the option to install for all users.

#ifndef AppVersion
  #error AppVersion must be defined: iscc /DAppVersion=<version>
#endif
#ifndef SourceDir
  #error SourceDir must be defined: the folder PyInstaller made
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif
#ifndef OutputBaseFilename
  #define OutputBaseFilename "napari-storm-" + AppVersion + "-windows-x64-setup"
#endif

[Setup]
; Never change AppId: it is how a newer installer finds and replaces an
; installed older version.
AppId={{6E3C2A51-8F0B-4C47-9B8E-5A1D2F7C9E34}
AppName=napari-storm
AppVersion={#AppVersion}
AppVerName=napari-storm {#AppVersion}
AppPublisher=napari-storm
AppPublisherURL=https://github.com/napari-storm/napari-storm
AppSupportURL=https://github.com/napari-storm/napari-storm/issues
AppUpdatesURL=https://github.com/napari-storm/napari-storm/releases
DefaultDirName={autopf}\napari-storm
DefaultGroupName=napari-storm
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
#ifdef IconFile
SetupIconFile={#IconFile}
#endif
UninstallDisplayIcon={app}\napari-storm.exe
LicenseFile=..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\napari-storm"; Filename: "{app}\napari-storm.exe"
Name: "{autodesktop}\napari-storm"; Filename: "{app}\napari-storm.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\napari-storm.exe"; Description: "{cm:LaunchProgram,napari-storm}"; Flags: nowait postinstall skipifsilent

[InstallDelete]
; A one-folder app replaced in place keeps files the new version no longer
; ships; clear the bundled libraries so an upgrade is a clean copy.
Type: filesandordirs; Name: "{app}\_internal"
