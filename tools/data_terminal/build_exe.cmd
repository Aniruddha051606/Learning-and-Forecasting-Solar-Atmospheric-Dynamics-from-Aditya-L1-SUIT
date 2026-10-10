@echo off
rem Builds tools\data_terminal\dist\SUIT Data Terminal.exe (single file, console window).
cd /d "%~dp0"
set PY="C:\Users\pawar\AppData\Local\Programs\Python\Python314\python.exe"
set ICON="%~dp0..\..\dashboard\build\suitdyn.ico"
%PY% -m PyInstaller --noconfirm --onefile --console --name "SUIT Data Terminal" --icon %ICON% ^
    --distpath "%~dp0dist" --workpath "%~dp0build" --specpath "%~dp0build" ^
    suit_data_terminal.py || exit /b 1
echo built: %~dp0dist\SUIT Data Terminal.exe
