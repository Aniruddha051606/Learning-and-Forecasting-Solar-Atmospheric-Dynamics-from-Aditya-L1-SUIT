@echo off
rem Builds dashboard\dist\SUIT-DYN Dashboard.exe (single file, no console window).
cd /d "%~dp0"
set PY="C:\Users\pawar\AppData\Local\Programs\Python\Python314\python.exe"
%PY% make_icon.py || exit /b 1
%PY% -m PyInstaller --noconfirm --onefile --windowed --name "SUIT-DYN Dashboard" --icon "%~dp0build\suitdyn.ico" ^
    --distpath "%~dp0dist" --workpath "%~dp0build" --specpath "%~dp0build" ^
    --exclude-module torch --exclude-module pandas --exclude-module matplotlib --exclude-module scipy ^
    --exclude-module astropy --exclude-module zarr --exclude-module pyarrow ^
    suitdyn_dashboard.py || exit /b 1
echo built: %~dp0dist\SUIT-DYN Dashboard.exe
