@echo off
setlocal
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found. Install Python 3.12 and select Add python.exe to PATH.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
if errorlevel 1 goto failed
if not exist ".env" copy ".env.example" ".env" >nul
echo.
echo Setup finished. Double-click run_windows.bat to open the app.
pause
exit /b 0
:failed
echo Setup failed. Copy the error above and ask for help.
pause
exit /b 1
