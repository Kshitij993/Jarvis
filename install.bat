@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Robotics Project - Windows Installer
echo ============================================================
echo.

:: ── Check Python ─────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [WARN] Python not found. Attempting automatic installation...
    call :install_python
    if errorlevel 1 (
        echo [ERROR] Automatic Python installation failed.
        echo         Please install Python 3.11 manually: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo [INFO] Restarting script with newly installed Python ...
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python still not on PATH. Open a new terminal and re-run install.bat.
        pause
        exit /b 1
    )
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Found Python %PY_VER%

:: ── Require Python 3.8+ ──────────────────────────────────────
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if %PY_MAJOR% LSS 3 (
    echo [ERROR] Python 3.8 or higher is required.
    pause
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 8 (
    echo [ERROR] Python 3.8 or higher is required.
    pause
    exit /b 1
)

:: ── Create virtual environment ────────────────────────────────
if not exist "venv\" (
    echo.
    echo [INFO] Creating virtual environment in .\venv ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [INFO] Virtual environment already exists, skipping creation.
)

:: ── Activate venv ─────────────────────────────────────────────
call venv\Scripts\activate.bat

:: ── Upgrade pip ───────────────────────────────────────────────
echo.
echo [INFO] Upgrading pip ...
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded.

:: ── Install project requirements ───────────────────────────
echo.
echo [INFO] Installing project requirements from requirements.txt ...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install one or more packages.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Installation complete!
echo.
echo  To activate the environment in a new terminal run:
echo      venv\Scripts\activate.bat
echo ============================================================
pause
goto :eof

:: ════════════════════════════════════════════════════════════
:: Subroutine — install Python automatically
:: Tries winget first, then falls back to a direct installer
::   download from python.org via PowerShell.
:: ════════════════════════════════════════════════════════════
:install_python
echo.
echo [INFO] Trying winget (Windows Package Manager) ...
winget --version >nul 2>&1
if not errorlevel 1 (
    winget install --id Python.Python.3.11 ^^
        --silent ^^
        --accept-source-agreements ^^
        --accept-package-agreements
    if not errorlevel 1 (
        echo [OK] Python installed via winget.
        :: Add common per-user install paths for current session
        set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python311"
        set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python311\Scripts"
        exit /b 0
    )
    echo [WARN] winget install returned an error, falling back to direct download.
)

echo [INFO] Downloading Python 3.11 installer from python.org ...
set "PY_INSTALLER=%TEMP%\python_installer.exe"
set "PY_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
powershell -NoProfile -Command ^
    "Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_INSTALLER%'"
if errorlevel 1 (
    echo [ERROR] Download failed. Check your internet connection.
    exit /b 1
)

echo [INFO] Running Python installer silently (current user only) ...
"%PY_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
if errorlevel 1 (
    echo [ERROR] Python installer exited with an error.
    del /f /q "%PY_INSTALLER%"
    exit /b 1
)
del /f /q "%PY_INSTALLER%"

:: Refresh PATH for the current session
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python311"
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python311\Scripts"
exit /b 0
