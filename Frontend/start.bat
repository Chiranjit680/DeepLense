@echo off
echo ================================
echo   DeepLense Frontend Launcher
echo ================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

:: Change to the Frontend directory
cd /d "%~dp0"

:: Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call myenv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
)

:: Run the launcher
echo Starting DeepLense Frontend...
python run.py

pause
