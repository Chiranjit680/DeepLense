@echo off
echo ================================
echo   DeepLense Backend Server
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

:: Change to the Backend directory
cd /d "%~dp0"

:: Check if virtual environment exists
if exist "..\..\..\myenv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call ..\..\..\myenv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
)

:: Start the server
echo Starting DeepLense Backend Server...
echo.
echo API will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python start_server.py

pause
