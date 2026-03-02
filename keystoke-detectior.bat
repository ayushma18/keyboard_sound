@echo off
REM Quick launcher for Keystroke Detector UI with virtual environment setup

echo ========================================
echo Keystroke Detector
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if pygame is installed
venv\Scripts\python -c "import pygame" 2>nul
if errorlevel 1 (
    echo Installing pygame for smooth audio playback...
    venv\Scripts\pip install pygame
)

REM Run the application
echo Starting application...
echo.
python inference\keystroke_detector_ui.py

REM Deactivate when done
deactivate
