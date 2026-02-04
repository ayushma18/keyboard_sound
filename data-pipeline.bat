@echo off
REM Quick launcher for Keyboard Acoustic Research Tool

echo ========================================
echo Keyboard Acoustic Research Tool
echo Modular Edition
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "myenv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv myenv
    echo Then: myenv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
call myenv\Scripts\activate.bat

REM Check if pygame is installed
python -c "import pygame" 2>nul
if errorlevel 1 (
    echo Installing pygame for smooth audio playback...
    pip install pygame
)

REM Run the application
echo Starting application...
echo.
python data-pipeline.py

REM Deactivate when done
deactivate
