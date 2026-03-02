@echo off
echo ========================================
echo   Starting Gradio Keystroke Detector
echo ========================================
echo.

REM Kill any existing Gradio processes on port 7860
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860') do (
    taskkill /F /PID %%a 2>nul
)

echo Cleaned up old processes...
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the Gradio app
python app_gradio.py --model model/CNN-Final.pkl

pause
