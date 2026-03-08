@echo off
echo.
echo ========================================
echo   🎹 Keyboard Keystroke Detection
echo   Starting Gradio Web Interface
echo ========================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run Gradio app (it will auto-detect your trained models)
python inference/app_gradio.py

pause
