@echo off
echo ========================================
echo   Starting Gradio App with Public URL
echo ========================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the Gradio app with public sharing
python app_gradio.py --model model/CNN-Final.pkl --share

pause
