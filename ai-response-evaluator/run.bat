@echo off
echo Starting AI Response Quality Evaluator Agent...

:: Check if requirements are installed
echo Verifying Python packages...
py -m pip install -r requirements.txt

:: Start FastAPI Backend in a new window
echo Starting API Backend...
start "Evaluator Backend" cmd /k "py -m backend.app.main"

:: Wait 3 seconds for backend to start up
timeout /t 3 /nobreak >nul

:: Start Streamlit Frontend
echo Starting Frontend Dashboard...
py -m streamlit run frontend/app.py
pause
