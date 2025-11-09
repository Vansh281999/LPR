@echo off
echo ============================================
echo Starting Smart LPR System
echo ============================================
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo Starting Flask server...
echo Open your browser to: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ============================================
python app.py
