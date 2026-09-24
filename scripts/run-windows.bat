@echo off
cd /d "%~dp0.."
"%~dp0..\PlateDetection\.venv\Scripts\python.exe" -m streamlit run app.py --browser.gatherUsageStats=false
pause
