@echo off
setlocal
cd /d "%~dp0"
echo ====================================================
echo Starting PayGuard AI Streamlit Dashboard...
echo Working Directory: "%CD%"
echo ====================================================

if exist "%~dp0.venv\Scripts\streamlit.exe" (
    call "%~dp0.venv\Scripts\streamlit.exe" run "%~dp0dashboard\app.py"
) else (
    call streamlit run "%~dp0dashboard\app.py"
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Dashboard stopped or failed to start.
)
pause
