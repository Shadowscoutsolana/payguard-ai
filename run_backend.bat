@echo off
setlocal
cd /d "%~dp0"
echo ====================================================
echo Starting PayGuard AI Backend Server on Port 8000...
echo Working Directory: "%CD%"
echo ====================================================

if exist "%~dp0.venv\Scripts\uvicorn.exe" (
    call "%~dp0.venv\Scripts\uvicorn.exe" backend.app.main:app --reload --port 8000
) else (
    call uvicorn backend.app.main:app --reload --port 8000
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server stopped or failed to start.
)
pause
