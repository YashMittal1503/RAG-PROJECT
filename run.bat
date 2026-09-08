@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Launching DocuChat RAG System (Backend + Frontend)
echo ============================================================
echo.

:: 1. Launch FastAPI Backend in a separate window
echo [1/2] Starting FastAPI Backend on http://localhost:8000 ...
start "DocuChat - Backend (FastAPI)" cmd /k "cd /d "%~dp0backend" && if exist venv\Scripts\activate.bat ( call venv\Scripts\activate.bat ) else ( echo [ERROR] backend\venv not found! && pause && exit /b ) && echo Backend virtualenv activated! && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

:: 2. Launch Next.js Frontend in a separate window
echo [2/2] Starting Next.js Frontend on http://localhost:3000 ...
start "DocuChat - Frontend (Next.js)" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo ============================================================
echo   Both services are launching in separate terminal windows!
echo.
echo   - Backend API & Docs: http://localhost:8000/docs
echo   - Frontend Web App:   http://localhost:3000
echo.
echo   Keep those terminal windows open while using the app.
echo   Press any key to close this launcher window.
echo ============================================================
pause >nul
