# DocuChat RAG System - PowerShell One-Click Launcher
$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Launching DocuChat RAG System (Backend + Frontend)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Launch FastAPI Backend in a new PowerShell window
Write-Host "`n[1/2] Starting FastAPI Backend on http://localhost:8000..." -ForegroundColor Green
$backendCmd = "cd '$rootDir\backend'; .\venv\Scripts\Activate.ps1; uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

# 2. Launch Next.js Frontend in a new PowerShell window
Write-Host "[2/2] Starting Next.js Frontend on http://localhost:3000..." -ForegroundColor Green
$frontendCmd = "cd '$rootDir\frontend'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  Both services launched in separate windows!" -ForegroundColor Cyan
Write-Host "  - Backend API:  http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "  - Frontend App: http://localhost:3000" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
