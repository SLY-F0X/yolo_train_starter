@echo off
set "ROOT=%~dp0"

if not exist "%ROOT%.venv\Scripts\activate.bat" (
  echo [ERROR] %ROOT%.venv\Scripts\activate.bat not found
  pause
  exit /b 1
)

start "YOLO venv" /D "%ROOT%" cmd /k "chcp 65001>nul & call .venv\Scripts\activate.bat"
