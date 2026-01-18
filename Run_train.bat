@echo off
chcp 65001 >nul
setlocal

if not exist ".venv\Scripts\activate.bat" (
echo [ERROR] .venv not exist
pause
)

call .venv\Scripts\activate.bat
chcp 65001 >nul
python yolo_train_starter.py
pause