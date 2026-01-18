@echo off
chcp 65001 >nul
setlocal

echo install uv
pip install -U uv

echo create uv venv
uv venv --python 3.12 --seed --no-cache-dir --relocatable --allow-existing --link-mode=copy --prompt "Yolo-Env"

call .venv\Scripts\activate.bat
chcp 65001 >nul

set UV_HTTP_TIMEOUT=600
set HTTP_TIMEOUT=600

echo uv pip install torch torchvision torchaudio
uv pip install torch torchvision torchaudio --torch-backend=cu130 --no-cache-dir --link-mode=copy

echo uv sync
uv sync --index https://download.pytorch.org/whl/cu130 --no-cache-dir --link-mode=copy

pause