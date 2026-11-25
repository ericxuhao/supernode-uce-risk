# 本机快速启动（Windows PowerShell）：首次运行会创建虚拟环境并安装依赖
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Test-Path .venv)) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install -q -r requirements-lock.txt
if (-not $?) { & .\.venv\Scripts\python.exe -m pip install -q -r requirements.txt }
& .\.venv\Scripts\python.exe supernode\server.py
