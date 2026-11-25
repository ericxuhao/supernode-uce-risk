#!/usr/bin/env bash
# 前台快速启动（不安装为系统服务；正式部署请用发布包根目录的 install.sh）。
# 首次运行创建 .venv 并安装依赖：有随包的 wheels/ 时离线安装，否则从 pip 源安装。
set -euo pipefail
cd "$(dirname "$0")/.."
PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
REQ=requirements-lock.txt; [ "$PYV" = "3.10" ] && REQ=requirements-lock-py310.txt
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  if [ -d wheels ]; then .venv/bin/pip install -q --no-index --find-links wheels -r "$REQ"
  else .venv/bin/pip install -q -r "$REQ" || .venv/bin/pip install -q -r requirements.txt; fi
fi
exec .venv/bin/python -W ignore supernode/server.py
