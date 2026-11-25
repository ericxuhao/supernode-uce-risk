#!/usr/bin/env bash
# 超节点计算卡故障风险预测 —— 监测服务一键部署（Ubuntu 22.04 / x86_64，支持无外网的内网环境）
#
#   sudo bash install.sh                      # 默认：端口 8765，安装到 /opt/supernode，数据在 /var/lib/supernode
#   sudo bash install.sh --port 9000          # 换端口
#   sudo bash install.sh --help               # 全部选项
#
# 完成后，内网用户直接用浏览器访问 http://<服务器IP>:<端口>。
# 依赖全部来自同目录的 wheels/（Python 包）和 debs/（系统库），安装过程不访问外网。
set -euo pipefail

PORT=8765
PREFIX=/opt/supernode
DATA_DIR=/var/lib/supernode
SVC_USER=supernode
SVC_NAME=supernode-server
SOURCE=auto            # 页面默认数据源：auto（本机有计算卡则用集群实时数据，否则用开源数据回放）/ cluster / replay
REPLAY=1
LOCAL_AGENT=1
FIREWALL=1
ONLINE=0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
用法: sudo bash install.sh [选项]
  --port N            监听端口（默认 8765）
  --prefix DIR        程序安装目录（默认 /opt/supernode）
  --data-dir DIR      运行数据目录：采集记录、设置（默认 /var/lib/supernode）
  --user NAME         服务运行用户（默认 supernode，自动创建）
  --source S          页面默认数据源：auto | cluster | replay（默认 auto）
  --no-replay         不启用开源数据回放，只运行集群实时监测
  --no-local-agent    不采集本机计算卡（默认：本机有 nvidia-smi 时自动作为一个节点接入）
  --no-firewall       不自动放行防火墙端口
  --online            允许从 pip 源下载缺少的依赖（默认只用随包的 wheels/）
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --user) SVC_USER="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --no-replay) REPLAY=0; shift ;;
    --no-local-agent) LOCAL_AGENT=0; shift ;;
    --no-firewall) FIREWALL=0; shift ;;
    --online) ONLINE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知选项: $1"; usage; exit 2 ;;
  esac
done

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32m✔\033[0m %s\n' "$*"; }
warn() { printf '    \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31m✘ %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. 环境检查
step "1/7 检查环境"
[ "$(id -u)" -eq 0 ] || die "需要 root 权限：sudo bash install.sh"
[[ "$PORT" =~ ^[0-9]+$ ]] && [ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || die "端口无效: $PORT"
[[ "$SOURCE" =~ ^(auto|cluster|replay)$ ]] || die "--source 只能是 auto / cluster / replay"
[ -f "$HERE/supernode/server.py" ] && [ -f "$HERE/models/models.joblib" ] || die "请在解压后的发布包目录中运行本脚本"
command -v systemctl >/dev/null && [ -d /run/systemd/system ] || die "需要 systemd"
. /etc/os-release 2>/dev/null || true
if [ "${ID:-}" = "ubuntu" ] && [ "${VERSION_ID:-}" = "22.04" ]; then ok "系统 ${PRETTY_NAME}"
else warn "系统为 ${PRETTY_NAME:-未知}；本脚本按 Ubuntu 22.04 验证，其他发行版可能需要 --online"; fi
ARCH="$(uname -m)"
command -v python3 >/dev/null || die "未找到 python3（Ubuntu 22.04 自带 Python 3.10）"
PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "需要 Python ≥ 3.10，当前 $PYV"
ok "Python $PYV ($ARCH)"

WHEELS="$HERE/wheels"
USE_INDEX=0
PYTAG="cp${PYV/./}"
REQ="$HERE/requirements-lock.txt"
[ "$PYV" = "3.10" ] && REQ="$HERE/requirements-lock-py310.txt"
if [ -d "$WHEELS" ] && ls "$WHEELS"/numpy-*"$PYTAG"*"$ARCH"*.whl >/dev/null 2>&1; then
  ok "离线依赖包 wheels/ 与本机匹配（$PYTAG / $ARCH）"
elif [ "$ONLINE" -eq 1 ]; then
  warn "没有与本机匹配的离线依赖（需要 $PYTAG / $ARCH），将从 pip 源下载 $(basename "$REQ")"
  USE_INDEX=1
else
  die "wheels/ 中没有适用于 Python $PYV / $ARCH 的依赖包。请使用 ubuntu22.04-x86_64 离线包，或加 --online 联网安装"
fi
PIP_WHL=""
[ "$USE_INDEX" -eq 0 ] && PIP_WHL="$(ls "$WHEELS"/pip-*.whl 2>/dev/null | head -1 || true)"

if ss -ltnH "( sport = :$PORT )" 2>/dev/null | grep -q .; then
  if systemctl is-active --quiet "$SVC_NAME"; then warn "端口 $PORT 被已安装的 $SVC_NAME 占用，将升级并重启"
  else die "端口 $PORT 已被其他程序占用（ss -ltnp | grep :$PORT），请用 --port 换一个"; fi
fi
NVSMI="$(command -v nvidia-smi || true)"
for c in /usr/bin/nvidia-smi /usr/local/nvidia/bin/nvidia-smi /usr/lib/wsl/lib/nvidia-smi; do
  [ -z "$NVSMI" ] && [ -x "$c" ] && NVSMI="$c"
done
SVC_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
if [ -n "$NVSMI" ] && "$NVSMI" -L >/dev/null 2>&1; then
  SVC_PATH="$(dirname "$NVSMI"):$SVC_PATH"          # 服务环境的 PATH 中也要能找到管理工具
  HAS_CARDS=1; ok "本机计算卡：$("$NVSMI" -L | wc -l) 张（$NVSMI）"
else HAS_CARDS=0; warn "本机未检测到计算卡管理工具（nvidia-smi），仅运行监测服务"; fi

# ---------------------------------------------------------------- 2. 系统库
step "2/7 检查系统库"
if ldconfig -p | grep -q 'libgomp.so.1'; then ok "libgomp1 已安装"
elif ls "$HERE"/debs/libgomp1_*.deb >/dev/null 2>&1; then
  dpkg -i "$HERE"/debs/libgomp1_*.deb >/dev/null && ok "已从 debs/ 安装 libgomp1（LightGBM 运行库）"
elif [ "$ONLINE" -eq 1 ]; then apt-get install -y libgomp1 >/dev/null && ok "已安装 libgomp1"
else die "缺少 libgomp1，且 debs/ 中没有安装包"; fi

# ---------------------------------------------------------------- 3. 运行用户
step "3/7 创建运行用户 $SVC_USER"
if id "$SVC_USER" >/dev/null 2>&1; then ok "用户已存在"
else useradd --system --home-dir "$DATA_DIR" --no-create-home --shell /usr/sbin/nologin "$SVC_USER"; ok "已创建系统用户"; fi
for g in systemd-journal video render; do        # 读取内核错误日志、访问计算卡设备
  getent group "$g" >/dev/null && usermod -aG "$g" "$SVC_USER"
done
ok "已加入组: $(id -nG "$SVC_USER")"

# ---------------------------------------------------------------- 4. 安装程序
step "4/7 安装程序到 $PREFIX"
if systemctl list-unit-files "$SVC_NAME.service" >/dev/null 2>&1 && systemctl is-active --quiet "$SVC_NAME"; then
  systemctl stop "$SVC_NAME"; ok "已停止正在运行的旧版本"
fi
mkdir -p "$PREFIX" "$DATA_DIR"
KEEP_CFG=""
if [ -f "$PREFIX/supernode/config.json" ]; then KEEP_CFG="$(mktemp)"; cp "$PREFIX/supernode/config.json" "$KEEP_CFG"; fi
for d in uce_risk supernode models demo_data docs deploy examples tests; do
  rm -rf "${PREFIX:?}/$d"
  [ -e "$HERE/$d" ] && cp -a "$HERE/$d" "$PREFIX/$d"
done
for f in README.md VERSION pyproject.toml requirements.txt requirements-lock.txt requirements-lock-py310.txt \
         requirements-agent.txt install.sh uninstall.sh; do
  [ -e "$HERE/$f" ] && cp -a "$HERE/$f" "$PREFIX/$f"
done
if [ -n "$KEEP_CFG" ]; then
  cp "$PREFIX/supernode/config.json" "$PREFIX/supernode/config.json.new"
  cp "$KEEP_CFG" "$PREFIX/supernode/config.json"; rm -f "$KEEP_CFG"
  ok "保留了原有配置 supernode/config.json（新版默认配置另存为 config.json.new）"
fi
python3 - "$PREFIX/supernode/config.json" "$PORT" "$REPLAY" "$LOCAL_AGENT" <<'PY'
import json, sys
p, port, replay, local = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4] == "1"
c = json.load(open(p, encoding="utf-8"))
c.setdefault("server", {}).update(host="0.0.0.0", port=port)
c["replay"] = replay
c["local_agent"] = local
json.dump(c, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY
[ "$REPLAY" -eq 1 ] || rm -rf "${PREFIX:?}/demo_data"
[ -d "$PREFIX/demo_data" ] || REPLAY=0               # 没有回放数据（如从源码仓库安装）时只运行集群实时监测
VER="$(head -1 "$HERE/VERSION" 2>/dev/null || true)"; VER="${VER%$'\r'}"
ok "程序版本 ${VER:-?}，监听 0.0.0.0:$PORT，回放 $([ "$REPLAY" -eq 1 ] && echo 开 || echo 关)"

# ---------------------------------------------------------------- 5. Python 依赖
step "5/7 安装 Python 依赖（$(basename "$REQ")）"
VENV="$PREFIX/.venv"
PY="$VENV/bin/python"
PYPATH=""
if [ ! -x "$PY" ] || ! "$PY" -c 'import sys' 2>/dev/null || [ "$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" != "$PYV" ]; then
  rm -rf "$VENV"
  if [ -z "$PIP_WHL" ]; then                      # 联网安装：虚拟环境自带 pip（需要 python3-venv）
    python3 -m venv "$VENV" >/dev/null 2>&1 || die "无法创建虚拟环境：请先 sudo apt install python3-venv，或使用离线包"
    "$VENV/bin/python" -m pip install -q --disable-pip-version-check --upgrade pip || true
    ok "已创建虚拟环境 $VENV"
  elif python3 -m venv --without-pip "$VENV" 2>/dev/null; then ok "已创建虚拟环境 $VENV"
  else                                            # 没有 venv 模块时：装到独立目录，用 PYTHONPATH 加载
    rm -rf "$VENV"; mkdir -p "$PREFIX/pylib"; PY="$(command -v python3)"; PYPATH="$PREFIX/pylib"
    warn "系统缺少 venv 模块，改为安装到 $PYPATH"
  fi
fi
[ -d "$PREFIX/pylib" ] && [ ! -d "$VENV" ] && { PY="$(command -v python3)"; PYPATH="$PREFIX/pylib"; }
pip_install() {
  local target=()
  [ -n "$PYPATH" ] && target=(--target "$PYPATH" --upgrade)
  if [ -n "$PIP_WHL" ]; then
    PYTHONPATH="$PIP_WHL" "$PY" -m pip "$@" "${target[@]}"
  else
    "$PY" -m pip "$@" "${target[@]}"
  fi
}
PIP_OPTS=(install --disable-pip-version-check --no-warn-script-location -q)
if [ "$USE_INDEX" -eq 1 ]; then
  pip_install "${PIP_OPTS[@]}" -r "$REQ"
else
  pip_install "${PIP_OPTS[@]}" --no-index --find-links "$WHEELS" -r "$REQ" $( [ -z "$PYPATH" ] && [ -n "$PIP_WHL" ] && echo "$PIP_WHL" )
fi
PYTHONPATH="$PYPATH" "$PY" -c "import numpy, pandas, sklearn, lightgbm, flask, waitress, pyarrow; print('    \033[32m✔\033[0m numpy', numpy.__version__, '| pandas', pandas.__version__, '| scikit-learn', sklearn.__version__, '| lightgbm', lightgbm.__version__)"
( cd "$PREFIX" && PYTHONPATH="$PREFIX/supernode:$PREFIX${PYPATH:+:$PYPATH}" "$PY" -W ignore -c "import joblib; joblib.load('models/models.joblib')" ) \
  || die "预训练模型加载失败（依赖版本不一致）。可在安装后用 supernode/retrain.py 重训"
ok "预训练模型可加载"
PYTHONPATH="$PYPATH" "$PY" -m compileall -q "$PREFIX/uce_risk" "$PREFIX/supernode" >/dev/null 2>&1 || true

# 权限：程序只读；数据目录与模型目录（重训会写入）属于运行用户
chown -R root:root "$PREFIX"
chown -R "$SVC_USER:$SVC_USER" "$DATA_DIR" "$PREFIX/models"
chmod 750 "$DATA_DIR"

# 页面默认数据源（只在首次安装时写入；之后在页面上切换）
if [ ! -f "$DATA_DIR/settings.json" ]; then
  SRC="$SOURCE"
  [ "$SRC" = "auto" ] && { [ "$HAS_CARDS" -eq 1 ] && [ "$LOCAL_AGENT" -eq 1 ] && SRC=cluster || SRC=replay; }
  [ "$REPLAY" -eq 1 ] || SRC=cluster
  printf '{\n "interval_s": 300,\n "retention_days": 30,\n "default_source": "%s"\n}\n' "$SRC" > "$DATA_DIR/settings.json"
  chown "$SVC_USER:$SVC_USER" "$DATA_DIR/settings.json"
  ok "页面默认数据源：$([ "$SRC" = cluster ] && echo 集群实时监测 || echo 开源数据回放)；采集频率 5 分钟，保留 30 天"
fi

# ---------------------------------------------------------------- 6. 系统服务与防火墙
step "6/7 注册系统服务 $SVC_NAME"
cat > "/etc/systemd/system/$SVC_NAME.service" <<EOF
[Unit]
Description=Supernode compute-card failure risk prediction service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$PREFIX
Environment=HOST=0.0.0.0
Environment=PORT=$PORT
Environment=SUPERNODE_DATA_DIR=$DATA_DIR
Environment=PATH=$SVC_PATH
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
${PYPATH:+Environment=PYTHONPATH=$PYPATH}
ExecStart=$PY -W ignore $PREFIX/supernode/server.py
Restart=always
RestartSec=5
User=$SVC_USER
Group=$SVC_USER
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable "$SVC_NAME" >/dev/null 2>&1
systemctl restart "$SVC_NAME"
ok "已设置开机自启"

if [ "$FIREWALL" -eq 1 ]; then
  if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "Status: active"; then
    ufw allow "$PORT/tcp" comment "supernode" >/dev/null && ok "ufw 已放行 TCP $PORT"
  elif command -v firewall-cmd >/dev/null && firewall-cmd --state >/dev/null 2>&1; then
    firewall-cmd --permanent --add-port="$PORT/tcp" >/dev/null && firewall-cmd --reload >/dev/null && ok "firewalld 已放行 TCP $PORT"
  else ok "未启用主机防火墙，无需放行"; fi
fi

# ---------------------------------------------------------------- 7. 健康检查
step "7/7 等待服务就绪"
ready=0
for _ in $(seq 1 90); do
  if PYTHONPATH="$PYPATH" "$PY" -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$PORT/api/status', timeout=3)" 2>/dev/null; then ready=1; break; fi
  systemctl is-active --quiet "$SVC_NAME" || { sleep 2; systemctl is-active --quiet "$SVC_NAME" || break; }
  sleep 2
done
if [ "$ready" -ne 1 ]; then
  journalctl -u "$SVC_NAME" -n 30 --no-pager || true
  die "服务未能启动，日志见上（journalctl -u $SVC_NAME）"
fi
ok "服务已就绪"

IPS="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.' | grep -v '^127\.' || true)"
[ -n "$IPS" ] || IPS="<服务器IP>"
printf '\n\033[1;32m部署完成。内网用户可用浏览器访问：\033[0m\n'
for ip in $IPS; do printf '    http://%s:%s\n' "$ip" "$PORT"; done
FIRST="$(echo "$IPS" | head -1)"
cat <<EOF

计算节点接入（在每台计算节点上执行一条命令；节点只需 Python 3 标准库，无需外网）：
    curl -fsSL http://$FIRST:$PORT/agent/install.sh | sudo bash
常用命令：
    systemctl status $SVC_NAME          # 状态
    journalctl -u $SVC_NAME -f          # 日志
    systemctl restart $SVC_NAME         # 重启
    sudo bash $PREFIX/uninstall.sh      # 卸载（保留数据；加 --purge 同时删除数据与用户）
程序目录 $PREFIX，数据目录 $DATA_DIR，配置 $PREFIX/supernode/config.json
EOF
