#!/usr/bin/env bash
# 卸载监测服务。默认保留运行数据（采集记录、设置）以便重装；--purge 同时删除数据目录与运行用户。
#   sudo bash uninstall.sh [--purge] [--prefix /opt/supernode] [--data-dir /var/lib/supernode] [--user supernode]
set -euo pipefail
PREFIX=/opt/supernode
DATA_DIR=/var/lib/supernode
SVC_USER=supernode
SVC_NAME=supernode-server
PURGE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --purge) PURGE=1; shift ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --user) SVC_USER="$2"; shift 2 ;;
    *) echo "未知选项: $1"; exit 2 ;;
  esac
done
[ "$(id -u)" -eq 0 ] || { echo "需要 root：sudo bash uninstall.sh"; exit 1; }
UNIT="/etc/systemd/system/$SVC_NAME.service"
PORT="$(grep -oP '^Environment=PORT=\K[0-9]+' "$UNIT" 2>/dev/null || true)"
systemctl disable --now "$SVC_NAME" 2>/dev/null || true
rm -f "$UNIT"
systemctl daemon-reload
if [ -n "$PORT" ] && command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "^$PORT/tcp.*supernode"; then
  ufw delete allow "$PORT/tcp" >/dev/null || true
fi
case "$PREFIX" in /|/usr|/usr/*|/etc|/var|/home) echo "拒绝删除 $PREFIX"; exit 1 ;; esac
rm -rf "$PREFIX"
echo "已卸载服务与程序目录 $PREFIX"
if [ "$PURGE" -eq 1 ]; then
  case "$DATA_DIR" in /|/var|/var/lib|/home) echo "拒绝删除 $DATA_DIR"; exit 1 ;; esac
  rm -rf "$DATA_DIR"
  id "$SVC_USER" >/dev/null 2>&1 && userdel "$SVC_USER" 2>/dev/null || true
  echo "已删除数据目录 $DATA_DIR 与用户 $SVC_USER"
else
  echo "运行数据保留在 $DATA_DIR（彻底删除：sudo bash uninstall.sh --purge）"
fi
