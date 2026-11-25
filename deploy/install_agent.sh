#!/usr/bin/env bash
# 在计算节点上安装 Agent：sudo ./install_agent.sh http://<监测服务器>:8765
# 只复制两个文件（agent.py 与采集器核心），只依赖 Python 3 标准库与厂商管理工具。
set -euo pipefail
SERVER="${1:?usage: sudo ./install_agent.sh http://<server>:8765}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }
command -v nvidia-smi >/dev/null || echo "warning: management tool (nvidia-smi) not found; only side channels will be probed"
install -d /opt/supernode-agent/collector /var/lib/supernode-agent
install -m 0644 "$HERE/supernode/agent.py" /opt/supernode-agent/agent.py
install -m 0644 "$HERE/supernode/collector/gpu_ecc_collector.py" /opt/supernode-agent/collector/gpu_ecc_collector.py
python3 /opt/supernode-agent/agent.py --print-capabilities --base-dir /var/lib/supernode-agent > /var/lib/supernode-agent/capabilities.json || true
echo "capabilities written to /var/lib/supernode-agent/capabilities.json"
sed "s#__SERVER__#${SERVER}#" "$HERE/deploy/supernode-agent.service" > /etc/systemd/system/supernode-agent.service
systemctl daemon-reload
systemctl enable --now supernode-agent
systemctl --no-pager status supernode-agent | head -5
