# 部署文档

## 0. 一键部署（Ubuntu 22.04 · A100 服务器 · 内网，推荐）

离线包 `supernode-uce-risk-<版本>-ubuntu22.04-cp310-x86_64-offline.tar.gz` 内含程序、模型、全部 Python 依赖（`wheels/`）和 LightGBM 所需的系统库（`debs/`）。**安装全程不访问外网**，也不需要配置 pip 源或 apt 源。

**第 1 步：在监测服务器上执行（一般选一台 A100 服务器或管理节点）**

```bash
# 把离线包拷到服务器（U 盘 / 跳板机 / scp 均可），然后：
sha256sum -c supernode-uce-risk-1.5.0-ubuntu22.04-cp310-x86_64-offline.tar.gz.sha256    # 可选：校验
tar xzf supernode-uce-risk-1.5.0-ubuntu22.04-cp310-x86_64-offline.tar.gz
cd supernode-uce-risk-1.5.0
sudo bash install.sh                 # 默认端口 8765；换端口：sudo bash install.sh --port 9000
```

脚本约 30 秒完成以下工作，结束时打印访问地址（本机每个网卡 IP 各一行）：

| 步骤 | 内容 |
|---|---|
| 1. 检查环境 | root 权限、systemd、Ubuntu 版本、Python ≥ 3.10、离线依赖与本机匹配、端口未被占用、本机计算卡 |
| 2. 系统库 | 缺少 `libgomp1` 时从 `debs/` 安装 |
| 3. 运行用户 | 创建系统用户 `supernode`，加入 `systemd-journal`（读内核错误日志）、`video` / `render`（访问计算卡） |
| 4. 安装程序 | 复制到 `/opt/supernode`；升级时保留原有 `config.json` |
| 5. Python 依赖 | 创建虚拟环境 `/opt/supernode/.venv`，从 `wheels/` 离线安装；检查预训练模型能加载 |
| 6. 系统服务 | 注册 `supernode-server`，开机自启、异常自动重启；ufw / firewalld 开启时自动放行端口 |
| 7. 健康检查 | 等待服务就绪，打印 `http://<IP>:<端口>` |

```
部署完成。内网用户可用浏览器访问：
    http://10.0.12.5:8765
计算节点接入（在每台计算节点上执行一条命令；节点只需 Python 3 标准库，无需外网）：
    curl -fsSL http://10.0.12.5:8765/agent/install.sh | sudo bash
```

**第 2 步：内网用户访问** —— 浏览器直接打开 `http://<服务器IP>:8765`。服务器本机的计算卡会自动作为一个节点接入（R01-N1）。

**第 3 步：接入其他计算节点** —— 在每台计算节点上执行上面打印的那一条命令即可：
- Agent 从监测服务器下载，不需要拷贝安装包，也不需要外网；
- 自动注册为 systemd 服务 `supernode-agent`；
- 几十秒后，该节点出现在页面"数据采集与处理"中。

批量接入示例（在跳板机上执行）：`for h in $(cat nodes.txt); do ssh root@$h "curl -fsSL http://10.0.12.5:8765/agent/install.sh | bash"; done`

### install.sh 选项

| 选项 | 默认 | 说明 |
|---|---|---|
| `--port N` | 8765 | 监听端口（始终监听 0.0.0.0，即本机所有网卡） |
| `--prefix DIR` | `/opt/supernode` | 程序目录 |
| `--data-dir DIR` | `/var/lib/supernode` | 运行数据：采集记录、设置、节点位置 |
| `--user NAME` | `supernode` | 服务运行用户 |
| `--source S` | `auto` | 页面默认数据源：`auto`（本机有计算卡时为集群实时监测，否则为开源数据回放）/ `cluster` / `replay`；之后可在页面上切换 |
| `--no-replay` | — | 不部署开源数据回放，只做集群实时监测 |
| `--no-local-agent` | — | 不采集服务器本机的计算卡 |
| `--no-firewall` | — | 不自动修改防火墙 |
| `--online` | — | 离线依赖与本机不匹配时（如非 Ubuntu 22.04），允许从 pip 源安装 |

### 日常运维

```bash
systemctl status supernode-server        # 状态
journalctl -u supernode-server -f        # 日志
sudo systemctl restart supernode-server  # 重启
sudo bash install.sh                     # 升级：在新版本的解压目录中重新执行（保留数据和配置）
sudo bash /opt/supernode/uninstall.sh    # 卸载（保留数据）；--purge 同时删除数据目录与运行用户
curl -fsSL http://<服务器IP>:8765/agent/install.sh | sudo bash -s -- --uninstall    # 卸载某个节点的 Agent
```

### 访问不了时

1. 在服务器上执行 `curl -s http://127.0.0.1:8765/api/status`：不通说明服务没起来，查看 `journalctl -u supernode-server -n 50`。
2. 本机通、别的机器不通：检查主机防火墙（`sudo ufw status`；脚本只在 ufw / firewalld 开启时自动放行），以及机房交换机 / 安全组是否放行该端口。
3. 确认使用的是服务器在内网中的 IP（`hostname -I`），而不是 docker、IB 等网卡的地址。

### 其他平台的离线包

离线包与目标机器的 Python 版本、CPU 架构绑定（Ubuntu 22.04 自带 Python 3.10，x86_64）。其他环境（如 Ubuntu 24.04 / Python 3.12，或 aarch64）可以：
- 在可以联网、且与目标环境相同的机器上制作 wheels，再打包：
  ```bash
  python3 -m pip download -r requirements-lock.txt pip -d wheelhouse/wheels    # Python 3.10 用 requirements-lock-py310.txt
  (mkdir -p wheelhouse/debs && cd wheelhouse/debs && apt-get download libgomp1)
  python tools/make_release.py --offline-wheels wheelhouse                       # 在源码仓库中执行
  ```
- 或者在能访问内网 pip 镜像的机器上，直接执行 `sudo bash install.sh --online`。

## 1. 组件与部署拓扑

```
 计算节点 × N（如 8 卡 A100 服务器）                    监测服务器（1 台）
 ┌──────────────────────────────┐   HTTP 推送        ┌─────────────────────────────────────────────┐
 │ supernode-agent              │ ───────────────▶  │ supernode-server（端口 8765）               │
 │ · 自动探测本机可采集的指标   │  /api/ingest       │ · 接收、转换、按字段生成模型输入            │
 │ · 按频率采集（默认 5 分钟）  │ ◀───────────────  │ · 按频率扫描 1/4/24 小时风险、故障树归因    │
 │ · 读内核错误日志、计数变化   │  /api/agent/config │ · Web 页面；开源数据回放（可选）            │
 │ · 断点续传                   │                    │ · 数据保留与清理                            │
 └──────────────────────────────┘                    └─────────────────────────────────────────────┘
```

- **节点 Agent**：只依赖 Python 3 标准库和厂商管理工具，可在任意 Linux 计算节点上运行。
- **监测服务**：需要 Python ≥ 3.10 和 `requirements.txt` 中的依赖。
- 监测服务器本机如果装有计算卡，会自动启动一个本机 Agent，作为一个节点接入。

## 2. 环境要求

| 项目 | 监测服务器 | 计算节点（Agent） |
|---|---|---|
| 操作系统 | Linux（推荐）/ Windows / macOS | Linux |
| Python | ≥ 3.10（已测试：Ubuntu 22.04 自带 3.10.12、Windows 3.13.9） | ≥ 3.8，仅标准库 |
| Python 依赖 | 离线包已附带；否则用 `requirements-lock-py310.txt`（Python 3.10）/ `requirements-lock.txt`（Python ≥ 3.11）/ `requirements.txt`（宽松版本） | 无；可选 `nvidia-ml-py`（设备事件监听） |
| 系统库 | `libgomp1`（LightGBM 需要；离线包附带） | — |
| 其他软件 | — | 厂商管理工具（如 `nvidia-smi`，随驱动安装）；`journalctl` 或 `dmesg` |
| 权限 | 普通用户 | 读取内核错误日志需要 root，或把运行用户加入 `systemd-journal` 组 |
| CPU / 内存 | 4 核 / 8 GB 可支持 1,024 张卡（一次扫描约 1 秒） | Agent 限制为 20% CPU、512 MB 内存 |
| 网络 | 开放 TCP 8765 给各节点与浏览器 | 能访问监测服务器的 8765 端口 |
| 磁盘 | 按采集频率估算，见 §6 | 本地缓存，按保留天数清理 |

**预训练模型的版本约束**：`models/models.joblib` 用 scikit-learn 1.7.2、lightgbm 4.7.0 保存（numpy 2.2 / 2.3 均可加载，已在 Ubuntu 22.04 上验证）。如果环境中这两个库的主版本不同，模型可能无法加载。此时：
- 安装 `requirements-lock.txt` 中的版本；
- 或者用 `supernode/retrain.py` 在自己的数据上重训（§8）。

## 3. 监测服务部署

Ubuntu 22.04 推荐直接使用 §0 的一键部署。以下为手动方式。

### 3.1 单机快速启动（前台运行，不注册服务）

```bash
cd supernode-uce-risk-<版本>
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt          # 不能联网时：先在可联网机器上 pip download -r requirements-lock.txt -d wheels，再 pip install --no-index -f wheels -r requirements-lock.txt
python supernode/server.py                    # 默认监听 0.0.0.0:8765
```

Windows 可以直接运行 `deploy\run_server.ps1`；Linux / macOS 可以运行 `deploy/run_server.sh`。

启动日志会打印以下信息，打开 `http://<服务器>:8765` 即可访问：
- 模型目录；
- 回放是否开启；
- 采集设置。

### 3.2 systemd（生产环境推荐）

```bash
sudo useradd -r -s /usr/sbin/nologin supernode
sudo mkdir -p /opt/supernode /var/lib/supernode && sudo cp -r . /opt/supernode
cd /opt/supernode && sudo python3 -m venv .venv && sudo .venv/bin/pip install -r requirements-lock.txt
sudo chown -R supernode:supernode /opt/supernode /var/lib/supernode
sudo cp deploy/supernode-server.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now supernode-server
journalctl -u supernode-server -f
```

服务的端口、监听地址和数据目录在 `supernode-server.service` 的 `Environment=` 中修改。

### 3.3 Docker

```bash
docker build -t supernode-server -f deploy/Dockerfile .
docker run -d --name supernode -p 8765:8765 -v $PWD/var:/data supernode-server
# 或：cd deploy && docker compose up -d
```

- 运行数据写入容器的 `/data`，请挂载到宿主机。
- 节点 Agent 需要访问宿主机的计算卡和内核日志，不建议放进容器，按 §4 部署。

## 4. 计算节点接入（Agent）

### 4.1 一键安装（推荐）

监测服务自带 Agent 安装脚本，在每个计算节点上执行：

```bash
curl -fsSL http://<监测服务器>:8765/agent/install.sh | sudo bash
```

没有 curl 时：`wget -qO- http://<监测服务器>:8765/agent/install.sh | sudo bash`。页面"数据采集与处理"中也显示这条命令。

### 4.2 从安装包安装（无法访问监测服务器的 HTTP 端口时）

在每个计算节点上执行（需要 root）：

```bash
scp -r supernode-uce-risk-<版本> node01:/tmp/ && ssh node01
cd /tmp/supernode-uce-risk-<版本> && sudo ./deploy/install_agent.sh http://<监测服务器>:8765
```

安装脚本做以下几件事：
1. 只复制 `agent.py` 和 `collector/gpu_ecc_collector.py` 两个文件到 `/opt/supernode-agent`；
2. 探测本机采集能力，结果写入 `/var/lib/supernode-agent/capabilities.json`；
3. 生成 `supernode-agent` 服务并启动。

几十秒内，该节点会出现在页面"数据采集与处理 → 采集节点与采集能力自动判定"中。

### 4.3 手动运行与自检

```bash
python3 agent.py --print-capabilities                    # 只查看本机能采集什么（指标、内核日志、互连）
python3 agent.py --server http://<监测服务器>:8765        # 前台运行
python3 agent.py --server ... --interval 60              # 固定频率（不跟随服务端设置）
python3 agent.py --server ... --simulate 4               # 模拟 4 个 8 卡节点（无硬件时测试整条链路）
```

### 4.4 能力判定与模型选择

服务端根据各节点上报的能力，判定能构建哪些模型输入，结果显示在页面上：

| 节点能提供的数据 | 判定 | 可用功能 |
|---|---|---|
| 温度 / 功率 + 可纠正错误计数（+ 内核错误日志、行重映射） | 方案 3（另含错误事件） | 完整风险预测与归因 |
| 温度 / 功率 + 内核错误日志 | 方案 1 | 风险预测与归因 |
| 只有温度 / 功率（如消费级卡没有 ECC） | 仅遥测 | 只能做热 / 供电归因，不给出有效的故障风险 |

### 4.5 节点在拓扑中的位置

- 默认按接入顺序依次占用 R01-N1、R01-N2……
- 如需固定位置，在 `<data_dir>/cluster/inventory.json` 中指定，例如 `{"gpu-node-01": "R01-N1", "gpu-node-02": "R01-N2"}`，然后重启服务。
- 已分配的位置记录在 `hosts.json`，重启后不变。

## 5. 配置

配置文件为 `supernode/config.json`，也可以用环境变量 `SUPERNODE_CONFIG` 指定另一份。相对路径相对于发布包根目录；如果设置了环境变量 `SUPERNODE_HOME`，则相对于它。

| 配置项 | 默认 | 说明 | 环境变量覆盖 |
|---|---|---|---|
| `server.host` / `server.port` | `0.0.0.0` / `8765` | 监听地址和端口。安装了 waitress（依赖中已包含）时使用 waitress 生产服务，否则退回 Flask 自带服务 | `HOST` / `PORT`；`THREADS`（waitress 线程数，默认 16） |
| `data_dir` | `var` | 运行数据：采集设置、集群原始记录（`cluster/raw/日期/节点.jsonl`）、节点位置、回放缓冲 | `SUPERNODE_DATA_DIR` |
| `model_dir` | `models` | `models.joblib`、`build_meta.json` | `SUPERNODE_MODEL_DIR` |
| `demo_dir` | `demo_data` | 开源数据回放；目录不存在或 `replay: false` 时只运行集群实时模式 | `SUPERNODE_DEMO_DIR` |
| `replay` | `true` | 是否启用开源数据回放 | — |
| `local_agent` | `true` | 服务器本机有计算卡时是否自动接入 | — |
| `topology` | 16 × 8 × 8 | 机柜数、每机柜节点数、每节点卡数 | — |
| `horizons` | 1h / 4h / 1d | 预测时间窗 | — |
| `sources.dcgm_exporter_urls`、`collector_spool_dirs` | — | 本机探测的遥测导出端点和采集器目录 | — |
| `open_data` | — | 仅 `build.py` 重建开源模型时使用 | — |

**采集频率和数据保留天数**在页面"数据采集与处理 → 集群实时采集设置"中修改：
- 保存到 `<data_dir>/settings.json`；
- 各节点 Agent 在 1 分钟内拉取生效；
- 默认采集频率 5 分钟，默认保留 30 天。

## 6. 存储估算与数据保留

- 单条 A100 类采样记录约 1 KB。页面会按实测的记录大小，显示"当前规模每天"和"保留期合计"。

  | 采集频率 | 1,024 张卡每天 | 保留 30 天合计 |
  |---|---|---|
  | 30 秒 | 约 3.1 GB | 约 93 GB |
  | 5 分钟（默认） | 约 300 MB | 约 9 GB |
  | 15 分钟 | 约 100 MB | 约 3 GB |

- 服务端每小时清理超过保留天数的原始记录；Agent 本地的数据也按保留期清理。
- 内核错误日志不受采集频率影响，始终实时读取。

## 7. 安全

- 原型**没有登录认证**，接收数据的接口也不校验来源。请只部署在内网，或者放在带认证的反向代理（如 Nginx + 统一认证）之后。
- 用防火墙限制 8765 端口，只允许计算节点和运维网段访问。
- 页面上的"保存并下发到各节点"会改变全集群的采集频率，请只开放给运维人员。

## 8. 升级与重训

- **升级**：
  1. 停止服务；
  2. 用新包覆盖 `uce_risk/`、`supernode/`、`models/`，保留 `var/`；
  3. 启动服务。
- **在自己集群的数据上重训**：集群累计到足够的历史数据后（至少 20 次以上不可纠正错误），执行：

  ```bash
  python supernode/retrain.py --val-start 2025-03-01      # 验证期起点：之前的数据用于训练
  sudo systemctl restart supernode-server
  ```

  重训会更新运行日志模型、故障树权重和报警线，旧模型自动备份为 `*.bak-<时间>`。
- **用开源数据重建模型和回放数据**：只在需要时使用，需要先下载开源原始数据并配置 `open_data`。

  ```bash
  python supernode/build.py && python supernode/build.py --ablation-only
  ```

## 9. 验证部署

```bash
python tests/test_uce_risk_features.py     # 特征计算与逐条暴力重算逐项比对
python tests/test_uce_risk_api.py          # 字段识别、三种输入方案、保存与加载
python tests/test_cluster_mode.py          # 集群模式：能力判定、接入、扫描、告警、设置、重启恢复
python tests/test_retrain.py               # 用采集记录重训
python tests/test_supernode_demo.py        # 回放模式下所有页面接口（需要 demo_data）
curl http://127.0.0.1:8765/api/status      # 服务在线
```

## 10. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `install.sh` 提示 wheels 不匹配 | 本机不是 Ubuntu 22.04 / Python 3.10 / x86_64。按 §0 "其他平台的离线包"制作，或用 `--online` |
| 内网其他机器打不开页面 | 见 §0 "访问不了时" |
| 启动时报 `models.joblib` 加载错误 | scikit-learn / lightgbm 版本与保存时不一致。安装 `requirements-lock.txt` 中的版本，或者重训 |
| 节点显示"离线" | 超过 3 个采集周期没有上报。检查 Agent 服务（`systemctl status supernode-agent`）、网络和 8765 端口 |
| 节点判定为"仅遥测" | 设备不支持 ECC，或 ECC 未开启（`nvidia-smi -q -d ECC`）；或者没有内核日志读权限 |
| 没有错误事件 | Agent 无权限读内核日志：用 root 运行，或把用户加入 `systemd-journal` 组；可选安装 `nvidia-ml-py` |
| 页面没有"开源数据回放"和"历史回溯" | `demo_data/` 不存在或 `replay: false`，此时只运行集群实时模式，属于正常情况 |
| 拓扑已满 | 节点数超过 `topology` 的容量。调整 `racks` / `nodes_per_rack` |
