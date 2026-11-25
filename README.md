# 超节点计算卡故障风险预测（算法包 + 原型）

对超节点内每张计算卡定时预测**未来 1 小时、4 小时、24 小时**发生不可纠正错误（UCE）的概率，并用故障树给出故障类别、根因和处置建议。

本包可以按两种方式使用：

| 用法 | 适合 | 入口 |
|---|---|---|
| **直接部署原型** | 在集群上运行实时监测：节点 Agent 采集、页面查看风险与归因、历史回溯 | `supernode/server.py`（监测服务）、`supernode/agent.py`（节点 Agent），见 [docs/DEPLOY.md](docs/DEPLOY.md) |
| **只用算法** | 用自己的数据（CSV / Parquet）训练与预测，或嵌入自己的系统 | `uce_risk` 算法库：命令行 `python -m uce_risk.cli` 或 Python 接口，见 [docs/BYO_DATA.md](docs/BYO_DATA.md)；现网 BD 数据（工单 + 遥测窄表 + 1 分钟 metric 表）见 [docs/BD_ADAPTER.md](docs/BD_ADAPTER.md) |

## 目录

```
uce_risk/            算法库：按字段自适应输入、7 组特征、分段风险模型、AI + 规则混合、评测、命令行、BD 现网数据适配（bd.py）
supernode/           原型：监测服务、实时引擎、集群采集、节点 Agent、故障树、页面（static/）
  collector/         节点采集器核心（只依赖 Python 标准库）
  benchmarks/        与论文对比所用的实验结果（帮助页“算法优势”读取）
  config.json        配置（拓扑、时间窗、端口、路径、采集源）
  build.py           用开源数据重建模型与回放数据（需要原始数据，一般不用）
  retrain.py         用自己集群采集到的数据重训模型
models/              预训练模型（models.joblib）与阈值、故障树权重（build_meta.json）
demo_data/           开源数据回放（约 16 MB；可删除，删除后只运行集群实时模式）
install.sh           监测服务一键部署（Ubuntu 22.04，离线）；uninstall.sh 卸载
deploy/              systemd 服务模板、Agent 安装脚本、Dockerfile、前台启动脚本
examples/            用自己的数据运行算法的示例
tests/               自动化测试
docs/                部署文档、自带数据接入、BD 现网数据接入、算法说明、原型说明、采集器说明
```

## 一键部署（Ubuntu 22.04 内网）

```bash
tar xzf supernode-uce-risk-<版本>-ubuntu22.04-cp310-x86_64-offline.tar.gz && cd supernode-uce-risk-<版本>
sudo bash install.sh            # 离线安装为系统服务，结束时打印 http://<服务器IP>:8765
# 其他计算节点接入：curl -fsSL http://<服务器IP>:8765/agent/install.sh | sudo bash
```

详见 [docs/DEPLOY.md §0](docs/DEPLOY.md)。

## 三分钟体验（任意系统，前台运行）

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt                 # Python 3.10 用 requirements-lock-py310.txt；或 requirements.txt（版本更宽松）
python supernode/server.py                           # 浏览器打开 http://<本机>:8765
python examples/byo_quickstart.py                    # 用示例数据走一遍“识别字段 → 训练 → 预测”
```

## 依赖

| 组件 | 要求 |
|---|---|
| 监测服务、算法库 | Python ≥ 3.10（已测试 Ubuntu 22.04 自带 3.10.12、Windows 3.13）；numpy、pandas、scikit-learn、lightgbm、pyarrow、joblib、flask、waitress；自动调参需要 optuna；系统库 libgomp1。离线包已附带全部依赖；否则见 `requirements-lock-py310.txt`（Python 3.10）/ `requirements-lock.txt`（≥ 3.11）/ `requirements.txt` |
| 节点 Agent | Python ≥ 3.8，仅标准库；厂商管理工具（如 nvidia-smi）；读取内核错误日志需要 root 或 `systemd-journal` 组权限 |
| 预训练模型 | 用 `requirements-lock.txt` 中的版本保存；scikit-learn / lightgbm 主版本不同可能无法加载，此时用 `supernode/retrain.py` 重训 |
| 硬件 | 监测服务：4 核、8 GB 内存可支持 1,024 张卡（一次扫描约 1 秒）；磁盘按采集频率估算：5 分钟频率下 1,024 张卡约 300 MB/天 |

## 文档

- [docs/DEPLOY.md](docs/DEPLOY.md)：部署（单机、systemd、Docker）、节点接入、配置、存储与安全、升级与重训、排错
- [docs/BYO_DATA.md](docs/BYO_DATA.md)：用自己的数据运行算法（字段识别规则、命令行、Python 接口）
- [docs/BD_ADAPTER.md](docs/BD_ADAPTER.md)：现网 BD 数据（工单表 + 遥测窄表 + 1 分钟 metric 表）接入：生成第三张表、UCE 标签、特征、小样本模型配置、训练与预测命令；各指标的用途见 [docs/BD_METRIC_USAGE.csv](docs/BD_METRIC_USAGE.csv)
- [docs/ALGORITHM.md](docs/ALGORITHM.md)：算法原理、输入输出、接口、精度与论文对比
- [docs/PROTOTYPE.md](docs/PROTOTYPE.md)：原型页面与架构说明
- [docs/COLLECTOR.md](docs/COLLECTOR.md)：节点采集器的指标、频率与输出格式
