# 用自己的数据运行算法

算法库 `uce_risk` 可以脱离原型单独使用：准备好数据表，用命令行或 Python 接口完成"识别字段 → 训练 → 预测"。不需要按数据集写适配代码，算法根据表中的字段决定启用哪些特征模块。

> 现网 BD 数据（工单表 + 遥测窄表 + 1 分钟 metric 表）有专门的适配层 `uce_risk/bd.py`，见 [BD_ADAPTER.md](BD_ADAPTER.md)。

## 1. 安装

```bash
pip install -r requirements.txt        # 或在发布包根目录：pip install .（安装 uce_risk 与命令 uce-risk）
```

## 2. 准备数据

只需要**一张带"设备、时间、错误编码"的错误日志**，其余数据都是可选的，有就会被利用：

| 表 | 必需 | 常见列 | 作用 |
|---|---|---|---|
| 错误日志 `--errors` | ✅ | 节点名、卡号、时间、错误编码 | `--uce-codes` 指定的编码作为不可纠正错误（标签和历史）；`--ce-codes` 指定的作为可纠正错误；其余作为错误事件 |
| 可纠正错误 `--ce` | | 节点名、卡号、时间，可选：计数 / 累计计数器、地址（dimm/rank/stack/sid/pc/bg/bank/row/col） | 可纠正错误计数；有地址时还会启用空间与故障模式特征 |
| 遥测 `--telemetry` | | 节点名、卡号、时间、数值列（温度、功率等） | 有时间列：时间序列特征；没有时间列：静态基线 |
| 设备清单 `--devices` | | 节点名、卡号、属性（型号、厂商、槽位……） | 补全没有出过错的设备；配置特征 |
| 下线记录 `--removed` | | 节点名、卡号、时间 | 设备下线后不再采样 |
| 不可纠正错误 `--uce` | | 节点名、卡号、时间 | 不用 `--errors` 时，单独提供标签 |

格式支持 CSV、Parquet、JSON lines，按扩展名自动识别。

### 字段自动识别

列名匹配时忽略大小写、下划线和连字符：

| 角色 | 识别的列名（部分） |
|---|---|
| 设备 | entity, device, device_id, gpu, gpu_id, gpu_uuid, uuid, name, card, gpu_index, sid |
| 节点 | host, hostname, node, server, machine, instance |
| 时间 | t, time, timestamp, ts, datetime, date, event_time, collect_time, time_ms, timestamp_ms, epoch（ISO 字符串、Unix 秒 / 毫秒 / 微秒 / 纳秒均可） |
| 错误编码 | code, xid, type, ecc_type, error_code, event_type |
| 可纠正错误计数 | n, count, cnt, ce_count, sbe_count, corrected_count（自动判断是否为累计计数器） |
| 地址层级 | dimm/memoryid → rank → stack → sid → pc/pcid → bg/bankgroup → bank/bankarray；行、列 |

识别不到时，用 `--device / --host / --time / --code` 指定列名。

训练前请先运行 `describe`，检查识别结果和将启用的模块。

## 3. 命令行

```bash
# 1) 看识别结果
python -m uce_risk.cli describe --errors errors.csv --uce-codes 48,94,95 --ce sbe.csv --telemetry tel.parquet

# 2) 训练：--val-start 之前的数据训练，之后的数据用于选择融合权重与报警阈值，再合并重训
python -m uce_risk.cli train --errors errors.csv --uce-codes 48,94,95 --ce sbe.csv --telemetry tel.parquet \
    --val-start 2024-03-01 --out model.joblib

# 3) 预测：给每张卡输出 1/4/24 小时风险、报警标志、规则依据
python -m uce_risk.cli predict --model model.joblib --errors errors.csv --uce-codes 48,94,95 --ce sbe.csv \
    --telemetry tel.parquet --at now --out risk.csv
```

### train 的可选参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--scenario` | `auto` | `S1_tel+xid` / `S2_ce` / `S3_tel+ce` / `ALL`；`auto` 表示用全部可用输入 |
| `--backend` | `lgbm` | 树模型：`lgbm`（LightGBM）或 `hgb`（scikit-learn） |
| `--tune` | 20 | 自动调参次数。验证期正样本少于 30 个时自动跳过；数据少时建议设为 0 |
| `--bags` | 3 | 装袋的随机种子数 |
| `--variant` | `hybrid` | `hybrid`（AI + 规则）/ `ai` / `rules`，用于对照实验 |

### 预测输出 `risk.csv` 的列

| 列 | 含义 |
|---|---|
| `host`、`device` | 原始的节点名和卡号 |
| `s` | 预测时刻（Unix 秒） |
| `p_1h`、`p_4h`、`p_1d` | 未来 1 小时 / 4 小时 / 24 小时发生不可纠正错误的概率（已校准，满足 1h ≤ 4h ≤ 24h） |
| `alarm_*` | 是否超过报警阈值。`--op fpr1` 表示验证期 1% 无故障设备误报（默认）；`--op valF1` 表示验证期 F1 最优 |
| `reason` | 规则评分卡的主要依据 |

### 训练数据量建议

- 至少几十次不可纠正错误事件；验证期中至少 10 次，用于选择报警阈值。
- 历史越长，"不可纠正错误历史"类特征越有效；在线预测时，输入数据至少包含过去 30 天，最好 180 天。

## 4. Python 接口

```python
from uce_risk.inputs import from_tables, describe
from uce_risk.api import UCERiskPredictor
import pandas as pd

errors = pd.read_csv("errors.csv")
data = from_tables(errors=errors, uce_codes=[48, 94, 95], ce=pd.read_csv("sbe.csv"),
                   telemetry=pd.read_parquet("tel.parquet"))
print(describe(data))

pred = UCERiskPredictor(scenario="auto").fit(data, val_start=int(pd.Timestamp("2024-03-01", tz="UTC").timestamp()))
pred.save("model.joblib")

# 在线：用新数据（字段名可以与训练时不同），沿用训练时的事件编码配置
pred = UCERiskPredictor.load("model.joblib")
new = from_tables(errors=pd.read_csv("errors_latest.csv"), uce_codes=[48, 94, 95], **pred.input_spec)
risk = pred.predict(new, at=int(pd.Timestamp.now(tz="UTC").timestamp()))   # 每张卡一行：p_1h / p_4h / p_1d / alarm_* / reason
```

更底层的接口，包括特征引擎、分段风险模型、规则评分卡和严格评测，见 [ALGORITHM.md](ALGORITHM.md)。

## 5. 把自己训练的模型接入原型

原型服务读取 `models/models.joblib`（键 `log` 为运行日志模型，键 `ce` 为显存 RAS 模型）和 `models/build_meta.json`（阈值、故障树权重）。如果数据来自集群 Agent 的采集，直接用：

```bash
python supernode/retrain.py --val-start <日期>
```

它会完成训练、计算故障树权重和报警线、备份旧模型。
