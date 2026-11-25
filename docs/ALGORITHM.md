# GPU UCE 风险预测算法（uce_risk）说明文档

| 项目 | 内容 |
|---|---|
| 代码 | `uce_risk/`（Python 3.10+，依赖 numpy、pandas、scikit-learn、lightgbm、optuna、pyarrow、joblib） |
| 验证报告 | `experiments/uce_unified/REPORT.md`；方法论推导见 `experiments/uce_unified/METHOD.md` |
| 测试 | `tests/test_uce_risk_features.py`（特征与逐条暴力重算逐项比对）、`tests/test_uce_risk_api.py`（字段识别、三种输入方案、保存与加载的端到端测试） |

---

## 1. 算法概述

本算法根据 GPU（或加速器、服务器内存）的历史错误记录和运行遥测，**在任意时刻为每个设备输出未来 1 小时、4 小时、1 天内发生不可纠正错误（UCE）的概率**，并给出报警标志和报警理由。

它面向三类常见的数据条件，用同一套代码处理：

| 输入方案 | 可用数据 | 典型来源 |
|---|---|---|
| 方案 1：遥测 + XID | 温度、功率等遥测；XID / 错误事件日志（含 UCE 事件） | DCGM 指标 + 内核日志（NVRM Xid） |
| 方案 2：只有 CE | CE 事件（可带地址）或 CE 计数器；UCE 记录 | mcelog / EDAC、DCGM ECC 计数、HBM 错误日志 |
| 方案 3：遥测 + CE | 方案 1 和方案 2 的组合 | 完整的监控平台 |

**算法不按数据集适配，而是按字段适配。** 用户只需提供原始表，算法从列名和取值中识别出设备、时间、错误编码、地址层级、CE 计数、遥测指标等字段，再据此决定启用哪些特征模块。换一个集群、换一种日志格式，通常只需要给出 UCE 的编码，不需要写适配代码。

---

## 2. 核心优势

| 优势 | 说明 | 依据 |
|---|---|---|
| **① 字段驱动，一套代码适配任意输入** | 内置 70 多个常见字段别名（匹配时忽略大小写、下划线和连字符）；缺少的字段对应的模块自动关闭；时间可以是 ISO 字符串、带时区的时间戳、秒或毫秒；地址可以是十进制或十六进制；CE 可以是逐条事件，也可以是累计计数器 | 同一个入口 `from_tables` 接入了 Summit XID 日志、阿里 DRAM mcelog、Calchas HBM 日志和模拟的 DCGM 风格数据。新入口在 Calchas 上生成的 41,031 个样本、全部特征，与原先的专用适配器逐一相同 |
| **② 多时间窗一致的概率** | 用分段风险模型（discrete-time hazard）一次给出 1 小时 / 4 小时 / 1 天三个概率，从模型结构上保证 P1h ≤ P4h ≤ P1d | 在全部 354 个"测试折 × 时间窗"组合上，概率合法性和单调性检查 100% 通过 |
| **③ 输出是校准过的概率，不只是分数** | 负样本降采样后做先验校正，输出可以直接解释为发生概率，用于计算预期故障数和备件需求 | DRAM 1 天窗口的预测均值是实际发生率的 0.84 倍；ECE 在 10⁻⁴ 量级 |
| **④ AI + 规则混合，小样本也稳定** | 规则评分卡（可解释、稳健）与带领域单调约束的梯度提升树融合，融合权重在验证集上选择 | 样本级 PR-AUC 比纯 GBDT 高：DRAM +12%，Calchas +22%，Calchas 严格口径 +18%；Summit 上纯 GBDT 的预测均值是实际发生率的 3.15 倍，混合模型降到 0.87 倍 |
| **⑤ 每次报警都附带理由** | 报警时给出贡献最大的规则，例如"行或列故障""近 7 天出现过 UCE""CE 突发"，运维人员可以直接复核 | DRAM 上得分最高的 1% 样本中，主要报警理由是"行或列故障"，与领域知识一致 |
| **⑥ 精度达到或超过已发表方法** | 在三个公开数据集上与原论文同口径对比 | Calchas：F1 **0.422**（论文 0.42，同口径重跑的 Calchas 随机森林 0.395）；阿里 DRAM：设备级 F1 15.6%–15.7%，与 SRDS'22（约 16%–17.5%）同一水平，PR-AUC 比此前基线高 16%，并新增 1 小时（F1 12.9%）和 4 小时（12.1%）预测；Summit：DBE 复发召回 72% |
| **⑦ 评测严格，结论可信** | 按时间切分；测试集保持真实正负比例；阈值只在验证集上确定；首次 UCE 和复发 UCE 分开报告 | 同时给出论文口径和严格口径的数字，不用复发 UCE 的效果掩盖首次 UCE |
| **⑧ 工程上可扩展** | 特征用"排序编码 + 二分查找"实现；按设备分片计算，可以写盘、断点续算；超大的 CE 表可以按分片从 parquet 读取 | 在可用内存约 3 GB 的机器上，处理了 7,500 万条 CE（单台服务器就有 3,160 万条）和 533 万个样本 |

---

## 3. 算法原理

### 3.1 问题定义

- **预测对象**：设备（GPU、加速器或服务器），可以带"所在节点"（host）信息。
- **预测时刻 s**：只使用 s 之前（t < s）的记录。
- **UCE 事件**：同一设备上相隔不超过 `episode_gap`（默认 1 小时）的 UCE 记录合并为一次事件。标签只看"s 之后下一次事件的开始时间"，所以同一次 UCE 的后续日志不会被当作"预测成功"。
- **标签**：设 lead = 下一次事件开始时间 − s，则 y_H = [lead ≤ H]，H ∈ {1 小时, 4 小时, 1 天}。
- **首次与复发**：设备第一次 UCE 和之后的 UCE 分开统计。前者是真正的"CE 或前兆 → UCE"预警，后者是复发预测，两者难度差别很大。

### 3.2 总体流程

```
 原始表（错误日志 / CE / 事件 / 遥测 / 设备清单 / 下线记录）
          │  inputs.from_tables：识别字段 → 统一格式 → 决定启用哪些模块
          ▼
 UCEData（统一数据格式）
          │  features.build：确定预测时刻 → 计算各特征组 → 生成多时间窗标签
          ▼
 样本 × 特征表（te / ev / ce / sp / hi / nb / cf 七组）
          │  api.UCERiskPredictor.fit
          │    ├─ 规则评分卡（rules.Scorecard）
          │    ├─ Optuna 自动调参（在验证期上搜索 LightGBM 超参数）
          │    ├─ 规则约束的分段风险 LightGBM（model.HazardModel，带单调约束）
          │    └─ 在验证集上选择融合权重和报警阈值；装袋；用训练期 + 验证期重新训练
          ▼
 predict：每个设备的 p_1h / p_4h / p_1d + 报警标志 + 报警理由
```

### 3.3 字段驱动的输入层（`uce_risk/inputs.py`）

`from_tables` 先按别名识别字段。别名匹配忽略大小写、下划线、连字符和空格，也可以显式指定列名。

| 角色 | 识别的列名（部分） | 处理 |
|---|---|---|
| 设备 | entity, device, device_id, gpu, gpu_id, gpu_uuid, uuid, name, card, gpu_index, sid | 与"节点"组合成设备主键；同一节点的设备编号连续 |
| 节点 | host, hostname, node, server, machine, instance | 用于同节点特征；只有节点列时，节点即设备 |
| 时间 | t, time, timestamp, ts, datetime, date, event_time, collect_time | ISO 字符串（带时区的统一转为 UTC，不带时区的按 UTC 处理）、Unix 秒、Unix 毫秒都可以 |
| 错误编码 / 类型 | code, xid, type, ecc_type, error_code, event_type | 配合 `uce_codes` / `ce_codes`，把一张错误日志拆成 UCE、CE、其他事件三类 |
| CE 地址层级（从粗到细） | dimm/memoryid → rank → stack/hbm → sid → pc/pcid/pseudo_channel → bg/bank_group → bank/bankid/bankarray；叶子为 row、col | 十进制或十六进制（0x…）都能解析；识别到地址列才启用空间模块 |
| CE 子类型 | sub, subtype, ce_type, detect_type, error_type, source | 取值不超过 8 种时，作为子类型（例如读 / 巡检） |
| CE 计数 | n, count, cnt, ce_count, sbe_count, corrected_count | 自动判断是"每条记录的计数"还是"累计计数器"；累计计数器先转换成增量 |
| 遥测指标 | 遥测表中其余的数值列 | 有时间列：作为时间序列；没有时间列：按设备或节点取均值，作为静态基线 |
| 设备属性 | 设备清单中其余的列 | 数值列作为数值特征；文本列做 one-hot |

**模块开关**由识别到的字段决定，而不是由数据集名称决定：

| 识别到的字段 | 启用的模块 |
|---|---|
| CE 记录 | CE 计数（`ce__`） |
| CE 地址列 | CE 空间分布与故障模式（`sp__`） |
| 其他错误编码 | 离散事件（`ev__`） |
| 带时间的遥测 | 遥测时间序列（`te__`） |
| 不带时间的遥测 | 静态遥测基线（`te__`） |
| 设备属性 | 配置（`cf__`） |
| 同一节点有多个设备 | 同节点上下文（`nbc__` / `nbe__` / `nbh__`） |
| UCE 记录（必需） | UCE 历史（`hi__`） |

**背景事件自动识别**：

- 在"从未发生 UCE 的设备"中，如果某个事件编码出现在 ≥ 5% 的设备上（设备总数 ≥ 500 时才启用），就标记为背景事件（前缀 `bg_`）。
- 背景事件不参与"硬件事件"规则和单调约束，也不会触发每日观察。
- 在 Summit 上，这个规则自动选出了 XID 31、13、43。这正是常见的应用类 XID，不需要人工写死。
- 也可以用 `background_codes` 手工指定。

### 3.4 预测时刻（采样）

| 触发方式 | 说明 | 目的 |
|---|---|---|
| 事件触发 | 每个出现任何记录（CE、事件、UCE）的 5 分钟窗口结束时预测一次 | 支撑 1 小时窗口：DRAM 上最后一次 CE 到 UE 的中位间隔只有 5 分钟 |
| 每日观察 | 出现非背景事件或 CE 后的 30 天内，每天 0 点预测一次；出现 UCE 后延长到 180 天（`uce_anchor_days`） | 覆盖"安静但仍有风险"的设备；复发往往间隔较长 |
| 全量每日（可选） | `anchor="all"`：所有设备每天预测一次 | 适合设备较少的场景 |
| 指定时刻 | `predict(at=...)`：任意时刻、任意设备 | 在线打分 |

设备下线（`removed`）之后不再产生样本。离线构建时，超出数据末尾 1 天的样本因标签不完整而被丢弃。

### 3.5 特征模块（`uce_risk/features.py`）

所有特征都只用 t < s 的记录。实现方式：

- 把（设备, 时间）编码成一个 64 位整数并排序；"某时间窗内的记录数"就变成两次二分查找。
- "时间窗内出错的不同部件数"：把每个部件的记录转换成互不重叠的覆盖区间，同样用两次二分查找求出。
- 因此特征计算的复杂度是 O(N log N)，与样本数和窗口数基本无关。

| 前缀 | 特征（窗口 1 小时 / 6 小时 / 1 天 / 7 天 / 30 天 / 累计） | 物理含义 |
|---|---|---|
| `ce__` | CE 次数（有计数列时按计数加权）、1 小时相对 1 天的加速度、距首次 / 上次 CE 的小时数、子类型数量和占比、近 1 天有 CE 的小时数、近 30 天有 CE 的天数、CE 风暴分钟数（1 分钟内 ≥ 10 次） | CE 的强度、趋势和新近程度 |
| `sp__` | 各层级出错部件数（1 天、累计）；单个单元 / 行 / 列 / bank 的最大 CE 数；故障模式：父部件出现第 2 个不同子部件的数量（如 bank 出现多行、行出现多个单元）；硬错误部件（隔 ≥ 1 天再次出错）；单元分散度 | 区分"单元级硬错误"和"行、列、bank 级结构性故障"，后者是 UE 的主要前兆 |
| `ev__` | 每种事件编码在 1 小时 / 1 天 / 7 天 / 30 天 / 累计的次数、距上次的小时数；全部事件的总数 | 各类 XID、UEO 等离散事件 |
| `hi__` | 各窗口内的 UCE 事件数、近 1 天 UCE 记录数、距上次 UCE 的小时数、出过 UCE 的部件数（有地址时） | UCE 复发：最强的信号 |
| `te__` | 每个指标的 1 小时 / 1 天均值、1 天标准差、1 小时相对 1 天的变化；或静态基线 | 温度、功率的水平和突变 |
| `nbc__` / `nbe__` / `nbh__` | 同节点其他设备的 CE、事件、UCE 数 | 节点级环境或供电问题 |
| `cf__` | 型号、厂商、槽位等 | 不同批次和型号的固有差异 |

### 3.6 分段风险模型（`model.HazardModel`）

把未来 1 天分为三段，各训练一个梯度提升树分类器。默认使用 **LightGBM**：原生支持缺失值、样本权重和单调约束（`monotone_constraints_method="advanced"`）。scikit-learn 的 HistGradientBoosting 作为可选后端（`backend="hgb"`）保留，用于对照：

```
h1 = P(UCE ∈ (s, s+1h])
h2 = P(UCE ∈ (s+1h, s+4h] | 前 1 小时未发生)        只用 lead > 1h 的样本训练
h3 = P(UCE ∈ (s+4h, s+1d] | 前 4 小时未发生)        只用 lead > 4h 的样本训练

P(1h) = h1
P(4h) = 1 − (1−h1)(1−h2)
P(1d) = 1 − (1−h1)(1−h2)(1−h3)
```

- **单调性**：三个概率天然满足 P1h ≤ P4h ≤ P1d，不需要事后修正。
- **先验校正**：每段训练时保留全部正样本，负样本只保留比例 q（正负比约 1:50），预测后按 p' = pq / (pq + 1 − p) 还原到真实发生率。训练前还会做一次预采样：每条样本按 1/q₀ 加权，以保持总体比例。
- **小样本回退**：某段正样本少于 30 个时，不单独训练，而是借用"1 天内是否发生 UCE"这个总体模型的排序，再按该段的实际发生率缩放（比例风险假设）。
- **缺失模态**：某个输入不存在时，对应特征组根本不会生成；单个特征缺失时用 NaN 表示，LightGBM 原生支持。

### 3.6b 自动调参（`model.tune_tree`）

在模型 A 阶段，用 Optuna 的 TPE 算法搜索 LightGBM 超参数：

| 超参数 | 搜索范围 | 默认值（第 1 组试验） |
|---|---|---|
| `learning_rate` | 0.01–0.2（对数尺度） | 0.05 |
| `n_estimators` | 100–800（步长 50） | 300 |
| `num_leaves` | 7–127（对数尺度） | 31 |
| `min_child_samples` | 10–300（对数尺度） | 40 |
| `subsample`（行采样，每轮） | 0.5–1.0 | 0.8 |
| `colsample_bytree` | 0.4–1.0 | 0.8 |
| `reg_lambda` | 0.001–10（对数尺度） | 1.0 |

- **目标**：验证期上"最长时间窗内是否发生 UCE"的 PR-AUC。训练集与正式训练使用同样的预采样、样本权重和单调约束。
- **试验次数**：默认 20 次。第 1 次固定为默认参数，因此调参结果在验证期上不会比默认参数差。
- **跳过条件**：验证期正样本少于 10 个时不调参，直接用默认参数，避免在噪声上过拟合。
- **测试集不参与调参**：调出的参数由同一折的其余装袋种子和重训的模型 B 共用。

### 3.7 AI + 规则混合（`model.HybridModel`、`rules.py`）

1. **规则评分卡**：
   - 把领域规则离散成档位。缺少输入的规则自动跳过。
   - 用 L2 正则的逻辑回归（C = 0.3）估计每档的权重，每个时间窗一套，并做先验校正。
   - 权重表可以直接阅读。

   | 规则 | 输入 | 档位 |
   |---|---|---|
   | 距上次 UCE | `hi__h_since_last` | ≤1h / ≤4h / ≤1d / ≤7d / ≤30d / 更早或从未 |
   | UCE 次数 | `hi__ep_cum` | 0 / 1 / 2 / ≥3 |
   | 同节点 UCE | `nbh__ep_30d` | 有 / 无 |
   | 每种非背景事件 | `ev__<code>_1d/_30d` | 1 天内 / 30 天内 / 无 |
   | 背景事件 | `ev__bg_*_1d` | 1 天内 / 无 |
   | CE 突发 | `ce__n_1h` | 0 / 低 / 中 / 高（按训练数据的分位数划分） |
   | CE 加速 | `ce__accel_1h_vs_1d` | 1 小时速率超过 1 天均速的 2 倍 / 否 |
   | 首次 CE 的新近程度 | `ce__h_since_first` | ≤1h / ≤1d / ≤7d / 更早 |
   | CE 风暴 | `ce__storm_min_1d` | 有 / 无 |
   | bank 级故障 | `sp__<层级>_multi_row/col` | 0 / 1 / ≥2 |
   | 行或列故障 | `sp__row/col_multi_cell` | 有 / 无 |
   | 单元分散 | 近 1 天出错单元 ≥ 5，且单个单元最大 CE ≤ 3 | 是 / 否 |
   | 硬错误 | `sp__hard_cell` | 0 / 1–10 / >10 |
   | 遥测偏高 | 每个 `te__` 指标 | 前 10% / 正常 / 缺失 |
   | 兜底 | 现成特征表中没有专门规则的特征组 | 有 / 无 |

2. **规则约束的 GBDT**：
   - 输入原始特征，加上评分卡在三个时间窗上的 logit。
   - 对以下特征施加**单调约束**：
     - 单调递增：UCE 次数、出过 UCE 的部件数、同节点 UCE、非背景事件数、bank / rank / DIMM 级故障模式数、评分卡输出；
     - 单调递减：距上次 UCE 的时间、距上次非背景事件的时间。
   - CE 计数和遥测**不加约束**，因为"单个单元 CE 很多"在 DRAM 上反而意味着低风险，约束会违背事实。
   - "距上次"类特征在缺失时（从未发生）填一个很大的值，让"从未发生"落在低风险一端。
3. **融合**：p = σ(w · logit(p_GBDT) + (1 − w) · logit(p_规则))。
   - w ∈ {0, 0.25, 0.5, 0.75, 1}，每个时间窗在验证集上按 PR-AUC 选择；验证集正样本少于 10 个时取 0.5。
   - 融合后再取累积最大值，保证三个时间窗单调。
4. **报警理由**：当前样本命中的档位中，评分卡权重最大的那一条。

### 3.8 训练流程（`api.UCERiskPredictor.fit`）

```
时间轴：  ──────── 训练期 ────────┤ val_start ├──── 验证期 ────┤
模型 A（装袋 3 个种子）：只用训练期    → 第 1 个种子在验证期上自动调参；在验证期选融合权重 w、确定报警阈值
模型 B（装袋 3 个种子）：用训练期 + 验证期重新训练，沿用 A 的超参数、w 和概率阈值
预测：使用模型 B，在 logit 空间对 3 个种子取平均
```

模型输出是校准过的概率，所以模型 A 上定的概率阈值可以直接用于模型 B。阈值有两种运行点：

- `fpr1`：验证期中 1% 的无 UCE 设备会产生报警。用于控制误报数量。
- `valF1`：验证期设备级 F1 最大。用于平衡精确率和召回率。

---

## 4. 输入规范

### 4.1 `inputs.from_tables(...)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `errors` | DataFrame | 一张混合错误日志，需要有编码列。编码在 `uce_codes` 中的行作为 UCE，在 `ce_codes` 中的作为 CE，其余作为事件 |
| `uce` / `ce` / `events` | DataFrame | 分开的表，可以任意组合；`ce` 也可以是 `ShardSource`（分片按需读取） |
| `telemetry` | DataFrame | 遥测：有时间列即为时间序列，没有时间列即为静态基线 |
| `devices` | DataFrame | 设备清单：主键加属性，用于补全没有出过错的设备和配置特征 |
| `removed` | DataFrame | 设备下线记录（主键 + 时间），下线后不再采样 |
| `exclude_train` | DataFrame | 不参与训练的设备（主键），例如因其他原因故障的设备 |
| `uce_codes` / `ce_codes` | list | 例如 `[48]`（XID 48），`["UER"]`，`["CE"]` |
| `device` / `host` / `time` / `code` | str | 别名识别失败时，显式指定列名 |
| `location` | dict | 地址层级的显式映射，例如 `{"bank": "BankArray", "row": "RowAddr"}` |
| `ce_cumulative` | bool / `"auto"` | CE 计数列是否为累计计数器，默认自动判断 |
| `background_codes` | list | 手工指定背景事件编码；默认由数据统计识别 |
| `warmup_days` | int | 数据开头不产生样本的天数，用于让历史特征"预热" |
| 其他关键字 | | 传给 `UCEData`：`episode_gap`（默认 3600 秒）、`storm_per_min`（10）、`uce_anchor_days`（180）、`anchor`（`"active"` / `"all"`）、`t_start` / `t_end`、`shards`（分片数） |

**最小输入**：一张带"设备、时间、编码"列的错误日志，加上 `uce_codes`。此时只启用 UCE 历史和事件模块。

### 4.2 三种方案的输入示例

```python
from uce_risk.inputs import from_tables, describe

# 方案 1：遥测 + XID（例如 DCGM 指标 + NVRM Xid 日志）
d1 = from_tables(errors=xid_log,              # 列：hostname, gpu, timestamp, xid
                 uce_codes=[48, 95],          # DBE / 不可抑制的 UCE
                 telemetry=dcgm,              # 列：hostname, gpu, timestamp, gpu_temp, mem_temp, power_usage
                 devices=inventory)           # 列：hostname, gpu, model, serial ...

# 方案 2：只有 CE（例如 HBM 错误日志：一张表中同时有 CE / UER / UEO）
d2 = from_tables(errors=hbm_log,              # 列：Server, Name, Time, EccType, Stack, SID, PcId, BankGroup, BankArray, Row, Col
                 uce_codes=["UER"], ce_codes=["CE"])

# 方案 2'：只有 CE 计数器（例如 DCGM 的 ECC 单比特累计计数）
d2b = from_tables(uce=uce_records,            # 列：host, gpu, time
                  ce=sbe_counter)             # 列：host, gpu, time, sbe_count（自动识别为累计计数器）

# 方案 3：遥测 + CE
d3 = from_tables(uce=uce_records, ce=ce_events, telemetry=dcgm, devices=inventory)

print(describe(d3))     # 识别到的字段和启用的模块，训练前请先检查
```

`describe` 的输出示例（Calchas HBM 日志）：

```
fields:  error_code=EccType, device=Name, host=Server, time=Time,
         ce_location={stack: Stack, sid: SID, pc: PcId, bg: BankGroup, bank: BankArray, row: Row, col: Col},
         event_codes=[UEO], background_codes=[]
modules: CE counts, CE spatial / fault modes, discrete events, same-node context, UCE history
```

### 4.3 现网 BD 数据（工单 + 遥测窄表 + 1 分钟指标表）

现网数据的指标以“一个指标一行”的窄表采集，ECC 信息是累计计数器而不是逐条记录，标签来自工单系统，因此用专门的适配层
`uce_risk/bd.py` 转成同一个 `UCEData`：窄表转 1 分钟宽表（第三张 metric 表），计数器增量转 CE 流（区分设备内存与 SRAM，
处理复位和采集缺口），双比特计数、错误码和不可纠正重映射转 UCE 标签并由工单确认时间，行重映射 / 备用行 / 链路 / 降频等状态
变化转事件，约 70 个派生遥测量转 5 分钟序列；另外每张有数据的卡每小时打一次分（`UCEData.grid`）。
完整说明、指标用法清单和命令行见 `docs/BD_ADAPTER.md`。

---

## 5. 输出规范

### 5.1 `UCERiskPredictor.predict(data, at=..., op="fpr1")` 返回的 DataFrame

| 列 | 含义 |
|---|---|
| `entity` | 内部设备编号 |
| `host`、`device` | 原始的节点名和设备名（由输入的主键得到） |
| `s` | 预测时刻（Unix 秒，UTC） |
| `p_1h`、`p_4h`、`p_1d` | 未来 1 小时 / 4 小时 / 1 天内发生 UCE 的概率（已校准，满足 p_1h ≤ p_4h ≤ p_1d） |
| `alarm_1h`、`alarm_4h`、`alarm_1d` | 是否超过所选运行点 `op` 的阈值（`"fpr1"` 或 `"valF1"`） |
| `reason` | 1 天风险的主要报警理由，例如 `rowcol_fault=yes`、`recur_time=<=7d`；`baseline` 表示没有命中增加风险的规则 |

`at` 有三种取法：

- 整数时间：所有设备在该时刻各打一次分；
- DataFrame（`entity, s`）：在指定的设备和时刻打分；
- `None`：按内置的采样方式输出所有预测时刻。

### 5.2 其他输出

| 接口 | 返回 |
|---|---|
| `predictor.info` | 方案、启用的特征组、特征数、树模型后端、调出的超参数（含调参前后的验证期 PR-AUC：`_val_ap_default` → `_val_ap`）、每个时间窗的融合权重（AI 占比）、训练 / 验证样本数、验证期正样本数 |
| `predictor.thresholds` | `{时间窗: {"fpr1": 阈值, "valF1": 阈值}}` |
| `predictor.explain(horizon="1d")` | 规则评分卡的权重（log-odds，相对于参照档位），可用于说明"什么提高风险、什么降低风险" |
| `inputs.describe(data)` | 识别到的字段、启用的模块、设备数、UCE 记录数 |

---

## 6. 接口用法

### 6.1 快速开始

```python
from uce_risk.inputs import from_tables, describe
from uce_risk.api import UCERiskPredictor

data = from_tables(errors=xid_log, uce_codes=[48], telemetry=dcgm, devices=inventory)
print(describe(data))

pred = UCERiskPredictor(scenario="auto", bags=3).fit(data, val_start=t_val)   # t_val：验证期开始时间（Unix 秒）
print(pred.info)

risk = pred.predict(data, at=t_now, op="fpr1")       # 每个设备一行
top = risk.sort_values("p_1d", ascending=False).head(20)

pred.save("uce_model.joblib")
pred = UCERiskPredictor.load("uce_model.joblib")
```

### 6.2 主要参数（`UCERiskPredictor`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `scenario` | `"auto"` | `"S1_tel+xid"` / `"S2_ce"` / `"S3_tel+ce"` / `"ALL"` / `"auto"`（使用全部可用输入）。指定的方案缺少输入时会报错 |
| `horizons` | `("1h","4h","1d")` | 预测时间窗 |
| `bags` | 3 | 装袋的种子数 |
| `drop_history` | False | True 时不用 UCE 历史，只看前兆，用于评估首次 UCE 预警能力 |
| `alarm_fpr` | 0.01 | `fpr1` 运行点的设备误报率 |
| `refit` | True | 是否用训练期 + 验证期重新训练 |
| `backend` | `"lgbm"` | 树模型：`"lgbm"`（LightGBM）或 `"hgb"`（scikit-learn HistGradientBoosting） |
| `tune` | 20 | Optuna 调参的试验次数；0 表示使用默认超参数 |
| `include` / `exclude` | 空 | 按正则保留 / 去掉特征列（在 `scenario` 选择之后），例如 `bd.COMPACT` |
| `params` | 空 | 指定树模型超参数（非空时直接使用、不再调参），例如小样本用的 `bd.SMALL_DATA_PARAMS` |
| `share` | 空 | 这些时间窗的风险段不单独训练树模型，而是复用合并模型（24 小时）的排序并按本段事件率缩放（比例风险）；正样本很少时更稳，BD 配置为 `("4h",)` |
| `interval_cols` | 空 | `{时间窗: 正则}`：该时间窗风险段的树模型只用匹配的列，例如 BD 配置中 1 小时段只用 CE 突发特征 `bd.BURST` |

### 6.3 在线打分

```python
# 每小时对全部设备打一次分：data 只需包含过去至少 30 天的记录（180 天最佳，用于 UCE 复发特征）
data_now = from_tables(errors=recent_xid_log, uce_codes=[48], telemetry=recent_dcgm, devices=inventory)
risk = pred.predict(data_now, at=int(time.time()))
alarms = risk[risk.alarm_1h | risk.alarm_1d]
```

在线数据的字段名可以与训练时不同：只要能被识别成同一种角色即可。缺少的特征会按"没有记录"处理（计数为 0）。

### 6.4 底层接口（自定义流程时使用）

| 接口 | 作用 |
|---|---|
| `features.build(data, out_dir=None, query=None)` | 生成样本 × 特征表和 UCE 事件表。`out_dir` 用于分片写盘、断点续算；`query` 用于在指定时刻计算 |
| `schema.scenario_columns(cols, scenario, drop_history)` | 按方案挑选特征列 |
| `model.HybridModel(cols, horizons).fit_sampled(X, lead, w, Xv, lead_v)` / `.predict(X)` / `.reasons(X)` | 单个混合模型 |
| `model.HybridModel.presample(lead, mask, Hmax)` | 训练行的预采样和权重 |
| `model.HazardModel` | 纯分段风险 GBDT（对照） |
| `model.tree(seed, mono, params, backend)` / `model.tune_tree(...)` | 树模型工厂；Optuna 调参 |
| `rules.Scorecard` | 纯规则评分卡（对照）；`weights()`、`reasons()` |
| `evaluate.sample_metrics / pick_threshold / best_f1 / entity_scores / operating_metrics` | 严格评测：PR-AUC、ROC、Brier、ECE、设备级 P/R/F1、事件召回（分首次和复发） |
| `schema.ShardSource(counts, load)` | 超大的 CE 表按设备区间分片读取 |

### 6.5 批量评测（复现报告中的结果）

```
python experiments/uce_unified/run.py --dataset {summit|dram|calchas_raw|calchas_table}
python experiments/uce_unified/calchas_compare.py
python experiments/uce_unified/summarize.py
```

---

## 7. 验证结果摘要

详细内容见 `experiments/uce_unified/REPORT.md`。评测均为严格口径：按时间切分、测试集保持真实比例、阈值在验证期确定。

| 数据集（输入方案） | 时间窗 | 本算法 | 对比 |
|---|---|---|---|
| 阿里 DRAM（方案 2） | 1 天 | 设备级 F1 15.6%–15.7%，PR-AUC 0.057 | SRDS'22：F1 约 16%（随机森林）/ 17.5%（LightGBM）；此前基线：F1 15.7%，PR-AUC 0.049 |
| 阿里 DRAM | 1 小时 / 4 小时 | F1 12.9% / 12.1% | 论文没有这两个时间窗 |
| Calchas HBM（论文口径） | 1 天，服务器级 | **F1 0.422**，PR-AUC 0.255 | 论文 F1 0.42；同口径重跑的 Calchas 随机森林 F1 0.395，PR-AUC 0.225 |
| Calchas HBM（严格口径） | 1 天 | PR-AUC **0.040** | Calchas 随机森林 0.034 |
| Calchas 方案对比 | 1 天 | 方案 3（遥测 + CE）PR-AUC 0.036 > 方案 1 0.032 > 方案 2 0.027 | 加入遥测带来 +33% |
| Summit（方案 1） | 1 天 | DBE 事件召回 47.5%（复发 72%），设备级精确率 28%，预测均值 / 实际发生率 0.87 | ISSRE'26（只用遥测）：AUROC 0.44–0.59，事件 F1 ≤ 0.35%；同数据上的纯 GBDT 事件召回同为 47.5%，但预测均值是实际发生率的 3.15 倍 |

---

## 8. 适用范围与注意事项

1. **首次 UCE 难以预测。** 三个数据集上首次 UCE 的召回都接近 0，能抓到的主要是复发和"短时间内快速恶化"的设备。要提高首次 UCE 的预警能力，需要更细的信号，例如 GPU 行重映射状态、DCGM 的 SBE / DBE 计数、XID 94/95 及对应地址、DRAM 的错误位模式。算法可以直接接入这些字段：计数器作为 CE，地址作为地址列，状态值作为遥测。
2. **时间序列遥测特征已经实现并通过单元测试**，但三个公开数据集都没有覆盖 UCE 发生前时间段的遥测，所以效果还没有在真实数据上验证。
3. **训练数据的正样本少于几十个时**（例如每个验证期只有十几个 UCE），融合权重会不稳定。这种情况下建议用规则评分卡，或把融合权重固定为 0.5。
4. **报警阈值依赖机群规模和误报预算**，换集群后应该用新集群的验证期重新确定。
5. **在线打分时**，输入数据要包含足够的历史：至少 30 天，用于 UCE 复发特征时 180 天最佳。否则历史类特征会偏低。
