# uce_risk：按字段自适应的 GPU UCE 多时间窗风险预测

- **完整的算法文档**（原理、输入输出、接口用法、优势）：`docs/ALGORITHM.md`
- 验证报告：`experiments/uce_unified/REPORT.md`

```python
from uce_risk.inputs import from_tables, describe
from uce_risk.api import UCERiskPredictor

data = from_tables(errors=xid_log, uce_codes=[48], telemetry=dcgm, devices=inventory)  # 按字段自动识别
print(describe(data))                                   # 识别到的字段和启用的模块
pred = UCERiskPredictor().fit(data, val_start=t_val)
risk = pred.predict(data, at=t_now)                     # p_1h ≤ p_4h ≤ p_1d、报警标志、报警理由
pred.save("uce_model.joblib")
```

现网 BD 数据（工单表 + 遥测窄表 + 1 分钟 metric 表）用专门的适配层，说明见 `docs/BD_ADAPTER.md`：

```python
from uce_risk import bd
from uce_risk.api import UCERiskPredictor

bd.build_metric_table("遥测/", "metric_1min/")               # 遥测窄表 -> 第三张 1 分钟宽表（按 date 分区）
data = bd.from_bd("工单.xlsx", "遥测维度.csv", "metric_1min/")  # CE 流、UCE 标签、事件、遥测特征
pred = UCERiskPredictor(alarm_fpr=0.05, **bd.PROFILE).fit(data, val_start=t_val)   # 小样本配置（几十台机器）
risk = bd.predict(pred, data)                                # 每张卡 p_1h / p_4h / p_24h、风险等级、中文理由
```

| 模块 | 文件 |
|---|---|
| 字段驱动的输入层 | `inputs.py`（`from_tables`、`describe`） |
| BD 现网数据适配 | `bd.py`（`metric_table`、`build_metric_table`、`from_bd`、`predict`、`metric_usage`） |
| 统一数据格式与输入方案 | `schema.py`（`UCEData`、`ShardSource`、`SCENARIOS`） |
| 采样、标签与 7 组特征 | `features.py`（`build`） |
| 规则评分卡 | `rules.py`（`RuleSet`、`Scorecard`） |
| 分段风险 GBDT 与 AI + 规则混合模型 | `model.py`（`HazardModel`、`HybridModel`、`ensemble_predict`） |
| 高层接口 | `api.py`（`UCERiskPredictor`） |
| 严格评测 | `evaluate.py` |
