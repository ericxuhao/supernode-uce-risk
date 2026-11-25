# gpu-ecc-collector：A100 GPU ECC 故障预测数据采集器

为 GPU 显存 ECC 故障预测（SBE 累积 → DBE/XID48/94/95、行重映射耗尽/失败）持续采集训练数据。
指标 ID 与语义遵循 NPUMonitor《14-GPU监控指标与告警规则说明表》（修订1.2）；选型依据见
[GPU ECC 故障预测数据需求与公开数据集分析](../../docs/gpu_ecc_prediction_data_zh.md)。

- 只需要 **Python ≥ 3.6 + NVIDIA 驱动(nvidia-smi)**，不依赖第三方包。
- 可选增强：`nvidia-ml-py`（NVML 实时 XID/SBE/DBE 事件）、DCGM Exporter（`MEMORY_UNREPAIRABLE`、`PROF_DRAM_ACTIVE` 等 nvidia-smi 拿不到的字段）。
- 启动时探测本机驱动支持的字段（`--help-query-gpu`），新旧驱动别名自动选择；不支持/`[N/A]` 存 `null` 并在 `na` 里写原因，**不会填 0 冒充健康**。

## 安装（每台 A100 服务器）

```bash
sudo ./install.sh              # 拷贝到 /opt，配置到 /etc/gpu-ecc-collector/config.json，自检后启用 systemd 服务
sudo ./install.sh --with-nvml  # 额外 pip 安装 nvidia-ml-py，开启 NVML 事件监听
```

手动运行 / 自检：

```bash
python3 gpu_ecc_collector.py --print-fields                 # 查看本机驱动下各指标映射与不支持项
python3 gpu_ecc_collector.py --base-dir /tmp/ecc --once     # 每档采一次后退出
python3 gpu_ecc_collector.py --config /etc/gpu-ecc-collector/config.json
```

## 采集内容与频率

| 档位 | 默认周期 | 内容 |
|---|---|---|
| fast | 10 s | ECC SBE/DBE（volatile/aggregate × total/显存DEV/SRAM）、行重映射计数/pending/failure、恢复动作、GPU/显存温度、功耗、SM/显存时钟、利用率、显存占用、P-state、时钟事件原因位图、PCIe 代/宽 |
| slow | 60 s | ECC 模式/待生效模式、bank 重映射资源直方图 Max/High/Partial/Low/None、非激活重映射行、页退役（旧架构）、按位置细分 ECC（寄存器/L1/L2/纹理/CBU/DRAM/SRAM 子类与 thresholdExceeded）、节流累计时长、NVLink 错误计数（每链路+合计）、GPU 上计算进程数与显存 |
| qfull | 300 s | 解析 `nvidia-smi -q`：PCIe Replay 计数（旧驱动时补 bank 直方图）；每 3600 s 归档原文 |
| inventory | 启动 + 6 h | 型号、序列号、VBIOS、驱动、InfoROM 版本、持久模式、MIG、GSP、功耗上限等；变化记 `inventory_change` 事件 |
| 事件 | 5 s 轮询 | 内核日志 `NVRM: Xid`、NVSwitch `SXid`、ECC/重映射相关 NVRM 行（journalctl 游标断点续读，无 journalctl 时用 dmesg）；可选 NVML 事件 |

**变化检测**：对 ECC/重映射/页退役/直方图 LOW/NONE/恢复动作等，计数增加写 `counter_increase`（含 delta），回退写 `counter_decrease`（驱动重载或 GPU reset 清零 volatile），布尔/枚举翻转写 `state_change`；GPU 从 nvidia-smi 消失写 `gpu_missing`。重启后与上次持久化值比较，停机期间的增量带 `since_previous_run=true`。

**触发快照**：内存类 XID（48/63/64/92/94/95/154/160/171/172/79）或 DBE、重映射、pending/failure、bank LOW/NONE 变化时，在 `snapshots/` 下保存：触发详情、事件前 30 分钟 fast 样本、最近内核 NVRM 行、该卡 `nvidia-smi -q` 全量。同一 GPU 冷却 300 s。启动回溯窗口里的历史 XID 只记 `historical=true`，不做快照。SBE 增量默认不触发快照（`snapshot_on_sbe`）。

## 输出目录（默认 `/var/lib/gpu-ecc-collector`）

```
samples/YYYY-MM-DD/fast-YYYYMMDDTHH.jsonl[.gz]   # 每 GPU 每次一行；跨小时自动 gzip
samples/YYYY-MM-DD/slow-YYYYMMDDTHH.jsonl[.gz]   # slow 与 qfull 两档
events/YYYY-MM-DD/events-YYYY-MM-DD.jsonl[.gz]   # XID/SXID/计数变化/掉卡/采集器启停与错误
inventory/YYYY-MM-DD/inventory-YYYY-MM-DD.jsonl  # 资产与版本、本机不支持指标清单
raw/YYYY-MM-DD/nvidia-smi-q_HHMMSSZ.txt.gz       # 每小时 nvidia-smi -q 原文
snapshots/YYYY-MM-DD/<HHMMSSZ>_gpu<i>_<reason>/  # 事件快照
dcgm/...                                         # 可选 DCGM Exporter 抓取
state/state.json, state/status.json              # 游标/上次计数值；各档最后成功时间、磁盘余量
logs/collector.log                               # 20MB×5 轮转
```

样本行示例（时间统一 UTC）：

```json
{"ts":"2026-11-16T13:04:21.262Z","host":"gpu-a100-01","boot_id":"…","tier":"fast","gpu":0,
 "uuid":"GPU-…","pci":"00000000:07:00.0",
 "m":{"DCGM_FI_DEV_ECC_SBE_VOL_TOTAL":12,"DCGM_FI_DEV_ECC_DBE_VOL_TOTAL":0,
      "DCGM_FI_DEV_ROW_REMAP_PENDING":0,"DCGM_FI_DEV_MEMORY_TEMP":47, "…":"…"},
 "na":{"DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL":"[N/A]"}}
```

读取：`pandas.read_json(path, lines=True)` 后 `pd.json_normalize(df["m"])` 展开。身份键为 `host + uuid`（对应规范的 instance/UUID）；计数器请按 `uuid + boot_id` 分段求增量，处理 `counter_decrease`。

## 资源与运维

- systemd 限制 `CPUQuota=20%`、`MemoryMax=512M`、`Nice=10`。8×A100 每小时约 6 MB 原始 JSONL，压缩后约 0.5–1 MB。
- 磁盘余量 < `min_free_pct`(10%) 停写原文与快照；< `critical_free_pct`(3%) 暂停样本写入。样本保留 90 天，事件保留 730 天。
- nvidia-smi 卡死/超时（掉卡常见前兆）会写 `query_error` 事件，fast 档连续 3 次失败触发主机级快照。
- 读内核日志与 NVML XID 事件需要 root。建议开启持久化 journal，否则崩溃/重启前的 XID 可能丢失。
- NVML 事件与内核日志会重复记录同一 XID（`source` 字段区分），下游按 `uuid+xid+时间±5s` 去重。
- **标签补充**：换卡/RMA/GPU reset/维修工单不在 GPU 侧，请从 CMDB/工单系统导出并按 `serial`/`uuid` 关联，它们是 ECC 预测的关键标签与删失信息。

## 测试

```bash
python -m unittest tests.test_gpu_ecc_collector -v   # 在仓库根目录运行；用伪造 A100 输出验证解析、变化检测与快照
```
