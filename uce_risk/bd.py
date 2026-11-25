"""BD 现网数据适配：工单表 + 遥测表 + 1 分钟指标宽表 -> UCEData（由 CE 预测 1 h / 4 h / 24 h 内的 UCE 风险）。

Three input tables. Each one can be a DataFrame, a CSV / Excel / Parquet file, or a directory of Parquet files
(Hive-style date=YYYYMMDD partitions are read one partition at a time):

  work orders  SEED fault events (ste_data.seed_karl_event_by_date): host key ipv4 / ip, fault_type,
               fault_sub_type (xid_48, NodeNotReady, ...), fault_timestamp / happened_time, parameters_json
  telemetry    the collector's narrow table (ste_data.sysaiops_sndc_offline_dwd): timestamp, uuid, dimensions
               (hostname, ipv4, gpu, rack, tray_index, scale_up_domain, ... and the tag JSON), metric, value
  metrics      the wide 1-minute table: timestamp, uuid, one column per metric. ``metric_table`` builds it from the
               telemetry table (the downstream wide table is not populated yet). Collector names
               (DCGM_FI_DEV_ECC_SBE_VOL_TOTAL) and the downstream snake_case names (ecc_sbe_vol_total) are both accepted.

What is taken from where
  device keys, topology  telemetry dimensions: uuid -> host (same-node features), ipv4 (joins the work orders)
  CE stream (ce__)       increments of the single-bit counters: device memory = TOTAL - SRAM (sub 1), SRAM (sub 2).
                         Volatile counters give the timing; aggregate counters add increments lost in gaps / resets.
  UCE labels             double-bit counter increments, uncorrectable-error codes (default 48 / 94 / 95) and
                         uncorrectable row remaps; UCE work orders confirm them and give the exact fault time. A UCE
                         work order without card evidence becomes a host-level event (WO_UCE_UNATTRIBUTED).
  events (ev__)          driver error codes mapped to vendor-neutral causes, state changes of the metrics (row remap
                         pending / failure, spare rows partial / low / exhausted, SRAM threshold, link and PCIe
                         degradation, throttling, driver reset, collector gaps) and non-UCE work orders (WO_<cause>),
                         the latter on every card of the host
  telemetry (te__)       ~70 derived gauges in 5-minute bins: temperatures and margins, power, workload and memory
                         activity, clocks and throttle reasons, links, and memory-health levels (lifetime CE, remapped
                         rows, spare-row histogram, pending / failure flags)
Every derived record is time-stamped at the end of its minute (or bin), so features at time s only use data < s.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .inputs import _to_seconds
from .schema import DAY, HOUR, UCEData

MIN = 60
TEL_BIN = 300
UCE_CODES = (48, 94, 95)

# ------------------------------------------------------------------ metric catalog
# '*' = in the 30-machine metric list (BD侧30台故障机器遥测指标说明); the others come from the full catalog
# (BD数据集V2) and are used when present.
# kind: static (configuration / threshold), gauge, ratio (0-1), pct (0-100), counter (persistent cumulative),
#       vcounter (cumulative since driver load, resets), flag (0/1 state), bitmask, enum, xid (last driver error code),
#       marker (reported as 1), auto (cumulative or per-interval: decided from the data), unknown
_CATALOG = """
*DCGM_FI_DEV_APP_MEM_CLOCK|app_mem_clock|static|clock|应用时钟配置的显存频率 (MHz)
*DCGM_FI_DEV_APP_SM_CLOCK|app_sm_clock|static|clock|应用时钟配置的 SM 频率 (MHz)
*DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_HIGH|banks_remap_rows_avail_high|gauge|remap|备用行可用性 High 档的 bank 数
*DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_LOW|banks_remap_rows_avail_low|gauge|remap|备用行可用性 Low 档的 bank 数
*DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_MAX|banks_remap_rows_avail_max|gauge|remap|备用行完整可用（Max 档）的 bank 数
*DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_NONE|banks_remap_rows_avail_none|gauge|remap|备用行耗尽（None 档）的 bank 数
*DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_PARTIAL|banks_remap_rows_avail_partial|gauge|remap|备用行部分可用（Partial 档）的 bank 数
*DCGM_FI_DEV_BAR1_FREE|bar1_free|gauge|memory|BAR1 映射空间空闲 (MB)
*DCGM_FI_DEV_BAR1_TOTAL|bar1_total|static|memory|BAR1 映射空间总量 (MB)
*DCGM_FI_DEV_BAR1_USED|bar1_used|gauge|memory|BAR1 映射空间已用 (MB)
*DCGM_FI_DEV_C2C_LINK_ERROR_INTR|c2c_link_error_intr|vcounter|c2c|C2C 链路 CRC 错误计数
*DCGM_FI_DEV_C2C_LINK_ERROR_REPLAY|c2c_link_error_replay|vcounter|c2c|C2C 链路重放计数
*DCGM_FI_DEV_C2C_LINK_ERROR_REPLAY_B2B|c2c_link_error_replay_b2b|vcounter|c2c|C2C 链路背靠背重放计数
*DCGM_FI_DEV_C2C_LINK_POWER_STATE|c2c_link_power_state|enum|c2c|C2C 链路功耗状态
*DCGM_FI_DEV_C2C_LINK_STATUS|c2c_link_status|flag|c2c|C2C 链路激活状态 (1=ACTIVE)
*DCGM_FI_DEV_C2C_MAX_BANDWIDTH|c2c_max_bandwidth|static|c2c|C2C 链路最大带宽 (MB/s)
*DCGM_FI_DEV_CLOCKS_EVENT_REASONS|clocks_event_reasons|bitmask|clock|时钟受限原因位掩码
*DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS|correctable_remapped_rows|counter|remap|因可纠正错误重映射的行数
*DCGM_FI_DEV_COUNT|gpu_count|gauge|host|节点识别到的计算卡数量
*DCGM_FI_DEV_DEC_UTIL|decoder_utilization|pct|workload|视频解码器利用率
*DCGM_FI_DEV_ECC_CURRENT|ecc_current|flag|ecc|ECC 开关 (1=开)
*DCGM_FI_DEV_ECC_DBE_AGG_TOTAL|ecc_dbe_agg_total|counter|ecc|持久累计双比特（不可纠正）错误
*DCGM_FI_DEV_ECC_DBE_VOL_TOTAL|ecc_dbe_vol_total|vcounter|ecc|易失累计双比特（不可纠正）错误
*DCGM_FI_DEV_ECC_SBE_AGG_SRM|ecc_sbe_agg_srm|counter|ecc|SRAM 持久累计单比特（可纠正）错误
*DCGM_FI_DEV_ECC_SBE_AGG_TOTAL|ecc_sbe_agg_total|counter|ecc|持久累计单比特（可纠正）错误
*DCGM_FI_DEV_ECC_SBE_VOL_SRM|ecc_sbe_vol_srm|vcounter|ecc|SRAM 易失累计单比特错误
*DCGM_FI_DEV_ECC_SBE_VOL_TOTAL|ecc_sbe_vol_total|vcounter|ecc|易失累计单比特错误
*DCGM_FI_DEV_ENC_UTIL|encoder_utilization|pct|workload|视频编码器利用率
*DCGM_FI_DEV_FB_FREE|fb_free|gauge|memory|设备内存空闲 (MiB)
*DCGM_FI_DEV_FB_TOTAL|fb_total|static|memory|设备内存总量 (MiB)
*DCGM_FI_DEV_FB_USED|fb_used|gauge|memory|设备内存已用 (MiB)
*DCGM_FI_DEV_FB_USED_PERCENT|fb_used_percent|ratio|memory|设备内存已用比例 (0-1)
*DCGM_FI_DEV_GPU_MAX_OP_TEMP|gpu_max_op_temperature|static|thermal|芯片最高运行温度阈值
*DCGM_FI_DEV_GPU_TEMP|gpu_temperature|gauge|thermal|芯片温度
*DCGM_FI_DEV_GPU_UTIL|gpu_utilization|pct|workload|芯片利用率
*DCGM_FI_DEV_MAX_MEM_CLOCK|max_mem_clock|static|clock|最大显存频率
*DCGM_FI_DEV_MAX_SM_CLOCK|max_sm_clock|static|clock|最大 SM 频率
*DCGM_FI_DEV_MAX_VIDEO_CLOCK|max_video_clock|static|clock|最大视频单元频率
*DCGM_FI_DEV_MEMORY_TEMP|memory_temperature|gauge|thermal|设备内存温度
*DCGM_FI_DEV_MEM_CLOCK|mem_clock|gauge|clock|显存频率
*DCGM_FI_DEV_MEM_COPY_UTIL|mem_copy_utilization|pct|workload|内存读写活动利用率
*DCGM_FI_DEV_MEM_MAX_OP_TEMP|mem_max_op_temperature|static|thermal|设备内存最高运行温度阈值
*DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_0|nvlink_count_fec_history_0|vcounter|link|片间链路 FEC 纠错直方图 bin 0
*DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_10|nvlink_count_fec_history_10|vcounter|link|片间链路 FEC 纠错直方图 bin 10
*DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_15|nvlink_count_fec_history_15|vcounter|link|片间链路 FEC 纠错直方图 bin 15
*DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_5|nvlink_count_fec_history_5|vcounter|link|片间链路 FEC 纠错直方图 bin 5
*DCGM_FI_DEV_P2P_NVLINK_STATUS|p2p_nvlink_status|bitmask|link|到同机其他卡的 P2P 连接位图
*DCGM_FI_DEV_PCIE_LINK_GEN|pcie_link_gen|enum|pcie|PCIe 当前代数
*DCGM_FI_DEV_PCIE_LINK_WIDTH|pcie_link_width|enum|pcie|PCIe 当前宽度
*DCGM_FI_DEV_PCIE_MAX_LINK_GEN|pcie_max_link_gen|static|pcie|PCIe 最大代数
*DCGM_FI_DEV_PCIE_MAX_LINK_WIDTH|pcie_max_link_width|static|pcie|PCIe 最大宽度
*DCGM_FI_DEV_POWER_USAGE|power_usage|gauge|power|功率 (W)
*DCGM_FI_DEV_POWER_USAGE_INSTANT|power_usage_instant|gauge|power|瞬时功率 (W)
*DCGM_FI_DEV_ROW_REMAP_FAILURE|row_remap_failure|flag|remap|行重映射失败 (0/1)
*DCGM_FI_DEV_ROW_REMAP_PENDING|row_remap_pending|flag|remap|有待完成的行重映射 (0/1)
*DCGM_FI_DEV_SHUTDOWN_TEMP|shutdown_temperature|static|thermal|关机保护温度阈值
*DCGM_FI_DEV_SLOWDOWN_TEMP|slowdown_temperature|static|thermal|降频保护温度阈值
*DCGM_FI_DEV_SM_CLOCK|sm_clock|gauge|clock|SM 频率
*DCGM_FI_DEV_SYSIO_POWER_UTIL_CURRENT|sysio_power_utilization_current|gauge|power|SoC/系统 I/O 功率 (W)
*DCGM_FI_DEV_THRESHOLD_SRM|threshold_srm|flag|ecc|SRAM 错误超阈值 (0/1)
*DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION|total_energy_consumption|vcounter|power|驱动加载以来累计能量 (mJ)
*DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS|uncorrectable_remapped_rows|counter|remap|因不可纠正错误重映射的行数
*DCGM_FI_DEV_VIDEO_CLOCK|video_clock|gauge|clock|视频单元频率
*DCGM_FI_DEV_XID_ERRORS|xid_errors|xid|event|驱动错误码（最近值）
*DCGM_FI_PROF_DRAM_ACTIVE|dram_active|ratio|workload|内存接口活跃周期比例
*DCGM_FI_PROF_GR_ENGINE_ACTIVE|gr_engine_active|ratio|workload|计算引擎活跃比例
*DCGM_FI_PROF_HOSTMEM_CACHE_HIT|hostmem_cache_hit|pct|workload|访问主机内存的缓存命中占比
*DCGM_FI_PROF_HOSTMEM_CACHE_MISS|hostmem_cache_miss|pct|workload|访问主机内存的缓存未命中占比
*DCGM_FI_PROF_PEERMEM_CACHE_HIT|peermem_cache_hit|pct|workload|访问对端内存的缓存命中占比
*DCGM_FI_PROF_PEERMEM_CACHE_MISS|peermem_cache_miss|pct|workload|访问对端内存的缓存未命中占比
*DCGM_FI_PROF_PIPE_FP16_ACTIVE|pipe_fp16_active|ratio|workload|FP16 管线活跃比例
*DCGM_FI_PROF_PIPE_FP32_ACTIVE|pipe_fp32_active|ratio|workload|FP32 管线活跃比例
*DCGM_FI_PROF_PIPE_FP64_ACTIVE|pipe_fp64_active|ratio|workload|FP64 管线活跃比例
*DCGM_FI_PROF_PIPE_INT_ACTIVE|pipe_int_active|ratio|workload|整数管线活跃比例
*DCGM_FI_PROF_PIPE_TENSOR_ACTIVE|pipe_tensor_active|ratio|workload|张量管线活跃比例
*DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE|pipe_tensor_dfma_active|ratio|workload|张量 DFMA 管线活跃比例
*DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE|pipe_tensor_hmma_active|ratio|workload|张量 HMMA 管线活跃比例
*DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE|pipe_tensor_imma_active|ratio|workload|张量 IMMA 管线活跃比例
*DCGM_FI_PROF_SM_ACTIVE|sm_active|ratio|workload|SM 活跃比例
*DCGM_FI_PROF_SM_OCCUPANCY|sm_occupancy|ratio|workload|SM 占用率
*cpu_power_usage_watts|cpu_power_usage_watts|gauge|host|CPU 功率 (W)
*ipmitool_sensor|ipmitool_sensor|unknown|host|IPMI 传感器读数（样本值为空）
*sdnc_cmem_rd_access|sdnc_cmem_rd_access|auto|host|CMem 读访问次数
*sdnc_cmem_rd_data|sdnc_cmem_rd_data|auto|host|CMem 读取数据量
*sdnc_cmem_wb_access|sdnc_cmem_wb_access|auto|host|CMem 写回访问次数
*sdnc_cmem_wr_access|sdnc_cmem_wr_access|auto|host|CMem 写访问次数
*sdnc_cmem_wr_total_bytes|sdnc_cmem_wr_total_bytes|auto|host|CMem 写入总字节数
*sdnc_cnvlink_read_bandwidth|sdnc_cnvlink_read_bandwidth|gauge|link|片间网络读带宽 (GB/s)
*sdnc_cnvlink_write_bandwidth|sdnc_cnvlink_write_bandwidth|gauge|link|片间网络写带宽 (GB/s)
*sdnc_cpu_memory_read_bandwidth|sdnc_cpu_memory_read_bandwidth|gauge|host|CPU 内存读带宽 (GB/s)
*sdnc_cpu_memory_read_utilization|sdnc_cpu_memory_read_utilization|gauge|host|CPU 内存读带宽利用率
*sdnc_cpu_memory_write_bandwidth|sdnc_cpu_memory_write_bandwidth|gauge|host|CPU 内存写带宽 (GB/s)
*sdnc_cpu_memory_write_utilization|sdnc_cpu_memory_write_utilization|gauge|host|CPU 内存写带宽利用率
*sdnc_nvlink_c2c_read_bandwidth|sdnc_nvlink_c2c_read_bandwidth|gauge|c2c|C2C 读带宽 (GB/s)
*sdnc_nvlink_c2c_write_bandwidth|sdnc_nvlink_c2c_write_bandwidth|gauge|c2c|C2C 写带宽 (GB/s)
*sdnc_pcie_read_bandwidth|sdnc_pcie_read_bandwidth|gauge|pcie|PCIe 读带宽 (GB/s)
*sdnc_pcie_read_utilization|sdnc_pcie_read_utilization|gauge|pcie|PCIe 读利用率
*sdnc_pcie_write_bandwidth|sdnc_pcie_write_bandwidth|gauge|pcie|PCIe 写带宽 (GB/s)
*sdnc_pcie_write_utilization|sdnc_pcie_write_utilization|gauge|pcie|PCIe 写利用率
*sndc_host_basic_info|sndc_host_basic_info|marker|host|主机基础信息上报标记
*sndc_host_heartbeat|sndc_host_heartbeat|marker|host|主机/采集器心跳
DCGM_FI_DEV_GPU_TEMP_LIMIT|gpu_temperature_limit|static|thermal|温度告警/降频限制值
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_3|nvlink_count_fec_history_3|vcounter|link|片间链路 FEC 纠错直方图 bin 3
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_11|nvlink_count_fec_history_11|vcounter|link|片间链路 FEC 纠错直方图 bin 11
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_14|nvlink_count_fec_history_14|vcounter|link|片间链路 FEC 纠错直方图 bin 14
DCGM_FI_DEV_CC_MODE|cc_mode|static|config|机密计算模式
DCGM_FI_DEV_COMPUTE_MODE|compute_mode|static|config|计算模式
DCGM_FI_DEV_CPU_AFFINITY_0|cpu_affinity_0|static|config|CPU 亲和性掩码 0
DCGM_FI_DEV_CPU_AFFINITY_1|cpu_affinity_1|static|config|CPU 亲和性掩码 1
DCGM_FI_DEV_CPU_AFFINITY_2|cpu_affinity_2|static|config|CPU 亲和性掩码 2
DCGM_FI_DEV_CPU_AFFINITY_3|cpu_affinity_3|static|config|CPU 亲和性掩码 3
DCGM_FI_DEV_MEM_AFFINITY_0|mem_affinity_0|static|config|内存亲和性掩码 0
DCGM_FI_DEV_MEM_AFFINITY_1|mem_affinity_1|static|config|内存亲和性掩码 1
DCGM_FI_DEV_MEM_AFFINITY_2|mem_affinity_2|static|config|内存亲和性掩码 2
DCGM_FI_DEV_MEM_AFFINITY_3|mem_affinity_3|static|config|内存亲和性掩码 3
DCGM_FI_DEV_CUDA_COMPUTE_CAPABILITY|cuda_compute_capability|static|config|计算能力版本号
DCGM_FI_DEV_ENFORCED_POWER_LIMIT|enforced_power_limit|gauge|power|当前强制功耗上限 (W)
DCGM_FI_DEV_FABRIC_MANAGER_STATUS|fabric_manager_status|enum|link|互连管理服务状态
DCGM_FI_DEV_GPM_SUPPORT|gpm_support|static|config|性能监控支持
DCGM_FI_DEV_INFOROM_CONFIG_CHECK|inforom_config_check|enum|config|InfoROM 配置校验
DCGM_FI_DEV_INFOROM_CONFIG_VALID|inforom_config_valid|flag|config|InfoROM 配置有效
DCGM_FI_DEV_MIG_MAX_SLICES|mig_max_slices|static|config|最大切分数
DCGM_FI_DEV_MIG_MODE|mig_mode|static|config|切分模式
DCGM_FI_DEV_MINOR_NUMBER|minor_number|static|config|次设备号
DCGM_FI_DEV_PCI_COMBINED_ID|pci_combined_id|static|config|PCI 合并标识
DCGM_FI_DEV_PCI_SUBSYS_ID|pci_subsys_id|static|config|PCI 子系统标识
DCGM_FI_DEV_PERSISTENCE_MODE|persistence_mode|static|config|驱动持久化模式
DCGM_FI_DEV_POWER_MGMT_LIMIT|power_mgmt_limit|gauge|power|功率管理上限 (W)
DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF|power_mgmt_limit_def|static|power|功率上限默认值
DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX|power_mgmt_limit_max|static|power|功率上限最大值
DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN|power_mgmt_limit_min|static|power|功率上限最小值
DCGM_FI_DEV_BOARD_LIMIT_VIOLATION|board_limit_violation|vcounter|throttle|整板功率极限受限时间
DCGM_FI_DEV_C2C_LINK_COUNT|c2c_link_count|gauge|c2c|激活的 C2C 链路数
DCGM_FI_DEV_ECC_DBE_AGG_DEV|ecc_dbe_agg_dev|counter|ecc|设备内存持久累计双比特错误
DCGM_FI_DEV_ECC_DBE_AGG_SRM|ecc_dbe_agg_srm|counter|ecc|SRAM 持久累计双比特错误
DCGM_FI_DEV_ECC_DBE_VOL_DEV|ecc_dbe_vol_dev|vcounter|ecc|设备内存易失累计双比特错误
DCGM_FI_DEV_ECC_DBE_VOL_SRM|ecc_dbe_vol_srm|vcounter|ecc|SRAM 易失累计双比特错误
DCGM_FI_DEV_ECC_PENDING|ecc_pending|flag|ecc|有待处理的 ECC 复位
DCGM_FI_DEV_ECC_SBE_AGG_DEV|ecc_sbe_agg_dev|counter|ecc|设备内存持久累计单比特错误
DCGM_FI_DEV_ECC_SBE_VOL_DEV|ecc_sbe_vol_dev|vcounter|ecc|设备内存易失累计单比特错误
DCGM_FI_DEV_FABRIC_MANAGER_ERROR_CODE|fabric_manager_error_code|enum|link|互连管理服务错误码
DCGM_FI_DEV_FB_RESERVED|fb_reserved|static|memory|预留设备内存 (MiB)
DCGM_FI_DEV_LOW_UTIL_VIOLATION|low_utilization_violation|vcounter|throttle|低利用率受限时间
DCGM_FI_DEV_PCIE_REPLAY_COUNTER|pcie_replay_counter|vcounter|pcie|PCIe 重传计数
DCGM_FI_DEV_POWER_VIOLATION|power_violation|vcounter|throttle|功耗墙受限时间
DCGM_FI_DEV_PSTATE|pstate|enum|clock|性能状态 (P0 最高)
DCGM_FI_DEV_RELIABILITY_VIOLATION|reliability_violation|vcounter|throttle|可靠性受限时间
DCGM_FI_DEV_SYNC_BOOST_VIOLATION|sync_boost_violation|vcounter|throttle|同步加速受限时间
DCGM_FI_DEV_THERMAL_VIOLATION|thermal_violation|vcounter|throttle|温度墙受限时间
DCGM_FI_DEV_TOTAL_APP_CLOCKS_VIOLATION|total_app_clocks_violation|vcounter|throttle|应用频率设置受限时间
DCGM_FI_DEV_TOTAL_BASE_CLOCKS_VIOLATION|total_base_clocks_violation|vcounter|throttle|基础频率设置受限时间
software_health_check|software_health_check|gauge|host|软件健康检查综合评分
"""


def _parse_catalog(text: str) -> pd.DataFrame:
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            name, snake, kind, group, zh = (x.strip() for x in line.lstrip("*").split("|"))
            rows.append((name, snake, kind, group, zh, line.startswith("*")))
    return pd.DataFrame(rows, columns=["metric", "snake", "kind", "group", "zh", "pdf1"]).set_index("metric")


CATALOG = _parse_catalog(_CATALOG)
PDF1_METRICS = CATALOG.index[CATALOG.pdf1].tolist()          # the 101 metrics of the 30-machine list
_P = "DCGM_FI_DEV_"
_F = "DCGM_FI_PROF_"
SBE_VOL, SBE_AGG, SBE_VOL_SRM, SBE_AGG_SRM = (_P + x for x in ("ECC_SBE_VOL_TOTAL", "ECC_SBE_AGG_TOTAL", "ECC_SBE_VOL_SRM",
                                                                "ECC_SBE_AGG_SRM"))
SBE_VOL_DEV, SBE_AGG_DEV = _P + "ECC_SBE_VOL_DEV", _P + "ECC_SBE_AGG_DEV"
DBE_VOL, DBE_AGG = _P + "ECC_DBE_VOL_TOTAL", _P + "ECC_DBE_AGG_TOTAL"
XID = _P + "XID_ERRORS"
BANKS = [_P + f"BANKS_REMAP_ROWS_AVAIL_{x}" for x in ("MAX", "HIGH", "PARTIAL", "LOW", "NONE")]

# driver error codes -> vendor-neutral causes (vendor codes stay inside this adapter)
XID_CAUSE = {13: "COMPUTE_ENGINE_EXCEPTION", 31: "DEVICE_MEMORY_ILLEGAL_ACCESS", 32: "COMMAND_STREAM_CORRUPTED",
             38: "DRIVER_FIRMWARE_ERROR", 43: "COMPUTE_TASK_ABORTED", 44: "CONTEXT_SWITCH_FAULT",
             45: "PREEMPTIVE_CLEANUP", 48: "MEMORY_UNCORRECTABLE_ERROR", 61: "MICROCONTROLLER_WARNING",
             62: "MICROCONTROLLER_HALT", 63: "MEMORY_PAGE_RETIRED", 64: "MEMORY_PAGE_RETIRE_FAILED",
             69: "COMPUTE_ENGINE_EXCEPTION", 74: "INTERCONNECT_LINK_ERROR", 79: "DEVICE_LOST",
             92: "MEMORY_CORRECTABLE_ERROR_STORM", 94: "MEMORY_UNCORRECTABLE_ERROR", 95: "MEMORY_UNCORRECTABLE_ERROR",
             109: "CONTEXT_SWITCH_FAULT", 119: "MICROCONTROLLER_HALT", 120: "MICROCONTROLLER_WARNING",
             121: "INTERCONNECT_LINK_ERROR"}
# application / runtime causes: frequent on healthy cards, featurised as background ('bg_') events
BACKGROUND_CAUSES = {"COMPUTE_ENGINE_EXCEPTION", "DEVICE_MEMORY_ILLEGAL_ACCESS", "COMMAND_STREAM_CORRUPTED",
                     "COMPUTE_TASK_ABORTED", "CONTEXT_SWITCH_FAULT", "PREEMPTIVE_CLEANUP"}
CODE_ZH = {
    "MEMORY_UNCORRECTABLE_ERROR": "设备内存不可纠正错误", "MEMORY_PAGE_RETIRED": "内存页退役/行重映射",
    "MEMORY_PAGE_RETIRE_FAILED": "内存页退役/行重映射失败", "MEMORY_CORRECTABLE_ERROR_STORM": "可纠正错误率过高",
    "INTERCONNECT_LINK_ERROR": "片间互连链路错误", "DEVICE_LOST": "设备掉卡", "MICROCONTROLLER_WARNING": "片上管理控制器告警",
    "MICROCONTROLLER_HALT": "片上管理控制器停机", "DRIVER_FIRMWARE_ERROR": "驱动/固件错误", "UNKNOWN_ERROR": "未知驱动错误",
    "COMPUTE_ENGINE_EXCEPTION": "计算引擎异常（应用）", "DEVICE_MEMORY_ILLEGAL_ACCESS": "设备内存非法访问（应用）",
    "COMMAND_STREAM_CORRUPTED": "命令流损坏", "COMPUTE_TASK_ABORTED": "计算任务中止（应用）",
    "CONTEXT_SWITCH_FAULT": "上下文切换故障", "PREEMPTIVE_CLEANUP": "抢占式清理（次生）",
    "CE_ROW_REMAPPED": "可纠正错误触发行重映射", "REMAP_PENDING": "出现待完成的行重映射", "REMAP_FAILURE": "行重映射失败",
    "SPARE_ROWS_PARTIAL": "bank 备用行降到 Partial", "SPARE_ROWS_LOW": "bank 备用行降到 Low",
    "SPARE_ROWS_EXHAUSTED": "bank 备用行耗尽", "SRAM_THRESHOLD": "SRAM 错误超阈值", "ECC_OFF": "ECC 被关闭",
    "ECC_RESET_PENDING": "ECC 待复位", "DEVICE_MISSING": "节点识别到的卡数减少", "PCIE_DEGRADED": "PCIe 链路降级",
    "PCIE_REPLAY": "PCIe 重传", "C2C_LINK_DOWN": "C2C 链路断开", "C2C_LINK_ERROR": "C2C 链路错误",
    "LINK_FEC_HIGH": "片间链路高位 FEC 纠错", "P2P_LINK_DROP": "P2P 互连减少", "THERMAL_THROTTLE": "温度降频",
    "HW_SLOWDOWN": "硬件降频/功率刹车", "DRIVER_RESET": "驱动重载或卡复位", "COLLECTOR_GAP": "采集中断 10 分钟以上",
    "UCE_UNATTRIBUTED": "无法定位到卡的不可纠正错误", "NodeNotReady": "节点 NotReady",
}

# derived gauges (te__): (name, how, metric arguments, 中文). An argument may be a tuple of alternatives.
TE_SPEC = [
    ("chip_temp", "col", [_P + "GPU_TEMP"], "芯片温度"),
    ("mem_temp", "col", [_P + "MEMORY_TEMP"], "设备内存温度"),
    ("chip_temp_margin", "diff", [(_P + "SLOWDOWN_TEMP", _P + "GPU_TEMP_LIMIT", _P + "GPU_MAX_OP_TEMP"), _P + "GPU_TEMP"],
     "芯片距降频温度的余量"),
    ("mem_temp_margin", "diff", [_P + "MEM_MAX_OP_TEMP", _P + "MEMORY_TEMP"], "设备内存距最高运行温度的余量"),
    ("power", "col", [_P + "POWER_USAGE"], "功率"),
    ("power_spike", "absdiff", [_P + "POWER_USAGE_INSTANT", _P + "POWER_USAGE"], "瞬时功率与平均功率之差"),
    ("energy_power", "rate_w", [_P + "TOTAL_ENERGY_CONSUMPTION"], "能量计数折算的功率"),
    ("power_limit_frac", "ratio", [_P + "POWER_USAGE", (_P + "ENFORCED_POWER_LIMIT", _P + "POWER_MGMT_LIMIT")], "功率/功耗上限"),
    ("sysio_power", "col", [_P + "SYSIO_POWER_UTIL_CURRENT"], "SoC/系统 I/O 功率"),
    ("cpu_power", "col", ["cpu_power_usage_watts"], "CPU 功率"),
    ("util", "col", [_P + "GPU_UTIL"], "芯片利用率"),
    ("mem_copy_util", "col", [_P + "MEM_COPY_UTIL"], "内存读写活动利用率"),
    ("dram_active", "col", [_F + "DRAM_ACTIVE"], "内存接口活跃比例"),
    ("fb_used_frac", "col", [_P + "FB_USED_PERCENT"], "设备内存已用比例"),
    ("bar1_used_frac", "ratio", [_P + "BAR1_USED", _P + "BAR1_TOTAL"], "BAR1 已用比例"),
    ("sm_active", "col", [_F + "SM_ACTIVE"], "SM 活跃比例"),
    ("sm_occupancy", "col", [_F + "SM_OCCUPANCY"], "SM 占用率"),
    ("gr_active", "col", [_F + "GR_ENGINE_ACTIVE"], "计算引擎活跃比例"),
    ("tensor_active", "col", [_F + "PIPE_TENSOR_ACTIVE"], "张量管线活跃比例"),
    ("tensor_hmma", "col", [_F + "PIPE_TENSOR_HMMA_ACTIVE"], "张量 HMMA 活跃比例"),
    ("tensor_imma", "col", [_F + "PIPE_TENSOR_IMMA_ACTIVE"], "张量 IMMA 活跃比例"),
    ("tensor_dfma", "col", [_F + "PIPE_TENSOR_DFMA_ACTIVE"], "张量 DFMA 活跃比例"),
    ("fp16", "col", [_F + "PIPE_FP16_ACTIVE"], "FP16 管线活跃比例"),
    ("fp32", "col", [_F + "PIPE_FP32_ACTIVE"], "FP32 管线活跃比例"),
    ("fp64", "col", [_F + "PIPE_FP64_ACTIVE"], "FP64 管线活跃比例"),
    ("int_active", "col", [_F + "PIPE_INT_ACTIVE"], "整数管线活跃比例"),
    ("hostmem_hit", "col", [_F + "HOSTMEM_CACHE_HIT"], "主机内存访问缓存命中占比"),
    ("hostmem_miss", "col", [_F + "HOSTMEM_CACHE_MISS"], "主机内存访问缓存未命中占比"),
    ("peermem_hit", "col", [_F + "PEERMEM_CACHE_HIT"], "对端内存访问缓存命中占比"),
    ("peermem_miss", "col", [_F + "PEERMEM_CACHE_MISS"], "对端内存访问缓存未命中占比"),
    ("dec_util", "col", [_P + "DEC_UTIL"], "解码器利用率"),
    ("enc_util", "col", [_P + "ENC_UTIL"], "编码器利用率"),
    ("sm_clock_frac", "ratio", [_P + "SM_CLOCK", (_P + "MAX_SM_CLOCK", _P + "APP_SM_CLOCK")], "SM 频率/最大频率"),
    ("mem_clock_frac", "ratio", [_P + "MEM_CLOCK", (_P + "MAX_MEM_CLOCK", _P + "APP_MEM_CLOCK")], "显存频率/最大频率"),
    ("video_clock_frac", "ratio", [_P + "VIDEO_CLOCK", _P + "MAX_VIDEO_CLOCK"], "视频频率/最大频率"),
    ("thr_idle", "bits", [_P + "CLOCKS_EVENT_REASONS", 0x1], "空闲降频"),
    ("thr_power_cap", "bits", [_P + "CLOCKS_EVENT_REASONS", 0x4], "功耗上限降频"),
    ("thr_hw_slowdown", "bits", [_P + "CLOCKS_EVENT_REASONS", 0x8 | 0x80], "硬件降频/功率刹车"),
    ("thr_thermal", "bits", [_P + "CLOCKS_EVENT_REASONS", 0x20 | 0x40], "温度降频"),
    ("thr_sync_boost", "bits", [_P + "CLOCKS_EVENT_REASONS", 0x10], "同步加速限制"),
    ("pstate", "col", [_P + "PSTATE"], "性能状态"),
    ("pcie_gen_deficit", "diff", [_P + "PCIE_MAX_LINK_GEN", _P + "PCIE_LINK_GEN"], "PCIe 代数低于最大值的差"),
    ("pcie_width_deficit", "diff", [_P + "PCIE_MAX_LINK_WIDTH", _P + "PCIE_LINK_WIDTH"], "PCIe 宽度低于最大值的差"),
    ("pcie_replay_rate", "rate", [_P + "PCIE_REPLAY_COUNTER"], "PCIe 重传速率（每分钟）"),
    ("c2c_up", "col", [_P + "C2C_LINK_STATUS"], "C2C 链路激活"),
    ("c2c_power_state", "col", [_P + "C2C_LINK_POWER_STATE"], "C2C 功耗状态"),
    ("c2c_err_rate", "rate", [_P + "C2C_LINK_ERROR_INTR", _P + "C2C_LINK_ERROR_REPLAY", _P + "C2C_LINK_ERROR_REPLAY_B2B"],
     "C2C 链路错误速率（每分钟）"),
    ("p2p_links", "popcount", [_P + "P2P_NVLINK_STATUS"], "P2P 互连数"),
    ("fec_low_rate", "lograte", [_P + "NVLINK_COUNT_FEC_HISTORY_0"], "片间链路 FEC bin 0 速率 (log)"),
    ("fec_mid_rate", "rate", [_P + "NVLINK_COUNT_FEC_HISTORY_3", _P + "NVLINK_COUNT_FEC_HISTORY_5"], "片间链路 FEC 中位 bin 速率"),
    ("fec_high_rate", "rate", [_P + "NVLINK_COUNT_FEC_HISTORY_10", _P + "NVLINK_COUNT_FEC_HISTORY_11",
                               _P + "NVLINK_COUNT_FEC_HISTORY_14", _P + "NVLINK_COUNT_FEC_HISTORY_15"], "片间链路 FEC 高位 bin 速率"),
    ("cnvlink_rd_bw", "col", ["sdnc_cnvlink_read_bandwidth"], "片间网络读带宽"),
    ("cnvlink_wr_bw", "col", ["sdnc_cnvlink_write_bandwidth"], "片间网络写带宽"),
    ("cpu_mem_rd_bw", "col", ["sdnc_cpu_memory_read_bandwidth"], "CPU 内存读带宽"),
    ("cpu_mem_wr_bw", "col", ["sdnc_cpu_memory_write_bandwidth"], "CPU 内存写带宽"),
    ("cpu_mem_rd_util", "col", ["sdnc_cpu_memory_read_utilization"], "CPU 内存读带宽利用率"),
    ("cpu_mem_wr_util", "col", ["sdnc_cpu_memory_write_utilization"], "CPU 内存写带宽利用率"),
    ("c2c_rd_bw", "col", ["sdnc_nvlink_c2c_read_bandwidth"], "C2C 读带宽"),
    ("c2c_wr_bw", "col", ["sdnc_nvlink_c2c_write_bandwidth"], "C2C 写带宽"),
    ("pcie_rd_bw", "col", ["sdnc_pcie_read_bandwidth"], "PCIe 读带宽"),
    ("pcie_wr_bw", "col", ["sdnc_pcie_write_bandwidth"], "PCIe 写带宽"),
    ("pcie_rd_util", "col", ["sdnc_pcie_read_utilization"], "PCIe 读利用率"),
    ("pcie_wr_util", "col", ["sdnc_pcie_write_utilization"], "PCIe 写利用率"),
    ("cmem_rd_access", "auto", ["sdnc_cmem_rd_access"], "CMem 读访问（每分钟）"),
    ("cmem_rd_data", "auto", ["sdnc_cmem_rd_data"], "CMem 读数据量（每分钟）"),
    ("cmem_wb_access", "auto", ["sdnc_cmem_wb_access"], "CMem 写回访问（每分钟）"),
    ("cmem_wr_access", "auto", ["sdnc_cmem_wr_access"], "CMem 写访问（每分钟）"),
    ("cmem_wr_bytes", "auto", ["sdnc_cmem_wr_total_bytes"], "CMem 写入字节（每分钟）"),
    ("thermal_violation_rate", "rate", [_P + "THERMAL_VIOLATION"], "温度墙受限时间增速"),
    ("power_violation_rate", "rate", [_P + "POWER_VIOLATION"], "功耗墙受限时间增速"),
    ("reliability_violation_rate", "rate", [_P + "RELIABILITY_VIOLATION"], "可靠性受限时间增速"),
    ("health_check", "col", ["software_health_check"], "软件健康检查评分"),
    # memory health levels
    ("ce_life", "col", [(SBE_AGG, SBE_VOL)], "累计可纠正错误数（含观测前历史）"),
    ("sram_ce_life", "col", [(SBE_AGG_SRM, SBE_VOL_SRM)], "SRAM 累计可纠正错误数"),
    ("ue_life", "col", [(DBE_AGG, DBE_VOL)], "累计不可纠正错误数（含观测前历史）"),
    ("ce_rows_remapped", "col", [_P + "CORRECTABLE_REMAPPED_ROWS"], "因可纠正错误重映射的行数"),
    ("ue_rows_remapped", "col", [_P + "UNCORRECTABLE_REMAPPED_ROWS"], "因不可纠正错误重映射的行数"),
    ("remap_pending", "col", [_P + "ROW_REMAP_PENDING"], "有待完成的行重映射"),
    ("remap_failure", "col", [_P + "ROW_REMAP_FAILURE"], "行重映射失败"),
    ("sram_threshold", "col", [_P + "THRESHOLD_SRM"], "SRAM 错误超阈值"),
    ("banks_used", "banks_used", BANKS, "已动用备用行的 bank 数"),
    ("banks_partial", "col", [BANKS[2]], "备用行 Partial 档 bank 数"),
    ("banks_low", "col", [BANKS[3]], "备用行 Low 档 bank 数"),
    ("banks_none", "col", [BANKS[4]], "备用行耗尽的 bank 数"),
    ("ecc_on", "col", [_P + "ECC_CURRENT"], "ECC 开启"),
    ("ecc_reset_pending", "col", [_P + "ECC_PENDING"], "ECC 待复位"),
    ("dev_count", "col", [_P + "COUNT"], "节点识别到的卡数"),
    ("heartbeat", "col", ["sndc_host_heartbeat"], "采集器心跳"),
]
TE_ZH = {n: zh for n, _, _, zh in TE_SPEC} | {"present": "分钟数据完整率（5 分钟内上报分钟数/5）"}
# event codes derived from metric changes
_EDGE_RISE = [("REMAP_PENDING", _P + "ROW_REMAP_PENDING"), ("REMAP_FAILURE", _P + "ROW_REMAP_FAILURE"),
              ("SRAM_THRESHOLD", _P + "THRESHOLD_SRM"), ("ECC_RESET_PENDING", _P + "ECC_PENDING")]
_EDGE_FALL = [("ECC_OFF", _P + "ECC_CURRENT"), ("C2C_LINK_DOWN", _P + "C2C_LINK_STATUS")]
_INCR = [("CE_ROW_REMAPPED", [_P + "CORRECTABLE_REMAPPED_ROWS"]),
         ("SPARE_ROWS_PARTIAL", [BANKS[2], BANKS[3], BANKS[4]]), ("SPARE_ROWS_LOW", [BANKS[3], BANKS[4]]),
         ("SPARE_ROWS_EXHAUSTED", [BANKS[4]]),
         ("C2C_LINK_ERROR", [_P + "C2C_LINK_ERROR_INTR", _P + "C2C_LINK_ERROR_REPLAY", _P + "C2C_LINK_ERROR_REPLAY_B2B"]),
         ("LINK_FEC_HIGH", [_P + "NVLINK_COUNT_FEC_HISTORY_10", _P + "NVLINK_COUNT_FEC_HISTORY_11",
                            _P + "NVLINK_COUNT_FEC_HISTORY_14", _P + "NVLINK_COUNT_FEC_HISTORY_15"]),
         ("PCIE_REPLAY", [_P + "PCIE_REPLAY_COUNTER"])]
GAP_EVENT = 600            # a collector gap of >= 10 minutes is an event

# Feature profile for small fleets (tens of machines, a few dozen UCEs). Every metric is still converted by the
# adapter; the model keeps the groups with a physical link to memory UCEs. On the 30-machine simulation the full
# feature set (~470 columns: every gauge x 4 statistics, link / workload / application events) fitted noise and lost
# to this profile at every horizon (experiments/bd_sample/REPORT.md). Use include=() once more data is available.
FEATURES_CE = (r"^ce__", r"^hi__", r"^nbh__", r"^nbc__")       # CE (incl. CEs per unit of memory activity), UCE history,
#                                                                same-host UCE / CE
FEATURES_HEALTH = (                                             # memory-health events and levels
    r"^ev__(MEMORY_|CE_ROW_REMAPPED|REMAP_|SPARE_ROWS_|SRAM_THRESHOLD|ECC_|DRIVER_RESET|WO_UCE).*_(1d|7d|30d|h_since)$",
    r"^te__(ce_life|sram_ce_life|ue_life|ce_rows_remapped|ue_rows_remapped|remap_pending|remap_failure|sram_threshold|"
    r"banks_used|banks_partial|banks_low|banks_none)_(mean1h|delta1h)$")
FEATURES_ENV = (r"^te__(mem_temp|mem_temp_margin|chip_temp|dram_active|util|power)_(mean1h|mean1d)$",   # thermal, exposure
                r"^te__present_mean1h$")
COMPACT = FEATURES_CE + FEATURES_HEALTH + FEATURES_ENV
# regularised trees for few positives (larger leaves, column subsampling, stronger L2)
SMALL_DATA_PARAMS = dict(learning_rate=0.03, n_estimators=300, num_leaves=15, min_child_samples=100, subsample=0.8,
                         colsample_bytree=0.5, reg_lambda=5.0)
# Short horizons hold only a few dozen positive samples per training run. The (0, 1 h] segment is learned from the
# CE-burst features alone (an imminent UCE shows as a burst; anything else there fitted noise); the (1 h, 4 h] segment
# reuses the ranking of the pooled 24 h model rescaled to its own event rate (proportional hazards); the (4 h, 24 h]
# segment uses the whole profile.
BURST = (r"^ce__n_(1h|6h)$", r"^ce__n_1h_per_access$", r"^ce__accel", r"^ce__storm_min_1d$", r"^ce__h_since_last$")
SHARE = ("4h",)
PROFILE = dict(include=COMPACT, params=SMALL_DATA_PARAMS, share=SHARE, interval_cols={"1h": BURST})   # UCERiskPredictor(**bd.PROFILE)


def derived(S: pd.DataFrame) -> pd.DataFrame:
    """exposure-normalised CE intensity: a CE is only detected when the memory is accessed, so the same CE count
    during light memory activity means a higher underlying error rate."""
    add = {}
    for w in ("1h", "1d"):
        n, a = f"ce__n_{w}", f"te__dram_active_mean{w}"
        if n in S and a in S:
            act = S[a].to_numpy(np.float64)
            add[f"ce__n_{w}_per_access"] = np.where(np.isnan(act), np.nan,
                                                    S[n].to_numpy(np.float64) / np.maximum(act, 0.05)).astype(np.float32)
    return pd.concat([S, pd.DataFrame(add, index=S.index)], axis=1) if add else S

DIMS = ["hostname", "ipv4", "ip", "ipv6", "gpu", "nvml_index", "rack", "facade_rack", "scale_up_domain", "tray_index",
        "chassis_slot_number", "chassis_serial_number", "model_name", "pci_bus_id", "device", "infiniband_guid", "psm",
        "package_name"]
_TAG_KEYS = {"gpu": "gpu", "uuid": "uuid", "pcibusid": "pci_bus_id", "device": "device", "modelname": "model_name",
             "hostname": "hostname_tag", "dcgmfidevnvmlindex": "nvml_index",
             "dcgmfidevplatformchassisserialnumber": "chassis_serial_number",
             "dcgmfidevplatformchassisslotnumber": "chassis_slot_number",
             "dcgmfidevplatforminfinibandguid": "infiniband_guid", "dcgmfidevplatformtrayindex": "tray_index",
             "rack": "rack", "facaderack": "facade_rack", "scaleupdomain": "scale_up_domain", "ip": "ip", "psm": "psm"}


def _norm(x) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


_METRIC_ALIAS = {}
for _m, _r in CATALOG.iterrows():
    _METRIC_ALIAS[_norm(_m)] = _m
    _METRIC_ALIAS.setdefault(_norm(_r.snake), _m)


def canonical_metric(name) -> str:
    """collector or downstream (snake_case) metric name -> collector name; unknown names are returned unchanged."""
    return _METRIC_ALIAS.get(_norm(name), str(name))


def metric_info(name) -> dict:
    m = canonical_metric(name)
    if m in CATALOG.index:
        r = CATALOG.loc[m]
        return {"metric": m, "kind": r.kind, "group": r.group, "zh": r.zh}
    return {"metric": m, "kind": "gauge", "group": "other", "zh": ""}


# ------------------------------------------------------------------ reading
def read_any(x) -> pd.DataFrame | None:
    """DataFrame | CSV / Excel / Parquet / JSON-lines file | directory of such files -> DataFrame."""
    if x is None or isinstance(x, pd.DataFrame):
        return x
    p = Path(x)
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in (".parquet", ".csv", ".xlsx", ".jsonl"))
        return pd.concat([read_any(f) for f in files], ignore_index=True) if files else None
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if suf in (".jsonl", ".json"):
        return pd.read_json(p, lines=True)
    return pd.read_csv(p, low_memory=False)


def iter_metric_chunks(metrics):
    """yields the wide metric table in chronological chunks (one per date partition / file)."""
    if metrics is None:
        return
    if isinstance(metrics, pd.DataFrame):
        yield metrics
        return
    p = Path(metrics)
    if not p.is_dir():
        yield read_any(p)
        return
    files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in (".parquet", ".csv"))
    by_part = {}
    for f in files:
        by_part.setdefault(f.parent, []).append(f)
    for part in sorted(by_part, key=lambda d: str(d)):
        yield pd.concat([read_any(f) for f in by_part[part]], ignore_index=True)


def _uuid_col(cols):
    for c in cols:
        if _norm(c) in ("uuid", "gpuuuid", "deviceuuid"):
            return c
    return None


def _time_col(cols):
    for want in ("timestamp", "ts", "time", "minute", "collecttime"):
        for c in cols:
            if _norm(c) == want:
                return c
    return None


# ------------------------------------------------------------------ telemetry (narrow) table
def _tags(s: pd.Series) -> pd.DataFrame:
    """tag column (JSON list of {key, value}) -> one column per known key."""
    rows = []
    for v in s.fillna("[]").astype(str):
        d = {}
        try:
            for kv in json.loads(v):
                k = _TAG_KEYS.get(_norm(kv.get("key")))
                if k:
                    d[k] = kv.get("value")
        except (ValueError, TypeError, AttributeError):
            pass
        rows.append(d)
    return pd.DataFrame(rows, index=s.index)


def inventory(telemetry: pd.DataFrame) -> pd.DataFrame:
    """device inventory from the telemetry table: one row per uuid with its latest dimensions (columns first, the
    tag JSON fills what is missing). 'host' is the grouping key: hostname, else ipv4 / ip."""
    if telemetry is None or not len(telemetry):
        return pd.DataFrame(columns=["uuid", "host"])
    t = telemetry
    uc, tc = _uuid_col(t.columns), _time_col(t.columns)
    lower = {_norm(c): c for c in t.columns}
    out = pd.DataFrame(index=t.index)
    for d in DIMS:
        if _norm(d) in lower:
            out[d] = t[lower[_norm(d)]]
    tagc = next((c for c in t.columns if _norm(c) in ("tag", "tags")), None)
    if tagc is not None:
        tg = _tags(t[tagc])
        for c in tg.columns:
            if c == "uuid" and uc is None:
                out["uuid"] = tg[c]
            elif c not in out or out[c].isna().all():
                out[c] = tg[c]
            elif c in out:
                out[c] = out[c].where(out[c].notna(), tg[c])
    if uc is not None:
        out["uuid"] = t[uc]
    if "uuid" not in out:
        raise ValueError("telemetry: no uuid column or tag")
    out = out.mask(out.isin(["NULL", "", "None"]))
    if tc is not None:
        out["_t"] = _to_seconds(t[tc])
        out = out.sort_values("_t")
    out = out[out.uuid.notna()]
    inv = out.groupby("uuid", sort=False).last().reset_index()      # latest non-null value of every dimension
    host = None
    for c in ("hostname", "hostname_tag", "ipv4", "ip"):
        if c in inv:
            host = inv[c] if host is None else host.where(host.notna(), inv[c])
    inv["host"] = host if host is not None else "uuid:" + inv.uuid.astype(str)
    inv["host"] = inv.host.where(inv.host.notna(), "uuid:" + inv.uuid.astype(str)).astype(str)
    if "_t" in inv:
        inv = inv.rename(columns={"_t": "last_seen"})
    return inv


def metric_table(telemetry: pd.DataFrame, freq: int = MIN, inv: pd.DataFrame | None = None) -> pd.DataFrame:
    """the third table: narrow telemetry (metric / value rows) -> wide table, one row per (uuid, minute).

    Each cell is the last value of the minute; for the driver error code the last non-zero value. Rows without a uuid
    (host-level metrics such as cpu_power_usage_watts) are copied to every card of the host. Output columns:
    timestamp (start of the minute, Unix seconds UTC), uuid, one column per metric, date (yyyymmdd), hour (HH)."""
    t = telemetry
    mc = next((c for c in t.columns if _norm(c) == "metric"), None)
    vc = next((c for c in t.columns if _norm(c) == "value"), None)
    tc, uc = _time_col(t.columns), _uuid_col(t.columns)
    if mc is None or vc is None or tc is None:
        raise ValueError("telemetry needs metric, value and timestamp columns")
    x = pd.DataFrame({"t": _to_seconds(t[tc]), "metric": t[mc].map(canonical_metric),
                      "value": pd.to_numeric(t[vc], errors="coerce")})
    x["uuid"] = t[uc].replace({"": np.nan, "NULL": np.nan}) if uc is not None else np.nan
    if x.uuid.isna().any():                                     # host-level rows -> every card of the host
        inv = inventory(t) if inv is None else inv
        hk = next((c for c in ("hostname", "ipv4", "ip") if c in t.columns), None)
        if hk is not None and len(inv):
            hrow = x[x.uuid.isna()].assign(host=t.loc[x.uuid.isna(), hk].astype(str).to_numpy())
            key = "host" if hk == "hostname" else hk
            m = inv[["uuid", key]].rename(columns={key: "host"}).astype({"host": str})
            hrow = hrow.drop(columns="uuid").merge(m, on="host").drop(columns="host")
            x = pd.concat([x[x.uuid.notna()], hrow], ignore_index=True)
        else:
            x = x[x.uuid.notna()]
    x["timestamp"] = x.t // freq * freq
    x = x.sort_values("t", kind="stable")
    isx = x.metric == XID
    out = x[~isx].groupby(["uuid", "timestamp", "metric"], sort=False).value.last().unstack("metric")
    if isx.any():
        xx = x[isx]
        nz = xx[xx.value.fillna(0) != 0].groupby(["uuid", "timestamp"]).value.last()
        anyx = xx.groupby(["uuid", "timestamp"]).value.size()
        xv = pd.Series(0.0, index=anyx.index)
        xv.loc[nz.index] = nz
        out = out.join(xv.rename(XID), how="outer") if len(out.columns) else xv.rename(XID).to_frame()
    out = out.reset_index().sort_values(["timestamp", "uuid"]).reset_index(drop=True)
    order = [m for m in CATALOG.index if m in out.columns] + sorted(c for c in out.columns
                                                                   if c not in CATALOG.index and c not in ("uuid", "timestamp"))
    out = out[["timestamp", "uuid"] + order]
    ts = pd.to_datetime(out.timestamp, unit="s", utc=True)
    out["date"] = ts.dt.strftime("%Y%m%d")
    out["hour"] = ts.dt.strftime("%H")
    return out


def write_metric_table(df: pd.DataFrame, out_dir, part: int = 0) -> list[Path]:
    """write the wide table as date=YYYYMMDD/part-<part>.parquet partitions (the layout ``from_bd`` streams)."""
    out = []
    for d, g in df.groupby("date"):
        p = Path(out_dir) / f"date={d}"
        p.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["date"]).to_parquet(p / f"part-{part}.parquet", index=False)
        out.append(p)
    return out


def build_metric_table(telemetry, out_dir, freq: int = MIN, verbose: bool = True) -> dict:
    """narrow telemetry -> wide 1-minute table on disk, one input file at a time (for tables that do not fit in
    memory: export the collector table partitioned by date / hour, one file per partition). A minute must not be split
    across input files. Returns row counts."""
    src = Path(telemetry) if not isinstance(telemetry, pd.DataFrame) else None
    files = [telemetry] if src is None else (sorted(f for f in src.rglob("*") if f.is_file() and f.suffix.lower() in
                                                    (".parquet", ".csv", ".xlsx")) if src.is_dir() else [src])
    n_in = n_out = 0
    for i, f in enumerate(files):
        t = read_any(f)
        if t is None or not len(t):
            continue
        w = metric_table(t, freq)
        write_metric_table(w, out_dir, part=i)
        n_in, n_out = n_in + len(t), n_out + len(w)
        if verbose:
            print(f"[bd-table] {f if src is not None else 'DataFrame'}: {len(t):,} rows -> {len(w):,} card-minutes",
                  flush=True)
    return {"input_rows": n_in, "card_minutes": n_out, "files": len(files)}


# ------------------------------------------------------------------ work orders
def _jload(v):
    try:
        return json.loads(v) if isinstance(v, str) and v.strip().startswith(("{", "[")) else {}
    except ValueError:
        return {}


def xid_cause(code: int) -> str:
    return XID_CAUSE.get(int(code), "UNKNOWN_ERROR")


def _sub_code(sub: str):
    m = re.fullmatch(r"(?i)xid[_ -]?(\d+)", str(sub).strip())
    return int(m.group(1)) if m else None


def work_orders(df: pd.DataFrame, uce_codes=UCE_CODES) -> pd.DataFrame:
    """SEED work orders -> one row per event: ip, t (fault time), sub (fault_sub_type), cause, role ('uce' | 'event'),
    fault_type, operation, level, event_id. Repeated emissions of one event_id are merged (earliest fault time)."""
    if df is None or not len(df):
        return pd.DataFrame(columns=["ip", "t", "sub", "cause", "role", "fault_type", "operation", "level", "event_id"])
    w = df.replace({"NULL": np.nan, "null": np.nan, "": np.nan, "None": np.nan}).copy()
    col = {_norm(c): c for c in w.columns}
    get = lambda *names: next((w[col[_norm(n)]] for n in names if _norm(n) in col), None)   # noqa: E731
    ip = get("ipv4")
    ip2 = get("ip")
    ip = ip2 if ip is None else (ip if ip2 is None else ip.where(ip.notna(), ip2))
    t = None
    for n in ("fault_timestamp", "happened_time", "received_time", "timestamp"):
        s = get(n)
        if s is not None:
            v = pd.Series(_to_seconds(pd.to_numeric(s, errors="coerce").fillna(0)), index=w.index).where(lambda z: z > 0)
            t = v if t is None else t.where(t.notna(), v)
    params = get("parameters_json")
    pj = params.map(_jload) if params is not None else pd.Series([{}] * len(w), index=w.index)
    sub = get("fault_sub_type")
    sub = sub if sub is not None else pj.map(lambda d: d.get("fault_sub_type") or d.get("name"))
    out = pd.DataFrame({"ip": ip, "t": t, "sub": sub.astype(str),
                        "fault_type": get("fault_type") if get("fault_type") is not None else pj.map(lambda d: d.get("fault_type")),
                        "operation": pj.map(lambda d: d.get("operation")),
                        "level": pj.map(lambda d: d.get("fault_level")),
                        "event_id": get("event_id") if get("event_id") is not None else np.nan})
    agg = get("is_aggregated")
    ips = get("ips")
    if agg is not None and ips is not None:                     # multi-host events: one row per host
        multi = agg.astype(str).str.lower().isin(["true", "1"])
        extra = []
        for i in np.flatnonzero(multi.to_numpy()):
            lst = _jload(ips.iloc[i])
            for h in (lst if isinstance(lst, list) else []):
                r = out.iloc[i].copy()
                r["ip"] = h
                extra.append(r)
        if extra:
            out = pd.concat([out, pd.DataFrame(extra)], ignore_index=True)
    out = out[out.ip.notna() & out.t.notna()].copy()
    out["t"] = out.t.astype(np.int64)
    code = out["sub"].map(_sub_code)
    ucs = {int(c) for c in uce_codes}
    out["cause"] = [xid_cause(c) if c is not None and c == c else s for c, s in zip(code, out["sub"])]
    out["role"] = np.where(code.isin(ucs) | out["sub"].str.contains(r"(?i)uncorrect|double.?bit|\bdbe\b"), "uce", "event")
    key = out.event_id.where(out.event_id.notna(), out.ip.astype(str) + "|" + out["sub"] + "|" + out.t.astype(str))
    out = out.assign(_k=key).sort_values("t").drop_duplicates(["_k", "ip"]).drop(columns="_k")
    return out.sort_values("t").reset_index(drop=True)


# ------------------------------------------------------------------ deriving streams from the 1-minute table
def _popcount(x: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(x, nan=0).astype(np.int64)
    c = np.zeros(len(v), np.int64)
    while v.any():
        c += v & 1
        v >>= 1
    return np.where(np.isnan(x), np.nan, c).astype(np.float64)


def _xid_persistent(e, t, code, gap: int = 2 * MIN) -> bool:
    """True when the error-code field keeps its last value (a non-zero code is usually followed, at the next sample
    of the same card, by the same code), False when a code is reported only when the error happens."""
    e, t, code = np.asarray(e), np.asarray(t), np.nan_to_num(np.asarray(code, np.float64))
    o = np.lexsort((t, e))
    e, t, code = e[o], t[o], code[o]
    nz = code[:-1] != 0
    if nz.sum() < 5:
        return False
    same = (e[1:] == e[:-1]) & (code[1:] == code[:-1]) & (t[1:] - t[:-1] <= gap)
    return bool(same[nz].mean() > 0.5)


def _xid_events(uuid, t, code, persistent: bool) -> np.ndarray:
    """mask of new errors in a per-card series of error codes (zeros included when present)."""
    o = np.lexsort((t, uuid))
    u, c = np.asarray(uuid)[o], np.nan_to_num(np.asarray(code, np.float64))[o]
    first = np.r_[True, u[1:] != u[:-1]]
    pc = np.r_[np.nan, c[:-1]]
    new = (c != 0) & (first | (pc != c)) if persistent else (c != 0)
    if persistent:
        new &= ~first                  # the first value seen may be stale (an error from before the window)
    m = np.zeros(len(o), bool)
    m[o] = new
    return m


class _Deriver:
    """streams the wide table chunk by chunk (chronological), keeping each card's last row so that deltas, resets
    and edges are exact across chunk borders."""

    PAIRS = {"ce_total": (SBE_VOL, SBE_AGG), "ce_srm": (SBE_VOL_SRM, SBE_AGG_SRM), "ce_dev": (SBE_VOL_DEV, SBE_AGG_DEV),
             "dbe": (DBE_VOL, DBE_AGG), "dbe_dev": (_P + "ECC_DBE_VOL_DEV", _P + "ECC_DBE_AGG_DEV"),
             "dbe_srm": (_P + "ECC_DBE_VOL_SRM", _P + "ECC_DBE_AGG_SRM")}

    def __init__(self, ent_of: dict, uce_codes=UCE_CODES, tel_bin=TEL_BIN):
        self.ent_of, self.uce_codes, self.tel_bin = ent_of, {int(c) for c in uce_codes}, tel_bin
        self.state = None
        self.auto = {}
        self.ce, self.uce, self.ev, self.te = [], [], [], []
        self.metrics_seen, self.unknown_uuids, self.rows = set(), set(), 0
        self.span = {}
        self.xid_persistent = None       # error-code semantics, decided from the data (see _xid_persistent)

    # -- helpers on the chunk (rows sorted by entity, ts; carry rows first in each entity)
    def _arr(self, R, m):
        return R[m].to_numpy(np.float64) if m in R.columns else None

    def feed(self, df: pd.DataFrame):
        if df is None or not len(df):
            return
        tc, uc = _time_col(df.columns), _uuid_col(df.columns)
        if tc is None or uc is None:
            raise ValueError("metric table needs timestamp and uuid columns")
        ren = {c: canonical_metric(c) for c in df.columns if c not in (tc, uc)}
        skip = {_norm(x) for x in DIMS} | {"date", "hour", "tag", "tags", "host", "uuid"}
        mcols = [c for c in df.columns if c not in (tc, uc) and _norm(c) not in skip and
                 (canonical_metric(c) in CATALOG.index or pd.api.types.is_numeric_dtype(df[c]))]
        u = df[uc].astype(str).to_numpy()
        ent = np.fromiter((self.ent_of.get(x, -1) for x in u), np.int64, len(u))
        if (ent < 0).any():
            self.unknown_uuids |= set(u[ent < 0])
        keep = ent >= 0
        R = pd.DataFrame({ren[c]: pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)[keep] for c in mcols})
        R.insert(0, "ts", _to_seconds(df[tc])[keep] // MIN * MIN)
        R.insert(0, "entity", ent[keep])
        R = R.loc[:, ~R.columns.duplicated()]
        self.metrics_seen |= set(R.columns[2:])
        R["_carry"] = False
        if self.state is not None:
            prev = self.state[self.state.index.isin(np.unique(R.entity.to_numpy()))]
            if len(prev):
                R = pd.concat([prev.reset_index().assign(_carry=True), R], ignore_index=True)
        R = R.sort_values(["entity", "ts", "_carry"], ascending=[True, True, False], kind="stable").reset_index(drop=True)
        R = R.drop_duplicates(["entity", "ts"], keep="last") if R.duplicated(["entity", "ts"]).any() else R
        R = R.reset_index(drop=True)
        self._derive(R)

    def _derive(self, R: pd.DataFrame):
        e = R.entity.to_numpy(np.int64)
        ts = R.ts.to_numpy(np.int64)
        carry = R._carry.to_numpy(bool)
        first = np.r_[True, e[1:] != e[:-1]]
        idx = np.flatnonzero(first)
        lens = np.diff(np.r_[idx, len(e)])
        newent = first & ~carry                         # first observation of a card: no previous row

        def prev(x):
            p = np.empty_like(x)
            p[0] = np.nan
            p[1:] = x[:-1]
            p[first] = np.nan
            return p

        def segsum(x):
            cs = np.cumsum(x)
            return cs - np.repeat(cs[idx] - x[idx], lens)

        dt = ts - np.where(first, ts, np.r_[ts[0], ts[:-1]])
        mins = np.maximum(dt / MIN, 1.0)
        live = ~carry
        st = {}                                         # new carry state columns

        # ---- counters: envelope of the volatile (timing) and aggregate (completeness) counters
        inc = {}
        for name, (vm, am) in self.PAIRS.items():
            v, a = self._arr(R, vm), self._arr(R, am)
            if v is None and a is None:
                continue
            X0 = R[f"_X_{name}"].to_numpy(np.float64) if f"_X_{name}" in R else np.zeros(len(R))
            Y0 = R[f"_Y_{name}"].to_numpy(np.float64) if f"_Y_{name}" in R else np.zeros(len(R))
            Z0 = R[f"_Z_{name}"].to_numpy(np.float64) if f"_Z_{name}" in R else np.zeros(len(R))
            if v is not None:
                dv = v - prev(v)
                dv = np.where(dv < 0, v, dv)            # reset (driver reload / card reset): value since reset
                dv = np.nan_to_num(dv, nan=0.0)
            else:
                dv = np.zeros(len(R))
            if a is not None:
                da = np.nan_to_num(np.clip(a - prev(a), 0, None), nan=0.0)
            else:
                da = np.zeros(len(R))
            dv = np.where(carry, np.nan_to_num(X0), np.where(newent, 0.0, dv))
            da = np.where(carry, np.nan_to_num(Y0), np.where(newent, 0.0, da))
            X, Y = segsum(dv), segsum(da)
            m = np.maximum(X, Y)
            m = np.where(carry, np.maximum(m, np.nan_to_num(Z0)), m)
            Z = pd.Series(m).groupby(e).cummax().to_numpy()
            n = Z - prev(Z)
            n = np.where(first, 0.0, np.nan_to_num(n, nan=0.0))
            inc[name] = n
            st[f"_X_{name}"], st[f"_Y_{name}"], st[f"_Z_{name}"] = X, Y, Z
        ce_srm = inc.get("ce_srm", np.zeros(len(R)))
        if "ce_dev" in inc:
            ce_dram = inc["ce_dev"]
        elif "ce_total" in inc:
            ce_dram = np.clip(inc["ce_total"] - ce_srm, 0, None)
        else:
            ce_dram = np.zeros(len(R))
        dbe = inc.get("dbe", inc.get("dbe_dev", np.zeros(len(R))) + inc.get("dbe_srm", np.zeros(len(R))))
        t_ev = ts + MIN - 1                             # a minute's increments are known at its end
        for n, sub in ((ce_dram, 1), (ce_srm, 2)):
            k = live & (n > 0)
            if k.any():
                self.ce.append(pd.DataFrame({"entity": e[k], "t": t_ev[k], "n": n[k], "sub": np.int8(sub)}))

        # ---- per-metric deltas (rates) for the other counters
        delta = {}

        def dlt(m):
            if m in delta:
                return delta[m]
            x = self._arr(R, m)
            if x is None:
                return None
            kind = CATALOG.kind.get(m, "gauge")
            d = x - prev(x)
            if kind == "auto":
                if m not in self.auto:
                    ok = d[np.isfinite(d)]
                    self.auto[m] = bool(len(ok) > 10 and (ok >= 0).mean() > 0.95 and (ok > 0).mean() > 0.3)
                if not self.auto[m]:
                    delta[m] = x                           # per-interval values: used as they are
                    return x
            if kind in ("vcounter", "auto"):
                d = np.where(d < 0, x, d)
            elif kind == "counter":
                d = np.clip(d, 0, None)
            delta[m] = d
            return d

        # ---- events
        evs = []

        def add(mask, code, tt=t_ev):
            k = live & mask
            if k.any():
                evs.append(pd.DataFrame({"entity": e[k], "t": tt[k], "code": code}))

        x = self._arr(R, XID)
        if x is not None:
            xn = np.nan_to_num(x)
            px = prev(x)
            if self.xid_persistent is None and int(((xn != 0) & live).sum()) >= 5:
                self.xid_persistent = _xid_persistent(e, ts, xn)
            if self.xid_persistent:      # the field keeps the last code: only a change of value is a new error
                rise = (xn != 0) & (np.where(np.isnan(px), xn, px) != xn) & ~newent
            else:                        # the code is reported only in the minute of the error
                rise = (xn != 0) & ((np.nan_to_num(px) != xn) | (dt > MIN)) & ~newent
            for c in np.unique(x[rise & live]).astype(int):
                mk = rise & (x == c)
                if c in self.uce_codes:
                    self._uce(e[mk & live], t_ev[mk & live], "code")
                else:
                    add(mk, xid_cause(c))
        uce_remap = self._arr(R, _P + "UNCORRECTABLE_REMAPPED_ROWS")
        k = live & (dbe > 0)
        if k.any():
            self._uce(e[k], t_ev[k], "counter")
        if uce_remap is not None:
            d = np.clip(np.nan_to_num(uce_remap - prev(uce_remap)), 0, None)
            k = live & (d > 0) & ~first
            if k.any():
                self._uce(e[k], t_ev[k], "remap")
        for code, m in _EDGE_RISE:
            v = self._arr(R, m)
            if v is not None:
                add((np.nan_to_num(v) > 0) & (np.nan_to_num(prev(v)) <= 0) & ~newent, code)
        for code, m in _EDGE_FALL:
            v = self._arr(R, m)
            if v is not None:
                add((np.nan_to_num(v, nan=1) <= 0) & (np.nan_to_num(prev(v), nan=0) > 0), code)
        for code, ms in _INCR:
            ds = [dlt(m) for m in ms if m in R.columns]
            if ds:
                add(np.nansum(np.stack(ds), 0) > 0, code)
        cnt = self._arr(R, _P + "COUNT")
        if cnt is not None:
            add(cnt < prev(cnt), "DEVICE_MISSING")
        p2p = self._arr(R, _P + "P2P_NVLINK_STATUS")
        if p2p is not None:
            pc = _popcount(p2p)
            add(pc < prev(pc), "P2P_LINK_DROP")
        gen_d = self._diff(R, [_P + "PCIE_MAX_LINK_GEN", _P + "PCIE_LINK_GEN"])
        wid_d = self._diff(R, [_P + "PCIE_MAX_LINK_WIDTH", _P + "PCIE_LINK_WIDTH"])
        if gen_d is not None or wid_d is not None:
            deg = (np.nan_to_num(gen_d) > 0 if gen_d is not None else False) | (np.nan_to_num(wid_d) > 0 if wid_d is not None else False)
            deg = np.asarray(deg, np.float64)
            add((deg > 0) & (prev(deg) == 0), "PCIE_DEGRADED")
        clk = self._arr(R, _P + "CLOCKS_EVENT_REASONS")
        if clk is not None:
            ci = np.nan_to_num(clk).astype(np.int64)
            pci = np.nan_to_num(prev(clk)).astype(np.int64)
            for code, mask in (("THERMAL_THROTTLE", 0x20 | 0x40), ("HW_SLOWDOWN", 0x8 | 0x80)):
                add(((ci & mask) > 0) & ((pci & mask) == 0) & ~newent, code)
        resets = np.zeros(len(R), bool)
        for m in (SBE_VOL, DBE_VOL, _P + "TOTAL_ENERGY_CONSUMPTION"):
            v = self._arr(R, m)
            if v is not None:
                resets |= np.nan_to_num(v - prev(v)) < 0
        add(resets, "DRIVER_RESET", ts)
        add(dt >= GAP_EVENT, "COLLECTOR_GAP", ts)
        if evs:
            self.ev.append(pd.concat(evs, ignore_index=True))

        # ---- gauges, aggregated to tel_bin bins
        G = {}
        for name, how, args, _ in TE_SPEC:
            v = self._gauge(R, how, args, dlt, mins, prev)
            if v is not None:
                G[name] = v
        for m in [c for c in R.columns[2:] if c not in CATALOG.index and not c.startswith("_")][:30]:
            G["x_" + re.sub(r"\W+", "_", str(m).lower())] = R[m].to_numpy(np.float64)   # metrics outside the catalog
        if G:
            T = pd.DataFrame(G)[live]
            T.insert(0, "bin", ts[live] // self.tel_bin)
            T.insert(0, "entity", e[live])
            g = T.groupby(["entity", "bin"], sort=False)
            B = g.mean()
            B["present"] = g.size() / (self.tel_bin / MIN)
            B = B.reset_index()
            B.insert(1, "t", B.pop("bin") * self.tel_bin + self.tel_bin - 1)
            self.te.append(B.astype({c: np.float32 for c in B.columns[2:]}))

        # ---- spans and new carry state: last row of every entity
        last = np.r_[idx[1:] - 1, len(e) - 1]
        for ee, a, b in zip(e[idx], ts[idx + (carry[idx]).astype(int).clip(max=lens - 1)], ts[last]):
            s0 = self.span.get(int(ee), (int(a), int(b)))
            self.span[int(ee)] = (min(s0[0], int(a)), max(s0[1], int(b)))
        S = R.iloc[last].drop(columns=["_carry"] + [c for c in R.columns if c.startswith(("_X_", "_Y_", "_Z_"))])
        for k, v in st.items():
            S[k] = v[last]
        self.state = S.set_index("entity") if self.state is None else \
            pd.concat([self.state[~self.state.index.isin(S.entity)], S.set_index("entity")])
        self.rows += int(live.sum())

    @staticmethod
    def _pick(R, a):
        if isinstance(a, tuple):
            return next((x for x in a if x in R.columns and R[x].notna().any()), None)
        return a if a in R.columns else None

    def _diff(self, R, args):
        a, b = self._pick(R, args[0]), self._pick(R, args[1])
        if a is None or b is None:
            return None
        return R[a].to_numpy(np.float64) - R[b].to_numpy(np.float64)

    def _gauge(self, R, how, args, dlt, mins, prev):
        if how == "col":
            m = self._pick(R, args[0])
            return None if m is None else R[m].to_numpy(np.float64)
        if how in ("diff", "absdiff"):
            d = self._diff(R, args)
            return None if d is None else (np.abs(d) if how == "absdiff" else d)
        if how == "ratio":
            a, b = self._pick(R, args[0]), self._pick(R, args[1])
            if a is None or b is None:
                return None
            den = R[b].to_numpy(np.float64)
            return np.where(den > 0, R[a].to_numpy(np.float64) / np.where(den > 0, den, 1), np.nan)
        if how == "bits":
            m = self._pick(R, args[0])
            if m is None:
                return None
            x = R[m].to_numpy(np.float64)
            return np.where(np.isnan(x), np.nan, ((np.nan_to_num(x).astype(np.int64) & args[1]) > 0).astype(np.float64))
        if how == "popcount":
            m = self._pick(R, args[0])
            return None if m is None else _popcount(R[m].to_numpy(np.float64))
        if how in ("rate", "lograte", "auto"):
            ds = [dlt(m) for m in args if m in R.columns]
            if not ds:
                return None
            d = np.nansum(np.stack(ds), 0)
            d = np.where(np.all(np.isnan(np.stack(ds)), 0), np.nan, d)
            if how == "auto" and not self.auto.get(args[0], True):
                return d                                    # per-interval value
            r = d / mins
            return np.log1p(np.clip(r, 0, None)) if how == "lograte" else r
        if how == "rate_w":
            d = dlt(args[0]) if args[0] in R.columns else None
            if d is None:
                return None
            x = R[args[0]].to_numpy(np.float64)
            ok = (x - prev(x)) >= 0
            return np.where(ok, d / (mins * MIN) / 1000.0, np.nan)
        if how == "banks_used":
            cols = [m for m in args if m in R.columns]
            if args[0] not in cols or len(cols) < 2:
                return None
            tot = np.nansum(np.stack([R[m].to_numpy(np.float64) for m in cols]), 0)
            return tot - R[args[0]].to_numpy(np.float64)
        return None

    def _uce(self, e, t, src):
        if len(e):
            self.uce.append(pd.DataFrame({"entity": e, "t": t, "src": src}))

    def result(self):
        cat = lambda xs, cols: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame(columns=cols)   # noqa: E731
        te = cat(self.te, ["entity", "t"])
        if len(te) and te.duplicated(["entity", "t"]).any():      # a bin split by a chunk border: weighted merge
            keys = ["entity", "t"]
            cols = [c for c in te.columns if c not in keys + ["present"]]
            w = te["present"].astype(np.float64)
            x = te[cols].astype(np.float64)
            g = te[keys]
            num = x.mul(w, axis=0).assign(**{k: g[k] for k in keys}).groupby(keys).sum(min_count=1)
            den = x.notna().mul(w, axis=0).assign(**{k: g[k] for k in keys}).groupby(keys).sum()
            m = (num / den.where(den > 0)).astype(np.float32)
            m["present"] = w.groupby([g[k] for k in keys]).sum().astype(np.float32)
            te = m.reset_index()[te.columns]
        return (cat(self.ce, ["entity", "t", "n", "sub"]), cat(self.uce, ["entity", "t", "src"]),
                cat(self.ev, ["entity", "t", "code"]), te)


def _merge_close(df: pd.DataFrame, keys: list[str], gap: int) -> pd.DataFrame:
    """drop records within `gap` seconds of the previous kept record of the same keys (sorted by time)."""
    if not len(df):
        return df
    df = df.sort_values(keys + ["t"], kind="stable").reset_index(drop=True)
    same = np.ones(len(df), bool)
    for k in keys:
        v = df[k].to_numpy()
        same[1:] &= v[1:] == v[:-1]
    same[0] = False
    t = df.t.to_numpy(np.int64)
    keep = ~(same & (np.r_[np.inf, np.diff(t)] <= gap))
    return df[keep].reset_index(drop=True)


# ------------------------------------------------------------------ the adapter
def from_bd(work_orders_table=None, telemetry=None, metrics=None, *, uce_codes=UCE_CODES, grid: int = HOUR,
            tel_bin: int = TEL_BIN, warmup_days: int = 3, label_complete: bool = True, keep_codes=None,
            background_codes=None, min_code_rows: int = 3, max_codes: int = 40, attrib_before: int = 30 * MIN,
            attrib_after: int = 2 * HOUR, name: str = "bd", verbose: bool = False, **options) -> UCEData:
    """work orders + telemetry + 1-minute metric table -> UCEData (see the module docstring).

    Any of the three inputs may be missing, but labels need the metric table (or work orders whose host has
    telemetry). metrics=None builds the metric table from the telemetry rows. grid: regular prediction times
    (seconds; every card with data in the previous period is scored); label_complete: drop samples whose 24 h label
    window runs past the data. keep_codes / background_codes: the event-code set of a trained model
    (``UCERiskPredictor.input_spec``) so that inference builds the same columns."""
    log = (lambda m: print(f"[bd] {m}", flush=True)) if verbose else (lambda m: None)
    report = {"inputs": {}, "labels": {}, "modules": []}
    tel = read_any(telemetry)
    wo = work_orders(read_any(work_orders_table), uce_codes)
    inv = inventory(tel) if tel is not None else pd.DataFrame(columns=["uuid", "host"])
    report["inputs"]["work_orders"] = int(len(wo))
    report["inputs"]["telemetry_rows"] = 0 if tel is None else int(len(tel))
    narrow_xid = None
    if tel is not None and any(_norm(c) == "metric" for c in tel.columns):
        mc = next(c for c in tel.columns if _norm(c) == "metric")
        vc = next(c for c in tel.columns if _norm(c) == "value")
        isx = tel[mc].map(canonical_metric) == XID
        if isx.any() and _uuid_col(tel.columns) is not None and _time_col(tel.columns) is not None:
            xx = tel[isx]
            narrow_xid = pd.DataFrame({"uuid": xx[_uuid_col(xx.columns)].astype(str).to_numpy(),
                                       "t": _to_seconds(xx[_time_col(xx.columns)]),
                                       "code": pd.to_numeric(xx[vc], errors="coerce").fillna(0).astype(int).to_numpy()})
        if metrics is None:
            metrics = metric_table(tel, inv=inv)
            log(f"metric table built from telemetry: {len(metrics):,} rows")

    # ---- pass 1: card registry (cards of one host get contiguous ids)
    uu = set(inv.uuid.astype(str))
    nrow = 0
    if isinstance(metrics, (str, Path)) and Path(metrics).is_dir():
        import pyarrow.parquet as pq
        for f in sorted(Path(metrics).rglob("*.parquet")):
            cols = pq.read_schema(f).names
            uc = _uuid_col(cols)
            tb = pq.read_table(f, columns=[uc])
            nrow += tb.num_rows
            uu |= set(tb.column(0).unique().to_pylist())
    else:
        for ch in iter_metric_chunks(metrics):
            uu |= set(ch[_uuid_col(ch.columns)].astype(str).unique())
            nrow += len(ch)
    host_of = dict(zip(inv.uuid.astype(str), inv.host.astype(str)))
    reg = pd.DataFrame({"device": sorted(uu)})
    reg["host"] = reg.device.map(host_of).fillna("uuid:" + reg.device)
    reg = reg.sort_values(["host", "device"]).reset_index(drop=True)
    reg["entity"] = np.arange(len(reg))
    reg["group"] = pd.factorize(reg.host)[0]
    ent_of = dict(zip(reg.device, reg.entity))
    report["inputs"].update(metric_rows=nrow, cards=len(reg), hosts=int(reg.host.nunique()),
                            cards_without_host=int(reg.host.str.startswith("uuid:").sum()))
    log(f"{len(reg)} cards on {reg.host.nunique()} hosts, {nrow:,} metric rows")

    # ---- pass 2: streams
    dv = _Deriver(ent_of, uce_codes, tel_bin)
    for i, ch in enumerate(iter_metric_chunks(metrics)):
        dv.feed(ch)
        if verbose and i % 7 == 6:
            log(f"  {i + 1} chunks, {dv.rows:,} rows")
    ce, uce, ev, te = dv.result()
    present = [m for m in dv.metrics_seen if m in CATALOG.index]
    report["inputs"]["metrics_found"] = len(present)
    report["inputs"]["pdf1_metrics_missing"] = sorted(set(PDF1_METRICS) - set(present))
    report["inputs"]["metrics_unknown"] = sorted(m for m in dv.metrics_seen if m not in CATALOG.index)
    report["inputs"]["auto_counters"] = {k: ("cumulative" if v else "per-interval") for k, v in dv.auto.items()}
    pers = dv.xid_persistent
    if narrow_xid is not None and len(narrow_xid):            # exact-time driver error codes from the narrow table
        nx = narrow_xid[narrow_xid.uuid.isin(ent_of)].reset_index(drop=True)
        uid = pd.factorize(nx.uuid)[0]
        if pers is None:
            pers = _xid_persistent(uid, nx.t.to_numpy(), nx.code.to_numpy())
        nx = nx[_xid_events(uid, nx.t.to_numpy(), nx.code.to_numpy(), pers)]
        ent = nx.uuid.map(ent_of).to_numpy(np.int64)
        isu = nx.code.isin(list(uce_codes)).to_numpy()
        if isu.any():
            dv._uce(ent[isu], nx.t.to_numpy(np.int64)[isu], "code")
            uce = pd.concat([uce, dv.uce[-1]], ignore_index=True)
        if (~isu).any():
            ev = pd.concat([ev, pd.DataFrame({"entity": ent[~isu], "t": nx.t.to_numpy(np.int64)[~isu],
                                              "code": [xid_cause(c) for c in nx.code.to_numpy()[~isu]]})],
                           ignore_index=True)

    if pers is not None:
        report["inputs"]["error_code_semantics"] = ("last value kept: a change of value is a new error" if pers else
                                                   "reported in the minute of the error")
    # ---- labels: card evidence + work orders
    uce = uce.sort_values(["entity", "t"]).reset_index(drop=True)
    src_counts = uce.groupby("src").size().to_dict() if len(uce) else {}
    uce_m = _merge_close(uce[["entity", "t", "src"]], ["entity"], 10 * MIN)
    ip_cols = [c for c in ("ipv4", "ip") if c in inv]
    host_of_ip = {}
    for c in ip_cols:
        host_of_ip.update({str(k): str(v) for k, v in zip(inv[c], inv.host) if pd.notna(k)})
    ents_of_host = reg.groupby("host").entity.apply(list).to_dict()
    wo["host"] = wo.ip.astype(str).map(host_of_ip)
    lab = {"card_evidence": int(len(uce_m)), "card_evidence_by_source": src_counts, "work_orders_uce": 0,
           "wo_confirmed": 0, "wo_unattributed": 0, "wo_no_host": 0}
    wo_ev = []
    ut = uce_m.t.to_numpy(np.int64).copy()
    ue = uce_m.entity.to_numpy(np.int64)
    for r in wo.itertuples():
        if not isinstance(r.host, str):
            lab["wo_no_host"] += r.role == "uce"
            continue
        ents = ents_of_host.get(r.host, [])
        if r.role == "uce":
            lab["work_orders_uce"] += 1
            m = np.isin(ue, ents) & (ut >= r.t - attrib_before) & (ut <= r.t + attrib_after)
            if m.any():
                lab["wo_confirmed"] += 1
                j = np.flatnonzero(m)
                ut[j] = np.minimum(ut[j], np.maximum(r.t, ut[j] - attrib_after))   # the work order times the fault
            else:
                lab["wo_unattributed"] += 1
                wo_ev += [(x, r.t, "WO_UCE_UNATTRIBUTED") for x in ents]
        else:
            wo_ev += [(x, r.t, "WO_" + re.sub(r"\W+", "_", str(r.cause))) for x in ents]
    uce_m = uce_m.assign(t=ut).sort_values(["entity", "t"]).reset_index(drop=True)
    lab["uce_records"] = int(len(uce_m))
    lab["cards_with_uce"] = int(uce_m.entity.nunique())
    report["labels"] = lab
    if wo_ev:
        ev = pd.concat([ev, pd.DataFrame(wo_ev, columns=["entity", "t", "code"])], ignore_index=True)
    ev = _merge_close(ev, ["entity", "code"], 2 * MIN) if len(ev) else ev

    # ---- event codes: background causes get the 'bg_' prefix; rare codes -> 'other'
    event_codes = []
    if len(ev):
        bg = set(BACKGROUND_CAUSES) | {"WO_" + c for c in BACKGROUND_CAUSES} | set(background_codes or [])
        cnt = ev.code.value_counts()
        if keep_codes is not None:
            keep = [c[3:] if c.startswith("bg_") else c for c in keep_codes]
        else:
            keep = list(ev.groupby("code").entity.nunique().loc[lambda x: cnt.reindex(x.index) >= min_code_rows]
                        .sort_values(ascending=False).index[:max_codes])
        ev["code"] = np.where(ev.code.isin(keep), ev.code, "other")
        ev["code"] = np.where(ev.code.isin(bg), "bg_" + ev.code, ev.code)
        event_codes = [("bg_" + c if c in bg else c) for c in keep] + (["other"] if (ev.code == "other").any() else [])
        report["modules"].append("events: driver error causes, metric state changes, work orders (ev__)")
    report["events"] = ev.code.value_counts().to_dict() if len(ev) else {}

    # ---- entities, telemetry, time range
    ent = reg.set_index("entity")[["group", "host", "device"]].copy()
    dims = inv.set_index(inv.uuid.astype(str))
    num_cols = []
    for c in ("gpu", "tray_index"):
        if c in dims:
            ent[c] = pd.to_numeric(ent.device.map(dims[c]), errors="coerce")
            if ent[c].notna().any():
                num_cols.append(c)
    for c in ("ipv4", "ip", "hostname", "rack", "scale_up_domain", "chassis_slot_number"):
        if c in dims:
            ent[c] = ent.device.map(dims[c])
    if len(te):
        te = te.sort_values(["entity", "t"]).reset_index(drop=True)
        const = [c for c in te.columns[2:] if te[c].nunique(dropna=True) <= 1]
        te = te.drop(columns=const)
        report["inputs"]["gauges_constant_dropped"] = const
        report["modules"].append(f"telemetry gauges, {te.shape[1] - 2} series in {tel_bin // MIN}-minute bins (te__)")
    if len(ce):
        report["modules"].insert(0, "CE from single-bit counters: device memory (sub 1) and SRAM (sub 2) (ce__)")
        ce = ce.astype({"n": np.float64}).sort_values(["entity", "t"]).reset_index(drop=True)
    tt = [x.t.to_numpy(np.int64) for x in (ce, uce_m, ev, te) if len(x)]
    if not tt:
        raise ValueError("no usable records in the inputs")
    t0 = int(min(a.min() for a in tt))
    t1 = int(max(sp[1] for sp in dv.span.values())) + MIN if dv.span else int(max(a.max() for a in tt))
    options.setdefault("t_start", t0 // DAY * DAY + warmup_days * DAY)
    options.setdefault("t_end", t1 if label_complete else t1 + DAY)
    report["modules"] += ["UCE history (hi__)", "same-host context (nb*__)"]
    report["time_range_utc"] = [str(pd.Timestamp(t0, unit="s")), str(pd.Timestamp(t1, unit="s"))]
    d = UCEData(name=name, entities=ent, uce=uce_m[["entity", "t"]].copy(), ce=ce if len(ce) else None,
                events=ev if len(ev) else None, telemetry=te if len(te) else None, num_cols=num_cols,
                event_codes=event_codes, anchor_codes=[c for c in event_codes if not c.startswith("bg_")] or None,
                grid=grid, post=derived, **options)
    d.report = report | {"fields": {"event_codes": event_codes,
                                    "background_codes": [c[3:] for c in event_codes if c.startswith("bg_")]}}
    d.bd = {"inventory": inv, "work_orders": wo, "span": pd.DataFrame(
        [(k, a, b) for k, (a, b) in dv.span.items()], columns=["entity", "first", "last"]).set_index("entity"),
        "uce_sources": uce}
    return d


def metric_usage(metrics=None) -> pd.DataFrame:
    """how every metric of the catalog (default: the 30-machine list) enters the model."""
    use = {}

    def add(m, txt):
        use.setdefault(m, []).append(txt)
    for name, (vm, am) in _Deriver.PAIRS.items():
        lab = {"ce_total": "CE 流（设备内存 = 总数 - SRAM）", "ce_srm": "CE 流（SRAM，sub 2）", "ce_dev": "CE 流（设备内存）",
               "dbe": "UCE 标签", "dbe_dev": "UCE 标签", "dbe_srm": "UCE 标签"}[name]
        add(vm, lab + "：易失计数给时间")
        add(am, lab + "：持久计数补齐缺口/复位")
    add(XID, "UCE 标签（48/94/95）+ 事件（其他错误码 -> 错误原因）")
    add(_P + "UNCORRECTABLE_REMAPPED_ROWS", "UCE 标签（不可纠正重映射）")
    for code, m in _EDGE_RISE + _EDGE_FALL:
        add(m, f"事件 {code}")
    for code, ms in _INCR:
        for m in ms:
            add(m, f"事件 {code}")
    add(_P + "COUNT", "事件 DEVICE_MISSING")
    add(_P + "P2P_NVLINK_STATUS", "事件 P2P_LINK_DROP")
    for m in (_P + "PCIE_MAX_LINK_GEN", _P + "PCIE_LINK_GEN", _P + "PCIE_MAX_LINK_WIDTH", _P + "PCIE_LINK_WIDTH"):
        add(m, "事件 PCIE_DEGRADED")
    add(_P + "CLOCKS_EVENT_REASONS", "事件 THERMAL_THROTTLE / HW_SLOWDOWN")
    for m in (SBE_VOL, DBE_VOL, _P + "TOTAL_ENERGY_CONSUMPTION"):
        add(m, "事件 DRIVER_RESET")
    for name, how, args, zh in TE_SPEC:
        for a in args:
            for m in (a if isinstance(a, tuple) else (a,)):
                if isinstance(m, str):
                    add(m, f"遥测特征 te__{name}（{zh}）")
    redundant = {_P + "FB_FREE": "冗余：已由 te__fb_used_frac（已用比例）覆盖", _P + "FB_USED": "冗余：已由 te__fb_used_frac 覆盖",
                 _P + "FB_TOTAL": "冗余：已由 te__fb_used_frac 覆盖", _P + "BAR1_FREE": "冗余：已由 te__bar1_used_frac 覆盖",
                 _P + "SHUTDOWN_TEMP": "静态阈值：温度余量用降频阈值（SLOWDOWN_TEMP）",
                 _P + "C2C_MAX_BANDWIDTH": "静态配置：C2C 实际带宽用 sdnc_nvlink_c2c_*",
                 "sndc_host_basic_info": "上报标记（恒为 1）：采集连续性由分钟数据完整率 te__present 和 COLLECTOR_GAP 体现"}
    for m, txt in redundant.items():
        use.setdefault(m, [txt])
    rows = []
    for m in (metrics or PDF1_METRICS):
        r = CATALOG.loc[m] if m in CATALOG.index else None
        rows.append(dict(metric=m, snake=None if r is None else r.snake, kind=None if r is None else r.kind,
                         group=None if r is None else r.group, zh=None if r is None else r.zh,
                         used_as="；".join(dict.fromkeys(use.get(m, []))) or
                                 ("未使用：值为空" if r is not None and r.kind == "unknown" else "未使用：静态配置/上报标记")))
    return pd.DataFrame(rows)


def describe_bd(d: UCEData) -> dict:
    r = dict(getattr(d, "report", {}))
    r["cards"] = len(d.entities)
    r["modalities"] = sorted(d.modalities())
    return r


# ------------------------------------------------------------------ scoring output
_RULE_ZH = [
    (r"^recur_time=<=1h$", "1 小时内发生过不可纠正错误"), (r"^recur_time=<=4h$", "4 小时内发生过不可纠正错误"),
    (r"^recur_time=<=1d$", "1 天内发生过不可纠正错误"), (r"^recur_time=<=7d$", "7 天内发生过不可纠正错误"),
    (r"^recur_time=<=30d$", "30 天内发生过不可纠正错误"), (r"^recur_count=(.+)$", r"累计不可纠正错误 \1 次"),
    (r"^node_uce=yes$", "同节点其他卡 30 天内发生过不可纠正错误"), (r"^ce_burst=high$", "近 1 小时可纠正错误数很高"),
    (r"^ce_burst=mid$", "近 1 小时可纠正错误数偏高"), (r"^ce_burst=low$", "近 1 小时出现可纠正错误"),
    (r"^ce_accel=rising$", "可纠正错误速率在上升"), (r"^ce_onset=<=1h$", "1 小时内首次出现可纠正错误"),
    (r"^ce_onset=<=1d$", "1 天内首次出现可纠正错误"), (r"^ce_onset=<=7d$", "7 天内首次出现可纠正错误"),
    (r"^ce_storm=yes$", "近 1 天出现可纠正错误风暴"), (r"^app_event=<=1d$", "近 1 天有应用类错误"),
]


def reason_zh(reason: str) -> str:
    """rule level behind an alarm -> Chinese text."""
    if not isinstance(reason, str) or not reason or reason == "baseline":
        return ""
    for pat, txt in _RULE_ZH:
        if re.match(pat, reason):
            return re.sub(pat, txt, reason)
    m = re.match(r"^hw_event_(.+)=(<=1d|<=30d)$", reason)
    if m:
        code = m.group(1)
        name = CODE_ZH.get(code[3:] if code.startswith("WO_") else code, code)
        name = ("工单：" if code.startswith("WO_") else "") + name
        return f"{'近 1 天' if m.group(2) == '<=1d' else '近 30 天'}{' ' if name[:1].isascii() else ''}{name}"
    m = re.match(r"^tel_high_(?:te__)?(.+)_(mean1h|mean1d|std1d|delta1h)=top10%$", reason)
    if m:
        stat = {"mean1h": "近 1 小时均值", "mean1d": "近 1 天均值", "std1d": "近 1 天波动", "delta1h": "近 1 小时较日均的变化"}[m.group(2)]
        return f"{TE_ZH.get(m.group(1), m.group(1))}：{stat}处于全体卡的前 10%"
    return reason


def predict(pred, d: UCEData, at=None, op: str = "fpr1", active_within: int = HOUR) -> pd.DataFrame:
    """risk of every card that reported within `active_within` before `at` (default: the end of the data).

    Output (one row per card): timestamp, uuid, hostname, ipv4, gpu, rack, p_1h, p_4h, p_24h, alarm_1h, alarm_4h,
    alarm_24h, risk_level (高 / 中 / 低), reason, reason_zh, date, hour."""
    sp = d.bd["span"]
    at = int(sp["last"].max() + MIN if at is None else at)
    ents = sp.index[(sp["last"] >= at - active_within) & (sp["first"] < at)].to_numpy()
    out = pred.predict(d, at=pd.DataFrame({"entity": ents, "s": at}), op=op)
    out = out.rename(columns={"p_1d": "p_24h", "alarm_1d": "alarm_24h", "device": "uuid"})
    for c in ("ipv4", "gpu", "rack", "scale_up_domain", "hostname"):
        if c in d.entities and c not in out:
            out[c] = out.entity.map(d.entities[c])
    # nested alarms: a UCE expected within 1 h is also expected within 4 h and 24 h
    if {"alarm_1h", "alarm_4h", "alarm_24h"} <= set(out.columns):
        out["alarm_4h"] = out.alarm_4h | out.alarm_1h
        out["alarm_24h"] = out.alarm_24h | out.alarm_4h
    a1, a4, a24 = (out.get(f"alarm_{h}", False) for h in ("1h", "4h", "24h"))
    out["risk_level"] = np.where(a1 | a4, "高", np.where(a24, "中", "低"))
    out["reason_zh"] = [reason_zh(r) if lv != "低" else "" for r, lv in zip(out.get("reason", ""), out.risk_level)]
    ts = pd.Timestamp(at, unit="s", tz="UTC")
    out.insert(0, "timestamp", at)
    out["date"], out["hour"] = ts.strftime("%Y%m%d"), ts.strftime("%H")
    if "hostname" not in out and "host" in out:
        out = out.rename(columns={"host": "hostname"})
    cols = ["timestamp", "uuid", "hostname", "ipv4", "gpu", "rack", "p_1h", "p_4h", "p_24h", "alarm_1h", "alarm_4h",
            "alarm_24h", "risk_level", "reason", "reason_zh", "date", "hour"]
    return out[[c for c in cols if c in out.columns]].sort_values("p_24h", ascending=False).reset_index(drop=True)
