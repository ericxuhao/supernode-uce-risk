#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A100 GPU ECC 故障预测数据采集器 (gpu-ecc-collector)

依据：NPUMonitor 文档《14-GPU监控指标与告警规则说明表》(修订1.2) 的指标 ID 与语义，
以及 NVIDIA《GPU Memory Error Management》(A100 行重映射) 与 XID 目录。

设计要点
- 仅依赖 Python 3.6+ 标准库与 nvidia-smi；pynvml (nvidia-ml-py) / DCGM Exporter 为可选增强。
- 三档采样：fast(默认10s：ECC总计数/行重映射/温度功耗等) | slow(默认60s：按位置细分ECC、
  bank重映射资源直方图、NVLink错误、进程上下文) | inventory(启动+每6h：资产与版本)。
- 事件：内核日志 XID/SXID/NVRM（journalctl 游标断点续读，退化到 dmesg），
  计数器增量/回退、状态翻转、GPU 掉卡；可选 NVML 事件（XID/SBE/DBE）。
- 触发快照：内存类 XID 或 DBE/行重映射类计数变化时，保存事件前窗口样本 + nvidia-smi -q 全量。
- 缺失/不支持保存为 null 并记录原因，绝不填 0（遵守规范“未知不能当健康”）。
- 时间戳统一 UTC（ISO8601 'Z'）。存储为按小时/天切分的 JSONL，跨小时自动 gzip，按天保留清理。
"""
import argparse
import collections
import csv
import datetime
import gzip
import io
import json
import logging
import logging.handlers
import os
import platform
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

VERSION = "1.0.0"
LOG = logging.getLogger("gpu-ecc-collector")

# --------------------------------------------------------------------------------------
# 指标映射：nvidia-smi 查询字段 -> metric_id。
# 文档14 已定义的使用其 DCGM_FI_* 稳定 ID；文档未定义的用 "nvsmi.<字段>"，保证可追溯。
# kind: counter(单调累计) / gauge / bool / enum / bitmask / str
# 每项可给多个候选字段名（新旧驱动别名），取第一个受支持的。
# --------------------------------------------------------------------------------------
def F(fields, metric_id, kind):
    if isinstance(fields, str):
        fields = (fields,)
    return {"fields": tuple(fields), "id": metric_id, "kind": kind}


KEY_FIELDS = ("index", "uuid", "pci.bus_id")

FAST_SPECS = [
    # ---- ECC 总计数（文档 A.2）----
    F("ecc.errors.corrected.volatile.total", "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL", "counter"),
    F("ecc.errors.uncorrected.volatile.total", "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL", "counter"),
    F("ecc.errors.corrected.aggregate.total", "DCGM_FI_DEV_ECC_SBE_AGG_TOTAL", "counter"),
    F("ecc.errors.uncorrected.aggregate.total", "DCGM_FI_DEV_ECC_DBE_AGG_TOTAL", "counter"),
    # ---- 设备显存(HBM/DRAM) 与 SRAM 细分 ----
    F("ecc.errors.corrected.volatile.device_memory", "DCGM_FI_DEV_ECC_SBE_VOL_DEV", "counter"),
    F("ecc.errors.uncorrected.volatile.device_memory", "DCGM_FI_DEV_ECC_DBE_VOL_DEV", "counter"),
    F("ecc.errors.corrected.aggregate.device_memory", "DCGM_FI_DEV_ECC_SBE_AGG_DEV", "counter"),
    F("ecc.errors.uncorrected.aggregate.device_memory", "DCGM_FI_DEV_ECC_DBE_AGG_DEV", "counter"),
    F("ecc.errors.corrected.volatile.sram", "DCGM_FI_DEV_ECC_SBE_VOL_SRM", "counter"),
    F("ecc.errors.uncorrected.volatile.sram", "DCGM_FI_DEV_ECC_DBE_VOL_SRM", "counter"),
    F("ecc.errors.corrected.aggregate.sram", "DCGM_FI_DEV_ECC_SBE_AGG_SRM", "counter"),
    F("ecc.errors.uncorrected.aggregate.sram", "DCGM_FI_DEV_ECC_DBE_AGG_SRM", "counter"),
    # ---- A100 行重映射（取代页退役）----
    F(("remapped_rows.correctable", "remapped_rows.sbe"), "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS", "counter"),
    F(("remapped_rows.uncorrectable", "remapped_rows.dbe"), "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS", "counter"),
    F("remapped_rows.pending", "DCGM_FI_DEV_ROW_REMAP_PENDING", "bool"),
    F("remapped_rows.failure", "DCGM_FI_DEV_ROW_REMAP_FAILURE", "bool"),
    F("gpu_recovery_action", "DCGM_FI_DEV_GPU_RECOVERY_ACTION", "enum"),
    # ---- 运行环境（温度/功耗/时钟/负载，作为 ECC 预测协变量）----
    F("temperature.gpu", "DCGM_FI_DEV_GPU_TEMP", "gauge"),
    F("temperature.memory", "DCGM_FI_DEV_MEMORY_TEMP", "gauge"),
    F("power.draw", "DCGM_FI_DEV_POWER_USAGE", "gauge"),
    F("clocks.sm", "DCGM_FI_DEV_SM_CLOCK", "gauge"),
    F("clocks.mem", "DCGM_FI_DEV_MEM_CLOCK", "gauge"),
    F("utilization.gpu", "DCGM_FI_DEV_GPU_UTIL", "gauge"),
    F("utilization.memory", "DCGM_FI_DEV_MEM_COPY_UTIL", "gauge"),
    F("memory.used", "DCGM_FI_DEV_FB_USED", "gauge"),
    F("memory.free", "DCGM_FI_DEV_FB_FREE", "gauge"),
    F("pstate", "nvsmi.pstate", "str"),
    F(("clocks_event_reasons.active", "clocks_throttle_reasons.active"), "DCGM_FI_DEV_CLOCKS_EVENT_REASONS", "bitmask"),
    F("pcie.link.gen.current", "DCGM_FI_DEV_PCIE_LINK_GEN", "gauge"),
    F("pcie.link.width.current", "DCGM_FI_DEV_PCIE_LINK_WIDTH", "gauge"),
]

SLOW_SPECS = [
    # ---- ECC 配置（文档 ECC_MODE / ECC_PENDING）----
    F("ecc.mode.current", "DCGM_FI_DEV_ECC_MODE", "bool"),
    F("ecc.mode.pending", "DCGM_FI_DEV_ECC_PENDING", "bool"),
    # ---- bank 重映射资源直方图（文档 BANK_REMAP_AVAIL_*）----
    F("remapped_rows.histogram.max", "DCGM_FI_DEV_BANK_REMAP_AVAIL_MAX", "gauge"),
    F("remapped_rows.histogram.high", "DCGM_FI_DEV_BANK_REMAP_AVAIL_HIGH", "gauge"),
    F("remapped_rows.histogram.partial", "DCGM_FI_DEV_BANK_REMAP_AVAIL_PARTIAL", "gauge"),
    F("remapped_rows.histogram.low", "DCGM_FI_DEV_BANK_REMAP_AVAIL_LOW", "gauge"),
    F("remapped_rows.histogram.none", "DCGM_FI_DEV_BANK_REMAP_AVAIL_NONE", "gauge"),
    F("remapped_rows.correctable_inactive", "nvsmi.remapped_rows.correctable_inactive", "counter"),
    F("remapped_rows.uncorrectable_inactive", "nvsmi.remapped_rows.uncorrectable_inactive", "counter"),
    # ---- 旧架构页退役（A100 通常不支持，保留以兼容 V100 等）----
    F(("retired_pages.sbe", "retired_pages.single_bit_ecc.count"), "DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL", "counter"),
    F(("retired_pages.dbe", "retired_pages.double_bit.count"), "DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL", "counter"),
    F("retired_pages.pending", "DCGM_FI_DEV_PAGE_RETIRED_PENDING", "bool"),
    # ---- 节流累计时长（热/功耗应力）----
    F("clocks_event_reasons_counters.hw_thermal_slowdown", "nvsmi.clocks_event_reasons_counters.hw_thermal_slowdown", "counter"),
    F("clocks_event_reasons_counters.sw_thermal_slowdown", "nvsmi.clocks_event_reasons_counters.sw_thermal_slowdown", "counter"),
    F("clocks_event_reasons_counters.hw_power_brake_slowdown", "nvsmi.clocks_event_reasons_counters.hw_power_brake_slowdown", "counter"),
    F("clocks_event_reasons_counters.sw_power_cap", "nvsmi.clocks_event_reasons_counters.sw_power_cap", "counter"),
    F("enforced.power.limit", "nvsmi.enforced.power.limit", "gauge"),
]
# 按位置细分的 ECC（寄存器/L1/L2/纹理/CBU/DRAM 及新驱动 SRAM 子类）
for _cls in ("corrected", "uncorrected"):
    for _scope in ("volatile", "aggregate"):
        for _loc in ("register_file", "l1_cache", "l2_cache", "texture_memory", "cbu", "dram"):
            _f = "ecc.errors.%s.%s.%s" % (_cls, _scope, _loc)
            SLOW_SPECS.append(F(_f, "nvsmi." + _f, "counter"))
for _f in ("volatile.sram.parity", "volatile.sram.secded", "aggregate.sram.parity", "aggregate.sram.secded",
           "aggregate.sram.thresholdExceeded", "aggregate.sram.l2", "aggregate.sram.sm",
           "aggregate.sram.mcu", "aggregate.sram.pcie", "aggregate.sram.other"):
    _f = "ecc.errors.uncorrected." + _f
    SLOW_SPECS.append(F(_f, "nvsmi." + _f, "bool" if _f.endswith("thresholdExceeded") else "counter"))

INVENTORY_SPECS = [
    F(("name", "gpu_name"), "nvsmi.name", "str"),
    F(("serial", "gpu_serial"), "nvsmi.serial", "str"),
    F("vbios_version", "nvsmi.vbios_version", "str"),
    F("driver_version", "nvsmi.driver_version", "str"),
    F(("inforom.img", "inforom.image"), "nvsmi.inforom.img", "str"),
    F("inforom.oem", "nvsmi.inforom.oem", "str"),
    F("inforom.ecc", "nvsmi.inforom.ecc", "str"),
    F(("inforom.pwr", "inforom.power"), "nvsmi.inforom.pwr", "str"),
    F("persistence_mode", "nvsmi.persistence_mode", "bool"),
    F("mig.mode.current", "nvsmi.mig.mode.current", "bool"),
    F("compute_cap", "nvsmi.compute_cap", "str"),
    F("memory.total", "nvsmi.memory.total", "gauge"),
    F(("power.limit", "gpu.power.requested_limit"), "nvsmi.power.limit", "gauge"),
    F("clocks.max.sm", "nvsmi.clocks.max.sm", "gauge"),
    F("clocks.max.memory", "nvsmi.clocks.max.memory", "gauge"),
    F("pcie.link.gen.max", "nvsmi.pcie.link.gen.max", "gauge"),
    F("pcie.link.width.max", "nvsmi.pcie.link.width.max", "gauge"),
    F("gsp.mode.current", "nvsmi.gsp.mode.current", "str"),
    F("ecc.mode.current", "DCGM_FI_DEV_ECC_MODE", "bool"),
    F("ecc.mode.pending", "DCGM_FI_DEV_ECC_PENDING", "bool"),
]

# 需要做“变化检测”的指标及其是否触发快照
WATCH = {
    "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL": False, "DCGM_FI_DEV_ECC_SBE_AGG_TOTAL": False,
    "DCGM_FI_DEV_ECC_SBE_VOL_DEV": False, "DCGM_FI_DEV_ECC_SBE_VOL_SRM": False,
    "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL": True, "DCGM_FI_DEV_ECC_DBE_AGG_TOTAL": True,
    "DCGM_FI_DEV_ECC_DBE_VOL_DEV": True, "DCGM_FI_DEV_ECC_DBE_VOL_SRM": True,
    "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS": True, "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS": True,
    "DCGM_FI_DEV_ROW_REMAP_PENDING": True, "DCGM_FI_DEV_ROW_REMAP_FAILURE": True,
    "DCGM_FI_DEV_GPU_RECOVERY_ACTION": True,
    "DCGM_FI_DEV_BANK_REMAP_AVAIL_LOW": True, "DCGM_FI_DEV_BANK_REMAP_AVAIL_NONE": True,
    "DCGM_FI_DEV_BANK_REMAP_AVAIL_PARTIAL": False,
    "DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL": True, "DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL": True,
    "DCGM_FI_DEV_PAGE_RETIRED_PENDING": True,
    "DCGM_FI_DEV_ECC_MODE": False, "DCGM_FI_DEV_ECC_PENDING": False,
    "nvsmi.ecc.errors.uncorrected.aggregate.sram.thresholdExceeded": True,
}
SBE_METRICS = {k for k in WATCH if "_SBE_" in k}

# XID 分类（NVIDIA XID 目录 / A100 Memory Error Management）
XID_CATEGORY = {
    48: ("ecc_dbe", True),                 # Double Bit ECC Error（不可纠正）
    63: ("row_remap_recorded", True),      # 行重映射条目写入 InfoROM
    64: ("row_remap_failure", True),       # 行重映射记录失败
    92: ("excessive_sbe_rate", True),      # 单比特 ECC 中断过多
    94: ("contained_ecc", True),           # 受控不可纠正 ECC，需重启应用
    95: ("uncontained_ecc", True),         # 非受控不可纠正 ECC，需 GPU reset
    160: ("memory_marked_for_repair", True),  # 新驱动
    171: ("uncorrectable_ecc_dram", True),    # 新驱动（Xid48 细分）
    172: ("uncorrectable_ecc_sram", True),    # 新驱动（Xid48 细分）
    154: ("recovery_action_changed", True),
    79: ("fallen_off_bus", True),
    74: ("nvlink_error", False),
    119: ("gsp_rpc_timeout", False),
    120: ("gsp_error", False),
    31: ("mmu_fault", False),
    13: ("graphics_engine_exception", False),
    43: ("gpu_stopped_processing", False),
    45: ("preemptive_cleanup", False),
    62: ("internal_microcontroller_halt", False),
}

NA_TOKENS = {"", "n/a", "[n/a]", "not supported", "[not supported]", "unknown error", "[unknown error]",
             "[insufficient permissions]", "[gpu is lost]", "[gpu requires reset]",
             "[requested functionality has been deprecated]", "[not applicable]"}
TRUE_TOKENS = {"enabled", "yes", "active", "true", "on"}
FALSE_TOKENS = {"disabled", "no", "not active", "false", "off"}

XID_RE = re.compile(r"NVRM: Xid \(PCI:(?P<pci>[0-9A-Fa-f:.]+)\): (?P<xid>\d+),\s*(?P<rest>.*)")
SXID_RE = re.compile(r"(?P<dev>nvidia-nvswitch\d+): SXid \(PCI:(?P<pci>[0-9A-Fa-f:.]+)\): (?P<xid>\d+),\s*(?P<rest>.*)")
NVRM_INTEREST_RE = re.compile(r"NVRM:.*(ECC|[Rr]emap|[Rr]etire|fallen off|DBE|SBE|InfoROM|poison)", re.I)
DMESG_RE = re.compile(r"^\[\s*(?P<mono>\d+\.\d+)\]\s?(?P<msg>.*)$")

DEFAULT_CONFIG = {
    "base_dir": "/var/lib/gpu-ecc-collector",
    "nvidia_smi": "nvidia-smi",
    "fast_interval_s": 10,
    "slow_interval_s": 60,
    "inventory_interval_s": 21600,
    "q_parse_interval_s": 300,
    "raw_q_interval_s": 3600,
    "kernel_log_poll_s": 5,
    "kernel_log_initial_lookback_s": 86400,
    "command_timeout_s": 30,
    "pre_event_window_s": 1800,
    "snapshot_cooldown_s": 300,
    "snapshot_on_sbe": False,
    "collect_nvlink_errors": True,
    "collect_compute_apps": True,
    "enable_nvml_events": True,
    "dcgm_exporter_url": "",
    "dcgm_metric_prefixes": ["DCGM_FI_DEV_ECC", "DCGM_FI_DEV_ROW_REMAP", "DCGM_FI_DEV_CORRECTABLE_REMAPPED",
                             "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED", "DCGM_FI_DEV_BANK_REMAP",
                             "DCGM_FI_DEV_XID", "DCGM_FI_DEV_MEMORY_TEMP", "DCGM_FI_DEV_GPU_RECOVERY_ACTION",
                             "DCGM_FI_DEV_MEMORY_UNREPAIRABLE", "DCGM_FI_DEV_PCIE_REPLAY",
                             "DCGM_FI_DEV_NVLINK", "DCGM_FI_PROF_DRAM_ACTIVE"],
    "retention_days": 90,
    "event_retention_days": 730,
    "min_free_pct": 10.0,
    "critical_free_pct": 3.0,
    "compress_rotated": True,
    "log_level": "INFO",
}


# --------------------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------------------
def utc_dt(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)


def iso(ts):
    return utc_dt(ts).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def parse_iso(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def read_text(path, default=None):
    try:
        with io.open(path, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return default


def boot_id():
    return read_text("/proc/sys/kernel/random/boot_id", "unknown")


def uptime_s():
    t = read_text("/proc/uptime")
    if t:
        try:
            return float(t.split()[0])
        except ValueError:
            pass
    return None


def norm_pci(bus_id):
    """'00000000:07:00.0' / '0000:07:00' -> '07:00'（总线:设备，忽略域与功能号）。"""
    if not bus_id:
        return None
    parts = bus_id.strip().lower().split(":")
    if len(parts) >= 3:
        bus, dev = parts[-2], parts[-1].split(".")[0]
    elif len(parts) == 2:
        bus, dev = parts[0], parts[1].split(".")[0]
    else:
        return bus_id.lower()
    return "%s:%s" % (bus.zfill(2)[-2:], dev.zfill(2)[-2:])


def parse_value(raw, kind):
    """返回 (value, na_reason)。不支持/缺失 -> (None, 原因)，不当作 0。"""
    s = (raw or "").strip()
    low = s.lower()
    if low in NA_TOKENS or (s.startswith("[") and s.endswith("]")):
        return None, s or "empty"
    if kind in ("str", "enum"):
        return s, None
    if kind == "bool":
        if low in TRUE_TOKENS:
            return 1, None
        if low in FALSE_TOKENS:
            return 0, None
    if low.startswith("0x"):
        try:
            return int(s, 16), None
        except ValueError:
            return s, None
    try:
        return int(s), None
    except ValueError:
        pass
    try:
        return float(s), None
    except ValueError:
        pass
    if low in TRUE_TOKENS:
        return 1, None
    if low in FALSE_TOKENS:
        return 0, None
    return s, None


def run_cmd(args, timeout):
    """运行命令，返回 (rc, stdout, stderr)；超时 rc=-9。兼容 Python 3.6。"""
    try:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as e:
        return -127, "", str(e)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        try:
            p.communicate(timeout=5)
        except Exception:
            pass
        return -9, "", "timeout after %ss" % timeout
    return p.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def parse_help_query_gpu(text):
    """从 `nvidia-smi --help-query-gpu` 输出中提取所有可查询字段名（含别名）。"""
    names = set()
    for line in text.splitlines():
        if line.startswith('"'):
            names.update(re.findall(r'"([^"]+)"', line))
    return names


def parse_smi_q(text):
    """把 `nvidia-smi -q` 文本解析为 {norm_pci: {"A/B/C": value}}。"""
    gpus = {}
    cur = None
    stack = []  # [(indent, key)]
    for line in text.splitlines():
        m = re.match(r"^GPU ([0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f])\s*$", line)
        if m:
            cur = gpus.setdefault(norm_pci(m.group(1)), {})
            stack = []
            continue
        if cur is None or not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if ":" in body:
            k, v = body.split(":", 1)
            k, v = k.strip(), v.strip()
            path = "/".join([s[1] for s in stack] + [k])
            if v:
                cur[path] = v
            else:
                stack.append((indent, k))
        else:
            stack.append((indent, body))
    return gpus


def q_lookup(section, suffix):
    for k, v in section.items():
        if k.endswith(suffix):
            return v
    return None


Q_EXTRA = [  # 从 -q 文本补充 query-gpu 拿不到/旧驱动不支持的项
    ("Bank Remap Availability Histogram/Max", "DCGM_FI_DEV_BANK_REMAP_AVAIL_MAX"),
    ("Bank Remap Availability Histogram/High", "DCGM_FI_DEV_BANK_REMAP_AVAIL_HIGH"),
    ("Bank Remap Availability Histogram/Partial", "DCGM_FI_DEV_BANK_REMAP_AVAIL_PARTIAL"),
    ("Bank Remap Availability Histogram/Low", "DCGM_FI_DEV_BANK_REMAP_AVAIL_LOW"),
    ("Bank Remap Availability Histogram/None", "DCGM_FI_DEV_BANK_REMAP_AVAIL_NONE"),
    ("Replays Since Reset", "DCGM_FI_DEV_PCIE_REPLAY_COUNTER"),
    ("Replay Number Rollovers", "nvsmi.pcie.replay_rollovers"),
]


def extract_q_extra(section):
    out = {}
    for suffix, mid in Q_EXTRA:
        v = q_lookup(section, suffix)
        if v is None:
            continue
        m = re.match(r"^\s*(\d+)", v)
        out[mid] = int(m.group(1)) if m else None
    return out


def parse_nvlink_errors(text):
    """解析 `nvidia-smi nvlink -e`，返回 {gpu_index: {"per_link": {...}, "sum": {...}}}。"""
    res = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^GPU (\d+):", line)
        if m:
            cur = res.setdefault(int(m.group(1)), {"per_link": {}, "sum": {}})
            continue
        m = re.match(r"^\s*Link (\d+):\s*(.+?)\s*:\s*(\d+)\s*$", line)
        if m and cur is not None:
            link, name, val = m.group(1), m.group(2), int(m.group(3))
            key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            cur["per_link"].setdefault(link, {})[key] = val
            cur["sum"][key] = cur["sum"].get(key, 0) + val
    return res


def parse_kernel_line(msg):
    """识别 XID/SXID/NVRM 行，返回事件 dict 或 None。"""
    m = XID_RE.search(msg)
    if m:
        xid = int(m.group("xid"))
        cat, mem = XID_CATEGORY.get(xid, ("other", False))
        return {"type": "xid", "xid": xid, "xid_category": cat, "memory_related": mem,
                "pci": norm_pci(m.group("pci")), "message": msg.strip()}
    m = SXID_RE.search(msg)
    if m:
        return {"type": "sxid", "xid": int(m.group("xid")), "nvswitch": m.group("dev"),
                "pci": norm_pci(m.group("pci")), "message": msg.strip()}
    if NVRM_INTEREST_RE.search(msg):
        pm = re.search(r"PCI:([0-9A-Fa-f:.]+)", msg)
        return {"type": "nvrm", "pci": norm_pci(pm.group(1)) if pm else None, "message": msg.strip()}
    return None


# --------------------------------------------------------------------------------------
# 存储
# --------------------------------------------------------------------------------------
def gzip_file(path):
    dst = path + ".gz"
    n = 1
    while os.path.exists(dst):
        dst = "%s.%d.gz" % (path, n)
        n += 1
    with open(path, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(path)
    return dst


class JsonlWriter(object):
    """按 UTC 小时/天切分 JSONL；切换文件时把旧文件 gzip。"""

    def __init__(self, base, sub, prefix, granularity="hour", compress=True):
        self.base, self.sub, self.prefix = base, sub, prefix
        self.granularity, self.compress = granularity, compress
        self.path, self.fh = None, None

    def _path_for(self, ts):
        d = utc_dt(ts)
        day = d.strftime("%Y-%m-%d")
        name = "%s-%s.jsonl" % (self.prefix, d.strftime("%Y%m%dT%H") if self.granularity == "hour" else day)
        return os.path.join(self.base, self.sub, day, name)

    def write(self, rec, ts):
        p = self._path_for(ts)
        if p != self.path:
            self._rotate(p)
        self.fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _rotate(self, newpath):
        old = self.path
        self.close()
        if old and self.compress and os.path.exists(old):
            try:
                gzip_file(old)
            except Exception as e:
                LOG.warning("gzip %s failed: %s", old, e)
        d = os.path.dirname(newpath)
        if not os.path.isdir(d):
            os.makedirs(d)
        self.path = newpath
        self.fh = io.open(newpath, "a", encoding="utf-8")

    def flush(self):
        if self.fh:
            self.fh.flush()

    def close(self):
        if self.fh:
            try:
                self.fh.close()
            finally:
                self.fh = None


def compress_leftovers(root, keep_paths):
    """启动时压缩上次运行遗留的非当前 .jsonl 文件。"""
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            if fn.endswith(".jsonl") and p not in keep_paths:
                try:
                    gzip_file(p)
                except Exception as e:
                    LOG.warning("compress leftover %s failed: %s", p, e)


def purge_old(root, days):
    if not os.path.isdir(root) or days <= 0:
        return
    cutoff = utc_dt(time.time() - days * 86400).strftime("%Y-%m-%d")
    for name in os.listdir(root):
        if re.match(r"^\d{4}-\d{2}-\d{2}$", name) and name < cutoff:
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)
            LOG.info("purged %s/%s", root, name)


# --------------------------------------------------------------------------------------
# 内核日志读取（XID / SXID / NVRM），独立线程，结果投递到队列
# --------------------------------------------------------------------------------------
class KernelLogReader(threading.Thread):
    def __init__(self, cfg, state, out_q, stop):
        threading.Thread.__init__(self, name="kernel-log")
        self.daemon = True
        self.cfg, self.state, self.q, self.stop = cfg, state, out_q, stop
        self.mode = None

    def run(self):
        if shutil.which("journalctl"):
            self.mode = "journalctl"
        elif shutil.which("dmesg"):
            self.mode = "dmesg"
        else:
            LOG.warning("neither journalctl nor dmesg found; XID log reading disabled")
            return
        LOG.info("kernel log reader mode=%s", self.mode)
        while not self.stop.is_set():
            try:
                if self.mode == "journalctl":
                    ok = self._poll_journal()
                    if not ok and shutil.which("dmesg"):
                        LOG.warning("journalctl failed, falling back to dmesg")
                        self.mode = "dmesg"
                else:
                    self._poll_dmesg()
            except Exception as e:
                LOG.exception("kernel log poll error: %s", e)
            self.stop.wait(self.cfg["kernel_log_poll_s"])

    def _emit(self, msg, event_ts, source):
        ev = parse_kernel_line(msg)
        if ev:
            ev["event_time"] = iso(event_ts) if event_ts else None
            ev["source"] = source
            self.q.put(("kernel", ev))

    def _poll_journal(self):
        args = ["journalctl", "-k", "-o", "json", "--no-pager", "-q"]
        cur = self.state.get("journal_cursor")
        if cur:
            args.append("--after-cursor=" + cur)  # 断点续读；持久化 journal 时跨重启也有效
        else:
            since = time.time() - self.cfg["kernel_log_initial_lookback_s"]
            args.append("--since=" + datetime.datetime.fromtimestamp(since).strftime("%Y-%m-%d %H:%M:%S"))
        rc, out, err = run_cmd(args, self.cfg["command_timeout_s"])
        if rc != 0 and not out:
            if cur and "cursor" in err.lower():
                LOG.warning("journal cursor invalid, resetting: %s", err.strip())
                self.state.pop("journal_cursor", None)
                return True
            LOG.warning("journalctl rc=%s err=%s", rc, err.strip()[:300])
            return False
        last = None
        for line in out.splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            last = e.get("__CURSOR", last)
            msg = e.get("MESSAGE")
            if isinstance(msg, list):
                msg = bytes(bytearray(msg)).decode("utf-8", "replace")
            if not msg:
                continue
            try:
                ts = int(e.get("__REALTIME_TIMESTAMP")) / 1e6
            except (TypeError, ValueError):
                ts = None
            self._emit(msg, ts, "journalctl")
        if last:
            self.q.put(("cursor", {"journal_cursor": last, "journal_boot_id": boot_id()}))
            self.state["journal_cursor"] = last
        return True

    def _poll_dmesg(self):
        rc, out, err = run_cmd(["dmesg"], self.cfg["command_timeout_s"])
        if rc != 0:
            LOG.warning("dmesg rc=%s err=%s", rc, err.strip()[:200])
            return
        bid = boot_id()
        last = self.state.get("dmesg_last_mono", -1.0) if self.state.get("dmesg_boot_id") == bid else -1.0
        up = uptime_s()
        boot_wall = (time.time() - up) if up is not None else None
        newest = last
        for line in out.splitlines():
            m = DMESG_RE.match(line)
            if not m:
                continue
            mono = float(m.group("mono"))
            if mono <= last:
                continue
            newest = max(newest, mono)
            self._emit(m.group("msg"), (boot_wall + mono) if boot_wall else None, "dmesg")
        if newest > last:
            upd = {"dmesg_last_mono": newest, "dmesg_boot_id": bid}
            self.state.update(upd)
            self.q.put(("cursor", upd))


# --------------------------------------------------------------------------------------
# 可选：NVML 事件（XID / SBE / DBE），需要 pip install nvidia-ml-py
# --------------------------------------------------------------------------------------
class NvmlEventThread(threading.Thread):
    def __init__(self, out_q, stop):
        threading.Thread.__init__(self, name="nvml-events")
        self.daemon = True
        self.q, self.stop = out_q, stop

    def run(self):
        try:
            import pynvml as N  # noqa
        except Exception:
            LOG.info("pynvml not installed; NVML event listening disabled (optional)")
            return
        try:
            N.nvmlInit()
            es = N.nvmlEventSetCreate()
            uuid_of = {}
            want = 0
            for name in ("nvmlEventTypeXidCriticalError", "nvmlEventTypeSingleBitEccError",
                         "nvmlEventTypeDoubleBitEccError"):
                want |= getattr(N, name, 0)
            for i in range(N.nvmlDeviceGetCount()):
                h = N.nvmlDeviceGetHandleByIndex(i)
                u = N.nvmlDeviceGetUUID(h)
                uuid_of[i] = u.decode() if isinstance(u, bytes) else u
                try:
                    mask = N.nvmlDeviceGetSupportedEventTypes(h) & want
                    if mask:
                        N.nvmlDeviceRegisterEvents(h, mask, es)
                except N.NVMLError as e:
                    LOG.info("NVML events unsupported on GPU %d: %s", i, e)
        except Exception as e:
            LOG.warning("NVML event init failed: %s", e)
            return
        names = {getattr(N, "nvmlEventTypeXidCriticalError", -1): "xid",
                 getattr(N, "nvmlEventTypeSingleBitEccError", -2): "sbe",
                 getattr(N, "nvmlEventTypeDoubleBitEccError", -3): "dbe"}
        LOG.info("NVML event listener started")
        while not self.stop.is_set():
            try:
                d = N.nvmlEventSetWait_v2(es, 1000) if hasattr(N, "nvmlEventSetWait_v2") else N.nvmlEventSetWait(es, 1000)
            except N.NVMLError as e:
                if getattr(N, "NVML_ERROR_TIMEOUT", None) == getattr(e, "value", None) or "Timeout" in str(e):
                    continue
                LOG.warning("NVML wait error: %s", e)
                self.stop.wait(5)
                continue
            try:
                u = N.nvmlDeviceGetUUID(d.device)
                u = u.decode() if isinstance(u, bytes) else u
            except Exception:
                u = None
            kind = names.get(d.eventType, str(d.eventType))
            ev = {"type": "nvml_event", "nvml_event": kind, "uuid": u, "event_time": iso(time.time()),
                  "source": "nvml"}
            if kind == "xid":
                ev["xid"] = int(d.eventData)
                ev["xid_category"], ev["memory_related"] = XID_CATEGORY.get(ev["xid"], ("other", False))
            else:
                ev["event_data"] = int(d.eventData)
            self.q.put(("nvml", ev))
        try:
            N.nvmlEventSetFree(es)
            N.nvmlShutdown()
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# 主采集器
# --------------------------------------------------------------------------------------
class Collector(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = cfg["base_dir"]
        self.host = socket.gethostname()
        self.stop = threading.Event()
        self.q = queue.Queue()
        self.state_path = os.path.join(self.base, "state", "state.json")
        self.state = self._load_state()
        comp = cfg["compress_rotated"]
        self.w_fast = JsonlWriter(self.base, "samples", "fast", "hour", comp)
        self.w_slow = JsonlWriter(self.base, "samples", "slow", "hour", comp)
        self.w_inv = JsonlWriter(self.base, "inventory", "inventory", "day", comp)
        self.w_ev = JsonlWriter(self.base, "events", "events", "day", comp)
        self.w_dcgm = JsonlWriter(self.base, "dcgm", "dcgm", "hour", comp)
        self.supported = None
        self.resolved = {}  # tier -> [(field, spec)]
        self.unsupported = {}
        self.ring = collections.defaultdict(lambda: collections.deque(
            maxlen=max(1, int(cfg["pre_event_window_s"] / max(1, cfg["fast_interval_s"])))))
        self.kernel_ring = collections.deque(maxlen=500)
        self.last_snapshot = {}
        self.pci_to_gpu = {}   # norm_pci -> (index, uuid)
        self.uuid_to_gpu = {}  # uuid -> (index, pci)
        self.disk_level = "ok"
        self.bid = boot_id()
        self.start_ts = time.time()
        self.last_raw = 0.0
        self.seen = collections.defaultdict(set)  # uuid -> 本次运行已见过的指标
        self.consecutive_fail = collections.Counter()
        self.status = {"version": VERSION, "host": self.host, "started": iso(time.time()), "tiers": {}}

    # ---------- state ----------
    def _load_state(self):
        try:
            with io.open(self.state_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_state(self):
        d = os.path.dirname(self.state_path)
        if not os.path.isdir(d):
            os.makedirs(d)
        tmp = self.state_path + ".tmp"
        with io.open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False)
        os.replace(tmp, self.state_path)
        st = os.path.join(d, "status.json")
        with io.open(st + ".tmp", "w", encoding="utf-8") as f:
            json.dump(self.status, f, ensure_ascii=False, indent=1)
        os.replace(st + ".tmp", st)

    # ---------- common ----------
    def envelope(self, ts, **kw):
        rec = {"ts": iso(ts), "host": self.host, "boot_id": self.bid}
        rec.update(kw)
        return rec

    def event(self, ts, etype, **kw):
        rec = self.envelope(ts, type=etype)
        rec.update({k: v for k, v in kw.items() if v is not None})
        self.w_ev.write(rec, ts)
        lvl = logging.WARNING if etype in ("xid", "counter_increase", "state_change", "gpu_missing",
                                           "query_error") else logging.INFO
        LOG.log(lvl, "event %s %s", etype, {k: v for k, v in kw.items() if k != "message"})
        return rec

    def smi(self, *args):
        return run_cmd([self.cfg["nvidia_smi"]] + list(args), self.cfg["command_timeout_s"])

    # ---------- field support ----------
    def detect_fields(self):
        rc, out, err = self.smi("--help-query-gpu")
        if rc == 0 and out:
            self.supported = parse_help_query_gpu(out)
        else:
            LOG.warning("--help-query-gpu failed (%s); probing fields individually", err.strip()[:200])
            self.supported = None
        for tier, specs in (("fast", FAST_SPECS), ("slow", SLOW_SPECS), ("inventory", INVENTORY_SPECS)):
            res = []
            for sp in specs:
                chosen = None
                for f in sp["fields"]:
                    if self._field_ok(f):
                        chosen = f
                        break
                if chosen:
                    res.append((chosen, sp))
                else:
                    self.unsupported[sp["id"]] = "field not supported by this driver: " + "|".join(sp["fields"])
            self.resolved[tier] = res
        LOG.info("fields resolved: fast=%d slow=%d inventory=%d unsupported=%d",
                 len(self.resolved["fast"]), len(self.resolved["slow"]), len(self.resolved["inventory"]),
                 len(self.unsupported))

    def _field_ok(self, f):
        if self.supported is not None:
            return f in self.supported
        rc, _o, _e = self.smi("--query-gpu=" + f, "--format=csv,noheader,nounits")
        return rc == 0

    def query(self, tier):
        pairs = self.resolved.get(tier) or []
        fields = list(KEY_FIELDS) + [f for f, _ in pairs]
        rc, out, err = self.smi("--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits")
        if rc != 0:
            if "not a valid field" in (err + out).lower() and pairs:
                LOG.warning("tier %s: invalid field reported, re-probing individually", tier)
                self.supported = None
                keep = [(f, sp) for f, sp in pairs if self._field_ok(f)]
                for f, sp in pairs:
                    if (f, sp) not in keep:
                        self.unsupported[sp["id"]] = "query failed for field " + f
                self.resolved[tier] = keep
                return self.query(tier)
            raise RuntimeError("nvidia-smi rc=%s %s" % (rc, (err or out).strip()[:300]))
        rows = []
        for row in csv.reader(io.StringIO(out), skipinitialspace=True):
            if len(row) < len(fields):
                continue
            idx, uuid, pci = row[0].strip(), row[1].strip(), row[2].strip()
            m, na = {}, {}
            for (f, sp), raw in zip(pairs, row[3:]):
                v, why = parse_value(raw, sp["kind"])
                m[sp["id"]] = v
                if why is not None:
                    na[sp["id"]] = why
            rows.append({"gpu": int(idx) if idx.isdigit() else idx, "uuid": uuid, "pci": pci, "m": m, "na": na})
        return rows

    # ---------- tiers ----------
    def tier_fast(self, ts):
        rows = self.query("fast")
        seen = set()
        for r in rows:
            seen.add(r["uuid"])
            self.uuid_to_gpu[r["uuid"]] = (r["gpu"], r["pci"])
            self.pci_to_gpu[norm_pci(r["pci"])] = (r["gpu"], r["uuid"])
            rec = self.envelope(ts, tier="fast", gpu=r["gpu"], uuid=r["uuid"], pci=r["pci"], m=r["m"])
            if r["na"]:
                rec["na"] = r["na"]
            self.w_fast.write(rec, ts)
            self.ring[r["uuid"]].append(rec)
            self.detect_changes(ts, r)
        known = set(self.state.get("known_uuids", []))
        for u in known - seen:
            if u not in self.state.get("missing_uuids", []):
                g = self.uuid_to_gpu.get(u, (None, None))
                self.event(ts, "gpu_missing", uuid=u, gpu=g[0], pci=g[1],
                           message="GPU absent from nvidia-smi output (fallen off bus / reset / removed?)")
                self.state.setdefault("missing_uuids", []).append(u)
                self.snapshot(ts, u, "gpu_missing", {"uuid": u})
        for u in seen:
            if u in self.state.get("missing_uuids", []):
                self.state["missing_uuids"].remove(u)
                self.event(ts, "gpu_reappeared", uuid=u)
        self.state["known_uuids"] = sorted(known | seen)
        self.w_fast.flush()

    def tier_slow(self, ts):
        rows = self.query("slow")
        nvl = {}
        if self.cfg["collect_nvlink_errors"] and not self.state.get("nvlink_unsupported"):
            rc, out, err = self.smi("nvlink", "-e")
            if rc == 0 and "Link" in out:
                nvl = parse_nvlink_errors(out)
            elif rc != 0 or not out.strip():
                LOG.info("nvidia-smi nvlink -e unavailable (PCIe card?): %s", (err or out).strip()[:120])
                self.state["nvlink_unsupported"] = True
        apps = {}
        if self.cfg["collect_compute_apps"]:
            rc, out, _e = self.smi("--query-compute-apps=gpu_uuid,used_memory", "--format=csv,noheader,nounits")
            if rc == 0:
                for row in csv.reader(io.StringIO(out), skipinitialspace=True):
                    if len(row) >= 2:
                        a = apps.setdefault(row[0].strip(), [0, 0.0])
                        a[0] += 1
                        v, _ = parse_value(row[1], "gauge")
                        if isinstance(v, (int, float)):
                            a[1] += v
        for r in rows:
            m = r["m"]
            if r["gpu"] in nvl:
                for k, v in nvl[r["gpu"]]["sum"].items():
                    m["nvsmi.nvlink." + k + "_total"] = v
            if self.cfg["collect_compute_apps"]:
                a = apps.get(r["uuid"], [0, 0.0])
                m["nvsmi.compute_apps.count"], m["nvsmi.compute_apps.used_memory_mib"] = a[0], a[1]
            rec = self.envelope(ts, tier="slow", gpu=r["gpu"], uuid=r["uuid"], pci=r["pci"], m=m)
            if r["na"]:
                rec["na"] = r["na"]
            if r["gpu"] in nvl:
                rec["nvlink_per_link"] = nvl[r["gpu"]]["per_link"]
            self.w_slow.write(rec, ts)
            self.detect_changes(ts, r)
        self.w_slow.flush()
        if self.cfg.get("dcgm_exporter_url"):
            self.scrape_dcgm(ts)

    def tier_inventory(self, ts):
        rows = self.query("inventory")
        rc, ver, _e = self.smi("--version")
        host = {"kernel": platform.release(), "os": self._os_release(), "python": platform.python_version(),
                "uptime_s": uptime_s(), "nvidia_smi_version": ver.strip() if rc == 0 else None,
                "collector_version": VERSION, "unsupported_metrics": self.unsupported,
                "dcgmi": bool(shutil.which("dcgmi"))}
        for r in rows:
            rec = self.envelope(ts, tier="inventory", gpu=r["gpu"], uuid=r["uuid"], pci=r["pci"], m=r["m"],
                                host_info=host)
            if r["na"]:
                rec["na"] = r["na"]
            prev = self.state.get("inventory", {}).get(r["uuid"])
            cur = {k: v for k, v in r["m"].items() if k.startswith("nvsmi.") and k not in ("nvsmi.memory.total",)}
            if prev and prev != cur:
                diff = {k: [prev.get(k), cur.get(k)] for k in set(prev) | set(cur) if prev.get(k) != cur.get(k)}
                self.event(ts, "inventory_change", uuid=r["uuid"], gpu=r["gpu"], diff=diff)
            self.state.setdefault("inventory", {})[r["uuid"]] = cur
            self.w_inv.write(rec, ts)
        self.w_inv.flush()

    def tier_qfull(self, ts):
        """全量 nvidia-smi -q：解析 PCIe replay 计数与（旧驱动）bank 直方图；按 raw 间隔归档原文。"""
        rc, out, err = self.smi("-q")
        if rc != 0:
            raise RuntimeError("nvidia-smi -q rc=%s %s" % (rc, err.strip()[:200]))
        have_hist = any(sp["id"] == "DCGM_FI_DEV_BANK_REMAP_AVAIL_MAX" for _f, sp in self.resolved.get("slow", []))
        for pci, sec in parse_smi_q(out).items():
            extra = extract_q_extra(sec)
            if have_hist:
                extra = {k: v for k, v in extra.items() if not k.startswith("DCGM_FI_DEV_BANK_REMAP")}
            if not extra or pci not in self.pci_to_gpu:
                continue
            gpu, uuid = self.pci_to_gpu[pci]
            r = {"gpu": gpu, "uuid": uuid, "pci": self.uuid_to_gpu.get(uuid, (None, pci))[1], "m": extra}
            self.w_slow.write(self.envelope(ts, tier="qfull", gpu=gpu, uuid=uuid, pci=r["pci"], m=extra), ts)
            self.detect_changes(ts, r)
        self.w_slow.flush()
        if self.disk_level != "ok" or ts - self.last_raw < self.cfg["raw_q_interval_s"]:
            return
        self.last_raw = ts
        d = utc_dt(ts)
        p = os.path.join(self.base, "raw", d.strftime("%Y-%m-%d"), "nvidia-smi-q_%s.txt.gz" % d.strftime("%H%M%SZ"))
        if not os.path.isdir(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p))
        with gzip.open(p, "wt", encoding="utf-8") as f:
            f.write(out)

    def tier_housekeeping(self, ts):
        try:
            du = shutil.disk_usage(self.base)
            free_pct = 100.0 * du.free / du.total
        except OSError:
            free_pct = 100.0
        lvl = "ok" if free_pct >= self.cfg["min_free_pct"] else (
            "low" if free_pct >= self.cfg["critical_free_pct"] else "critical")
        if lvl != self.disk_level:
            LOG.warning("disk level %s -> %s (free %.1f%%)", self.disk_level, lvl, free_pct)
            self.disk_level = lvl
        self.status["disk_free_pct"] = round(free_pct, 2)
        for sub in ("samples", "raw", "snapshots", "dcgm", "inventory"):
            purge_old(os.path.join(self.base, sub), self.cfg["retention_days"])
        purge_old(os.path.join(self.base, "events"), self.cfg["event_retention_days"])

    def scrape_dcgm(self, ts):
        try:
            from urllib.request import urlopen
            body = urlopen(self.cfg["dcgm_exporter_url"], timeout=5).read().decode("utf-8", "replace")
        except Exception as e:
            self.consecutive_fail["dcgm"] += 1
            if self.consecutive_fail["dcgm"] in (1, 10, 100):
                LOG.warning("dcgm exporter scrape failed: %s", e)
            return
        self.consecutive_fail["dcgm"] = 0
        pref = tuple(self.cfg["dcgm_metric_prefixes"])
        lines = [l for l in body.splitlines() if l and not l.startswith("#") and l.startswith(pref)]
        if lines:
            self.w_dcgm.write(self.envelope(ts, lines=lines), ts)
            self.w_dcgm.flush()

    @staticmethod
    def _os_release():
        t = read_text("/etc/os-release", "")
        m = re.search(r'^PRETTY_NAME="?([^"\n]*)', t or "", re.M)
        return m.group(1) if m else platform.platform()

    # ---------- change detection & snapshots ----------
    def detect_changes(self, ts, r):
        last_all = self.state.setdefault("last", {})
        last = last_all.setdefault(r["uuid"], {})
        for mid, snap in WATCH.items():
            if mid not in r["m"]:
                continue
            new = r["m"][mid]
            old = last.get(mid, "__unset__")
            if new is None:
                continue  # 缺失不更新基线，也不当恢复
            last[mid] = new
            if old == "__unset__" or old is None or old == new:
                continue
            first = mid not in self.seen[r["uuid"]]
            kw = dict(gpu=r["gpu"], uuid=r["uuid"], pci=r["pci"], metric=mid, old=old, new=new,
                      since_previous_run=True if first else None)
            if isinstance(new, (int, float)) and isinstance(old, (int, float)) and not mid.endswith(
                    ("_PENDING", "_FAILURE", "_MODE", "thresholdExceeded")):
                if new > old:
                    self.event(ts, "counter_increase", delta=new - old, **kw)
                else:
                    self.event(ts, "counter_decrease", delta=new - old,
                               message="counter went backwards: volatile reset (driver reload/GPU reset) or rollover", **kw)
                    continue
            else:
                self.event(ts, "state_change", **kw)
            if snap or (mid in SBE_METRICS and self.cfg["snapshot_on_sbe"]):
                self.snapshot(ts, r["uuid"], "metric_" + mid, kw)
        self.seen[r["uuid"]].update(r["m"])

    def snapshot(self, ts, uuid, reason, detail):
        if self.disk_level == "critical":
            return
        key = uuid or "host"
        if ts - self.last_snapshot.get(key, 0) < self.cfg["snapshot_cooldown_s"]:
            return
        self.last_snapshot[key] = ts
        g = self.uuid_to_gpu.get(uuid, (None, None))
        d = utc_dt(ts)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", reason)[:60]
        sdir = os.path.join(self.base, "snapshots", d.strftime("%Y-%m-%d"),
                            "%s_gpu%s_%s" % (d.strftime("%H%M%SZ"), g[0] if g[0] is not None else "x", safe))
        try:
            os.makedirs(sdir)
        except OSError:
            return
        with io.open(os.path.join(sdir, "trigger.json"), "w", encoding="utf-8") as f:
            json.dump({"ts": iso(ts), "host": self.host, "uuid": uuid, "gpu": g[0], "pci": g[1],
                       "reason": reason, "detail": detail}, f, ensure_ascii=False, indent=1, default=str)
        with gzip.open(os.path.join(sdir, "pre_window_fast.jsonl.gz"), "wt", encoding="utf-8") as f:
            for rec in (self.ring.get(uuid) or []):
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        with io.open(os.path.join(sdir, "kernel_tail.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.kernel_ring))
        args = ["-q"] + (["-i", uuid] if uuid and uuid.startswith(("GPU-", "MIG-")) and g[0] is not None else [])
        rc, out, err = self.smi(*args)
        with gzip.open(os.path.join(sdir, "nvidia-smi-q.txt.gz"), "wt", encoding="utf-8") as f:
            f.write(out if rc == 0 else "rc=%s\n%s" % (rc, err))
        LOG.warning("snapshot written: %s", sdir)

    # ---------- queue from threads ----------
    def drain(self):
        while True:
            try:
                kind, ev = self.q.get_nowait()
            except queue.Empty:
                return
            ts = time.time()
            if kind == "cursor":
                self.state.update(ev)
                continue
            if kind == "kernel":
                self.kernel_ring.append("%s %s" % (ev.get("event_time"), ev.get("message")))
                g = self.pci_to_gpu.get(ev.get("pci"), (None, None))
                ev["gpu"], ev["uuid"] = g[0], g[1]
            etype = ev.pop("type")
            et = parse_iso(ev.get("event_time"))
            if et is not None and et < self.start_ts - 60:
                ev["historical"] = True  # 启动回溯窗口内的旧事件：只记录，不做快照
            rec = self.event(ts, etype, **ev)
            if etype in ("xid", "nvml_event") and rec.get("memory_related") and not rec.get("historical"):
                self.snapshot(ts, rec.get("uuid"), "xid%s_%s" % (rec.get("xid"), rec.get("xid_category")), rec)

    # ---------- main loop ----------
    def run(self, once=False, duration=None):
        for sub in ("samples", "events", "inventory", "raw", "snapshots", "state", "dcgm"):
            p = os.path.join(self.base, sub)
            if not os.path.isdir(p):
                os.makedirs(p)
        self.bid = boot_id()
        if self.state.get("boot_id") and self.state.get("boot_id") != self.bid:
            self.state["missing_uuids"] = []
        self.state["boot_id"] = self.bid
        compress_leftovers(os.path.join(self.base, "samples"), set())
        compress_leftovers(os.path.join(self.base, "dcgm"), set())
        self.detect_fields()
        self.event(time.time(), "collector_start", version=VERSION, config=self.cfg,
                   unsupported_metrics=self.unsupported)
        KernelLogReader(self.cfg, dict(self.state), self.q, self.stop).start()
        if self.cfg["enable_nvml_events"]:
            NvmlEventThread(self.q, self.stop).start()

        tiers = [("housekeeping", 300, self.tier_housekeeping),
                 ("inventory", self.cfg["inventory_interval_s"], self.tier_inventory),
                 ("fast", self.cfg["fast_interval_s"], self.tier_fast),
                 ("slow", self.cfg["slow_interval_s"], self.tier_slow),
                 ("qfull", self.cfg["q_parse_interval_s"], self.tier_qfull)]
        due = {name: 0.0 for name, _i, _f in tiers}
        t_end = time.monotonic() + duration if duration else None
        last_save = 0.0
        while not self.stop.is_set():
            now_m = time.monotonic()
            for name, interval, fn in tiers:
                if now_m < due[name]:
                    continue
                due[name] = now_m + interval
                if name in ("fast", "slow") and self.disk_level == "critical":
                    continue
                ts = time.time()
                t0 = time.monotonic()
                try:
                    fn(ts)
                    self.consecutive_fail[name] = 0
                    self.status["tiers"][name] = {"last_ok": iso(ts), "duration_s": round(time.monotonic() - t0, 3)}
                except Exception as e:
                    self.consecutive_fail[name] += 1
                    n = self.consecutive_fail[name]
                    self.status["tiers"].setdefault(name, {})["last_error"] = "%s: %s" % (iso(ts), e)
                    if n in (1, 3, 10) or n % 100 == 0:
                        self.event(ts, "query_error", tier=name, consecutive=n, message=str(e)[:500])
                    if name == "fast" and n == 3:
                        self.snapshot(ts, None, "nvidia_smi_failing", {"error": str(e)[:500]})
            self.drain()
            self.w_ev.flush()
            if time.time() - last_save > 30:
                self.save_state()
                last_save = time.time()
            if once and all(due[n] > 0 for n in due):
                time.sleep(1.0)  # 给日志线程一次读取机会
                self.drain()
                break
            if t_end and time.monotonic() >= t_end:
                break
            nxt = min(due.values()) - time.monotonic()
            self.stop.wait(max(0.2, min(1.0, nxt)))
        self.shutdown()

    def shutdown(self):
        self.stop.set()
        try:
            self.drain()
            self.event(time.time(), "collector_stop")
        except Exception:
            pass
        for w in (self.w_fast, self.w_slow, self.w_inv, self.w_ev, self.w_dcgm):
            w.close()
        self.save_state()
        LOG.info("collector stopped")


# --------------------------------------------------------------------------------------
def load_config(path, overrides):
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with io.open(path, encoding="utf-8") as f:
            user = json.load(f)
        unknown = set(user) - set(DEFAULT_CONFIG) - {"_comment"}
        if unknown:
            raise SystemExit("unknown config keys: %s" % ", ".join(sorted(unknown)))
        cfg.update({k: v for k, v in user.items() if k != "_comment"})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def setup_logging(cfg, to_stderr=True):
    LOG.setLevel(getattr(logging, str(cfg["log_level"]).upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s")
    logdir = os.path.join(cfg["base_dir"], "logs")
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    fh = logging.handlers.RotatingFileHandler(os.path.join(logdir, "collector.log"), maxBytes=20 * 1024 * 1024,
                                              backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    LOG.addHandler(fh)
    if to_stderr:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        LOG.addHandler(sh)


def main(argv=None):
    ap = argparse.ArgumentParser(description="A100 GPU ECC 故障预测数据采集器")
    ap.add_argument("--config", help="JSON 配置文件路径")
    ap.add_argument("--base-dir", help="数据输出根目录（覆盖配置）")
    ap.add_argument("--nvidia-smi", dest="nvidia_smi", help="nvidia-smi 可执行文件路径")
    ap.add_argument("--fast-interval", dest="fast_interval_s", type=float)
    ap.add_argument("--slow-interval", dest="slow_interval_s", type=float)
    ap.add_argument("--log-level", dest="log_level")
    ap.add_argument("--once", action="store_true", help="每档各采一次后退出（安装自检用）")
    ap.add_argument("--duration", type=float, help="运行指定秒数后退出")
    ap.add_argument("--print-fields", action="store_true", help="打印本机驱动支持/不支持的指标映射后退出")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, {"base_dir": a.base_dir, "nvidia_smi": a.nvidia_smi,
                                 "fast_interval_s": a.fast_interval_s, "slow_interval_s": a.slow_interval_s,
                                 "log_level": a.log_level})
    if a.print_fields:
        logging.basicConfig(level=logging.WARNING)
        c = Collector.__new__(Collector)
        c.cfg, c.unsupported, c.resolved = cfg, {}, {}
        c.detect_fields()
        for tier in ("fast", "slow", "inventory"):
            print("[%s]" % tier)
            for f, sp in c.resolved[tier]:
                print("  %-58s <- %s" % (sp["id"], f))
        print("[unsupported]")
        for k, v in sorted(c.unsupported.items()):
            print("  %-58s %s" % (k, v))
        return 0
    setup_logging(cfg)
    col = Collector(cfg)

    def _sig(signum, _frame):
        LOG.info("signal %s received, stopping", signum)
        col.stop.set()
    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, _sig)
        except (ValueError, AttributeError):
            pass
    LOG.info("gpu-ecc-collector %s starting, base_dir=%s", VERSION, cfg["base_dir"])
    col.run(once=a.once, duration=a.duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
