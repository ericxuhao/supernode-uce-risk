"""Shared constants: topology naming, time windows, paths, vendor-neutral cause mapping.

Every text shown in the UI is vendor neutral: devices are '计算卡', error codes are mapped to causes
through configs/error_taxonomy.json (vendor codes stay inside that file and the collectors).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent                               # repository / release root
if importlib.util.find_spec("uce_risk") is None:   # not pip-installed: use the copy next to this package
    sys.path.insert(0, str(ROOT))

# configuration: SUPERNODE_CONFIG overrides the bundled config.json; relative paths are resolved against
# SUPERNODE_HOME (default: the release root)
CFG_PATH = Path(os.environ.get("SUPERNODE_CONFIG", PKG / "config.json"))
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
HOME = Path(os.environ.get("SUPERNODE_HOME", ROOT))


def cfg_path(value) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return p if p.is_absolute() else (HOME / p).resolve()


DATA = cfg_path(os.environ.get("SUPERNODE_DATA_DIR", CFG["data_dir"]))          # runtime: settings, cluster data, spool
MODEL_DIR = cfg_path(os.environ.get("SUPERNODE_MODEL_DIR", CFG.get("model_dir", CFG["data_dir"])))   # models.joblib, build_meta.json
DEMO_DIR = cfg_path(os.environ.get("SUPERNODE_DEMO_DIR", CFG.get("demo_dir", CFG["data_dir"])))      # replay data (optional)
SPOOL = DATA / "spool"
_tax = PKG / "error_taxonomy.json"
TAXONOMY = json.loads((_tax if _tax.exists() else ROOT / "configs" / "error_taxonomy.json").read_text(encoding="utf-8"))


def replay_available() -> bool:
    return all((DEMO_DIR / f).exists() for f in ("cards.parquet", "events_demo.parquet", "history_risk.parquet",
                                                  "episodes.parquet", "telemetry_library.npz"))

RACKS, NODES, CARDS = CFG["topology"]["racks"], CFG["topology"]["nodes_per_rack"], CFG["topology"]["cards_per_node"]
N_CARDS = RACKS * NODES * CARDS
HOUR, DAY = 3600, 86400


def ts(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp())


DEMO_START, LIVE_START, DEMO_END = (ts(CFG["window"][k]) for k in ("demo_start", "live_start", "demo_end"))


def node_name(r: int, n: int) -> str:
    return f"R{r + 1:02d}-N{n + 1}"


def card_name(r: int, n: int, c: int) -> str:
    return f"{node_name(r, n)}-C{c}"


def topology() -> pd.DataFrame:
    rows = [(i, r, n, c, f"R{r + 1:02d}", node_name(r, n), card_name(r, n, c))
            for i, (r, n, c) in enumerate((r, n, c) for r in range(RACKS) for n in range(NODES) for c in range(CARDS))]
    return pd.DataFrame(rows, columns=["card", "rack_i", "node_i", "slot", "rack", "node", "name"])


# ----------------------------------------------------------------------------- causes
UCE_CAUSE = "MEMORY_UNCORRECTABLE_ERROR"
CE_CAUSE = "MEMORY_CORRECTABLE_ERROR"
EXTRA_CAUSES = {      # demo-only causes not in the shared taxonomy (address-level memory RAS logs)
    "MEMORY_CORRECTABLE_ERROR": {"name_zh": "设备内存可纠正错误", "layer": "hardware_memory", "severity": "info",
                                 "action": "持续观察单元/行/列分布；出现结构性模式时安排诊断。"},
    "MEMORY_UNCORRECTABLE_DEFERRED": {"name_zh": "巡检发现的不可纠正错误", "layer": "hardware_memory", "severity": "major",
                                      "action": "安排停机窗口执行内存诊断与隔离，避免作业读到坏块。"},
    "JOB_START": {"name_zh": "作业开始", "layer": "application", "severity": "info", "action": ""},
    "JOB_END": {"name_zh": "作业结束", "layer": "application", "severity": "info", "action": ""},
}
CAUSES = {**TAXONOMY["causes"], **EXTRA_CAUSES}
XID_TO_CAUSE = {int(k): v for k, v in TAXONOMY["vendor_codes"]["nvidia_xid"].items()}
RAS_TO_CAUSE = {"CE": CE_CAUSE, "UER": UCE_CAUSE, "UEO": "MEMORY_UNCORRECTABLE_DEFERRED"}
LAYER_ZH = TAXONOMY["layers"]


def cause_zh(c: str) -> str:
    return CAUSES.get(c, {}).get("name_zh", c)


def cause_of_xid(x) -> str:
    try:
        return XID_TO_CAUSE.get(int(x), "UNKNOWN_ERROR")
    except (TypeError, ValueError):
        return "UNKNOWN_ERROR"
