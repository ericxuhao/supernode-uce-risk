"""Fault-tree attribution for a risky card.

Top event: 计算卡不可纠正错误. Five OR gates (fault categories), each with basic events that are read
from the model features (error history, memory errors, events, same-node context) or from live
telemetry. The weight of a basic event is the conditional probability p = P(UCE within 24 h | event
present), estimated on the training samples (data-driven fault tree); its relative risk RR is reported as
evidence. Basic events without labelled history (telemetry) use an expert-prior RR times the base rate and
are marked as such. Gate probability = 1 - prod(1 - p_i) over active events (OR gate); the category is the
gate with the largest probability and the root cause the basic event with the largest p.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GATES = {
    "MEM": {"name": "显存子系统故障", "action": "迁移作业并隔离该计算卡；执行显存诊断，检查行重映射/页退役余量，必要时更换。"},
    "DEV": {"name": "芯片本体/互连故障", "action": "检查供电、链路与固件日志，复位或重新插拔；反复出现时报修更换。"},
    "THERM": {"name": "热与供电应力", "action": "检查节点风扇、液冷流量与进风温度；降低功耗上限或迁移高负载作业。"},
    "NODE": {"name": "节点/环境共因", "action": "排查同节点主板、电源与散热；考虑整节点下线检修。"},
    "SW": {"name": "软件/作业诱发", "action": "核对作业代码、计算库与驱动版本；与作业所有者确认后再判断硬件。"},
}


def _ev(f, code, win):
    return max(f.get(f"ev__{code}_{win}", 0) or 0, f.get(f"ev__bg_{code}_{win}", 0) or 0)


def _sum(f, prefix, suffix):
    return sum((v or 0) for k, v in f.items() if k.startswith(prefix) and k.endswith(suffix))


def _bank(f):
    return max([(v or 0) for k, v in f.items() if k.startswith("sp__") and ("_multi_row" in k or "_multi_col" in k)
                and not k.startswith(("sp__row_", "sp__col_"))] or [0])


# id, gate, name, source, test(features, telemetry) -> bool, evidence text, expert prior RR (telemetry only)
BASIC = [
    ("M_RECUR", "MEM", "近期发生过不可纠正错误（复发）", "model", lambda f, t: (f.get("hi__h_since_last") or 1e9) <= 168,
     lambda f, t: f"距上次不可纠正错误 {f.get('hi__h_since_last', 0):.0f} 小时", None),
    ("M_MULTI", "MEM", "多次不可纠正错误", "model", lambda f, t: (f.get("hi__ep_cum") or 0) >= 2,
     lambda f, t: f"累计 {f.get('hi__ep_cum', 0):.0f} 次", None),
    ("M_RETIRE", "MEM", "显存页退役/行重映射", "model", lambda f, t: _ev(f, "MEMORY_PAGE_RETIRED", "30d") > 0,
     lambda f, t: f"近30天 {_ev(f, 'MEMORY_PAGE_RETIRED', '30d'):.0f} 次", None),
    ("M_RETIRE_FAIL", "MEM", "页退役失败（修复资源耗尽）", "model", lambda f, t: _ev(f, "MEMORY_PAGE_RETIRE_FAILED", "30d") > 0,
     lambda f, t: f"近30天 {_ev(f, 'MEMORY_PAGE_RETIRE_FAILED', '30d'):.0f} 次", None),
    ("M_CE_STORM", "MEM", "可纠正错误率过高", "model",
     lambda f, t: _ev(f, "MEMORY_CORRECTABLE_ERROR_STORM", "30d") > 0 or (f.get("ce__storm_min_1d") or 0) > 0,
     lambda f, t: "可纠正错误风暴", None),
    ("M_CE_BURST", "MEM", "可纠正错误突发/加速", "model",
     lambda f, t: (f.get("ce__n_1h") or 0) > 0 and (f.get("ce__accel_1h_vs_1d") or 0) > 2,
     lambda f, t: f"近1小时 {f.get('ce__n_1h', 0):.0f} 次，速率为日均的 {f.get('ce__accel_1h_vs_1d', 0):.1f} 倍", None),
    ("M_ROWCOL", "MEM", "行/列结构性故障", "model",
     lambda f, t: (f.get("sp__row_multi_cell") or 0) + (f.get("sp__col_multi_cell") or 0) > 0,
     lambda f, t: f"多单元行 {f.get('sp__row_multi_cell', 0):.0f}，多单元列 {f.get('sp__col_multi_cell', 0):.0f}", None),
    ("M_BANK", "MEM", "Bank 级故障", "model", lambda f, t: _bank(f) > 0, lambda f, t: f"多行/多列 bank {_bank(f):.0f} 个", None),
    ("M_DEFERRED", "MEM", "巡检发现不可纠正错误", "model", lambda f, t: _ev(f, "MEMORY_UNCORRECTABLE_DEFERRED", "30d") > 0,
     lambda f, t: f"近30天 {_ev(f, 'MEMORY_UNCORRECTABLE_DEFERRED', '30d'):.0f} 次", None),
    ("D_LOST", "DEV", "掉卡（设备丢失）", "model", lambda f, t: _ev(f, "DEVICE_LOST", "30d") > 0,
     lambda f, t: f"近30天 {_ev(f, 'DEVICE_LOST', '30d'):.0f} 次", None),
    ("D_LINK", "DEV", "片间互连链路错误", "model", lambda f, t: _ev(f, "INTERCONNECT_LINK_ERROR", "30d") > 0,
     lambda f, t: f"近30天 {_ev(f, 'INTERCONNECT_LINK_ERROR', '30d'):.0f} 次", None),
    ("D_MCU", "DEV", "片上管理控制器告警/停机", "model",
     lambda f, t: _ev(f, "MICROCONTROLLER_WARNING", "30d") + _ev(f, "MICROCONTROLLER_HALT", "30d") > 0,
     lambda f, t: "管理控制器异常", None),
    ("T_MEMHOT", "THERM", "显存温度偏高", "telemetry", lambda f, t: (t.get("mem_z") or 0) >= 3,
     lambda f, t: f"近1小时显存温度 {t.get('mem_temp', 0):.1f}℃，高于全超节点 {t.get('mem_z', 0):.1f} 个标准差", 3.0),
    ("T_COREHOT", "THERM", "核心温度偏高", "telemetry", lambda f, t: (t.get("core_z") or 0) >= 3,
     lambda f, t: f"近1小时核心温度 {t.get('core_temp', 0):.1f}℃，高于全超节点 {t.get('core_z', 0):.1f} 个标准差", 2.0),
    ("T_POWER", "THERM", "功率剧烈波动", "telemetry", lambda f, t: (t.get("power_cv") or 0) >= 0.35,
     lambda f, t: f"近1小时功率变异系数 {t.get('power_cv', 0):.2f}", 1.5),
    ("T_NODEHOT", "THERM", "节点整体散热异常", "telemetry", lambda f, t: (t.get("node_hot_cards") or 0) >= 4,
     lambda f, t: f"同节点 {t.get('node_hot_cards', 0):.0f} 张卡显存温度偏高", 2.5),
    ("N_UCE", "NODE", "同节点其他计算卡近期不可纠正错误", "model", lambda f, t: (f.get("nbh__ep_30d") or 0) > 0,
     lambda f, t: f"近30天同节点 {f.get('nbh__ep_30d', 0):.0f} 次", None),
    ("N_HW", "NODE", "同节点错误事件频发", "model", lambda f, t: (f.get("nbe__ev_7d") or 0) >= 20,
     lambda f, t: f"近7天同节点 {f.get('nbe__ev_7d', 0):.0f} 条", None),
    ("S_APP", "SW", "应用类错误频发", "model", lambda f, t: _sum(f, "ev__bg_", "_1d") > 0,
     lambda f, t: f"近1天应用类错误 {_sum(f, 'ev__bg_', '_1d'):.0f} 条", None),
    ("S_DRV", "SW", "驱动/固件异常", "model",
     lambda f, t: sum(_ev(f, c, "30d") for c in ("DRIVER_FIRMWARE_ERROR", "CONTEXT_SWITCH_FAULT", "COMMAND_STREAM_CORRUPTED")) > 0,
     lambda f, t: "驱动/固件错误", None),
]
BASIC_BY_ID = {b[0]: b for b in BASIC}


def estimate_weights(S: pd.DataFrame, horizon_s: int = 86400) -> dict:
    """attributable fraction of every model-based basic event on training samples (vectorised)."""
    y = (S.lead.to_numpy() <= horizon_s)
    base = y.mean()
    g = lambda c: S[c].to_numpy(np.float64) if c in S else np.zeros(len(S))           # noqa: E731
    ev = lambda code, w: np.maximum(g(f"ev__{code}_{w}"), g(f"ev__bg_{code}_{w}"))     # noqa: E731
    bank_cols = [c for c in S.columns if c.startswith("sp__") and ("_multi_row" in c or "_multi_col" in c)
                 and not c.startswith(("sp__row_", "sp__col_"))]
    bg1d = [c for c in S.columns if c.startswith("ev__bg_") and c.endswith("_1d")]
    act = {
        "M_RECUR": np.nan_to_num(g("hi__h_since_last"), nan=1e9) <= 168,
        "M_MULTI": g("hi__ep_cum") >= 2,
        "M_RETIRE": ev("MEMORY_PAGE_RETIRED", "30d") > 0,
        "M_RETIRE_FAIL": ev("MEMORY_PAGE_RETIRE_FAILED", "30d") > 0,
        "M_CE_STORM": (ev("MEMORY_CORRECTABLE_ERROR_STORM", "30d") > 0) | (g("ce__storm_min_1d") > 0),
        "M_CE_BURST": (g("ce__n_1h") > 0) & (g("ce__accel_1h_vs_1d") > 2),
        "M_ROWCOL": (g("sp__row_multi_cell") + g("sp__col_multi_cell")) > 0,
        "M_BANK": (S[bank_cols].max(axis=1).to_numpy() > 0) if bank_cols else np.zeros(len(S), bool),
        "M_DEFERRED": ev("MEMORY_UNCORRECTABLE_DEFERRED", "30d") > 0,
        "D_LOST": ev("DEVICE_LOST", "30d") > 0,
        "D_LINK": ev("INTERCONNECT_LINK_ERROR", "30d") > 0,
        "D_MCU": (ev("MICROCONTROLLER_WARNING", "30d") + ev("MICROCONTROLLER_HALT", "30d")) > 0,
        "N_UCE": g("nbh__ep_30d") > 0,
        "N_HW": g("nbe__ev_7d") >= 20,
        "S_APP": (S[bg1d].sum(axis=1).to_numpy() > 0) if bg1d else np.zeros(len(S), bool),
        "S_DRV": sum(ev(c, "30d") for c in ("DRIVER_FIRMWARE_ERROR", "CONTEXT_SWITCH_FAULT", "COMMAND_STREAM_CORRUPTED")) > 0,
    }
    w = {}
    for bid, a in act.items():
        n1, n0 = a.sum(), (~a).sum()
        if n1 < 20 or n0 == 0 or y[~a].mean() == 0:
            w[bid] = {"af": None, "rr": None, "support": int(n1)}
            continue
        rr = y[a].mean() / y[~a].mean()
        w[bid] = {"af": float(np.clip((rr - 1) / rr, 0, 0.95)) if rr > 1 else 0.0, "rr": float(rr), "support": int(n1),
                  "p_uce": float(y[a].mean())}
    w["_base"] = float(base)
    return w


def attribute(f: dict, tel: dict, weights: dict) -> dict:
    """evaluate the fault tree for one card. f: feature dict, tel: live telemetry stats."""
    base = weights.get("_base", 3e-4)
    events, gates = [], {g: [] for g in GATES}
    for bid, gate, name, src, test, evid, prior in BASIC:
        try:
            on = bool(test(f, tel))
        except Exception:
            on = False
        wt = weights.get(bid, {}) if src == "model" else {}
        if src == "model":
            p, rr = wt.get("p_uce"), wt.get("rr")
            basis = "历史数据估计" if p is not None else "数据不足（按 2 倍基线）"
            if p is None:
                p, rr = 2 * base, None
        else:
            p, rr, basis = min(0.5, base * prior), prior, "专家先验"
        item = {"id": bid, "gate": gate, "name": name, "active": on, "p": float(p), "basis": basis,
                "rr": None if rr is None else round(float(rr), 1), "evidence": evid(f, tel) if on else ""}
        events.append(item)
        if on:
            gates[gate].append(item)
    gate_p = {g: 1 - float(np.prod([1 - e["p"] for e in v])) if v else 0.0 for g, v in gates.items()}
    top = 1 - float(np.prod([1 - q for q in gate_p.values()]))
    if any(gate_p.values()):
        cat = max(gate_p, key=gate_p.get)
        root = max(gates[cat], key=lambda e: e["p"])
        category, cause, action = GATES[cat]["name"], root["name"], GATES[cat]["action"]
    else:
        cat, category, cause = None, "未见明显前兆", "模型依据整体统计给出风险，无单一主因"
        action = "保持观察；如风险持续上升，安排巡检。"
    return {"category": category, "category_id": cat, "root_cause": cause, "action": action, "top": round(top, 4),
            "base": base,
            "gates": [{"id": g, "name": GATES[g]["name"], "p": round(gate_p[g], 4),
                       "events": [e for e in events if e["gate"] == g]} for g in GATES]}
