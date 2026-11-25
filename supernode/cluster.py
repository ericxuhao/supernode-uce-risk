"""Cluster mode: real-time collection from node agents and scheduled risk scans.

Agents (agent.py) register their capabilities and push collector records to the server. This engine:
  * maps every real node onto the supernode topology (racks x nodes x 8 cards): from
    <data_dir>/cluster/inventory.json ({"hostname": "R03-N5"}) if present, otherwise in registration order;
  * chooses the model of every card from the fields its node can deliver (field-driven: address-level memory
    logs -> memory-RAS model; error events / counters -> error-log model);
  * keeps the ingested events (all) and telemetry (last 7 days) in memory, writes the raw records per day
    under <data_dir>/cluster/raw and deletes them after the retention period;
  * scans every card at the collection interval set by the user (default 300 s).
"""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from common import CARDS, DATA, DAY, HOUR, NODES, topology
from engine import Engine
from ingest import ADDR, records_to_tables
from sources import probe_sources

SETTINGS_PATH = DATA / "settings.json"
DEFAULT_SETTINGS = {"interval_s": 300, "retention_days": 30, "default_source": "replay"}
RAW = DATA / "cluster" / "raw"
EV_COLS = ["host", "device", "t", "cause", *ADDR, "job", "n"]

# metric id -> vendor-neutral capability group
GROUPS = [
    ("温度（核心/显存）", ("DCGM_FI_DEV_GPU_TEMP", "DCGM_FI_DEV_MEMORY_TEMP")),
    ("功率", ("DCGM_FI_DEV_POWER_USAGE", "nvsmi.enforced.power.limit")),
    ("时钟与利用率", ("DCGM_FI_DEV_SM_CLOCK", "DCGM_FI_DEV_MEM_CLOCK", "DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_MEM_COPY_UTIL",
                   "DCGM_FI_DEV_FB_USED", "DCGM_FI_DEV_FB_FREE", "nvsmi.pstate")),
    ("可纠正错误计数", ("DCGM_FI_DEV_ECC_SBE_",)),
    ("不可纠正错误计数", ("DCGM_FI_DEV_ECC_DBE_",)),
    ("片上 SRAM / 缓存错误明细", ("nvsmi.ecc.errors.",)),
    ("行重映射计数与状态", ("DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS", "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS",
                       "DCGM_FI_DEV_ROW_REMAP_", "nvsmi.remapped_rows.")),
    ("重映射资源余量", ("DCGM_FI_DEV_BANK_REMAP_AVAIL_",)),
    ("页退役（旧架构）", ("DCGM_FI_DEV_PAGE_RETIRED_",)),
    ("降频原因与累计", ("DCGM_FI_DEV_CLOCKS_EVENT_REASONS", "nvsmi.clocks_event_reasons_counters.")),
    ("PCIe 链路", ("DCGM_FI_DEV_PCIE_",)),
    ("恢复动作", ("DCGM_FI_DEV_GPU_RECOVERY_ACTION",)),
    ("ECC 模式", ("DCGM_FI_DEV_ECC_MODE", "DCGM_FI_DEV_ECC_PENDING")),
]


def load_settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        s.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    return s


def save_settings(s: dict):
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")


def capability_summary(caps: dict) -> dict:
    """per capability group: supported metrics / unavailable on the device; plus the model inputs they give."""
    sup = set(caps.get("metrics_supported", []))
    na = set(caps.get("metrics_na_on_device", [])) | set(caps.get("metrics_unsupported", []))
    groups = []
    for name, prefixes in GROUPS:
        ok = sorted(m for m in sup if m.startswith(prefixes))
        no = sorted(m for m in na if m.startswith(prefixes))
        groups.append({"group": name, "supported": len(ok), "unavailable": len(no),
                       "status": "支持" if ok else ("不支持" if no else "未知")})
    has = lambda g: any(x["group"] == g and x["supported"] for x in groups)          # noqa: E731
    side = [{"group": "内核错误日志", "status": "支持" if caps.get("kernel_log") else "不支持"},
            {"group": "设备事件监听", "status": "支持" if caps.get("device_events") else "不支持"},
            {"group": "片间互连错误", "status": "支持" if caps.get("interconnect") else "不支持"}]
    inputs = {"遥测（温度/功率）": has("温度（核心/显存）") or has("功率"),
              "错误事件": bool(caps.get("kernel_log")) or has("行重映射计数与状态") or has("不可纠正错误计数"),
              "可纠正错误统计": has("可纠正错误计数"),
              "不可纠正错误标签": bool(caps.get("kernel_log")) or has("不可纠正错误计数"),
              "地址级显存日志": False}
    if inputs["遥测（温度/功率）"] and inputs["可纠正错误统计"]:
        scenario = "方案 3：遥测 + 可纠正错误统计" + ("（另含错误事件）" if inputs["错误事件"] else "")
    elif inputs["遥测（温度/功率）"] and inputs["错误事件"]:
        scenario = "方案 1：遥测 + 错误事件"
    elif inputs["可纠正错误统计"]:
        scenario = "方案 2：只有可纠正错误统计"
    elif inputs["遥测（温度/功率）"]:
        scenario = "仅遥测：可监测与归因热/供电应力；没有错误事件和错误计数，无法给出有效的故障风险"
    else:
        scenario = "无可用数据"
    return {"groups": groups + side, "inputs": inputs, "scenario": scenario,
            "model": "ce" if inputs["地址级显存日志"] else "log"}


class ClusterEngine(Engine):
    mode = "cluster"

    def __init__(self, models, meta):
        self.models, self.meta = models, meta
        self.thr = meta["demo_thresholds"]
        self.settings = load_settings()
        topo = topology()
        topo["node_g"] = topo.rack_i * NODES + topo.node_i
        topo["source"], topo["ohost"], topo["odev"] = "", "", ""
        topo["device"] = "C" + topo.slot.astype(str)
        self.cards = topo
        self.events = pd.DataFrame(columns=EV_COLS)
        self.future = pd.DataFrame(columns=EV_COLS)
        self.telemetry = pd.DataFrame(columns=["host", "device", "t", "core_temp", "mem_temp", "power"])
        self.history = pd.DataFrame({"card": pd.Series(dtype=np.int16), "s": pd.Series(dtype=np.int64),
                                     **{f"p_{h}": pd.Series(dtype=np.float32) for h in ("1h", "4h", "1d")}})
        self.episodes = pd.DataFrame(columns=["entity", "t", "first", "ep", "card", "source"])
        self.key, self.host_node, self.agents = {}, {}, {}
        inv = DATA / "cluster" / "inventory.json"
        self.inventory = json.loads(inv.read_text(encoding="utf-8")) if inv.exists() else {}
        self.sim_t = int(time.time())
        self.playing, self.tick_s = True, self.settings["interval_s"]
        self.next_scan = time.time() + 10
        self.live, self.scans, self.current, self.attrib, self.cur_tel = [], [], None, {}, {}
        self.ingest_stats = {"files": 0, "records": 0, "telemetry_rows": 0, "event_rows": 0, "last_file": "",
                             "last_records": 0, "bytes": 0}
        self.probes = probe_sources()
        self.local = SimpleNamespace(ok=False, host="")
        self.local_series = []
        self.lock = threading.RLock()
        self.last_cleanup = 0.0
        hp = DATA / "cluster" / "hosts.json"
        if hp.exists():                      # keep every node at the same topology position across restarts
            self.host_node = json.loads(hp.read_text(encoding="utf-8"))
        self.reload_raw()

    def reload_raw(self):
        """rebuild the in-memory state from the raw records kept on disk (within the retention period)."""
        if not RAW.exists():
            return
        cut = time.strftime("%Y-%m-%d", time.gmtime(time.time() - self.settings["retention_days"] * DAY))
        n = 0
        for d in sorted(p for p in RAW.iterdir() if p.is_dir() and p.name >= cut):
            for f in sorted(d.glob("*.jsonl")):
                with open(f, encoding="utf-8") as fh:
                    recs = [json.loads(x) for x in fh if x.strip()]
                if recs:
                    n += self.ingest(f.stem, recs, persist=False)
        for a in self.agents.values():         # reloaded nodes count as offline until their agent reports again
            a["last_seen"] = 0
        self.ingest_stats["reloaded"] = n

    # ------------------------------------------------------------------ registration / ingest
    def _node_for(self, host):
        if host in self.host_node:
            return self.host_node[host]
        used = set(self.host_node.values())
        want = self.inventory.get(host)
        if want and want not in used:
            node = want
        else:
            free = [n for n in self.cards.node.drop_duplicates() if n not in used]
            if not free:
                return None
            node = free[0]
        self.host_node[host] = node
        hp = DATA / "cluster" / "hosts.json"
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text(json.dumps(self.host_node, ensure_ascii=False, indent=1), encoding="utf-8")
        return node

    def register(self, host, caps, interval_s=None, version=None):
        with self.lock:
            node = self._node_for(host)
            summ = capability_summary(caps or {})
            a = self.agents.setdefault(host, {"host": host, "records": 0, "first_seen": time.time()})
            a.update(node=node, caps=caps, summary=summ, interval_s=interval_s, version=version, last_seen=time.time())
            if node is None:
                a["warning"] = "超节点拓扑已满"
                return a
            devs = [d["index"] for d in (caps or {}).get("devices", [])][:CARDS]
            for i in devs:
                m = (self.cards.node == node) & (self.cards.slot == i)
                self.cards.loc[m, ["source", "ohost", "odev"]] = [summ["model"], host, str(i)]
                self.key[(node, f"C{i}")] = int(self.cards.card[m].iloc[0])
            a["cards"] = len(devs)
            return a

    def ingest(self, host, records, persist=True):
        node = self._node_for(host)
        if node is None:
            return 0
        day = time.strftime("%Y-%m-%d", time.gmtime())
        blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        if persist:
            RAW.joinpath(day).mkdir(parents=True, exist_ok=True)
            with open(RAW / day / f"{host}.jsonl", "a", encoding="utf-8") as f:
                f.write(blob)
        E, T = records_to_tables(records)
        E["host"], T["host"] = node, node
        with self.lock:
            if host not in self.agents or not self.agents[host].get("cards"):
                devs = sorted({int(r["gpu"]) for r in records if r.get("gpu") is not None})
                self.register(host, {"devices": [{"index": i} for i in devs]})
            if len(E):
                self.events = pd.concat([self.events, E.reindex(columns=EV_COLS)], ignore_index=True)
                self.events["t"] = self.events.t.astype(np.int64)
            if len(T):
                self.telemetry = pd.concat([self.telemetry, T], ignore_index=True)
                self.telemetry = self.telemetry.astype({"t": np.int64, "core_temp": float, "mem_temp": float, "power": float})
                self.telemetry = self.telemetry[self.telemetry.t > time.time() - 7 * DAY]
            st = self.ingest_stats
            st.update(records=st["records"] + len(records), telemetry_rows=st["telemetry_rows"] + len(T),
                      event_rows=st["event_rows"] + len(E), last_records=len(records), bytes=st["bytes"] + len(blob),
                      last_file=f"{day}/{host}.jsonl")
            a = self.agents[host]
            a["records"] = a.get("records", 0) + len(records)
            a["bytes"] = a.get("bytes", 0) + len(blob)
            a["last_seen"] = time.time()
            a["last_data"] = time.time()
        return len(records)

    # ------------------------------------------------------------------ scanning
    def registered(self):
        return self.cards[self.cards.source != ""]

    def scan(self, t):
        if self.registered().empty:
            return
        self.sim_t = int(t)
        super().scan(int(t))

    def step(self):
        self.scan(int(time.time()))
        self.next_scan = time.time() + self.settings["interval_s"]

    def run_forever(self):
        while True:
            now = time.time()
            if self.playing and now >= self.next_scan:
                try:
                    self.step()
                except Exception as e:
                    import traceback
                    self.scans.append({"t": int(now), "error": repr(e), "trace": traceback.format_exc()[-1500:]})
                    self.next_scan = now + self.settings["interval_s"]
            if now - self.last_cleanup > 3600:
                self.cleanup()
                self.last_cleanup = now
            self.sim_t = int(now)                    # cluster mode: the monitoring clock is real time
            time.sleep(1.0)

    def cleanup(self):
        """delete raw records older than the retention period."""
        keep = self.settings["retention_days"]
        cut = time.strftime("%Y-%m-%d", time.gmtime(time.time() - keep * DAY))
        if RAW.exists():
            for d in RAW.iterdir():
                if d.is_dir() and d.name < cut:
                    for f in d.iterdir():
                        f.unlink()
                    d.rmdir()

    # ------------------------------------------------------------------ UI queries
    def card_telemetry(self, card, t0, t1):
        c = self.cards[self.cards.card == card].iloc[0]
        T = self.telemetry[(self.telemetry.host == c.node) & (self.telemetry.device == c.device) &
                           (self.telemetry.t > t0) & (self.telemetry.t <= t1)].sort_values("t")
        f = lambda s: [None if pd.isna(v) else round(float(v), 1) for v in s]          # noqa: E731
        return {"t": T.t.astype(int).tolist(), "core_temp": f(T.core_temp), "mem_temp": f(T.mem_temp),
                "power": f(T.power), "busy": [0] * len(T)}

    def explain(self, card, t):
        import fta
        c = self.cards[self.cards.card == card].iloc[0]
        if not c.source:
            return {"risk": {h: 0.0 for h in ("1h", "4h", "1d")}, "level": {h: 0 for h in ("1h", "4h", "1d")},
                    "reason": "", "fta": fta.attribute({}, {}, self.meta["fta_weights"]["log"]), "telemetry": {},
                    "features": {}}
        R, F = self.score(t, cards_only=[card], with_features=True)
        R = self.levels(R)
        tel = self.tel_stats(t)
        a = fta.attribute(F.get(card, {}), tel.get(card, {}), self.meta["fta_weights"][c.source])
        row = R.iloc[0]
        return {"risk": {h: float(row[f"p_{h}"]) for h in ("1h", "4h", "1d")},
                "level": {h: int(row[f"lvl_{h}"]) for h in ("1h", "4h", "1d")}, "reason": row.reason, "fta": a,
                "telemetry": tel.get(card, {}), "features": {}}

    def agent_rows(self):
        now = time.time()
        iv = self.settings["interval_s"]
        out = []
        for h, a in sorted(self.agents.items()):
            seen = a.get("last_seen", 0)
            out.append({"host": h, "node": a.get("node"), "cards": a.get("cards", 0),
                        "online": now - seen < max(3 * iv, 180), "last_seen": int(seen), "records": a.get("records", 0),
                        "interval_s": a.get("interval_s"), "summary": a.get("summary"), "warning": a.get("warning")})
        return out

    def storage_estimate(self):
        """bytes per day at the current interval, from the measured record size (or a typical full record)."""
        cards = int((self.cards.source != "").sum())
        recs = self.ingest_stats["records"]
        per = self.ingest_stats["bytes"] / recs if recs else 1200.0
        per_day = cards * (DAY / self.settings["interval_s"]) * per
        return {"cards": cards, "bytes_per_record": round(per), "mb_per_day": round(per_day / 2**20, 1),
                "gb_retained": round(per_day * self.settings["retention_days"] / 2**30, 2),
                "mb_per_day_1024": round(1024 * (DAY / self.settings["interval_s"]) * per / 2**20, 1)}
