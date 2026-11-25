"""Data acquisition: probe local collectors, read collector spools, or replay open data sets.

Order of preference (see config.json 'sources'):
  1. local collectors on this host: nvidia-smi / dcgmi / DCGM exporter endpoint / kernel log (journalctl)
     and gpu-ecc-collector output directories (collectors/gpu_ecc, JSONL);
  2. replay of open data sets, written to the spool in the same JSONL envelope as the collector, so the
     processing path is identical:
       - error logs: OLCF Summit kernel error log (per card, real timestamps),
       - address-level memory RAS logs: Calchas HBM log (CE / UER / UEO with stack/SID/PC/BG/bank/row/col),
       - telemetry + job log: Summit artifact 10-s node telemetry (Jan 2020), 5-min means, replayed day by
         day onto the demo nodes; job start / end from the allocation id.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from common import CFG, DATA, DEMO_DIR, DAY, SPOOL

TEL_STEP = CFG["scan"]["telemetry_step_s"]
SLOTS_PER_DAY = DAY // TEL_STEP


# ----------------------------------------------------------------------------- local probes
def probe_sources() -> list[dict]:
    out = []
    for name, cmd, label in [("nvidia-smi", "nvidia-smi", "计算卡管理工具"), ("dcgmi", "dcgmi", "数据中心管理工具"),
                             ("journalctl", "journalctl", "内核日志")]:
        path = shutil.which(cmd)
        ok = False
        if path and name != "journalctl":
            try:
                ok = subprocess.run([path, "-L"] if name == "nvidia-smi" else [path, "discovery", "-l"],
                                    capture_output=True, timeout=5).returncode == 0
            except Exception:
                ok = False
        elif path:
            ok = True
        out.append({"source": label, "probe": label, "found": bool(path), "usable": ok,
                    "detail": ("已检测到，可采集" if ok else "已安装但无法访问") if path else "本机未安装"})
    for url in CFG["sources"]["dcgm_exporter_urls"]:
        ok = False
        try:
            with urllib.request.urlopen(url, timeout=1.5) as r:
                ok = r.status == 200
        except Exception:
            pass
        out.append({"source": "遥测导出端点", "probe": url.split("//")[-1], "found": ok, "usable": ok, "detail": "可访问" if ok else "无响应"})
    for i, d in enumerate(CFG["sources"]["collector_spool_dirs"]):
        p = Path(d)
        ok = p.exists() and any(p.glob("samples/*/fast-*.jsonl*"))
        out.append({"source": "采集器输出目录", "probe": f"配置路径 {i + 1}", "found": p.exists(), "usable": ok,
                    "detail": "有样本" if ok else ("目录为空" if p.exists() else "不存在")})
    return out


# ----------------------------------------------------------------------------- local live collection
LOCAL_FIELDS = [("temperature.gpu", "DCGM_FI_DEV_GPU_TEMP", "核心温度 ℃"),
                 ("temperature.memory", "DCGM_FI_DEV_MEMORY_TEMP", "显存温度 ℃"),
                 ("power.draw", "DCGM_FI_DEV_POWER_USAGE", "功率 W"),
                 ("utilization.gpu", "DCGM_FI_DEV_GPU_UTIL", "计算利用率 %"),
                 ("memory.used", "DCGM_FI_DEV_FB_USED", "显存占用 MiB"),
                 ("ecc.errors.corrected.volatile.total", "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL", "可纠正错误计数"),
                 ("ecc.errors.uncorrected.volatile.total", "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL", "不可纠正错误计数"),
                 ("retired_pages.pending", "DCGM_FI_DEV_RETIRED_PENDING", "待退役页")]


class LocalCollector:
    """live samples of the cards on this host through the vendor management tool, written in the collector
    JSONL envelope (unsupported fields are null, never 0)."""

    def __init__(self):
        import socket
        self.tool = shutil.which("nvidia-smi")
        self.host = socket.gethostname()
        self.ok = False
        if self.tool:
            try:
                self.ok = bool(self.sample())
            except Exception:
                self.ok = False

    def sample(self) -> list[dict]:
        if not self.tool:
            return []
        q = ",".join(["index", "uuid"] + [f for f, _, _ in LOCAL_FIELDS])
        r = subprocess.run([self.tool, f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=10)
        if r.returncode != 0:
            return []
        now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        recs = []
        for line in r.stdout.strip().splitlines():
            v = [x.strip() for x in line.split(",")]
            m, na = {}, {}
            for (f, key, _), x in zip(LOCAL_FIELDS, v[2:]):
                try:
                    m[key] = float(x)
                except ValueError:
                    m[key], na[key] = None, x
            recs.append({"ts": now, "host": self.host, "tier": "fast", "gpu": int(v[0]), "uuid": v[1], "m": m, "na": na})
        SPOOL.mkdir(parents=True, exist_ok=True)
        with open(SPOOL / f"local-{now[:10]}.jsonl", "a", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return recs


# ----------------------------------------------------------------------------- telemetry library
class TelemetryLibrary:
    """5-minute node-day profiles from real 10-s telemetry; demo node n on day d replays profile
    (n * 37 + d * 11) mod P; cards 0-5 use the profile's 6 devices, cards 6-7 use devices 0-1 of the next
    profile."""

    def __init__(self, path: Path = DEMO_DIR / "telemetry_library.npz"):
        z = np.load(path, allow_pickle=True)
        self.power, self.core, self.mem, self.busy, self.alloc = z["power"], z["core"], z["mem"], z["busy"], z["alloc"]
        self.P = self.power.shape[0]

    def profile(self, node: int, day: int) -> int:
        return (node * 37 + day * 11) % self.P

    def window(self, nodes: np.ndarray, slots: np.ndarray, t0: int, t1: int):
        """telemetry of cards (node index, slot) for 5-min steps in (t0, t1]: arrays [cards, steps]."""
        steps = np.arange((t0 // TEL_STEP) + 1, (t1 // TEL_STEP) + 1)
        day, k = steps // SLOTS_PER_DAY, steps % SLOTS_PER_DAY
        prof = (nodes[:, None] * 37 + day[None, :] * 11) % self.P
        dev = slots[:, None] % 6
        prof = np.where(slots[:, None] >= 6, (prof + 1) % self.P, prof)
        kk = np.broadcast_to(k, prof.shape)
        return (steps * TEL_STEP, self.core[prof, kk, dev].astype(np.float32), self.mem[prof, kk, dev].astype(np.float32),
                self.power[prof, kk, dev].astype(np.float32), self.busy[prof, kk].astype(bool), self.alloc[prof, kk])


def build_telemetry_library(src_dir: Path, out: Path):
    files = sorted(src_dir.glob("2020*.parquet"))
    pw = ["p0_gpu0_power", "p0_gpu1_power", "p0_gpu2_power", "p1_gpu0_power", "p1_gpu1_power", "p1_gpu2_power"]
    ct = [f"gpu{i}_core_temp" for i in range(6)]
    mt = [f"gpu{i}_mem_temp" for i in range(6)]
    P, C, M, B, A = [], [], [], [], []
    for f in files:
        d = pd.read_parquet(f, columns=["timestamp", "hostname", "allocation_id", *pw, *ct, *mt])
        d["k"] = (d.timestamp.dt.hour * 3600 + d.timestamp.dt.minute * 60 + d.timestamp.dt.second) // TEL_STEP
        g = d.groupby(["hostname", "k"])
        num = g[pw + ct + mt].mean()
        alloc = g.allocation_id.max()
        for h in num.index.get_level_values(0).unique():
            x = num.loc[h].reindex(range(SLOTS_PER_DAY)).interpolate(limit_direction="both")
            if x.isna().any().any():
                continue
            a = alloc.loc[h].reindex(range(SLOTS_PER_DAY)).ffill().bfill().fillna(0).to_numpy(np.int64)
            P.append(x[pw].to_numpy(np.float16)); C.append(x[ct].to_numpy(np.float16)); M.append(x[mt].to_numpy(np.float16))
            B.append(a > 0); A.append(a)
        print(f"   {f.name}: {len(P)} profiles", flush=True)
    np.savez_compressed(out, power=np.stack(P), core=np.stack(C), mem=np.stack(M), busy=np.stack(B), alloc=np.stack(A))


# ----------------------------------------------------------------------------- replay -> spool
class Replayer:
    """writes collector-format JSONL for (t0, t1] into the spool: one file per simulated hour."""

    def __init__(self, cards: pd.DataFrame, events: pd.DataFrame, tel: TelemetryLibrary):
        self.cards, self.tel = cards, tel
        self.events = events.sort_values("t").reset_index(drop=True)
        self.et = self.events.t.to_numpy()
        SPOOL.mkdir(parents=True, exist_ok=True)

    def emit(self, t0: int, t1: int) -> Path:
        recs = []
        c = self.cards
        steps, core, mem, power, busy, alloc = self.tel.window(c.node_g.to_numpy(), c.slot.to_numpy(), t0, t1)
        iso = pd.to_datetime(steps, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
        hosts, slots = c.node.to_numpy(), c.slot.to_numpy()
        for j, s in enumerate(iso):
            for i in range(len(c)):
                recs.append({"ts": s, "host": hosts[i], "tier": "fast", "gpu": int(slots[i]),
                             "m": {"DCGM_FI_DEV_GPU_TEMP": round(float(core[i, j]), 1),
                                   "DCGM_FI_DEV_MEMORY_TEMP": round(float(mem[i, j]), 1),
                                   "DCGM_FI_DEV_POWER_USAGE": round(float(power[i, j]), 1)}})
        # job log: allocation changes per node (first card of the node carries the node-level record)
        first = c.slot.to_numpy() == 0
        for i in np.flatnonzero(first):
            a = alloc[i]
            for j in range(1, len(a)):
                if a[j] != a[j - 1]:
                    if a[j - 1] > 0:
                        recs.append({"ts": iso[j], "host": hosts[i], "type": "job_end", "gpu": 0, "job": int(a[j - 1])})
                    if a[j] > 0:
                        recs.append({"ts": iso[j], "host": hosts[i], "type": "job_start", "gpu": 0, "job": int(a[j])})
        lo, hi = np.searchsorted(self.et, t0, "right"), np.searchsorted(self.et, t1, "right")
        for r in self.events.iloc[lo:hi].itertuples(index=False):
            s = pd.Timestamp(r.t, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            if r.kind == "xid":
                recs.append({"ts": s, "host": r.host, "type": "xid", "gpu": int(r.device[1:]), "xid": int(r.code),
                             "message": f"RAS: device error code {int(r.code)}"})
            else:
                recs.append({"ts": s, "host": r.host, "type": "mem_ras", "gpu": int(r.device[1:]), "severity": r.code,
                             **{k: int(getattr(r, k)) for k in ("stack", "sid", "pc", "bg", "bank", "row", "col")}})
        path = SPOOL / f"{pd.Timestamp(t1, unit='s', tz='UTC'):%Y%m%dT%H%M}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    @staticmethod
    def cleanup(keep_hours: int):
        files = sorted(p for p in SPOOL.glob("*.jsonl") if not p.name.startswith("local-"))
        for p in files[:-keep_hours]:
            try:
                p.unlink()
            except OSError:
                pass
