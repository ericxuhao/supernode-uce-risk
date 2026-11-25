"""Live engine: simulated clock, scheduled collection -> processing -> risk scan -> fault-tree attribution.

Each tick (default every 3 s of real time = 1 simulated hour):
  1. collect: the replayer writes the hour's records into the spool (collector JSONL format)
  2. process: the spool file is parsed into vendor-neutral event / telemetry tables (ingest.py)
  3. scan: for every card, risk of a UCE within 1 h / 4 h / 24 h (field-driven input -> model), alarm
     level from the thresholds fixed on the validation period
  4. attribute: fault-tree analysis for every card above the watch level
History before live_start comes from the offline build (same models, hourly scans).
"""
from __future__ import annotations

import json
import threading
import time

import joblib
import numpy as np
import pandas as pd

import fta
from common import CFG, DATA, DEMO_DIR, MODEL_DIR, DAY, DEMO_END, DEMO_START, HOUR, LIVE_START, cause_zh, CAUSES
from ingest import make_input, read_spool, records_to_tables
from sources import LOCAL_FIELDS, LocalCollector, Replayer, TelemetryLibrary, probe_sources
from uce_risk import features
from uce_risk.model import ensemble_predict

HZ = CFG["horizons"]
LEVELS = ["正常", "关注", "高风险"]


class Engine:
    def __init__(self):
        self.cards = pd.read_parquet(DEMO_DIR / "cards.parquet")
        self.models = joblib.load(MODEL_DIR / "models.joblib")
        self.meta = json.loads((MODEL_DIR / "build_meta.json").read_text(encoding="utf-8"))
        ev = pd.read_parquet(DEMO_DIR / "events_demo.parquet")
        self.future = ev[ev.t >= LIVE_START]                        # what the replayer will "collect"
        self.events = ev[ev.t < LIVE_START][["host", "device", "t", "cause", "stack", "sid", "pc", "bg", "bank", "row",
                                             "col"]].copy()
        self.events["job"] = np.nan
        self.history = pd.read_parquet(DEMO_DIR / "history_risk.parquet")
        self.episodes = pd.read_parquet(DEMO_DIR / "episodes.parquet")
        self.tel = TelemetryLibrary()
        self.replayer = Replayer(self.cards, self.future, self.tel)
        self.telemetry = pd.DataFrame(columns=["host", "device", "t", "core_temp", "mem_temp", "power"])
        self.key = {(h, d): c for h, d, c in zip(self.cards.node, self.cards.device, self.cards.card)}
        self.sim_t = LIVE_START
        self.playing, self.tick_s = True, CFG["scan"]["tick_real_s"]
        self.live = []                           # scan results (DataFrames)
        self.scans = []                          # scan log
        self.current = None                      # latest scan per card
        self.attrib = {}                         # card -> fault-tree result of the latest scan
        self.ingest_stats = {"files": 0, "records": 0, "telemetry_rows": 0, "event_rows": 0, "last_file": "",
                             "last_records": 0}
        self.probes = probe_sources()
        self.local = LocalCollector()             # real samples of this host's cards (if any)
        self.local_series = []
        self.lock = threading.RLock()
        self.thr = self.meta["demo_thresholds"]          # policy.thresholds: high / watch per horizon
        # warm up: last day of telemetry so the first scan has a telemetry window
        self._collect(LIVE_START - DAY, LIVE_START)
        self.scan(LIVE_START)

    # ------------------------------------------------------------------ pipeline steps
    def _collect(self, t0, t1):
        for a in range(t0, t1, HOUR):
            path = self.replayer.emit(a, a + HOUR)
            recs = read_spool([path])
            E, T = records_to_tables(recs)
            self.ingest_stats.update(files=self.ingest_stats["files"] + 1, records=self.ingest_stats["records"] + len(recs),
                                     telemetry_rows=self.ingest_stats["telemetry_rows"] + len(T),
                                     event_rows=self.ingest_stats["event_rows"] + len(E), last_file=path.name,
                                     last_records=len(recs))
            self.events = pd.concat([self.events, E], ignore_index=True)
            self.telemetry = pd.concat([self.telemetry, T], ignore_index=True)
        self.telemetry = self.telemetry[self.telemetry.t > t1 - 2 * DAY]
        Replayer.cleanup(CFG["scan"]["spool_keep_hours"])

    def tel_stats(self, t, cards=None, tel=None):
        """per-card telemetry statistics over (t - 1 h, t] plus fleet z-scores."""
        if tel is None:
            T = self.telemetry[(self.telemetry.t > t - HOUR) & (self.telemetry.t <= t)]
        else:
            T = tel
        if T.empty:
            return {}
        g = T.groupby(["host", "device"]).agg(core_temp=("core_temp", "mean"), mem_temp=("mem_temp", "mean"),
                                               power=("power", "mean"), power_sd=("power", "std")).reset_index()
        for c in ("core_temp", "mem_temp"):
            sd = g[c].std() or 1.0
            g[c[:-5] + "_z"] = (g[c] - g[c].mean()) / sd
        g["power_cv"] = g.power_sd / g.power.clip(lower=1)
        g["hot"] = g.mem_z >= 2
        g["node_hot_cards"] = g.groupby("host").hot.transform("sum")
        g["card"] = [self.key.get((h, d)) for h, d in zip(g.host, g.device)]
        return {int(r.card): {k: float(getattr(r, k)) for k in ("core_temp", "mem_temp", "power", "core_z", "mem_z",
                                                                "power_cv", "node_hot_cards")}
                for r in g.itertuples() if r.card is not None}

    def score(self, t, cards_only=None, with_features=False):
        """risk of every card (or the given cards) at time t from the events ingested before t."""
        out, feats = [], {}
        for src, pred in self.models.items():
            sub = self.cards[self.cards.source == src]
            if cards_only is not None:
                sub = sub[sub.card.isin(cards_only)]
                if sub.empty:
                    continue
            allsub = self.cards[self.cards.source == src]
            if allsub.empty:
                continue
            ev = self.events[(self.events.t < t)]
            ev = ev.merge(allsub[["node", "device"]], left_on=["host", "device"], right_on=["node", "device"]).drop(columns="node")
            devs = allsub[["node", "device"]].rename(columns={"node": "host"})
            d = make_input(ev, devs, spec=pred.input_spec, t_end=t + DAY, name=src)
            ent = d.entities.reset_index()[["entity", "host", "device"]]
            ent["card"] = [self.key[(h, x)] for h, x in zip(ent.host, ent.device)]
            ent = ent[ent.card.isin(sub.card)]
            S, _ = features.build(d, verbose=False, query=pd.DataFrame({"entity": ent.entity, "s": t}))
            X = pred.matrix(S)
            p = ensemble_predict(pred.models, X, list(pred.horizons))
            why = pred.models[0].reasons(X, "1d")
            r = S[["entity"]].merge(ent[["entity", "card"]], on="entity")
            for h in pred.horizons:
                r[f"p_{h}"] = p[h]
            r["reason"] = why
            r["source"] = src
            out.append(r.drop(columns="entity"))
            if with_features:
                cols = [c for c in S.columns if "__" in c]
                for card, row in zip(r.card, S[cols].to_dict("records")):
                    feats[int(card)] = row
        R = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["card", "p_1h", "p_4h", "p_1d", "reason", "source"])
        R["s"] = t
        return (R, feats) if with_features else R

    def levels(self, R):
        for h in HZ:
            th = R.source.map(lambda s: self.thr[s][h]["high"])
            tw = R.source.map(lambda s: self.thr[s][h]["watch"])
            R[f"lvl_{h}"] = np.where(R[f"p_{h}"] > th, 2, np.where(R[f"p_{h}"] > tw, 1, 0))
        return R

    def scan(self, t):
        t_start = time.time()
        R, F = self.score(t, with_features=True)
        R = self.levels(R)
        tel = self.tel_stats(t)
        watch = R[(R[[f"lvl_{h}" for h in HZ]].max(axis=1) > 0)]
        attrib = {}
        for r in watch.itertuples():
            w = self.meta["fta_weights"][r.source]
            attrib[int(r.card)] = fta.attribute(F.get(int(r.card), {}), tel.get(int(r.card), {}), w)
        with self.lock:
            self.current, self.attrib, self.cur_tel = R, attrib, tel
            self.live.append(R[["card", "s", "p_1h", "p_4h", "p_1d"]].astype(
                {"card": np.int16, "p_1h": np.float32, "p_4h": np.float32, "p_1d": np.float32}))
            self.scans.append({"t": t, "seconds": round(time.time() - t_start, 2), "watch": int((R.lvl_1d >= 1).sum()),
                               "high_1h": int((R.lvl_1h == 2).sum()), "high_4h": int((R.lvl_4h == 2).sum()),
                               "high_1d": int((R.lvl_1d == 2).sum()),
                               "uce": int(((self.events.t > t - HOUR) & (self.events.t <= t) &
                                           (self.events.cause == "MEMORY_UNCORRECTABLE_ERROR")).sum())})

    def collect_local(self):
        if not self.local.ok:
            return
        try:
            recs = self.local.sample()
        except Exception:
            return
        E, T = records_to_tables(recs)                   # same processing path as the replayed data
        for r in recs:
            self.local_series.append({"ts": r["ts"], "gpu": r["gpu"], **r["m"]})
        self.local_series = self.local_series[-600:]
        self.ingest_stats["local_records"] = self.ingest_stats.get("local_records", 0) + len(recs)
        self.ingest_stats["local_rows"] = len(T)

    def step(self):
        self.collect_local()
        t0 = self.sim_t
        if t0 >= DEMO_END:
            self.playing = False
            return
        self._collect(t0, t0 + HOUR)
        self.sim_t = t0 + HOUR
        self.scan(self.sim_t)

    def run_forever(self):
        while True:
            if not self.playing:
                self.collect_local()
            if self.playing:
                try:
                    self.step()
                except Exception as e:                       # keep the service alive, report on the page
                    self.scans.append({"t": self.sim_t, "error": repr(e)})
                    self.playing = False
            time.sleep(self.tick_s)

    def start(self):
        threading.Thread(target=self.run_forever, daemon=True).start()

    # ------------------------------------------------------------------ queries for the UI
    def risk_series(self, card, t0, t1):
        h = self.history[(self.history.card == card) & (self.history.s > t0) & (self.history.s <= t1)]
        live = [x[(x.card == card) & (x.s > t0) & (x.s <= t1)] for x in self.live[-24 * 30:]] if self.live else []
        return pd.concat([h] + live, ignore_index=True).sort_values("s")

    def card_events(self, card, t0, t1):
        c = self.cards[self.cards.card == card].iloc[0]
        allev = pd.concat([self.events, self.future[self.future.t <= self.sim_t]], ignore_index=True)
        e = allev[(allev.host == c.node) & ((allev.device == c.device) | allev.cause.isin(["JOB_START", "JOB_END"]))
                  & (allev.t > t0) & (allev.t <= t1)].drop_duplicates(["t", "cause", "device"])
        return [{"t": int(r.t), "cause": cause_zh(r.cause), "code": r.cause,
                 "severity": CAUSES.get(r.cause, {}).get("severity", "info"),
                 "node_level": r.cause in ("JOB_START", "JOB_END")} for r in e.sort_values("t").itertuples()]

    def card_telemetry(self, card, t0, t1):
        c = self.cards[self.cards.card == card].iloc[0]
        steps, core, mem, power, busy, _ = self.tel.window(np.array([c.node_g]), np.array([c.slot]), t0, t1)
        return {"t": steps.tolist(), "core_temp": core[0].round(1).tolist(), "mem_temp": mem[0].round(1).tolist(),
                "power": power[0].round(1).tolist(), "busy": busy[0].astype(int).tolist()}

    def explain(self, card, t):
        """fault-tree attribution of one card at any time (live or history)."""
        c = self.cards[self.cards.card == card].iloc[0]
        R, F = self.score(t, cards_only=[card], with_features=True)
        R = self.levels(R)
        # telemetry statistics at t from the replayed library (works for history too)
        T = []
        steps, core, mem, power, _, _ = self.tel.window(self.cards.node_g.to_numpy(), self.cards.slot.to_numpy(), t - HOUR, t)
        T = pd.DataFrame({"host": np.repeat(self.cards.node.to_numpy(), len(steps)),
                          "device": np.repeat(self.cards.device.to_numpy(), len(steps)),
                          "t": np.tile(steps, len(self.cards)), "core_temp": core.ravel(), "mem_temp": mem.ravel(),
                          "power": power.ravel()})
        tel = self.tel_stats(t, tel=T)
        a = fta.attribute(F.get(card, {}), tel.get(card, {}), self.meta["fta_weights"][c.source])
        row = R.iloc[0]
        return {"risk": {h: float(row[f"p_{h}"]) for h in HZ}, "level": {h: int(row[f"lvl_{h}"]) for h in HZ},
                "reason": row.reason, "fta": a, "telemetry": tel.get(card, {}),
                "features": {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
                             for k, v in F.get(card, {}).items()}}
