"""Offline build of the supernode demo (run once, ~15-30 min):

  1. telemetry library from the Summit artifact (real 10-s node telemetry -> 5-min node-day profiles)
  2. card population: 1,024 card slots (16 racks x 8 nodes x 8 cards) filled from open data
       - rack 16, first 51 slots: Calchas HBM devices (address-level CE / UER / UEO log, time-shifted)
       - all other slots: Summit cards (kernel error log), enriched with every card that had a UCE or a
         hardware error inside the demo window, the rest drawn at random
  3. two risk models through the same ingest path the live engine uses (field-driven input):
       - 'log' model: Summit, all 28k cards, data before demo_start (inputs: error events + history)
       - 'ce'  model: Calchas, data before the split (inputs: address-level CE + events + history)
  4. fault-tree weights (attributable fractions) on each model's training samples
  5. history: hourly risk for every card from demo_start to live_start + effectiveness metrics

    python supernode/build.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import (CFG, DATA, DEMO_DIR, MODEL_DIR, cfg_path, DAY, DEMO_END, DEMO_START, HOUR, LIVE_START, N_CARDS, ROOT, UCE_CAUSE, cause_of_xid,
                    topology, RAS_TO_CAUSE)
import fta
import policy
from ingest import make_input
from sources import TelemetryLibrary, build_telemetry_library
from uce_risk import features
from uce_risk.api import UCERiskPredictor
from uce_risk.schema import HORIZONS

CAL_SPLIT = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp())     # Calchas: train before, replay after
CAL_VAL = int(pd.Timestamp("2022-10-15", tz="UTC").timestamp())
CAL_SHIFT = DEMO_START - CAL_SPLIT
SUMMIT_VAL = int(pd.Timestamp("2021-04-01", tz="UTC").timestamp())
ADDR = ["stack", "sid", "pc", "bg", "bank", "row", "col"]
HW_XID = {63, 64, 79, 74, 92, 61, 62, 48}
HW_CAUSES = {UCE_CAUSE, "MEMORY_PAGE_RETIRED", "MEMORY_PAGE_RETIRE_FAILED", "DEVICE_LOST", "INTERCONNECT_LINK_ERROR",
             "MEMORY_UNCORRECTABLE_DEFERRED"}
# open data sets used to (re)build the models and the replay data (config.json "open_data")
OPEN = CFG.get("open_data", {})
SUMMIT_DIR = cfg_path(OPEN.get("summit_dir", ROOT / "dataset"))
CALCHAS_RAW = cfg_path(OPEN.get("calchas_raw", "open_data/calchas/dataset(opensource).csv"))
ARTIFACT_TEL = cfg_path(OPEN.get("artifact_telemetry", "open_data/summit_artifact_202001"))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_summit():
    f = pd.read_csv(SUMMIT_DIR / "failures.csv", usecols=["timestamp", "hostname", "GPU", "xid"])
    f = f[f.GPU.notna()]
    t = (pd.to_datetime(f.timestamp, utc=True, format="ISO8601").astype("int64") // 10**9).to_numpy(np.int64)
    return pd.DataFrame({"ohost": f.hostname.to_numpy(), "oslot": f.GPU.astype(int).to_numpy(), "t": t,
                         "xid": f.xid.astype(int).to_numpy()})


def load_calchas():
    r = pd.read_csv(CALCHAS_RAW)
    out = pd.DataFrame({"ohost": r.Server, "odev": r.Name, "t": r.Time.astype(np.int64), "sev": r.EccType})
    for c, src in zip(ADDR, ["Stack", "SID", "PcId", "BankGroup", "BankArray", "Row", "Col"]):
        out[c] = r[src].map(lambda x: int(str(x), 16)).astype(np.int64)
    return out


def select_cards(sm: pd.DataFrame, cal: pd.DataFrame, rng) -> pd.DataFrame:
    topo = topology()
    topo["node_g"] = topo.rack_i * CFG["topology"]["nodes_per_rack"] + topo.node_i
    cal_devs = cal[["ohost", "odev"]].drop_duplicates().sort_values(["ohost", "odev"]).to_numpy()
    n_cal = len(cal_devs)
    cal_slots = topo.index[topo.rack_i == topo.rack_i.max()][:n_cal]
    win = sm[(sm.t >= DEMO_START) & (sm.t < DEMO_END)]
    key = lambda d: set(map(tuple, d[["ohost", "oslot"]].drop_duplicates().to_numpy()))          # noqa: E731
    U = key(win[win.xid == 48])
    H = key(win[win.xid.isin(HW_XID)]) - U
    hosts = open(SUMMIT_DIR / "hostnames.txt").read().split()
    allg = {(h, s) for h in hosts for s in range(6)}
    n_sum = N_CARDS - n_cal
    Hs = sorted(H)
    Hpick = [Hs[i] for i in rng.choice(len(Hs), min(CFG["selection"]["hardware_event_cards"], len(Hs)), replace=False)]
    rest = sorted(allg - U - set(Hpick))
    fill = [rest[i] for i in rng.choice(len(rest), n_sum - len(U) - len(Hpick), replace=False)]
    sel = sorted(list(U) + Hpick + fill)                     # keep cards of one original host together
    free = [i for i in topo.index if i not in set(cal_slots)]
    # spread original hosts over random demo nodes: shuffle node order, then fill node by node
    nodes = topo.loc[free, "node_g"].drop_duplicates().to_numpy()
    order = {n: k for k, n in enumerate(rng.permutation(nodes))}
    free = sorted(free, key=lambda i: (order[topo.node_g[i]], topo.slot[i]))
    topo["source"], topo["ohost"], topo["odev"] = "", "", ""
    for i, (h, s) in zip(free, sel):
        topo.loc[i, ["source", "ohost", "odev"]] = ["log", h, str(s)]
    for i, (h, d) in zip(cal_slots, cal_devs):
        topo.loc[i, ["source", "ohost", "odev"]] = ["ce", h, d]
    topo["device"] = "C" + topo.slot.astype(str)
    log(f"cards: {len(U)} with UCE in the window, {len(Hpick)} with hardware errors, {len(fill)} random, {n_cal} memory-RAS")
    return topo


def summit_events(sm, cards=None):
    """vendor-neutral event table; with `cards`, re-keyed to the demo topology."""
    if cards is None:
        e = pd.DataFrame({"host": sm.ohost, "device": "C" + sm.oslot.astype(str), "t": sm.t, "cause": sm.xid.map(cause_of_xid)})
        return e
    m = cards[cards.source == "log"][["ohost", "odev", "node", "device"]]
    e = sm.assign(odev=sm.oslot.astype(str)).merge(m, on=["ohost", "odev"])
    return pd.DataFrame({"host": e.node, "device": e.device, "t": e.t, "cause": e.xid.map(cause_of_xid), "kind": "xid",
                         "code": e.xid.astype(str)})


def calchas_events(cal, cards=None):
    if cards is None:
        e = pd.DataFrame({"host": cal.ohost, "device": cal.odev, "t": cal.t, "cause": cal.sev.map(RAS_TO_CAUSE)})
        for c in ADDR:
            e[c] = cal[c].to_numpy()
        return e
    m = cards[cards.source == "ce"][["ohost", "odev", "node", "device"]]
    e = cal.merge(m, on=["ohost", "odev"])
    out = pd.DataFrame({"host": e.node, "device": e.device, "t": e.t + CAL_SHIFT, "cause": e.sev.map(RAS_TO_CAUSE),
                        "kind": "ras", "code": e.sev.astype(str)})
    for c in ADDR:
        out[c] = e[c].to_numpy()
    return out


def train(events, devices, val_start, t_end, name, bags, tune, variant="hybrid"):
    d = make_input(events[events.t < t_end], devices, t_end=t_end, name=name)
    S, ep = features.build(d, verbose=False)
    if variant != "hybrid":
        p = UCERiskPredictor(scenario="auto", bags=bags, tune=tune, variant=variant).fit(d, val_start=val_start,
                                                                                         samples=(S, ep))
        log(f"   {name}/{variant}: params {p.info['tree_params']}")
        return p, None, d.report
    log(f"   {name}: {len(S):,} training samples, {int((S.lead <= DAY).sum()):,} with a UCE within 24 h; "
        f"modules {d.report['modules']}")
    p = UCERiskPredictor(scenario="auto", bags=bags, tune=tune).fit(d, val_start=val_start, samples=(S, ep))
    w = fta.estimate_weights(S)
    log(f"   {name}: params {p.info['tree_params']}, fusion {p.info['fusion_weight_ai']}")
    return p, w, d.report


def history(pred, events, devices, cards_sub, t0, t1, chunk_h=240):
    d = make_input(events[events.t < t1], devices, spec=pred.input_spec, t_end=t1 + DAY, name="history")
    ent = d.entities.reset_index()[["entity", "host", "device"]].merge(cards_sub[["card", "node", "device"]],
                                                                         left_on=["host", "device"], right_on=["node", "device"])
    hours = np.arange(t0 + HOUR, t1 + 1, HOUR)
    out = []
    for k in range(0, len(hours), chunk_h):
        hh = hours[k:k + chunk_h]
        q = pd.DataFrame({"entity": np.repeat(ent.entity.to_numpy(), len(hh)), "s": np.tile(hh, len(ent))})
        r = pred.predict(d, at=q, explain=False)
        out.append(r[["entity", "s", "p_1h", "p_4h", "p_1d"]])
        log(f"      history {pd.Timestamp(hh[-1], unit='s'):%Y-%m-%d}: {len(r):,} card-hours")
    h = pd.concat(out).merge(ent[["entity", "card"]], on="entity").drop(columns="entity")
    ep = features.episodes(d.uce, d.episode_gap).merge(ent[["entity", "card"]], on="entity")
    return h, ep


def apply_policy(cards, ev_demo, hist, eps, preds):
    hw = ev_demo[ev_demo.cause.isin(HW_CAUSES)].merge(cards[["card", "node", "device"]], left_on=["host", "device"],
                                                      right_on=["node", "device"])
    hw_times = hw.groupby("card").t.apply(np.sort).to_dict()
    rule = policy.rule_alarms(hist, hw_times)
    eff, demo_thr = {}, {}
    for src, pred in preds.items():
        cs = set(cards.card[cards.source == src])
        hs = hist[hist.card.isin(cs)]
        demo_thr[src] = policy.thresholds(pred.thresholds, hs)
        eff[src] = policy.effectiveness(hs, eps[eps.card.isin(cs)], demo_thr[src], rule[rule.card.isin(cs)],
                                        DEMO_START, LIVE_START, hw_times=hw_times)
    return eff, demo_thr


def recompute_policy():
    cards = pd.read_parquet(DEMO_DIR / "cards.parquet")
    ev = pd.read_parquet(DEMO_DIR / "events_demo.parquet")
    hist = pd.read_parquet(DEMO_DIR / "history_risk.parquet")
    eps = pd.read_parquet(DEMO_DIR / "episodes.parquet")
    preds = joblib.load(MODEL_DIR / "models.joblib")
    meta = json.loads((MODEL_DIR / "build_meta.json").read_text(encoding="utf-8"))
    meta["effectiveness"], meta["demo_thresholds"] = apply_policy(cards, ev, hist, eps, preds)
    (MODEL_DIR / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for src, e in meta["effectiveness"].items():
        for h, v in e.items():
            log(f"{src} {h}: " + ", ".join(f"{k} R {x['recall']:.2f} P {x['precision']:.3f} days {x['alarm_card_days']}"
                                          for k, x in v.items()))


def with_lead(hist, eps):
    """attach the time to the next UCE episode (lead, seconds) to every scored card-hour."""
    h = hist.astype({"card": np.int64, "s": np.int64}).sort_values("s").reset_index(drop=True)
    e = eps[["card", "t"]].rename(columns={"t": "t_ep"}).astype({"card": np.int64, "t_ep": np.int64}).sort_values("t_ep")
    m = pd.merge_asof(h, e, left_on="s", right_on="t_ep", by="card", direction="forward", allow_exact_matches=False)
    m["lead"] = m.t_ep - m.s
    return m


def ablation_metrics(hist, eps, model_thr, rule, t0, t1):
    from sklearn.metrics import average_precision_score
    h = with_lead(hist[(hist.s > t0) & (hist.s <= t1)], eps)
    out = {}
    for hz, sec in policy.HSEC.items():
        y = (h.lead <= sec).to_numpy()
        p = h[f"p_{hz}"].to_numpy()
        out[hz] = {"pr_auc": float(average_precision_score(y, p)) if y.any() else None,
                   "calib": float(p.mean() / max(y.mean(), 1e-12)), "pos": int(y.sum())}
    thr = policy.thresholds(model_thr, hist)
    eff = policy.effectiveness(hist, eps, thr, rule, t0, t1)
    for hz in out:
        out[hz].update(eff[hz])
    return out


def run_ablation(cards, ev_demo, sm, cal, preds, hist_hybrid, eps, bags, tune):
    """hybrid (already built) vs AI only vs rule scorecard only vs the simple rule, on the backtrack window."""
    hw = ev_demo[ev_demo.cause.isin(HW_CAUSES)].merge(cards[["card", "node", "device"]], left_on=["host", "device"],
                                                      right_on=["node", "device"])
    rule = policy.rule_alarms(hist_hybrid, hw.groupby("card").t.apply(np.sort).to_dict())
    sm_dev = pd.DataFrame({"host": np.repeat(open(SUMMIT_DIR / "hostnames.txt").read().split(), 6)})
    sm_dev["device"] = "C" + (np.arange(len(sm_dev)) % 6).astype(str)
    cal_dev = cal[["ohost", "odev"]].drop_duplicates().rename(columns={"ohost": "host", "odev": "device"})
    setup = {"log": (summit_events(sm), sm_dev, SUMMIT_VAL, DEMO_START),
             "ce": (calchas_events(cal), cal_dev, CAL_VAL, CAL_SPLIT)}
    res = {}
    for src in ("log", "ce"):
        sub = cards[cards.source == src]
        cs = set(sub.card)
        e_src, r_src, h_src = eps[eps.card.isin(cs)], rule[rule.card.isin(cs)], hist_hybrid[hist_hybrid.card.isin(cs)]
        res[src] = {"hybrid": ablation_metrics(h_src, e_src, preds[src].thresholds, r_src, DEMO_START, LIVE_START)}
        evs = ev_demo.merge(sub[["node", "device"]], left_on=["host", "device"], right_on=["node", "device"]).drop(columns="node")
        hist_ev = evs if src == "ce" else pd.concat([summit_events(sm, cards)[lambda x: x.t < DEMO_START].pipe(
            lambda x: x[x.host.isin(sub.node)]), evs[evs.t >= DEMO_START]])
        devs = sub[["node", "device"]].rename(columns={"node": "host"})
        for variant in ("ai", "rules"):
            log(f"ablation {src}/{variant}")
            p, _, _ = train(*setup[src], src, bags, tune, variant=variant)
            h, _ = history(p, hist_ev.drop_duplicates(), devs, sub, DEMO_START, LIVE_START)
            res[src][variant] = ablation_metrics(h, e_src, p.thresholds, r_src, DEMO_START, LIVE_START)
        log(f"   {src}: " + ", ".join(f"{v} PR-AUC(24h) {m['1d']['pr_auc']:.3f} same-budget R "
                                     f"{m['1d'].get('model_same_budget', {}).get('recall', 0):.2f}" for v, m in res[src].items()))
    return res


def ablation_only(quick):
    bags, tune = (1, 0) if quick else (3, 20)
    cards = pd.read_parquet(DEMO_DIR / "cards.parquet")
    ev = pd.read_parquet(DEMO_DIR / "events_demo.parquet")
    hist = pd.read_parquet(DEMO_DIR / "history_risk.parquet")
    eps = pd.read_parquet(DEMO_DIR / "episodes.parquet")
    preds = joblib.load(MODEL_DIR / "models.joblib")
    sm, cal = load_summit(), load_calchas()
    meta = json.loads((MODEL_DIR / "build_meta.json").read_text(encoding="utf-8"))
    meta["ablation"] = run_ablation(cards, ev, sm, cal, preds, hist, eps, bags, tune)
    (MODEL_DIR / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log("ablation saved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 bag, no tuning (for testing)")
    ap.add_argument("--policy-only", action="store_true", help="recompute alarm thresholds / metrics from saved history")
    ap.add_argument("--ablation-only", action="store_true", help="train AI-only / rules-only models and compare")
    a = ap.parse_args()
    if a.policy_only:
        return recompute_policy()
    if a.ablation_only:
        return ablation_only(a.quick)
    bags, tune = (1, 0) if a.quick else (3, 20)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(CFG["selection"]["seed"])
    if not (DEMO_DIR / "telemetry_library.npz").exists():
        log("telemetry library")
        build_telemetry_library(ARTIFACT_TEL, DEMO_DIR / "telemetry_library.npz")
    log("loading open data")
    sm, cal = load_summit(), load_calchas()
    cards = select_cards(sm, cal, rng)
    cards.to_parquet(DEMO_DIR / "cards.parquet", index=False)
    ev_demo = pd.concat([summit_events(sm, cards), calchas_events(cal, cards)], ignore_index=True)
    ev_demo = ev_demo[ev_demo.t < DEMO_END].sort_values("t").reset_index(drop=True)
    ev_demo.to_parquet(DEMO_DIR / "events_demo.parquet", index=False)
    log(f"demo events: {len(ev_demo):,}")

    log("training the log model (all Summit cards, before the demo window)")
    sm_dev = pd.DataFrame({"host": np.repeat(open(SUMMIT_DIR / "hostnames.txt").read().split(), 6)})
    sm_dev["device"] = "C" + (np.arange(len(sm_dev)) % 6).astype(str)
    p_log, w_log, rep_log = train(summit_events(sm), sm_dev, SUMMIT_VAL, DEMO_START, "log", bags, tune)
    log("training the memory-RAS model (Calchas, before the split)")
    cal_dev = cal[["ohost", "odev"]].drop_duplicates().rename(columns={"ohost": "host", "odev": "device"})
    p_ce, w_ce, rep_ce = train(calchas_events(cal), cal_dev, CAL_VAL, CAL_SPLIT, "ce", bags, tune)
    joblib.dump({"log": p_log, "ce": p_ce}, MODEL_DIR / "models.joblib", compress=3)

    log("history (hourly scans) and effectiveness")
    H, EP = [], []
    for src, pred in (("log", p_log), ("ce", p_ce)):
        sub = cards[cards.source == src]
        evs = ev_demo[ev_demo.host.isin(sub.node) & ev_demo.device.isin(sub.device)]
        evs = evs.merge(sub[["node", "device"]], left_on=["host", "device"], right_on=["node", "device"]).drop(columns="node")
        hist_ev = evs if src == "ce" else pd.concat([summit_events(sm, cards)[lambda x: x.t < DEMO_START].pipe(
            lambda x: x[x.host.isin(sub.node)]), evs[evs.t >= DEMO_START]])
        devs = sub[["node", "device"]].rename(columns={"node": "host"})
        h, ep = history(pred, hist_ev.drop_duplicates(), devs, sub, DEMO_START, LIVE_START)
        h["source"], ep["source"] = src, src
        H.append(h)
        EP.append(ep)
    hist = pd.concat(H, ignore_index=True)
    eps = pd.concat(EP, ignore_index=True)
    hist = hist.astype({"card": np.int16, "s": np.int64, "p_1h": np.float32, "p_4h": np.float32, "p_1d": np.float32})
    hist.to_parquet(DEMO_DIR / "history_risk.parquet", index=False)
    eps.to_parquet(DEMO_DIR / "episodes.parquet", index=False)
    eff, demo_thr = apply_policy(cards, ev_demo, hist, eps, {"log": p_log, "ce": p_ce})
    meta = {"built": time.strftime("%Y-%m-%d %H:%M"), "bags": bags, "tune": tune,
            "models": {"log": {"info": p_log.info, "thresholds": p_log.thresholds, "report": rep_log},
                       "ce": {"info": p_ce.info, "thresholds": p_ce.thresholds, "report": rep_ce}},
            "fta_weights": {"log": w_log, "ce": w_ce}, "effectiveness": eff, "demo_thresholds": demo_thr,
            "cal_shift_days": CAL_SHIFT / DAY}
    (MODEL_DIR / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log("done: " + json.dumps(eff, ensure_ascii=False, default=str)[:1500])


if __name__ == "__main__":
    main()
