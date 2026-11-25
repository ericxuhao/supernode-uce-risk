"""Alarm policy and effectiveness metrics, shared by the offline build and the live engine.

Thresholds per model group and horizon:
  24 h  high  = validation-F1 threshold of the model;  watch = min(1 %-false-alarm threshold, high / 3)
  1 h / 4 h  = thresholds with the same alarm rate as the 24 h ones on the scored history (quantile
               matching). Validation periods hold only a handful of 1 h / 4 h positives, so thresholds fitted
               there directly are unstable; the matching uses model outputs only (no labels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY, HOUR = 86400, 3600
HSEC = {"1h": HOUR, "4h": 4 * HOUR, "1d": DAY}


def thresholds(model_thr: dict, hist: pd.DataFrame) -> dict:
    """24 h thresholds from the model; 1 h / 4 h thresholds at the same alarm rate (quantile matching on the
    scored card-hours, no labels), because the short-horizon probabilities have much narrower tails."""
    hi1d = model_thr["1d"]["valF1"]
    wa1d = min(model_thr["1d"]["fpr1"], hi1d / 3)
    p1d = hist.p_1d.to_numpy()
    r_hi, r_wa = float((p1d > hi1d).mean()), float((p1d > wa1d).mean())
    out = {}
    for h in HSEC:
        if h == "1d":
            hi, wa = hi1d, wa1d
        else:
            p = hist[f"p_{h}"].to_numpy()
            hi = float(np.quantile(p, 1 - max(r_hi, 1e-7)))
            wa = min(float(np.quantile(p, 1 - max(r_wa, 1e-7))), hi)
        out[h] = {"high": hi, "watch": wa, "alarm_rate_high": r_hi, "alarm_rate_watch": r_wa,
                  "valF1": model_thr[h]["valF1"], "fpr1": model_thr[h]["fpr1"]}
    return out


def _caught(alarm_s: dict, E: pd.DataFrame, Hs: int):
    caught, leads = [], []
    for e in E.itertuples():
        s = alarm_s.get(e.card, np.array([]))
        w = s[(s >= e.t - Hs) & (s < e.t)]
        caught.append(len(w) > 0)
        w24 = s[(s >= e.t - DAY) & (s < e.t)]
        leads.append((e.t - w24.min()) / HOUR if len(w24) else np.nan)
    return np.array(caught, bool), np.array(leads, float)


def _precision(A: pd.DataFrame, E: pd.DataFrame, Hs: int):
    ad = A.assign(day=A.s // DAY).groupby(["card", "day"]).s.min().reset_index()
    et = E.groupby("card").t.apply(np.array).to_dict()
    ok = [bool(((et.get(c, np.array([])) > s) & (et.get(c, np.array([])) <= s + max(Hs, DAY))).any())
          for c, s in zip(ad.card, ad.s)]
    return (float(np.mean(ok)) if ok else 0.0), int(len(ad))


def _summ(caught, leads, E, prec, days):
    first = E["first"].to_numpy(bool)
    return {"episodes": int(len(E)), "caught": int(caught.sum()), "recall": float(caught.mean()) if len(E) else 0.0,
            "first_recall": float(caught[first].mean()) if first.any() else None,
            "recur_recall": float(caught[~first].mean()) if (~first).any() else None,
            "median_lead_h": float(np.nanmedian(leads)) if np.isfinite(leads).any() else None,
            "alarm_card_days": days, "precision": prec}


def _same_budget(hist, E, h, Hs, days):
    """model with the same number of alarmed card-days as a baseline (top card-days by daily max risk)."""
    cd = hist.assign(day=hist.s // DAY).groupby(["card", "day"])[f"p_{h}"].max()
    if not days or len(cd) < days:
        return None
    q = float(np.sort(cd.to_numpy())[-days])
    B = hist[hist[f"p_{h}"] >= q][["card", "s"]]
    c, l = _caught(B.groupby("card").s.apply(np.array).to_dict(), E, Hs)
    return _summ(c, l, E, *_precision(B, E, Hs))


def effectiveness(hist: pd.DataFrame, eps: pd.DataFrame, thr: dict, rule_alarm: pd.DataFrame, t0: int, t1: int,
                  hw_times: dict | None = None) -> dict:
    """per horizon: model (own threshold); the 30-day rule and the model at its alarm budget; and, when hw_times
    is given, the horizon-matched rule (hardware-type error in the previous H hours) and the model at its budget.

    The 30-day rule is a state that stays on for 30 days, so its recall hardly depends on the horizon; the
    horizon-matched rule is the baseline for how well the *timing* of a failure is anticipated."""
    hist = hist[(hist.s > t0) & (hist.s <= t1)]
    E = eps[(eps.t > t0) & (eps.t <= t1)]
    R = rule_alarm[(rule_alarm.s > t0) & (rule_alarm.s <= t1)][["card", "s"]]
    res = {}
    for h, Hs in HSEC.items():
        A = hist[hist[f"p_{h}"] > thr[h]["high"]][["card", "s"]]
        out = {}
        c, l = _caught(A.groupby("card").s.apply(np.array).to_dict(), E, Hs)
        out["model"] = _summ(c, l, E, *_precision(A, E, Hs))
        c, l = _caught(R.groupby("card").s.apply(np.array).to_dict(), E, Hs)
        rp, rdays = _precision(R, E, Hs)
        out["rule"] = _summ(c, l, E, rp, rdays)
        sb = _same_budget(hist, E, h, Hs, rdays)               # model with the rule's alarm volume
        if sb:
            out["model_same_budget"] = sb
        if hw_times is not None:
            W = rule_alarms(hist[["card", "s"]], hw_times, window_s=Hs)
            c, l = _caught(W.groupby("card").s.apply(np.array).to_dict(), E, Hs)
            wp, wdays = _precision(W, E, Hs)
            out["rule_window"] = _summ(c, l, E, wp, wdays)
            sb = _same_budget(hist, E, h, Hs, wdays)
            if sb:
                out["model_same_budget_window"] = sb
        res[h] = out
    return res


def rule_alarms(hist: pd.DataFrame, hw_times: dict, window_s: int = 30 * DAY) -> pd.DataFrame:
    """rule baseline: any hardware-type error on the card in the previous window_s seconds (default 30 days)."""
    base = hist[["card", "s"]]
    flag = np.zeros(len(base), bool)
    s_all = base.s.to_numpy()
    for c, idx in base.groupby("card").indices.items():
        tt = hw_times.get(c)
        if tt is None:
            continue
        s = s_all[idx]
        j = np.searchsorted(tt, s, "left") - 1
        flag[idx] = (j >= 0) & (s - tt[np.clip(j, 0, None)] <= window_s)
    return base[flag]
