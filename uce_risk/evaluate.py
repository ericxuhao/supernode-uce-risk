"""Strict evaluation for UCE risk scores.

* sample level, natural class ratio: PR-AUC, ROC-AUC, Brier score, expected calibration error (ECE) and
  calibration ratio (mean predicted / observed rate);
* operating point: the threshold is fixed on the validation period so that 1 % of the entities without
  a UCE raise at least one alarm (falls back to 1 % of negative samples when there are < 100 negative
  entities); on the test period we report
    - entity level precision / recall / F1 (a UCE entity is caught if it is alarmed within H before a
      test-period UCE; a non-UCE entity with any alarm is a false positive),
    - episode recall, split into first UCE on the entity and recurrent UCE.
UCE entities that were never sampled (no activity before the UCE) count as misses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .schema import HORIZONS


def ece(y, p, bins=10):
    if len(p) == 0:
        return np.nan
    o = np.argsort(p)
    parts = np.array_split(o, bins)
    return float(sum(abs(p[i].mean() - y[i].mean()) * len(i) for i in parts if len(i)) / len(p))


def sample_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = y.astype(bool)
    r = dict(n=len(y), pos=int(y.sum()), pos_rate=float(y.mean()) if len(y) else np.nan)
    if y.any() and (~y).any():
        r.update(PR_AUC=average_precision_score(y, p), ROC_AUC=roc_auc_score(y, p))
    else:
        r.update(PR_AUC=np.nan, ROC_AUC=np.nan)
    r.update(Brier=float(np.mean((p - y) ** 2)), ECE=ece(y, p), calib_ratio=float(p.mean() / max(y.mean(), 1e-12)))
    return r


def neg_entities(S: pd.DataFrame, ep: pd.DataFrame | None, t0, t1) -> np.ndarray:
    ents = S.entity.unique()
    if ep is None:
        return ents
    bad = ep[(ep.t > t0) & (ep.t <= t1 + HORIZONS["1d"])].entity.unique()
    return np.setdiff1d(ents, bad)


def pick_threshold(S: pd.DataFrame, p: np.ndarray, H: str, ep, t0, t1, fpr=0.01) -> float:
    neg = neg_entities(S, ep, t0, t1)
    if len(neg) >= 100:
        m = pd.Series(p).groupby(S.entity.to_numpy()).max()
        return float(np.quantile(m.reindex(neg).dropna(), 1 - fpr))
    y = (S.lead.to_numpy() <= HORIZONS[H])
    return float(np.quantile(p[~y], 1 - fpr)) if (~y).any() else 1.0


def operating_metrics(S: pd.DataFrame, p: np.ndarray, thr: float, H: str, ep: pd.DataFrame | None, t0, t1,
                      test_entities=None) -> dict:
    Hs = HORIZONS[H]
    lead = S.lead.to_numpy()
    a = p > thr
    inwin = lead <= Hs
    r = dict(thr=thr)
    if ep is None:      # no entity / episode structure: sample level
        y = inwin
        tp, fp, fn = int((a & y).sum()), int((a & ~y).sum()), int((~a & y).sum())
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        r.update(P=P, R=R, F1=2 * P * R / (P + R) if P + R else 0.0, alarms=int(a.sum()))
        return r
    E = ep[(ep.t > t0) & (ep.t <= t1)]
    if test_entities is not None:
        E = E[E.entity.isin(test_entities)]
    hit_ep = set(S.ep.to_numpy()[a & inwin])
    cover_ep = set(S.ep.to_numpy()[inwin])
    caught = E.ep.isin(hit_ep).to_numpy()
    cover = E.ep.isin(cover_ep).to_numpy()
    first = E["first"].to_numpy(bool)
    pos_ent = E.entity.unique()
    cov_ent = E.entity[cover].unique()
    tp_ent = E.entity[caught].unique()
    neg = neg_entities(S, ep, t0, t1)
    alarmed = np.unique(S.entity.to_numpy()[a])
    fp = len(np.intersect1d(alarmed, neg))
    tp = len(tp_ent)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / len(pos_ent) if len(pos_ent) else np.nan
    Rc = tp / len(cov_ent) if len(cov_ent) else np.nan
    r.update(R_cov=Rc, F1_cov=2 * P * Rc / (P + Rc) if P + Rc else 0.0, cov_entities=len(cov_ent))
    r.update(tp_entities=tp, caught=int(caught.sum()), covered=int(cover.sum()), first_caught=int(caught[first].sum()),
             recur_caught=int(caught[~first].sum()))
    r.update(P=P, R=R, F1=2 * P * R / (P + R) if P + R else 0.0, pos_entities=len(pos_ent), neg_entities=len(neg),
             fp_entities=fp, episodes=len(E), ep_recall=caught.mean() if len(E) else np.nan,
             ep_coverable=cover.mean() if len(E) else np.nan,
             first_eps=int(first.sum()), first_recall=caught[first].mean() if first.any() else np.nan,
             recur_eps=int((~first).sum()), recur_recall=caught[~first].mean() if (~first).any() else np.nan)
    return r


def entity_scores(S: pd.DataFrame, p: np.ndarray, H: str, ep: pd.DataFrame | None, t0, t1, test_entities=None):
    """max score per positive entity inside its warning windows (-inf if never sampled there) and per
    negative entity over the whole period; sample level when there is no entity structure."""
    inwin = S.lead.to_numpy() <= HORIZONS[H]
    if ep is None:
        return p[inwin], p[~inwin]
    E = ep[(ep.t > t0) & (ep.t <= t1)]
    if test_entities is not None:
        E = E[E.entity.isin(test_entities)]
    m = inwin & S.ep.isin(set(E.ep)).to_numpy()
    ps = pd.Series(p[m]).groupby(S.entity.to_numpy()[m]).max().reindex(E.entity.unique()).fillna(-np.inf).to_numpy()
    neg = neg_entities(S, ep, t0, t1)
    ns = pd.Series(p).groupby(S.entity.to_numpy()).max().reindex(neg).dropna().to_numpy()
    return ps, ns


def best_f1(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """(threshold, F1) maximising F1 when alarming on score > threshold."""
    if not len(pos):
        return 1.0, np.nan
    cand = np.unique(pos[np.isfinite(pos)])
    if not len(cand):
        return 1.0, 0.0
    ps, ns = np.sort(pos), np.sort(neg)
    tp = len(ps) - np.searchsorted(ps, cand, "left")      # score >= cand
    fp = len(ns) - np.searchsorted(ns, cand, "left")
    P = tp / np.maximum(tp + fp, 1)
    R = tp / len(ps)
    F = np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0)
    i = int(np.argmax(F))
    return float(np.nextafter(cand[i], -np.inf)), float(F[i])
