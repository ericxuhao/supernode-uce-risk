"""Domain rules for UCE risk (the "rule" half of the AI + rule hybrid).

Each rule maps the feature table to a small number of interpretable levels (bins). A rule is used only
when its input columns exist, so the same rule set serves every input scenario:

  rule                   inputs                 meaning (prior knowledge it encodes)
  recur_time             hi__h_since_last       UCEs cluster in time (Calchas F6, Summit: 76 % of repeats)
  recur_count            hi__ep_cum             devices with several UCEs keep failing
  node_uce               nbh__ep_30d            another device on the node had a UCE
  hw_event_*             ev__<code>_*           rare (device-specific) event codes, e.g. Xid 63/64/92/79, HBM UEO
  app_event              ev__bg_<code>_1d       background codes seen on many devices (e.g. Xid 13/31), tagged by the
                                                input layer from the data, not hard-coded
  ce_burst / ce_accel    ce__n_1h, accel        CE burst in the last hour, rate above the daily level
  ce_onset               ce__h_since_first      new CE source: UE servers fail soon after the first CE
  ce_storm               ce__storm_min_1d       CE storm (Calchas F7)
  bank_fault             sp__<bank>_multi_row/col  multi-row / multi-column bank fault (SRDS'22, Calchas F2/F5)
  rowcol_fault           sp__row/col_multi_cell row / column fault mode
  scatter                sp__n_cell_1d, max_per_cell  many distinct cells, few repeats (soft, scattered)
  hard_fault             sp__hard_cell          repeating cell: usually handled by page offlining
  tel_high_*             te__* (temperature / power)  top-decile thermal / power level or 1 h rise
  <group>_any            fallback for ready-made feature tables (e.g. Calchas counts)

The scorecard is an L2 logistic regression on the one-hot rule levels, one per horizon, trained on all
positives plus a sample of negatives and prior-corrected to the natural base rate. Its weights are a
readable table, and the active level with the largest weight is reported as the alarm reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .schema import HORIZONS

BACKGROUND = ("bg_", "other")       # event codes tagged as background by the input layer
BASELINE = {"none", "no", "0", "flat", "older/none", ">30d/none", "other", "normal"}
INF = np.inf


def _cut(x, edges, labels):
    x = np.asarray(x, np.float64)
    out = np.full(len(x), labels[-1], dtype=object)
    for e, lab in zip(edges, labels):
        m = (x <= e) & (out == labels[-1])
        out[m] = lab
    return out


class RuleSet:
    """fit() learns data-driven cut points (quantiles) on training rows; transform() returns one-hot levels."""

    def fit(self, df: pd.DataFrame):
        c = set(df.columns)
        self.cols = c
        self.q = {}
        for col in [x for x in df.columns if x.startswith("te__")]:
            v = df[col].to_numpy(np.float64)
            v = v[np.isfinite(v)]
            if len(v) > 100 and np.nanstd(v) > 0:
                self.q[col] = np.quantile(v, 0.9)
        for col in ["ce__n_1h", "ce__n_1d"]:
            if col in c:
                v = df[col].to_numpy(np.float64)
                v = v[v > 0]
                self.q[col] = (np.quantile(v, [0.5, 0.9]) if len(v) > 100 else np.array([5.0, 50.0]))
        self.levels = None
        lv = self._levels(df)
        # the baseline level ("nothing happened") is the reference (first, dropped in transform), so that a positive
        # weight reads "riskier than baseline" and the alarm reason is never a baseline level
        self.levels = {k: sorted(set(v), key=lambda x: (x not in BASELINE, x)) for k, v in lv.items()}
        return self

    def _levels(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        c, L, n = self.cols, {}, len(df)
        g = lambda col: df[col].to_numpy(np.float64)   # noqa: E731
        if "hi__h_since_last" in c:
            h = np.nan_to_num(g("hi__h_since_last"), nan=INF)
            L["recur_time"] = _cut(h, [1, 4, 24, 168, 720, INF], ["<=1h", "<=4h", "<=1d", "<=7d", "<=30d", ">30d/none"])
        if "hi__ep_cum" in c:
            L["recur_count"] = _cut(g("hi__ep_cum"), [0, 1, 2, INF], ["0", "1", "2", ">=3"])
        if "nbh__ep_30d" in c:
            L["node_uce"] = np.where(g("nbh__ep_30d") > 0, "yes", "no")
        hw = sorted({x.split("__")[1].rsplit("_", 1)[0] for x in c if x.startswith("ev__") and x.endswith("_30d")
                     and not x[4:].startswith(BACKGROUND)})
        for code in hw:
            d1, d30 = g(f"ev__{code}_1d"), g(f"ev__{code}_30d")
            L[f"hw_event_{code}"] = np.where(d1 > 0, "<=1d", np.where(d30 > 0, "<=30d", "none"))
        app = [x for x in c if x.startswith("ev__") and x.endswith("_1d") and x[4:].startswith(BACKGROUND)]
        if app:
            L["app_event"] = np.where(df[app].sum(axis=1).to_numpy() > 0, "<=1d", "none")
        if "ce__n_1h" in c:
            q1 = self.q["ce__n_1h"]
            L["ce_burst"] = _cut(g("ce__n_1h"), [0, q1[0], q1[1], INF], ["0", "low", "mid", "high"])
        if "ce__accel_1h_vs_1d" in c:
            L["ce_accel"] = np.where(g("ce__accel_1h_vs_1d") > 2, "rising", "flat")
        if "ce__h_since_first" in c:
            L["ce_onset"] = _cut(np.nan_to_num(g("ce__h_since_first"), nan=INF), [1, 24, 168, INF], ["<=1h", "<=1d", "<=7d", "older/none"])
        if "ce__storm_min_1d" in c:
            L["ce_storm"] = np.where(g("ce__storm_min_1d") > 0, "yes", "no")
        bank = [x for x in c if x.startswith("sp__") and ("_multi_row" in x or "_multi_col" in x)
                and not x.startswith(("sp__row_", "sp__col_"))]
        if bank:
            L["bank_fault"] = _cut(df[bank].max(axis=1).to_numpy(), [0, 1, INF], ["0", "1", ">=2"])
        rc = [x for x in ("sp__row_multi_cell", "sp__col_multi_cell") if x in c]
        if rc:
            L["rowcol_fault"] = np.where(df[rc].sum(axis=1).to_numpy() > 0, "yes", "no")
        if "sp__n_cell_1d" in c and "sp__max_per_cell" in c:
            L["scatter"] = np.where((g("sp__n_cell_1d") >= 5) & (g("sp__max_per_cell") <= 3), "scattered", "other")
        if "sp__hard_cell" in c:
            L["hard_fault"] = _cut(g("sp__hard_cell"), [0, 10, INF], ["0", "1-10", ">10"])
        for col, thr in self.q.items():
            if col.startswith("te__"):
                L[f"tel_high_{col[4:]}"] = np.where(g(col) > thr, "top10%", np.where(np.isnan(g(col)), "missing", "normal"))
        # fallback for groups without a dedicated rule (ready-made feature tables)
        for pre in ("hi__", "ev__", "ce__", "sp__"):
            if not any(k.startswith(t) for k in L for t in self._rule_prefix(pre)):
                cols = [x for x in c if x.startswith(pre)]
                if cols:
                    L[f"{pre[:-2]}_any"] = np.where(df[cols].sum(axis=1).to_numpy() > 0, "yes", "no")
        return {k: np.asarray(v).astype(str) for k, v in L.items()}

    @staticmethod
    def _rule_prefix(pre):
        return {"hi__": ("recur",), "ev__": ("hw_event", "app_event"), "ce__": ("ce_",),
                "sp__": ("bank_fault", "rowcol_fault", "scatter", "hard_fault")}[pre]

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        L = self._levels(df)
        mats, names = [], []
        for k, levels in self.levels.items():
            v = L[k]
            for lev in levels[1:] if len(levels) > 1 else levels:     # first level = reference
                mats.append((v == lev).astype(np.float32))
                names.append(f"{k}={lev}")
        return (np.stack(mats, 1) if mats else np.zeros((len(df), 0), np.float32)), names


class Scorecard:
    """per-horizon L2 logistic regression on rule levels (direct horizons, made monotone by cummax)."""

    def __init__(self, horizons=("1h", "4h", "1d"), neg_ratio=50, C=0.3, seed=0):
        self.horizons, self.neg_ratio, self.C, self.seed = list(horizons), neg_ratio, C, seed

    def fit(self, df: pd.DataFrame, lead: np.ndarray, pool: np.ndarray, w: np.ndarray | None = None):
        rng = np.random.default_rng(self.seed)
        w = np.ones(len(pool)) if w is None else np.asarray(w, np.float64)
        self.rules = RuleSet().fit(df.iloc[pool])
        R, self.names = self.rules.transform(df.iloc[pool])
        lead = np.where(np.isnan(lead), np.inf, lead)[pool]
        self.m = {}
        for h in self.horizons:
            y = lead <= HORIZONS[h]
            pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
            if len(pos) < 3 or R.shape[1] == 0:
                self.m[h] = ("const", (w[pos].sum() + 0.5) / (w.sum() + 1))
                continue
            k = min(len(neg), self.neg_ratio * len(pos))
            sel = np.r_[pos, rng.choice(neg, k, replace=False)]
            lr = LogisticRegression(C=self.C, max_iter=2000).fit(R[sel], y[sel], sample_weight=w[sel] / w[sel].mean())
            self.m[h] = ("lr", lr, k / max(len(neg), 1))
        return self

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        R, _ = self.rules.transform(df)
        out, prev = {}, None
        for h in self.horizons:
            m = self.m[h]
            if m[0] == "const":
                p = np.full(len(df), m[1])
            else:
                p = m[1].predict_proba(R)[:, 1]
                p = p * m[2] / (p * m[2] + 1 - p + 1e-12)
            p = p if prev is None else np.maximum(p, prev)
            out[h] = prev = p
        return out

    def weights(self, h="1d") -> pd.Series:
        m = self.m[h]
        if m[0] != "lr":
            return pd.Series(dtype=float)
        return pd.Series(m[1].coef_[0], index=self.names).sort_values(ascending=False)

    def reasons(self, df: pd.DataFrame, h="1d") -> np.ndarray:
        """the active rule level with the largest positive weight (alarm explanation)."""
        R, names = self.rules.transform(df)
        m = self.m[h]
        if m[0] != "lr" or R.shape[1] == 0:
            return np.full(len(df), "", dtype=object)
        contrib = R * m[1].coef_[0][None, :]
        i = contrib.argmax(1)
        return np.where(contrib.max(1) > 0, np.asarray(names, dtype=object)[i], "baseline")
