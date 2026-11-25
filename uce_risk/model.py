"""Multi-horizon UCE risk models.

HazardModel -- discrete-time hazard over (0,1h], (1h,4h], (4h,1d]:

  h1 = P(UCE in (s, s+1h])
  h2 = P(UCE in (s+1h, s+4h] | none in the first hour)      trained on samples still at risk at 1 h
  h3 = P(UCE in (s+4h, s+1d] | none in the first 4 hours)
  P(1h) = h1,  P(4h) = 1-(1-h1)(1-h2),  P(1d) = 1-(1-h1)(1-h2)(1-h3)

so the three risk probabilities are consistent (P1h <= P4h <= P1d) by construction. Each hazard is a
gradient-boosted tree classifier trained on all positives and a random share q of negatives; its output
is mapped back to the natural base rate with the prior correction p' = p q / (p q + 1 - p).
Missing modalities are NaN columns (HGB routes NaN natively), so one model class serves every input
scenario. ``mode='direct'`` trains one classifier per horizon instead (ablation).

Small-sample fallback: an interval with fewer than ``min_pos`` positives borrows the ranking of a pooled
model (y = UCE within 1 d) and only rescales it to the interval's observed rate (proportional hazards),
instead of fitting a tree model on a handful of positives.

Tree learner: LightGBM (default; native NaN handling, sample weights, monotone constraints) or
scikit-learn HistGradientBoosting (backend='hgb', kept for comparison). ``tune_tree`` searches LightGBM
hyper-parameters with Optuna (TPE) on the validation period.

HybridModel -- AI + rule hybrid built on top of it (see its docstring).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from .schema import HORIZONS

INTERVALS = {"1h": (0, HORIZONS["1h"]), "4h": (HORIZONS["1h"], HORIZONS["4h"]), "1d": (HORIZONS["4h"], HORIZONS["1d"])}


LGBM_DEFAULT = dict(learning_rate=0.05, n_estimators=300, num_leaves=31, min_child_samples=40, subsample=0.8,
                    colsample_bytree=0.8, reg_lambda=1.0)


def hgb(seed=0, mono=None):
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, early_stopping=False, random_state=seed,
                                          monotonic_cst=mono)


def tree(seed=0, mono=None, params=None, backend="lgbm"):
    """gradient-boosted tree classifier: LightGBM (default) or HistGradientBoosting."""
    if backend == "hgb":
        return hgb(seed, mono)
    import lightgbm as lgb
    p = dict(LGBM_DEFAULT, **(params or {}))
    return lgb.LGBMClassifier(objective="binary", subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1,
                              monotone_constraints=list(mono) if mono is not None and any(mono) else None,
                              monotone_constraints_method="advanced", **p)


def tune_tree(Z, y, w, Zv, yv, mono=None, n_trials=20, seed=0, neg_ratio=50, min_val_pos=30) -> dict:
    """Optuna (TPE) search of LightGBM hyper-parameters, objective = validation PR-AUC of the pooled
    (y = UCE within the longest horizon) model. The first trial is the default setting, so the result is
    never worse than the defaults on the validation period. Returns {} when validation has too few positives."""
    if n_trials <= 0 or yv.sum() < min_val_pos or y.sum() < 5:
        return {}
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    sel = np.sort(np.r_[pos, rng.choice(neg, min(len(neg), neg_ratio * len(pos)), replace=False)])
    ww = None if w is None else w[sel]

    def objective(trial):
        params = dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                      n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
                      num_leaves=trial.suggest_int("num_leaves", 7, 127, log=True),
                      min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
                      subsample=trial.suggest_float("subsample", 0.5, 1.0),
                      colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                      reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
        clf = tree(seed, mono, params).fit(Z[sel], y[sel], sample_weight=ww)
        return average_precision_score(yv, clf.predict_proba(Zv)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(dict(LGBM_DEFAULT))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params)
    best["_val_ap"] = float(study.best_value)
    best["_val_ap_default"] = float(study.trials[0].value)
    return best


class HazardModel:
    def __init__(self, horizons=("1h", "4h", "1d"), mode="hazard", neg_ratio=50, min_pos=30, seed=0, mono=None,
                 params=None, backend="lgbm", share=(), cols=None):
        self.horizons, self.mode, self.neg_ratio, self.min_pos, self.seed = list(horizons), mode, neg_ratio, min_pos, seed
        self.share = tuple(share)       # horizons whose interval reuses the pooled ranking, rescaled to its rate
        self.cols = dict(cols or {})    # horizon -> column indices its interval model uses (default: all)
        self.mono, self.w, self.backend = mono, None, backend
        self.params = {k: v for k, v in (params or {}).items() if not k.startswith("_")}
        if mode == "hazard":
            assert self.horizons == ["1h", "4h", "1d"][: len(self.horizons)], "hazard mode needs nested horizons"

    def _fit_one(self, X, y, idx_pool, rng, shared=False, cidx=None):
        pos, neg = idx_pool[y[idx_pool]], idx_pool[~y[idx_pool]]
        if len(pos) < self.min_pos or (shared and self.pooled is not None):
            ww = np.ones(len(idx_pool)) if self.w is None else self.w[idx_pool]
            rate = (ww[y[idx_pool]].sum() + 0.5) / (ww.sum() + 1.0)
            if self.pooled is None:
                return ("const", rate)
            base = np.average(self._proba(self.pooled, X[idx_pool]), weights=ww)
            return ("scaled", rate / max(base, 1e-12), self.pooled)
        k = min(len(neg), self.neg_ratio * len(pos))
        q = k / max(len(neg), 1)
        sel = np.sort(np.r_[pos, rng.choice(neg, k, replace=False)])
        mono = self.mono if cidx is None or self.mono is None else [self.mono[i] for i in cidx]
        Xs = X[sel] if cidx is None else X[np.ix_(sel, cidx)]
        clf = tree(self.seed, mono, self.params, self.backend).fit(
            Xs, y[sel], sample_weight=None if self.w is None else self.w[sel])
        return ("hgb", clf, q) if cidx is None else ("hgb", clf, q, np.asarray(cidx))

    @staticmethod
    def _proba(m, X):
        if m[0] == "const":
            return np.full(len(X), m[1])
        if m[0] == "scaled":
            return np.clip(HazardModel._proba(m[2], X) * m[1], 0, 1)
        p = m[1].predict_proba(X if len(m) < 4 else X[:, m[3]])[:, 1]
        q = m[2]
        return p * q / (p * q + 1 - p + 1e-12)

    def fit(self, X: np.ndarray, lead: np.ndarray, weight_mask: np.ndarray | None = None, w: np.ndarray | None = None):
        """weight_mask: rows usable for training; w: base sample weights (1 / keep-rate of a pre-subsampling)."""
        rng = np.random.default_rng(self.seed)
        self.w = w
        lead = np.where(np.isnan(lead), np.inf, lead)
        pool = np.flatnonzero(weight_mask) if weight_mask is not None else np.arange(len(X))
        self.models, self.pooled = {}, None
        ylong = lead <= max(HORIZONS[h] for h in self.horizons)
        pooled = self._fit_one(X, ylong, pool, rng)
        self.pooled = pooled if pooled[0] == "hgb" else None
        for h in self.horizons:
            if self.mode == "hazard":
                a, b = INTERVALS[h]
                at_risk = pool[lead[pool] > a]
                y = lead <= b
                self.models[h] = self._fit_one(X, y, at_risk, rng, shared=h in getattr(self, "share", ()),
                                               cidx=getattr(self, "cols", {}).get(h))
            else:
                y = lead <= HORIZONS[h]
                self.models[h] = self._fit_one(X, y, pool, rng)
        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        out, surv = {}, np.ones(len(X))
        for h in self.horizons:
            p = self._proba(self.models[h], X)
            if self.mode == "hazard":
                surv = surv * (1 - p)
                out[h] = 1 - surv
            else:
                out[h] = p if not out else np.maximum(p, out[list(out)[-1]])
        return out


# ----------------------------------------------------------------------------- AI + rule hybrid
MONO_POS = ("hi__ep_", "hi__n_", "hi__events_", "nbh__", "rule__", "sp__bank_multi", "sp__rank_multi", "sp__dimm_multi")
MONO_NEG = ("hi__h_since_last",)
APP_EVENTS = ("ev__bg_", "ev__other", "ev__all")     # background codes (tagged from the data) stay free


def monotone(cols: list[str]) -> list[int]:
    """domain constraints: more / more recent UCEs, hardware events and bank-level fault modes never lower
    the risk. Everything else (CE counts, telemetry, ...) is left free: e.g. many CEs on one cell is
    *negatively* related to UE in DRAM."""
    out = []
    for c in cols:
        hw_ev = c.startswith("ev__") and not c.startswith(APP_EVENTS)
        if c.startswith(MONO_NEG) or (hw_ev and c.endswith("_h_since")):
            out.append(-1)
        elif c.startswith(MONO_POS) or hw_ev:
            out.append(1)
        else:
            out.append(0)
    return out


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


class HybridModel:
    """AI + rule hybrid.

    1. Scorecard on rule levels (``rules.py``): interpretable, robust with few positives, gives the
       alarm reason.
    2. Rule-informed GBDT hazard model: features + scorecard logits, with monotone constraints that encode
       domain knowledge (``monotone``); 'since' features are filled with a large value instead of NaN so
       that "never happened" sits at the low-risk end of the constraint.
    3. Fusion in logit space, p = sigmoid(w * logit(p_gbdt) + (1 - w) * logit(p_rule)); w per horizon is
       chosen on the validation period by PR-AUC (w = 0.5 when it has < 10 positives); P1h <= P4h <= P1d
       is enforced by a cumulative max.
    Training rows are pre-subsampled (every row with a UCE within 1 d, other rows at rate q0) and weighted
    by 1/q0, so the probabilities stay on the natural base rate.
    """

    GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

    def __init__(self, cols, horizons=("1h", "4h", "1d"), mode="hazard", seed=0, use_rules=True, use_gbdt=True,
                 use_mono=True, pre_ratio=200, backend="lgbm", params=None, tune=0, share=(), interval_cols=None):
        """backend: 'lgbm' | 'hgb'; params: tree hyper-parameters (e.g. from an earlier tuning);
        tune: Optuna trials on the validation rows passed to fit_sampled (0 = no tuning)."""
        self.cols, self.horizons, self.mode, self.seed = list(cols), list(horizons), mode, seed
        self.backend, self.params, self.tune = backend, dict(params or {}), tune
        self.use_rules, self.use_gbdt, self.use_mono, self.pre_ratio = use_rules, use_gbdt, use_mono, pre_ratio
        self.share = tuple(share)
        self.interval_cols = dict(interval_cols or {})   # horizon -> regexes of the columns its interval model uses
        self.since = [i for i, c in enumerate(self.cols) if c.endswith(("h_since_last", "_h_since"))]

    def _prep(self, X):
        if not self.since:
            return X
        X = X.copy()
        for i in self.since:
            X[:, i] = np.nan_to_num(X[:, i], nan=1e5)
        return X

    def _df(self, X):
        return pd.DataFrame(X, columns=self.cols)

    def _rule_feats(self, X):
        p = self.sc.predict(self._df(X))
        return np.stack([_logit(p[h]) for h in self.horizons], 1).astype(np.float32), p

    @staticmethod
    def presample(lead, train_mask, Hmax, ratio=200, seed=0):
        """training rows: every row with a UCE within Hmax, other rows at rate q0 (weight 1/q0)."""
        rng = np.random.default_rng(seed)
        ld = np.where(np.isnan(lead), np.inf, lead)
        tr = np.flatnonzero(train_mask)
        hot, cold = tr[ld[tr] <= Hmax], tr[ld[tr] > Hmax]
        k = min(len(cold), ratio * max(len(hot), 1))
        q0 = k / max(len(cold), 1)
        idx = np.sort(np.r_[hot, rng.choice(cold, k, replace=False)])
        return idx, np.where(ld[idx] <= Hmax, 1.0, 1.0 / q0)

    def fit(self, X, lead, train_mask, val_mask=None):
        idx, w = self.presample(lead, train_mask, HORIZONS[self.horizons[-1]], self.pre_ratio, self.seed)
        va = np.flatnonzero(val_mask) if val_mask is not None else None
        return self.fit_sampled(X[idx], lead[idx], w, None if va is None else X[va], None if va is None else lead[va])

    def fit_sampled(self, Xs, lead_s, w, Xv=None, lead_v=None):
        """Xs, lead_s, w: pre-sampled training rows (see presample); Xv, lead_v: validation rows."""
        from .rules import Scorecard
        Xs = self._prep(Xs)
        if self.use_rules:
            self.sc = Scorecard(self.horizons, seed=self.seed).fit(self._df(Xs), lead_s, np.arange(len(Xs)), w=w)
        if self.use_gbdt:
            Z, names = Xs, list(self.cols)
            if self.use_rules:
                Z = np.hstack([Xs, self._rule_feats(Xs)[0]])
                names += [f"rule__{h}" for h in self.horizons]
            mono = monotone(names) if self.use_mono else None
            if self.tune and self.backend == "lgbm" and Xv is not None and not self.params:
                Xvp = self._prep(Xv)
                Zv = np.hstack([Xvp, self._rule_feats(Xvp)[0]]) if self.use_rules else Xvp
                Hl = HORIZONS[self.horizons[-1]]
                y = np.where(np.isnan(lead_s), np.inf, lead_s) <= Hl
                yv = np.where(np.isnan(lead_v), np.inf, lead_v) <= Hl
                self.params = tune_tree(Z, y, w, Zv, yv, mono, self.tune, self.seed)
                del Zv
            ic = {h: [i for i, c in enumerate(names) if any(re.search(p, c) for p in pats)]
                  for h, pats in getattr(self, "interval_cols", {}).items()}
            self.gb = HazardModel(self.horizons, self.mode, seed=self.seed, mono=mono, params=self.params,
                                  backend=self.backend, share=getattr(self, "share", ()), cols={h: v for h, v in ic.items() if v})
            self.gb.fit(Z, lead_s, None, w=w)
        self.wf = {h: (1.0 if not self.use_rules else 0.0 if not self.use_gbdt else 0.5) for h in self.horizons}
        if self.use_rules and self.use_gbdt and Xv is not None:
            ld = np.where(np.isnan(lead_v), np.inf, lead_v)
            parts = [self._components(Xv[i:i + 200_000]) for i in range(0, len(Xv), 200_000)]
            comp = {k: {h: np.concatenate([p[k][h] for p in parts]) for h in self.horizons} for k in parts[0]}
            for h in self.horizons:
                y = ld <= HORIZONS[h]
                if y.sum() < 10:
                    continue
                self.wf[h] = max(self.GRID, key=lambda a: (round(average_precision_score(y, self._fuse(comp, h, a)), 4),
                                                         -abs(a - 0.5)))
        return self

    def _components(self, X):
        Xp = self._prep(X)
        out = {}
        if self.use_rules:
            rf, out["rule"] = self._rule_feats(Xp)
        if self.use_gbdt:
            out["gbdt"] = self.gb.predict(np.hstack([Xp, rf]) if self.use_rules else Xp)
        return out

    @staticmethod
    def _fuse(comp, h, a):
        if "rule" not in comp:
            return comp["gbdt"][h]
        if "gbdt" not in comp:
            return comp["rule"][h]
        return 1 / (1 + np.exp(-(a * _logit(comp["gbdt"][h]) + (1 - a) * _logit(comp["rule"][h]))))

    def predict(self, X, return_components=False, chunk=200_000):
        if len(X) > chunk and not return_components:
            parts = [self.predict(X[i:i + chunk]) for i in range(0, len(X), chunk)]
            return {h: np.concatenate([p[h] for p in parts]) for h in self.horizons}
        comp = self._components(X)
        out, prev = {}, None
        for h in self.horizons:
            p = self._fuse(comp, h, self.wf[h])
            out[h] = prev = p if prev is None else np.maximum(p, prev)
        return (out, comp) if return_components else out

    def reasons(self, X, h="1d"):
        return self.sc.reasons(self._df(self._prep(X)), h) if self.use_rules else np.full(len(X), "", dtype=object)


def ensemble_predict(models, X, horizons):
    """average of bagged models in logit space, then P1h <= P4h <= P1d."""
    ps = [m.predict(X) for m in models]
    out, prev = {}, None
    for h in horizons:
        z = np.mean([_logit(p[h]) for p in ps], 0)
        v = 1 / (1 + np.exp(-z))
        out[h] = prev = v if prev is None else np.maximum(v, prev)
    return out
