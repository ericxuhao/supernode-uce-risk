"""High-level interface: train once, score any device at any time, save / load.

    from uce_risk import adapters
    from uce_risk.api import UCERiskPredictor

    data = adapters.calchas_raw()                         # or your own UCEData
    pred = UCERiskPredictor(scenario="auto").fit(data, val_start=t_val)
    risk = pred.predict(data, at=t_now)                   # one row per device
    pred.save("uce_model.joblib"); pred = UCERiskPredictor.load("uce_model.joblib")

``fit`` follows the validated protocol: model A is trained on samples up to ``val_start`` and fixes the
fusion weights and alarm thresholds on the validation period; model B is then refit on all samples with
the same weights (``refit=True``). Both are bags of ``bags`` seeds averaged in logit space.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import evaluate, features, schema
from .model import HybridModel, ensemble_predict
from .schema import HORIZONS, UCEData

KEEP_NAN = ("h_since", "h_since_last", "h_since_first")


@dataclass
class UCERiskPredictor:
    scenario: str = "auto"              # 'S1_tel+xid' | 'S2_ce' | 'S3_tel+ce' | 'ALL' | 'auto' (all available inputs)
    horizons: tuple = ("1h", "4h", "1d")
    bags: int = 3
    drop_history: bool = False          # True: precursors only (no past-UCE features)
    alarm_fpr: float = 0.01             # operating point 'fpr1': 1 % of UCE-free devices alarm in validation
    refit: bool = True
    backend: str = "lgbm"               # tree learner: 'lgbm' (LightGBM) | 'hgb' (scikit-learn)
    tune: int = 20                      # Optuna trials for the tree hyper-parameters (0 = defaults)
    variant: str = "hybrid"             # 'hybrid' (AI + rules) | 'ai' (tree model only) | 'rules' (scorecard only)
    seed: int = 0
    verbose: bool = False
    include: tuple = ()                 # regexes: keep only matching feature columns (empty = all of the scenario)
    exclude: tuple = ()                 # regexes: drop matching feature columns
    params: dict = field(default_factory=dict)   # base tree parameters (non-empty: used as given, no tuning)
    share: tuple = ()                   # horizons whose hazard interval reuses the pooled ranking (few positives)
    interval_cols: dict = field(default_factory=dict)   # horizon -> regexes: columns of that interval's tree model
    # learned state
    cols: list = field(default_factory=list)
    models: list = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    input_spec: dict = field(default_factory=dict)   # event-code set of the training input; pass to from_tables(**spec)

    # ------------------------------------------------------------------ training
    def fit(self, data: UCEData, val_start: int, samples: tuple[pd.DataFrame, pd.DataFrame] | None = None):
        """data: canonical input; val_start: samples with s <= val_start train model A, later samples validate.
        samples: optional pre-built (S, episodes) from features.build to avoid recomputing."""
        S, ep = samples if samples is not None else features.build(data, verbose=self.verbose)
        present = data.modalities()
        sc = self._scenario(present)
        feat = [c for c in S.columns if "__" in c]
        self.cols = schema.scenario_columns(feat, sc, drop_history=self.drop_history)
        if self.include:
            self.cols = [c for c in self.cols if any(re.search(p, c) for p in self.include)]
        if self.exclude:
            self.cols = [c for c in self.cols if not any(re.search(p, c) for p in self.exclude)]
        if not self.cols:
            raise ValueError(f"no features for scenario {sc}; available inputs: {sorted(present)}")
        hz, mode = list(self.horizons), ("hazard" if list(self.horizons) == ["1h", "4h", "1d"] else "direct")
        X = S[self.cols].to_numpy(np.float32)
        lead, s = S.lead.to_numpy(), S.s.to_numpy()
        ok = ~S.excl.to_numpy(bool)
        tr, va = (s <= val_start) & ok, s > val_start
        if not va.any():
            raise ValueError("no samples after val_start")
        A = []
        for k in range(self.bags):      # the first bag tunes the trees, the others (and the refit) reuse its params
            idx, w = HybridModel.presample(lead, tr, HORIZONS[hz[-1]], 200, self.seed + k)
            m = HybridModel(self.cols, hz, mode, seed=self.seed + k, backend=self.backend,
                            params=A[0].params if A else (dict(self.params) or None),
                            tune=0 if (A or self.params) else self.tune, **self._flags())
            A.append(m.fit_sampled(X[idx], lead[idx], w, X[va], lead[va]))
        pv = ensemble_predict(A, X[va], hz)
        L = S[["entity", "s", "lead", "ep"]]
        v0, v1 = int(val_start), int(s[va].max())
        self.thresholds = {}
        for h in hz:
            self.thresholds[h] = {
                "fpr1": evaluate.pick_threshold(L[va], pv[h], h, ep, v0, v1, fpr=self.alarm_fpr),
                "valF1": evaluate.best_f1(*evaluate.entity_scores(L[va], pv[h], h, ep, v0, v1))[0]}
        self.models = A
        if self.refit:
            self.models = []
            for k, ma in enumerate(A):
                idx, w = HybridModel.presample(lead, (tr | va) & ok, HORIZONS[hz[-1]], 200, self.seed + k)
                mb = HybridModel(self.cols, hz, mode, seed=self.seed + k, backend=self.backend,
                                 params=A[0].params, **self._flags()).fit_sampled(X[idx], lead[idx], w)
                mb.wf = dict(ma.wf)
                self.models.append(mb)
        rep = getattr(data, "report", {}).get("fields", {})
        self.input_spec = {"keep_codes": [c[3:] if c.startswith("bg_") else c for c in rep.get("event_codes", []) if c != "other"],
                           "background_codes": list(rep.get("background_codes", []))}
        self.info = dict(scenario=sc, inputs=sorted(present), n_features=len(self.cols), backend=self.backend,
                         tree_params=dict(A[0].params),
                         groups=sorted({c.split("__")[0] for c in self.cols}),
                         fusion_weight_ai={h: float(np.mean([m.wf[h] for m in A])) for h in hz},
                         train_samples=int(tr.sum()), val_samples=int(va.sum()),
                         val_positives={h: int((lead[va] <= HORIZONS[h]).sum()) for h in hz})
        return self

    def _flags(self) -> dict:
        f = {"hybrid": {}, "ai": {"use_rules": False, "use_mono": False}, "rules": {"use_gbdt": False}}[self.variant]
        return dict(f, share=tuple(getattr(self, "share", ())), interval_cols=dict(getattr(self, "interval_cols", {})))

    def _scenario(self, present):
        if self.scenario != "auto":
            if not schema.runnable(self.scenario, present):
                raise ValueError(f"scenario {self.scenario} needs inputs that are missing: have {sorted(present)}")
            return self.scenario
        return "ALL"

    # ------------------------------------------------------------------ scoring
    def predict(self, data: UCEData, at=None, op: str = "fpr1", explain: bool = True) -> pd.DataFrame:
        """Risk of a UCE within each horizon.

        at: None (every built-in sample time), an int time (every device at that time) or a DataFrame with
            columns entity, s (explicit prediction times). Only records with t < s are used.
        returns: entity, s, p_1h, p_4h, p_1d, alarm_<h> (bool, threshold of operating point `op`),
                 reason (top rule behind the 1-day risk, if explain).
        """
        if at is None:
            query = None
        elif isinstance(at, pd.DataFrame):
            query = at[["entity", "s"]]
        else:
            ids = data.entities.index.to_numpy()
            query = pd.DataFrame({"entity": ids, "s": np.full(len(ids), int(at))})
        S, _ = features.build(data, verbose=False, query=query)
        X = self.matrix(S)
        p = ensemble_predict(self.models, X, list(self.horizons))
        out = S[["entity", "s"]].copy()
        for h in self.horizons:
            out[f"p_{h}"] = p[h]
            out[f"alarm_{h}"] = p[h] > self.thresholds[h][op]
        if explain:
            out["reason"] = self.models[0].reasons(X, list(self.horizons)[-1])
        keys = [c for c in ("host", "device") if c in data.entities]
        if keys:                                   # original device keys of the input
            out = out.merge(data.entities[keys], left_on="entity", right_index=True, how="left")
        return out.reset_index(drop=True)

    def matrix(self, S: pd.DataFrame) -> np.ndarray:
        """align a feature table to the training columns; absent count features mean 'no records' -> 0."""
        X = S.reindex(columns=self.cols)
        for c in self.cols:
            if c not in S.columns and not c.endswith(KEEP_NAN) and not c.startswith((schema.G_TE, schema.G_CF)):
                X[c] = 0.0
        return X.to_numpy(np.float32)

    def explain(self, horizon: str = "1d", top: int = 15) -> pd.Series:
        """rule-scorecard weights (log-odds) of the first bag: what raises / lowers the risk."""
        if not self.models[0].use_rules:
            return pd.Series(dtype=float)
        return self.models[0].sc.weights(horizon).head(top)

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        import joblib
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "UCERiskPredictor":
        import joblib
        return joblib.load(path)
