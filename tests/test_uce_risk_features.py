"""Brute-force check of the vectorised feature engine on a small random event log."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uce_risk import features, schema  # noqa: E402
from uce_risk.model import HazardModel  # noqa: E402

DAY, HOUR = schema.DAY, schema.HOUR


def make(seed=0):
    rng = np.random.default_rng(seed)
    n = 3000
    ce = pd.DataFrame({"entity": rng.integers(0, 6, n), "t": rng.integers(0, 40 * DAY, n),
                       "bank": rng.integers(0, 3, n), "row": rng.integers(0, 6, n), "col": rng.integers(0, 4, n),
                       "sub": rng.integers(1, 3, n)})
    uce = pd.DataFrame({"entity": rng.integers(0, 6, 25), "t": rng.integers(0, 40 * DAY, 25)})
    ev = pd.DataFrame({"entity": rng.integers(0, 6, 400), "t": rng.integers(0, 40 * DAY, 400), "code": rng.choice(["a", "b"], 400)})
    tel = pd.DataFrame({"entity": np.repeat(np.arange(6), 2000), "t": np.tile(np.arange(2000) * 1800, 6)})
    tel["temp"] = rng.normal(60, 5, len(tel))
    ents = pd.DataFrame({"group": [0, 0, 0, 1, 1, 1], "kind": list("xyxyxy")}, index=range(6))
    return schema.UCEData(name="toy", entities=ents, uce=uce, ce=ce, events=ev, telemetry=tel, t_start=0, t_end=40 * DAY,
                          hier=["bank"], leaf=("row", "col"), cat_cols=["kind"], event_codes=["a", "b"], storm_per_min=2, shards=2)


def test_features_match_bruteforce():
    d = make()
    S, ep = features.build(d, verbose=False)
    ce, uce, ev, tel = d.ce, d.uce, d.events, d.telemetry
    rng = np.random.default_rng(1)
    for i in rng.choice(len(S), 150, replace=False):
        e, s = int(S.entity[i]), int(S.s[i])
        c = ce[(ce.entity == e) & (ce.t < s)]
        w = c[c.t >= s - DAY]
        assert S["ce__n_1d"][i] == len(w)
        assert S["ce__n_cum"][i] == len(c)
        assert S["ce__sub1_1h"][i] == ((c.t >= s - HOUR) & (c["sub"] == 1)).sum()
        assert S["sp__n_cell_1d"][i] == len(w.groupby(["bank", "row", "col"]))
        assert S["sp__n_row_cum"][i] == len(c.groupby(["bank", "row"]))
        assert S["sp__max_per_cell"][i] == (c.groupby(["bank", "row", "col"]).size().max() if len(c) else 0)
        assert S["sp__bank_multi_row"][i] == (c.groupby("bank").row.nunique() >= 2).sum()
        assert S["sp__row_multi_cell"][i] == (c.groupby(["bank", "row"]).col.nunique() >= 2).sum()
        hard = c.groupby(["bank", "row", "col"]).t.agg(lambda x: (x >= x.min() + DAY).any()).sum() if len(c) else 0
        assert S["sp__hard_cell"][i] == hard
        allc = ce[ce.entity == e]
        cnt = allc.groupby(allc.t // 60).size()
        ends = (cnt[cnt >= d.storm_per_min].index.to_numpy() + 1) * 60
        assert S["ce__storm_min_30d"][i] == ((ends >= s - 30 * DAY) & (ends < s)).sum()
        last = c.t.max() if len(c) else np.nan
        assert np.isclose(S["ce__h_since_last"][i], (s - last) / HOUR, equal_nan=True)
        x = ev[(ev.entity == e) & (ev.t < s) & (ev.code == "a")]
        assert S["ev__a_7d"][i] == (x.t >= s - 7 * DAY).sum()
        u = uce[(uce.entity == e) & (uce.t < s)]
        assert S["hi__h_since_last"][i] == np.float32((s - u.t.max()) / HOUR) or (len(u) == 0 and np.isnan(S["hi__h_since_last"][i]))
        # neighbours: other entities of the same group
        g = d.entities.group[e]
        sib = d.entities.index[(d.entities.group == g) & (d.entities.index != e)]
        nc = ce[ce.entity.isin(sib) & (ce.t < s) & (ce.t >= s - DAY)]
        assert S["nbc__ce_1d"][i] == len(nc)
        tt = tel[(tel.entity == e) & (tel.t < s) & (tel.t >= s - DAY)]
        assert np.isclose(S["te__temp_mean1d"][i], tt.temp.mean() if len(tt) else np.nan, equal_nan=True, rtol=1e-4)
        # label: next UCE episode start strictly after s
        ue = np.sort(uce[uce.entity == e].t.to_numpy())
        starts = [t for k, t in enumerate(ue) if k == 0 or t - ue[k - 1] > d.episode_gap]
        nxt = [t for t in starts if t > s]
        assert np.isclose(S.lead[i], nxt[0] - s if nxt else np.nan, equal_nan=True)


def test_hazard_monotone():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 5)).astype(np.float32)
    lead = np.where(rng.random(20000) < 0.05 + 0.1 * (X[:, 0] > 1), rng.uniform(0, 2 * DAY, 20000), np.nan)
    p = HazardModel().fit(X, lead).predict(X)
    assert (p["1h"] <= p["4h"] + 1e-12).all() and (p["4h"] <= p["1d"] + 1e-12).all()
    y = lead <= DAY
    assert abs(p["1d"].mean() - np.nanmean(y)) < 0.02          # prior correction keeps the base rate


if __name__ == "__main__":
    test_features_match_bruteforce()
    test_hazard_monotone()
    print("ok")
