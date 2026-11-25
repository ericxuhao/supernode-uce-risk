"""End-to-end check of the field-driven input layer and the high-level predictor on synthetic GPU data
with non-standard column names: telemetry time series, a cumulative CE counter, an Xid log with UCEs."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uce_risk import inputs  # noqa: E402
from uce_risk.api import UCERiskPredictor  # noqa: E402

DAY = 86400


def synthetic(seed=0, hosts=6, gpus=4, days=90):
    rng = np.random.default_rng(seed)
    t0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp())
    dev = pd.DataFrame([(f"node{h:02d}", g) for h in range(hosts) for g in range(gpus)], columns=["Hostname", "GPU_Index"])
    dev["Model"] = np.where(dev.index % 3 == 0, "A100", "H100")
    weak = rng.random(len(dev)) < 0.35                       # devices with a degrading memory
    tel, cnt, xid = [], [], []
    for i, r in dev.iterrows():
        ts = t0 + np.arange(0, days * DAY, 600)
        temp = 60 + 5 * weak[i] + rng.normal(0, 2, len(ts))
        tel.append(pd.DataFrame({"hostname": r.Hostname, "gpu_index": r.GPU_Index,
                                 "collect_time": pd.to_datetime(ts, unit="s", utc=True).astype(str),
                                 "mem_temp_c": temp, "power_w": 250 + rng.normal(0, 20, len(ts))}))
        rate = (3.0 if weak[i] else 0.2) / DAY
        ce_t = np.sort(rng.uniform(t0, t0 + days * DAY, rng.poisson(rate * days * DAY)))
        cnt.append(pd.DataFrame({"Host": r.Hostname, "GPU": r.GPU_Index, "time_stamp": ce_t * 1000,   # epoch ms
                                 "SBE_Count": np.arange(1, len(ce_t) + 1)}))                         # cumulative
        n_u = rng.poisson(6 if weak[i] else 0.3)
        for tu in rng.uniform(t0 + 5 * DAY, t0 + days * DAY, n_u):
            xid.append((r.Hostname, r.GPU_Index, tu, 48))
            if rng.random() < 0.7:
                xid.append((r.Hostname, r.GPU_Index, tu - rng.uniform(600, 6 * 3600), 63))
        for tx in rng.uniform(t0, t0 + days * DAY, rng.poisson(20)):
            xid.append((r.Hostname, r.GPU_Index, tx, 31))
    xl = pd.DataFrame(xid, columns=["node", "gpu", "ts", "XID"])
    xl["ts"] = pd.to_datetime(xl.ts, unit="s", utc=True)
    return dev, pd.concat(tel), pd.concat(cnt), xl, t0


def test_field_driven_predictor():
    dev, tel, cnt, xl, t0 = synthetic()
    d = inputs.from_tables(errors=xl, uce_codes=[48], ce=cnt, telemetry=tel, devices=dev, min_devices_for_background=10)
    rep = inputs.describe(d)
    mods = " ".join(rep["modules"])
    for m in ("CE counts", "discrete events", "telemetry time series", "device configuration", "same-node"):
        assert m in mods, (m, rep)
    assert "cumulative counter" in rep["fields"]["ce_count"]
    assert "spatial" not in mods                                   # no address fields -> no spatial module
    assert rep["fields"]["background_codes"] == ["xid31"]          # seen on every device -> background
    assert (d.ce.n >= 0).all() and d.ce.n.max() == 1               # counter turned into per-record deltas
    val_start = t0 + 60 * DAY
    for sc in ("S1_tel+xid", "S2_ce", "S3_tel+ce", "auto"):
        p = UCERiskPredictor(scenario=sc, bags=2).fit(d, val_start=val_start)
        groups = set(p.info["groups"])
        if sc == "S2_ce":
            assert "te" not in groups and "ev" not in groups and "ce" in groups
        if sc == "S1_tel+xid":
            assert "ce" not in groups and {"te", "ev"} <= groups
        if sc == "S3_tel+ce":
            assert {"te", "ce"} <= groups and "ev" not in groups
        out = p.predict(d, at=t0 + 80 * DAY)
        assert len(out) == len(d.entities)
        P = out[["p_1h", "p_4h", "p_1d"]].to_numpy()
        assert np.isfinite(P).all() and (P >= 0).all() and (P <= 1).all()
        assert (np.diff(P, axis=1) >= -1e-9).all()
        assert out.reason.notna().all()
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "m.joblib"
        p.save(f)
        q = UCERiskPredictor.load(f)
        o1, o2 = p.predict(d, at=t0 + 80 * DAY), q.predict(d, at=t0 + 80 * DAY)
        assert np.allclose(o1.p_1d, o2.p_1d)
    # a scenario whose inputs are missing must be refused
    d2 = inputs.from_tables(errors=xl, uce_codes=[48])
    try:
        UCERiskPredictor(scenario="S3_tel+ce").fit(d2, val_start=val_start)
        raise AssertionError("expected refusal")
    except ValueError:
        pass
    return rep, p


if __name__ == "__main__":
    rep, p = test_field_driven_predictor()
    print(rep["modules"])
    print(p.info)
    print(p.explain().head(8))
    print("ok")
