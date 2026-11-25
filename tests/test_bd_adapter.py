"""BD adapter: narrow -> wide table, counter deltas and resets, chunk carry-over, labels and work orders, end to end."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uce_risk import bd, features  # noqa: E402

T0 = 1788739200            # 2026-09-07 00:00 UTC
P = "DCGM_FI_DEV_"


def narrow(rows):
    return pd.DataFrame(rows, columns=["timestamp", "uuid", "hostname", "ipv4", "gpu", "metric", "value"])


def test_metric_table_pivot_and_host_rows():
    t = narrow([
        (T0 + 2, "u0", "h", "ip", 0, P + "GPU_TEMP", 40), (T0 + 47, "u0", "h", "ip", 0, P + "GPU_TEMP", 42),
        (T0 + 5, "u1", "h", "ip", 1, P + "GPU_TEMP", 50), (T0 + 10, "u0", "h", "ip", 0, P + "XID_ERRORS", 48),
        (T0 + 40, "u0", "h", "ip", 0, P + "XID_ERRORS", 0), (T0 + 70, "u0", "h", "ip", 0, P + "XID_ERRORS", 0),
        (T0 + 8, None, "h", "ip", None, "cpu_power_usage_watts", 300.5), (T0 + 65, "u1", "h", "ip", 1, "gpu_temperature", 51),
    ])
    w = bd.metric_table(t).set_index(["uuid", "timestamp"])
    assert w.loc[("u0", T0), P + "GPU_TEMP"] == 42                      # last value of the minute
    assert w.loc[("u0", T0), P + "XID_ERRORS"] == 48                     # last non-zero error code of the minute
    assert w.loc[("u0", T0 + 60), P + "XID_ERRORS"] == 0
    assert w.loc[("u1", T0 + 60), P + "GPU_TEMP"] == 51                  # snake_case name mapped to the collector name
    assert w.loc[("u0", T0), "cpu_power_usage_watts"] == w.loc[("u1", T0), "cpu_power_usage_watts"] == 300.5
    assert set(w.date) == {"20260907"} and set(w.hour) == {"00"}


def wide(uuid, vol, agg, start=0, gaps=(), **cols):
    n = len(vol)
    ts = T0 + 60 * (start + np.arange(n) + np.cumsum([g for g in gaps] + [0] * (n - len(gaps))))
    df = pd.DataFrame({"timestamp": ts, "uuid": uuid, P + "ECC_SBE_VOL_TOTAL": vol, P + "ECC_SBE_AGG_TOTAL": agg})
    for k, v in cols.items():
        df[P + k] = v
    return df


def derive(*chunks, inv=None):
    ent = {"u0": 0, "u1": 1}
    dv = bd._Deriver(ent)
    for c in chunks:
        dv.feed(c)
    return dv.result()


def test_counter_reset_and_gap():
    # minutes 0..4: volatile counter resets at minute 3, the aggregate one keeps counting
    ce, uce, ev, te = derive(wide("u0", [0, 1, 3, 0, 2], [10, 11, 13, 13, 15]))
    assert ce.n.tolist() == [1, 2, 2] and (ce.t - T0).tolist() == [119, 179, 299]
    assert "DRIVER_RESET" in set(ev.code)
    # errors during a 20-minute collector gap followed by a reset: only the aggregate counter shows them
    ce, uce, ev, te = derive(wide("u0", [0, 2, 0], [0, 2, 6], gaps=[0, 20]))
    assert ce.n.tolist() == [2, 4]
    assert {"COLLECTOR_GAP", "DRIVER_RESET"} <= set(ev.code)


def test_chunks_equal_single_pass():
    rng = np.random.default_rng(0)
    inc = rng.poisson(0.3, 300)
    vol = np.cumsum(inc)
    vol[150:] -= vol[149]                                    # reset in the middle
    df = wide("u0", vol, 5 + np.cumsum(inc), ECC_DBE_VOL_TOTAL=np.r_[np.zeros(200), np.ones(100)],
              ROW_REMAP_PENDING=np.r_[np.zeros(120), np.ones(180)])
    one = derive(df)
    two = derive(df.iloc[:137], df.iloc[137:])
    for a, b in zip(one, two):
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
    assert int(one[0].n.sum()) == int(inc[1:].sum())
    assert len(one[1]) == 1 and (one[1].t - T0).iloc[0] == 200 * 60 + 59
    assert (one[2].code == "REMAP_PENDING").sum() == 1


def test_labels_and_work_orders(tmp_path):
    w = pd.concat([wide("u0", np.zeros(180, int), np.zeros(180, int), ECC_DBE_VOL_TOTAL=np.r_[np.zeros(100), np.ones(80)],
                        XID_ERRORS=np.r_[np.zeros(100), [48, 94], np.zeros(78)]),
                   wide("u1", np.zeros(180, int), np.zeros(180, int))], ignore_index=True)
    tel = narrow([(T0, "u0", "h0", "ip0", 0, P + "COUNT", 2), (T0, "u1", "h0", "ip0", 1, P + "COUNT", 2)])
    t_uce = T0 + 100 * 60 + 12
    wo = pd.DataFrame([dict(fault_type="GPU", ipv4="ip0", fault_sub_type="xid_48", fault_timestamp=t_uce, event_id="e1",
                            timestamp=t_uce + 65, parameters_json=json.dumps({"operation": "OFFLINE"})),
                       dict(fault_type="GPU", ipv4="ip0", fault_sub_type="xid_48", fault_timestamp=t_uce, event_id="e1",
                            timestamp=t_uce + 164, parameters_json="{}"),
                       dict(fault_type="GPU", ipv4="ip0", fault_sub_type="xid_48", fault_timestamp=T0 + 3 * 3600,
                            event_id="e2", timestamp=T0 + 3 * 3600, parameters_json="{}"),
                       dict(fault_type="Other", ipv4="ip0", fault_sub_type="NodeNotReady", fault_timestamp=T0 + 600,
                            event_id="e3", timestamp=T0 + 600, parameters_json="{}")])
    d = bd.from_bd(wo, tel, w, warmup_days=0, min_code_rows=1)
    lab = d.report["labels"]
    assert lab["uce_records"] == 1 and lab["wo_confirmed"] == 1 and lab["wo_unattributed"] == 1
    assert int(d.uce.t.iloc[0]) == t_uce                     # the work order times the fault exactly
    assert d.entities.loc[d.uce.entity.iloc[0], "device"] == "u0"
    codes = set(d.events.code)
    assert {"WO_NodeNotReady", "WO_UCE_UNATTRIBUTED"} <= codes
    assert d.entities.group.nunique() == 1                   # both cards on one host


def test_persistent_error_code():
    """the error-code field keeps its last value: a collector gap or repeated scrapes must not create new UCEs."""
    xid = np.r_[np.zeros(30), np.full(150, 48)]
    w = wide("u0", np.zeros(180, int), np.zeros(180, int), gaps=[0] * 90 + [30], XID_ERRORS=xid,
             ECC_DBE_VOL_TOTAL=np.r_[np.zeros(30), np.ones(150)])
    tel = narrow([(T0 + 30 * 60 + 15 * k, "u0", "h0", "ip0", 0, P + "XID_ERRORS", 48) for k in range(200)] +
                 [(T0, "u0", "h0", "ip0", 0, P + "XID_ERRORS", 0)])
    d = bd.from_bd(None, tel, w, warmup_days=0, label_complete=False)
    assert d.report["inputs"]["error_code_semantics"].startswith("last value")
    assert len(d.uce) == 1 and abs(int(d.uce.t.iloc[0]) - (T0 + 30 * 60)) <= 60


def test_reason_zh():
    assert bd.reason_zh("hw_event_REMAP_PENDING=<=1d") == "近 1 天出现待完成的行重映射"
    assert bd.reason_zh("ce_burst=high") == "近 1 小时可纠正错误数很高"
    assert bd.reason_zh("tel_high_mem_temp_mean1h=top10%").startswith("设备内存温度")
    assert bd.reason_zh("baseline") == ""


@pytest.mark.skipif(not (ROOT / "dataset" / "real_sample" / "遥测数据样例.csv").exists(), reason="real sample not present")
def test_real_sample_files():
    S = ROOT / "dataset" / "real_sample"
    tel = pd.read_csv(S / "遥测数据样例.csv", dtype=str)
    inv = bd.inventory(tel)
    assert len(inv) == 3 and inv.host.nunique() == 1 and inv.tray_index.notna().all()
    w = bd.metric_table(tel)
    assert len(w) == 3 and w[P + "P2P_NVLINK_STATUS"].dropna().tolist() == [11]
    wo = bd.work_orders(pd.read_excel(S / "工单样例.xlsx"))
    assert len(wo) == 2 and set(wo.role) == {"uce", "event"}


def test_end_to_end_small():
    rng = np.random.default_rng(1)
    n = 3 * 1440
    frames, inv = [], []
    for h in range(3):
        for g in range(4):
            u = f"u{h}{g}"
            faulty = (h, g) in ((0, 1), (2, 3))
            lam = np.where(np.arange(n) > n - 1500, 0.2, 0.002) if faulty else np.full(n, 0.001)
            inc = rng.poisson(lam)
            dbe = np.zeros(n)
            if faulty:
                dbe[n - 300:] = 1
            frames.append(pd.DataFrame({"timestamp": T0 + 60 * np.arange(n), "uuid": u, P + "ECC_SBE_VOL_TOTAL": np.cumsum(inc),
                                        P + "ECC_SBE_AGG_TOTAL": np.cumsum(inc), P + "ECC_DBE_VOL_TOTAL": dbe,
                                        P + "GPU_TEMP": 60 + rng.normal(0, 2, n), P + "MEMORY_TEMP": 65 + rng.normal(0, 2, n),
                                        "DCGM_FI_PROF_DRAM_ACTIVE": np.clip(0.5 + rng.normal(0, 0.1, n), 0, 1)}))
            inv.append((T0, u, f"h{h}", f"ip{h}", g, P + "COUNT", 4))
    d = bd.from_bd(None, narrow(inv), pd.concat(frames, ignore_index=True), warmup_days=0, label_complete=False)
    S, ep = features.build(d, verbose=False)
    assert len(ep) == 2 and {"ce__n_1h", "te__mem_temp_mean1h", "hi__ep_cum"} <= set(S.columns)
    assert (S.s % 3600 == 0).sum() > 12 * 60                    # hourly grid for every reporting card
    from uce_risk.api import UCERiskPredictor
    pred = UCERiskPredictor(bags=1, **bd.PROFILE).fit(d, val_start=T0 + 2 * 86400 + 3600, samples=(S, ep))
    assert all(any(re.search(p, c) for p in bd.COMPACT) for c in pred.cols) and "ce__n_1d_per_access" in pred.cols
    out = bd.predict(pred, d, at=T0 + n * 60 - 400 * 60)
    assert len(out) == 12 and {"p_1h", "p_4h", "p_24h", "risk_level", "reason_zh"} <= set(out.columns)
    assert (out.p_1h <= out.p_4h + 1e-9).all() and (out.p_4h <= out.p_24h + 1e-9).all()
