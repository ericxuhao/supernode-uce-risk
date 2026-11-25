"""Cluster mode: capability judgement, agent registration, ingest of collector records, scheduled scan,
alarms with fault-tree attribution, collection-interval settings and restart recovery (temporary data dir)."""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ("supernode" if (ROOT / "supernode").exists() else "supernode_demo")))
import joblib  # noqa: E402
import json  # noqa: E402

import cluster  # noqa: E402
from common import MODEL_DIR  # noqa: E402

A100_CAPS = {"tool": True, "devices": [{"index": i, "uuid": f"DEV-{i}"} for i in range(8)],
             "metrics_supported": ["DCGM_FI_DEV_GPU_TEMP", "DCGM_FI_DEV_MEMORY_TEMP", "DCGM_FI_DEV_POWER_USAGE",
                                   "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL", "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL",
                                   "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS", "DCGM_FI_DEV_ROW_REMAP_FAILURE"],
             "metrics_na_on_device": [], "metrics_unsupported": [], "kernel_log": "journalctl",
             "device_events": False, "interconnect": True}
NO_ECC_CAPS = {"tool": True, "devices": [{"index": 0, "uuid": "DEV-0"}],
               "metrics_supported": ["DCGM_FI_DEV_GPU_TEMP", "DCGM_FI_DEV_POWER_USAGE",
                                     "nvsmi.ecc.errors.uncorrected.volatile.sram.parity"],
               "metrics_na_on_device": ["DCGM_FI_DEV_ECC_SBE_VOL_TOTAL", "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL"],
               "metrics_unsupported": [], "kernel_log": None, "device_events": False, "interconnect": False}


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def records(host, now):
    out = []
    for k in range(12):                     # one hour of 5-minute samples for 8 cards
        t = now - 3600 + 300 * k
        for i in range(8):
            out.append({"ts": iso(t), "host": host, "tier": "fast", "gpu": i,
                        "m": {"DCGM_FI_DEV_GPU_TEMP": 55.0, "DCGM_FI_DEV_MEMORY_TEMP": 60.0 + (8 if i == 5 else 0),
                              "DCGM_FI_DEV_POWER_USAGE": 300.0}})
    out.append({"ts": iso(now - 1800), "host": host, "type": "counter_increase", "gpu": 2,
                "metric": "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL", "delta": 40})
    out.append({"ts": iso(now - 1200), "host": host, "type": "counter_increase", "gpu": 2,
                "metric": "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS", "delta": 1})
    out.append({"ts": iso(now - 600), "host": host, "type": "xid", "gpu": 3, "xid": 48,
                "event_time": iso(now - 610), "message": "uncorrectable"})
    return out


def test_cluster_mode():
    tmp = Path(tempfile.mkdtemp(prefix="cluster_"))
    cluster.RAW = tmp / "raw"
    cluster.SETTINGS_PATH = tmp / "settings.json"
    cluster.DATA = tmp                       # hosts.json / inventory under the temp dir
    models = joblib.load(MODEL_DIR / "models.joblib")
    meta = json.loads((MODEL_DIR / "build_meta.json").read_text(encoding="utf-8"))

    s1, s2 = cluster.capability_summary(A100_CAPS), cluster.capability_summary(NO_ECC_CAPS)
    assert s1["scenario"].startswith("方案 3") and s1["inputs"]["不可纠正错误标签"], s1
    assert s2["scenario"].startswith("仅遥测") and not s2["inputs"]["可纠正错误统计"], s2

    C = cluster.ClusterEngine(models, meta)
    assert C.settings["interval_s"] == 300                       # default collection interval: 5 minutes
    C.register("node-a", A100_CAPS, 300)
    now = int(time.time())
    C.ingest("node-a", records("node-a", now))
    C.step()
    sc = C.scans[-1]
    assert not sc.get("error"), sc
    R = C.current.set_index("card")
    card3 = C.key[(C.host_node["node-a"], "C3")]
    card2 = C.key[(C.host_node["node-a"], "C2")]
    card0 = C.key[(C.host_node["node-a"], "C0")]
    assert R.loc[card3, "lvl_1d"] >= 1 and R.loc[card3, "p_1d"] > 100 * R.loc[card0, "p_1d"]   # recent UCE
    assert R.loc[card2, "p_1d"] > 20 * R.loc[card0, "p_1d"]                        # CE + row remap -> elevated
    a = C.attrib[card3]
    assert a["category"] == "显存子系统故障" and "复发" in a["root_cause"], a
    tel = C.card_telemetry(card3, now - 86400, now)
    assert len(tel["t"]) == 12

    # settings: interval and retention, persisted
    C.settings["interval_s"] = 600
    cluster.save_settings(C.settings)
    assert cluster.load_settings()["interval_s"] == 600
    est = C.storage_estimate()
    assert est["cards"] == 8 and est["mb_per_day"] > 0

    # restart: state rebuilt from the raw records, same topology position
    C2 = cluster.ClusterEngine(models, meta)
    assert C2.host_node["node-a"] == C.host_node["node-a"] and len(C2.events) == len(C.events)
    C2.step()
    assert C2.current.set_index("card").loc[card3, "lvl_1d"] >= 1
    return C


if __name__ == "__main__":
    C = test_cluster_mode()
    print(C.current.sort_values("p_1d", ascending=False)[["card", "p_1h", "p_4h", "p_1d", "lvl_1d"]].head(4).to_string())
    print("ok")
