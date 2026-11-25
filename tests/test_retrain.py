"""Retraining the cluster model from collector records (synthetic 4 nodes x 8 cards, 75 days)."""
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / ("supernode" if (ROOT / "supernode").exists() else "supernode_demo")
sys.path.insert(0, str(APP))


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def write_raw(raw: Path, days=75, seed=1):
    rng = random.Random(seed)
    t0 = int(time.time()) - days * 86400
    weak = {("n1", 2), ("n1", 5), ("n2", 0), ("n3", 7), ("n4", 3), ("n4", 6)}
    for host in ("n1", "n2", "n3", "n4"):
        recs = []
        for h in range(days * 24):
            t = t0 + h * 3600
            for g in range(8):
                bad = (host, g) in weak
                recs.append({"ts": iso(t), "host": host, "tier": "fast", "gpu": g,
                             "m": {"DCGM_FI_DEV_GPU_TEMP": 55 + rng.gauss(0, 2), "DCGM_FI_DEV_MEMORY_TEMP": 60 + 6 * bad,
                                   "DCGM_FI_DEV_POWER_USAGE": 300 + rng.gauss(0, 20)}})
                if rng.random() < (0.3 if bad else 0.01):
                    recs.append({"ts": iso(t), "host": host, "type": "counter_increase", "gpu": g,
                                 "metric": "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL", "delta": rng.randint(1, 30 if bad else 2)})
                if bad and rng.random() < 0.02:
                    recs.append({"ts": iso(t + 60), "host": host, "type": "xid", "gpu": g, "xid": 63})
                if bad and rng.random() < 0.012:
                    recs.append({"ts": iso(t + 1800), "host": host, "type": "xid", "gpu": g, "xid": 48})
        d = raw / time.strftime("%Y-%m-%d", time.gmtime(t0))
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{host}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    return t0


def test_retrain():
    import retrain
    from common import MODEL_DIR
    tmp = Path(tempfile.mkdtemp(prefix="retrain_"))
    raw, mdir = tmp / "raw", tmp / "models"
    mdir.mkdir()
    for f in ("models.joblib", "build_meta.json"):
        shutil.copy2(MODEL_DIR / f, mdir / f)
    t0 = write_raw(raw)
    val = time.strftime("%Y-%m-%d", time.gmtime(t0 + 50 * 86400))
    assert retrain.main(["--raw-dir", str(raw), "--model-dir", str(mdir), "--val-start", val, "--bags", "1",
                         "--tune", "0", "--min-uce", "10"]) == 0
    meta = json.loads((mdir / "build_meta.json").read_text(encoding="utf-8"))
    assert meta["retrained"]["cards"] == 32 and meta["retrained"]["uce_episodes"] >= 10
    assert set(meta["demo_thresholds"]["log"]) == {"1h", "4h", "1d"}
    assert any(p.name.startswith("models.joblib.bak-") for p in mdir.iterdir())
    import joblib
    m = joblib.load(mdir / "models.joblib")
    assert "ce" in m and m["log"].info["n_features"] > 10
    return meta["retrained"]


if __name__ == "__main__":
    print(test_retrain())
    print("ok")
