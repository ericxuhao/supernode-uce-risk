"""Retrain the cluster model on data collected from your own cluster.

The service ships with models trained on open data. Once the cluster has collected enough history
(collector records under <data_dir>/cluster/raw, including some uncorrectable errors), retrain the error-log
model on it:

    python supernode/retrain.py --val-start 2025-03-01                 # raw data from the configured data_dir
    python supernode/retrain.py --raw-dir /data/raw --val-start 2025-03-01 --model-dir /opt/supernode/models

What it does: collector records -> vendor-neutral tables (ingest.py) -> field-driven model input -> AI + rule
hybrid with 1 h / 4 h / 24 h risk (uce_risk) -> fault-tree weights on the training samples -> alarm thresholds
(validation period). The previous models.joblib / build_meta.json are kept as *.bak-<timestamp>.
Restart the service afterwards.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fta  # noqa: E402
import policy  # noqa: E402
from common import DATA, MODEL_DIR, cfg_path  # noqa: E402
from ingest import make_input, records_to_tables  # noqa: E402
from uce_risk import features  # noqa: E402
from uce_risk.api import UCERiskPredictor  # noqa: E402
from uce_risk.model import ensemble_predict  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_raw(raw_dir: Path):
    E, devs = [], set()
    files = sorted(glob.glob(str(raw_dir / "*" / "*.jsonl")))
    if not files:
        raise SystemExit(f"no collector records under {raw_dir}")
    for f in files:
        with open(f, encoding="utf-8") as fh:
            recs = [json.loads(x) for x in fh if x.strip()]
        e, t = records_to_tables(recs)
        E.append(e)
        devs |= set(zip(t.host, t.device)) | set(zip(e.host, e.device))
    ev = pd.concat(E, ignore_index=True)
    ev["t"] = ev.t.astype(np.int64)
    devices = pd.DataFrame(sorted(devs), columns=["host", "device"])
    return ev, devices


def main(argv=None):
    ap = argparse.ArgumentParser(description="retrain the cluster model on collected data")
    ap.add_argument("--raw-dir", default=str(DATA / "cluster" / "raw"))
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--val-start", required=True, help="start of the validation period (date)")
    ap.add_argument("--backend", default="lgbm", choices=["lgbm", "hgb"])
    ap.add_argument("--tune", type=int, default=20)
    ap.add_argument("--bags", type=int, default=3)
    ap.add_argument("--min-uce", type=int, default=20, help="refuse to train with fewer UCE episodes than this")
    a = ap.parse_args(argv)
    raw, out = cfg_path(a.raw_dir), cfg_path(a.model_dir)
    val_start = int(pd.Timestamp(a.val_start, tz="UTC").timestamp())

    log(f"reading collector records from {raw}")
    ev, devices = load_raw(raw)
    t_end = int(ev.t.max()) + 1
    log(f"{len(ev):,} events on {len(devices)} cards, {ev.cause.value_counts().to_dict()}")
    d = make_input(ev, devices, t_end=t_end, name="cluster")
    log(f"model input modules: {d.report['modules']}")
    S, ep = features.build(d, verbose=False)
    if len(ep) < a.min_uce:
        raise SystemExit(f"only {len(ep)} uncorrectable-error episodes in the data (need >= {a.min_uce}); "
                         "collect longer before retraining")
    pred = UCERiskPredictor(scenario="auto", backend=a.backend, tune=a.tune, bags=a.bags).fit(
        d, val_start=val_start, samples=(S, ep))
    log(f"trained: {pred.info['n_features']} features, validation positives {pred.info['val_positives']}")
    weights = fta.estimate_weights(S)
    va = S[S.s > val_start]
    p = ensemble_predict(pred.models, pred.matrix(va), list(pred.horizons))
    hist = pd.DataFrame({"card": va.entity.to_numpy(), "s": va.s.to_numpy(), **{f"p_{h}": p[h] for h in pred.horizons}})
    thr = policy.thresholds(pred.thresholds, hist)

    out.mkdir(parents=True, exist_ok=True)
    mpath, jpath = out / "models.joblib", out / "build_meta.json"
    models = joblib.load(mpath) if mpath.exists() else {}
    meta = json.loads(jpath.read_text(encoding="utf-8")) if jpath.exists() else {}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for f in (mpath, jpath):
        if f.exists():
            shutil.copy2(f, f.with_name(f.name + f".bak-{stamp}"))
    models["log"] = pred
    meta.setdefault("models", {})["log"] = {"info": pred.info, "thresholds": pred.thresholds, "report": d.report}
    meta.setdefault("fta_weights", {})["log"] = weights
    meta.setdefault("demo_thresholds", {})["log"] = thr
    meta["retrained"] = {"at": time.strftime("%Y-%m-%d %H:%M"), "raw_dir": str(raw), "val_start": a.val_start,
                         "cards": int(len(devices)), "uce_episodes": int(len(ep))}
    meta["built"] = meta["retrained"]["at"]
    joblib.dump(models, mpath, compress=3)
    jpath.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"saved {mpath} and {jpath} (previous versions kept as *.bak-{stamp}); restart the service to use them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
