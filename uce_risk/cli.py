"""Command line for running the algorithm on your own data.

    python -m uce_risk.cli describe --errors errors.csv --uce-codes 48
    python -m uce_risk.cli train    --errors errors.csv --uce-codes 48 --telemetry tel.csv --val-start 2024-03-01 --out model.joblib
    python -m uce_risk.cli predict  --model model.joblib --errors errors.csv --telemetry tel.csv --at now --out risk.csv

Inputs (any subset, CSV / Parquet / JSON-lines by extension):
  --errors     one error log with a code / type column; rows in --uce-codes are uncorrectable errors (labels),
               rows in --ce-codes are correctable errors, all other rows are events
  --uce / --ce / --events   separate tables instead of (or in addition to) --errors
  --telemetry  numeric metrics with a time column (time series) or without (static baseline)
  --devices    device inventory (keys + attributes), --removed  devices that left service (keys + time)
Column names are recognised automatically (see `describe`); override with --device/--host/--time/--code.

BD production data (work orders + telemetry narrow table + 1-minute metric table, see uce_risk/bd.py):
    python -m uce_risk.cli bd-table    --telemetry 遥测/ --out metric_1min/          # narrow -> 1-minute wide table
    python -m uce_risk.cli bd-describe --work-orders 工单.xlsx --telemetry 遥测.csv --metrics metric_1min/
    python -m uce_risk.cli bd-train    --work-orders ... --telemetry ... --metrics ... --val-start 2026-09-12 --out bd.joblib
    python -m uce_risk.cli bd-predict  --model bd.joblib --work-orders ... --telemetry ... --metrics ... --at now --out risk.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .api import UCERiskPredictor
from .inputs import describe, from_tables


def read_table(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    suf = "".join(p.suffixes).lower()
    if suf.endswith(".parquet"):
        return pd.read_parquet(p)
    if suf.endswith((".jsonl", ".json", ".jsonl.gz", ".json.gz")):
        return pd.read_json(p, lines=True)
    return pd.read_csv(p)


def codes(v):
    if not v:
        return None
    out = []
    for x in v.split(","):
        x = x.strip()
        out.append(int(x) if x.lstrip("-").isdigit() else x)
    return out


def load_input(a, spec: dict | None = None):
    kw = {k: getattr(a, k) for k in ("device", "host", "time", "code") if getattr(a, k)}
    if a.warmup_days:
        kw["warmup_days"] = a.warmup_days
    return from_tables(errors=read_table(a.errors), uce=read_table(a.uce), ce=read_table(a.ce), events=read_table(a.events),
                       telemetry=read_table(a.telemetry), devices=read_table(a.devices), removed=read_table(a.removed),
                       uce_codes=codes(a.uce_codes), ce_codes=codes(a.ce_codes), name="cli", **(spec or {}), **kw)


def to_time(v: str) -> int:
    if v == "now":
        return int(pd.Timestamp.now(tz="UTC").timestamp())
    if v.lstrip("-").isdigit():
        return int(v)
    return int(pd.Timestamp(v, tz="UTC").timestamp() if pd.Timestamp(v).tzinfo is None else pd.Timestamp(v).timestamp())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="uce-risk", description="UCE risk prediction on your own data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("describe", "train", "predict"):
        p = sub.add_parser(name)
        for k in ("errors", "uce", "ce", "events", "telemetry", "devices", "removed"):
            p.add_argument(f"--{k}")
        p.add_argument("--uce-codes", help="comma-separated codes / types of uncorrectable errors in --errors, e.g. 48,94,95")
        p.add_argument("--ce-codes", help="comma-separated codes / types of correctable errors in --errors")
        for k in ("device", "host", "time", "code"):
            p.add_argument(f"--{k}", help=f"column name of the {k} (if not recognised automatically)")
        p.add_argument("--warmup-days", type=int, default=0)
        if name == "train":
            p.add_argument("--val-start", required=True, help="start of the validation period (date or epoch seconds)")
            p.add_argument("--out", required=True, help="model file (.joblib)")
            p.add_argument("--scenario", default="auto", help="auto | S1_tel+xid | S2_ce | S3_tel+ce | ALL")
            p.add_argument("--backend", default="lgbm", choices=["lgbm", "hgb"])
            p.add_argument("--tune", type=int, default=20)
            p.add_argument("--bags", type=int, default=3)
            p.add_argument("--variant", default="hybrid", choices=["hybrid", "ai", "rules"])
        if name == "predict":
            p.add_argument("--model", required=True)
            p.add_argument("--at", default="now", help="prediction time (date / epoch seconds / now); default now")
            p.add_argument("--op", default="fpr1", choices=["fpr1", "valF1"], help="alarm operating point")
            p.add_argument("--out", help="output CSV (default: print the top 20)")
    for name in ("bd-table", "bd-describe", "bd-train", "bd-predict"):
        p = sub.add_parser(name)
        p.add_argument("--telemetry", help="telemetry narrow table: file or directory (metric / value rows + dimensions)")
        if name != "bd-table":
            p.add_argument("--work-orders", help="work orders (SEED events): CSV / Excel / Parquet")
            p.add_argument("--metrics", help="1-minute metric table: directory of date=YYYYMMDD partitions or a file; "
                                             "omitted: built from --telemetry")
            p.add_argument("--uce-codes", default="48,94,95", help="driver error codes that are UCEs")
        if name == "bd-table":
            p.add_argument("--out", required=True, help="output directory (date=YYYYMMDD/part-*.parquet)")
        if name == "bd-train":
            p.add_argument("--val-start", required=True)
            p.add_argument("--out", required=True)
            p.add_argument("--fpr", type=float, default=0.05, help="alarm operating point: share of UCE-free cards "
                                                                   "allowed to alarm in validation")
            p.add_argument("--bags", type=int, default=3)
            p.add_argument("--features", default="compact", choices=["compact", "all"],
                           help="compact: memory-related features with regularised trees (default, for tens of "
                                "machines); all: every feature, tuned trees (for large fleets)")
            p.add_argument("--tune", type=int, default=20, help="Optuna trials (only with --features all)")
        if name == "bd-predict":
            p.add_argument("--model", required=True)
            p.add_argument("--at", default="end", help="prediction time (date / epoch seconds / now / end of data)")
            p.add_argument("--op", default="fpr1", choices=["fpr1", "valF1"])
            p.add_argument("--out", help="output CSV (default: print the top 20)")
    a = ap.parse_args(argv)

    if a.cmd.startswith("bd-"):
        return _bd(a)
    if a.cmd == "describe":
        d = load_input(a)
        print(json.dumps(describe(d), ensure_ascii=False, indent=1, default=str))
        return 0
    if a.cmd == "train":
        d = load_input(a)
        print(json.dumps(describe(d)["modules"], ensure_ascii=False))
        pred = UCERiskPredictor(scenario=a.scenario, backend=a.backend, tune=a.tune, bags=a.bags, variant=a.variant,
                                verbose=True).fit(d, val_start=to_time(a.val_start))
        pred.save(a.out)
        print(json.dumps({"saved": a.out, **pred.info, "thresholds": pred.thresholds}, ensure_ascii=False, indent=1,
                         default=str))
        return 0
    pred = UCERiskPredictor.load(a.model)
    d = load_input(a, spec=pred.input_spec)          # same event-code set as at training time
    out = pred.predict(d, at=to_time(a.at), op=a.op)
    out = out.sort_values("p_1d", ascending=False)
    if a.out:
        out.to_csv(a.out, index=False)
        print(f"{len(out)} devices -> {a.out}; alarms (24 h): {int(out['alarm_1d'].sum())}")
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(out.head(20).to_string(index=False))
    return 0


def _bd(a) -> int:
    from . import bd
    if a.cmd == "bd-table":
        print(json.dumps(bd.build_metric_table(a.telemetry, a.out), ensure_ascii=False))
        return 0
    spec = {}
    if a.cmd == "bd-predict":
        pred = UCERiskPredictor.load(a.model)
        spec = pred.input_spec
    d = bd.from_bd(a.work_orders, a.telemetry, a.metrics, uce_codes=codes(a.uce_codes), verbose=True,
                   label_complete=a.cmd != "bd-predict", **spec)
    if a.cmd == "bd-describe":
        print(json.dumps(bd.describe_bd(d), ensure_ascii=False, indent=1, default=str))
        return 0
    if a.cmd == "bd-train":
        prof = dict(bd.PROFILE) if a.features == "compact" else dict(tune=a.tune)
        pred = UCERiskPredictor(bags=a.bags, alarm_fpr=a.fpr, verbose=True, **prof).fit(d, val_start=to_time(a.val_start))
        pred.save(a.out)
        print(json.dumps({"saved": a.out, **pred.info, "thresholds": pred.thresholds}, ensure_ascii=False, indent=1,
                         default=str))
        return 0
    out = bd.predict(pred, d, at=None if a.at == "end" else to_time(a.at), op=a.op)
    if a.out:
        out.to_csv(a.out, index=False, encoding="utf-8-sig")
        print(f"{len(out)} cards -> {a.out}; high {int((out.risk_level == '高').sum())}, "
              f"medium {int((out.risk_level == '中').sum())}")
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(out.drop(columns=["hostname", "ipv4", "rack"], errors="ignore").head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
