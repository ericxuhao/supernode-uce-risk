"""Collected records -> model input.

Records follow the gpu-ecc-collector JSONL envelope (collectors/gpu_ecc): telemetry samples
(tier 'fast', DCGM field names under 'm') and events (kernel error lines, address-level memory RAS
records, job start / end). They are turned into vendor-neutral tables:

  events     host, device, t, cause [, stack, sid, pc, bg, bank, row, col]
  telemetry  host, device, t, core_temp, mem_temp, power

and handed to the field-driven input layer ``uce_risk.inputs.from_tables``; the model input is whatever
modules the present fields switch on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import CE_CAUSE, RAS_TO_CAUSE, UCE_CAUSE, cause_of_xid
from uce_risk.inputs import from_tables

ADDR = ["stack", "sid", "pc", "bg", "bank", "row", "col"]
TEL_FIELDS = {"DCGM_FI_DEV_GPU_TEMP": "core_temp", "DCGM_FI_DEV_MEMORY_TEMP": "mem_temp",
              "DCGM_FI_DEV_POWER_USAGE": "power"}
NON_MODEL = {"JOB_START", "JOB_END"}


# counter metrics of the collector -> vendor-neutral causes (A100-class devices)
CE_COUNTERS = ("DCGM_FI_DEV_ECC_SBE_VOL_TOTAL", "DCGM_FI_DEV_ECC_SBE_AGG_TOTAL")
UE_COUNTERS = ("DCGM_FI_DEV_ECC_DBE_VOL_TOTAL", "DCGM_FI_DEV_ECC_DBE_AGG_TOTAL")
REMAP_COUNTERS = ("DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS", "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS",
                  "DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL", "DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL")


def _t(r):
    v = r.get("event_time") or r["ts"]
    return int(pd.Timestamp(v).timestamp()) if isinstance(v, str) else int(v)


def records_to_tables(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """collector / replay JSONL records -> (events, telemetry).

    events: kernel error lines (code -> cause), address-level memory RAS records, job start / end, and the
    collector's change events: correctable-error counter increases become CE rows with the increment in
    column n, uncorrectable-error counter increases become UCE rows, row-remap / page-retirement increases
    become MEMORY_PAGE_RETIRED, a remap-failure flag becomes MEMORY_PAGE_RETIRE_FAILED and a card that
    vanished becomes DEVICE_LOST."""
    ev, tel = [], []
    for r in records:
        g = r.get("gpu", r.get("card"))
        if g is None:
            continue
        dev, host, typ = f"C{g}", r["host"], r.get("type")
        if r.get("tier") == "fast":
            m = r.get("m", {})
            tel.append((host, dev, _t(r), *((m.get(k) if m.get(k) is not None else np.nan) for k in TEL_FIELDS)))
            continue
        t = _t(r)
        if typ == "xid":
            ev.append({"host": host, "device": dev, "t": t, "cause": cause_of_xid(r.get("xid"))})
        elif typ == "mem_ras":
            e = {"host": host, "device": dev, "t": t, "cause": RAS_TO_CAUSE.get(r.get("severity"), "UNKNOWN_ERROR")}
            e.update({k: r.get(k) for k in ADDR})
            ev.append(e)
        elif typ in ("job_start", "job_end"):
            ev.append({"host": host, "device": dev, "t": t, "cause": typ.upper(), "job": r.get("job")})
        elif typ == "counter_increase":
            mid, d = r.get("metric", ""), r.get("delta") or 0
            if mid in CE_COUNTERS:
                ev.append({"host": host, "device": dev, "t": t, "cause": CE_CAUSE, "n": float(d), "_m": mid})
            elif mid in UE_COUNTERS:
                ev.append({"host": host, "device": dev, "t": t, "cause": UCE_CAUSE})
            elif mid in REMAP_COUNTERS:
                ev.append({"host": host, "device": dev, "t": t, "cause": "MEMORY_PAGE_RETIRED"})
        elif typ == "state_change":
            if r.get("metric") == "DCGM_FI_DEV_ROW_REMAP_FAILURE" and r.get("new"):
                ev.append({"host": host, "device": dev, "t": t, "cause": "MEMORY_PAGE_RETIRE_FAILED"})
        elif typ == "gpu_missing":
            ev.append({"host": host, "device": dev, "t": t, "cause": "DEVICE_LOST"})
    E = pd.DataFrame(ev, columns=["host", "device", "t", "cause", *ADDR, "job", "n", "_m"])
    if len(E):
        # volatile and aggregate counters move together: keep one CE row per card and timestamp
        ce = E[E.cause == CE_CAUSE]
        if len(ce):
            ce = ce.sort_values("_m", key=lambda x: x != CE_COUNTERS[0]).drop_duplicates(["host", "device", "t"])
        E = pd.concat([E[E.cause != CE_CAUSE].drop_duplicates(["host", "device", "t", "cause"]), ce], ignore_index=True)
    E = E.drop(columns="_m")
    T = pd.DataFrame(tel, columns=["host", "device", "t", *TEL_FIELDS.values()])
    return E, T


def read_spool(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


def make_input(events: pd.DataFrame, devices: pd.DataFrame, spec: dict | None = None, t_end: int | None = None,
               name: str = "supernode"):
    """vendor-neutral event table -> UCEData through the field-driven input layer."""
    e = events[~events.cause.isin(NON_MODEL)]
    addr = [c for c in ADDR if c in e and e[c].notna().any()]
    cnt = ["n"] if "n" in e and e["n"].notna().any() else []
    e = e[["host", "device", "t", "cause", *addr, *cnt]]
    if cnt:                        # CE statistics: CEs per record (counter increments); event rows count 1
        e = e.copy()
        e["n"] = e["n"].fillna(1.0)
    if addr:                       # address columns only exist for address-level memory logs
        e = e.copy()
        for c in addr:
            e[c] = e[c].fillna(-1).astype(np.int64)
    kw = dict(spec or {})
    opts = {"t_end": t_end} if t_end is not None else {}
    return from_tables(errors=e, code="cause", uce_codes=[UCE_CAUSE], ce_codes=[CE_CAUSE], devices=devices[["host", "device"]],
                       time="t", name=name, uce_anchor_days=180, ce_cumulative=False, **kw, **opts)
