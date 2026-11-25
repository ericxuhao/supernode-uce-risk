"""Field-driven input layer: raw tables in, canonical ``UCEData`` out.

The algorithm is configured by the *fields present in the input*, not by the data set it came from.
``from_tables`` recognises fields by name (case-insensitive aliases, overridable), converts them, and
switches feature modules on or off accordingly:

  field found                                   module switched on
  device key (+ optional host key)              always (host key -> same-node features)
  time                                          always (timestamps, ISO strings, epoch s/ms)
  error code / type with uce_codes, ce_codes    splits one error log into UCE labels, CEs, other events
  CE records                                    CE counting module (ce__)
  CE count / counter column                     CE statistics (weights, cumulative counters -> deltas)
  CE subtype (e.g. read / scrub)                subtype shares
  location columns (dimm/rank/stack/sid/pc/bg/bank, row, col)   spatial + fault-mode module (sp__)
  other event codes (Xid, UEO, ...)             event module (ev__); codes seen on many devices are
                                                tagged 'background' (bg_) from the data itself
  numeric columns of a timestamped table        telemetry time series (te__ mean / std / change)
  numeric columns of an untimed table           static telemetry baseline (te__)
  device attributes (model, vendor, slot, ...)  configuration (cf__)

``describe(data)`` returns the detected fields and enabled modules, so a deployment can check what the
algorithm will use before training.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from .schema import DAY, ShardSource, UCEData

ALIASES = {
    "time": ["t", "time", "timestamp", "ts", "datetime", "date", "event_time", "time_stamp", "collect_time",
             "time_ms", "timestamp_ms", "ts_ms", "time_s", "timestamp_s", "epoch", "unix_time"],
    "host": ["host", "hostname", "node", "server", "machine", "instance", "node_name"],
    "device": ["entity", "device", "device_id", "gpu", "gpu_id", "gpu_uuid", "uuid", "dev", "name", "card", "gpu_index",
               "sid"],
    "code": ["code", "xid", "type", "ecc_type", "error_code", "err_type", "event_type", "event", "error_type"],
    "sub": ["sub", "subtype", "ce_type", "detect_type", "error_type", "source"],
    "count": ["n", "count", "cnt", "ce_count", "sbe_count", "corrected_count", "ce_cnt"],
}
HIER = [("dimm", ["dimm", "memoryid", "dimm_id", "dimm_no"]), ("rank", ["rank", "rankid", "rank_id"]),
        ("stack", ["stack", "hbm", "hbm_stack", "stack_id"]), ("sid", ["sid", "stackid", "hbm_sid"]),
        ("pc", ["pc", "pcid", "pseudo_channel", "pseudochannel", "channel"]),
        ("bg", ["bg", "bankgroup", "bank_group"]), ("bank", ["bank", "bankid", "bank_id", "bankarray", "bank_array"])]
LEAF = [("row", ["row", "rowid", "row_id"]), ("col", ["col", "column", "colid", "col_id"])]


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find(cols, names, exclude=()):
    """first column whose name matches an alias, ignoring case, '_' , '-' and spaces."""
    low = {}
    for c in cols:
        if c not in exclude:
            low.setdefault(_norm(c), c)
    for n in names:
        if _norm(n) in low:
            return low[_norm(n)]
    return None


def _to_seconds(x: pd.Series) -> np.ndarray:
    """time column -> Unix seconds. Accepts datetime dtypes, ISO strings (with or without zone; naive = UTC) and
    epoch numbers in s / ms / us / ns (also when stored as text)."""
    if len(x) == 0:
        return np.zeros(0, np.int64)
    if pd.api.types.is_datetime64_any_dtype(x):
        return (pd.to_datetime(x, utc=True).astype("int64") // 10**9).to_numpy(np.int64)
    if x.dtype == object or pd.api.types.is_string_dtype(x):     # numbers stored as text / objects
        num = pd.to_numeric(x, errors="coerce")
        if num.notna().all():
            x = num
    if pd.api.types.is_numeric_dtype(x):
        v = x.to_numpy(np.float64)
        m = np.nanmax(np.abs(v)) if len(v) else 0
        div = 1e9 if m > 1e17 else 1e6 if m > 1e14 else 1e3 if m > 1e11 else 1.0
        return (v / div).astype(np.int64)
    ts = pd.to_datetime(x, utc=True, format="ISO8601", errors="coerce")
    return (ts.astype("int64") // 10**9).to_numpy(np.int64)


def _to_int(x: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(x):
        return x.to_numpy(np.int64)
    return x.astype(str).map(lambda v: int(v, 16) if v.lower().startswith("0x") else int(float(v))).to_numpy(np.int64)


def _locations(raw: pd.DataFrame, used: set, location: dict | None) -> dict:
    loc = dict(location or {})
    for lvl, al in HIER + LEAF:
        if lvl not in loc:
            col = _find(raw.columns, al, exclude=used | set(loc.values()))
            if col is not None:
                loc[lvl] = col
    return {k: v for k, v in loc.items() if v in raw.columns}


class _Keys:
    """device / host key resolution and integer entity ids (devices of one host get contiguous ids)."""

    def __init__(self, device, host, identity=False):
        self.device, self.host, self.identity = device, host, identity
        self.table = None

    def detect(self, df: pd.DataFrame):
        host = self.host or _find(df.columns, ALIASES["host"])
        dev = self.device or _find(df.columns, ALIASES["device"], exclude=(host,))
        if dev is None and host is not None:
            dev, host = host, None
        if dev is None:
            raise ValueError(f"no device column found among {list(df.columns)}; pass device=...")
        return dev, host

    def register(self, frames, extra_ids=None):
        rows = []
        if extra_ids is not None:
            rows.append(pd.DataFrame({"device": pd.Index(extra_ids).astype(str), "host": pd.Index(extra_ids).astype(str)}))
        for df, dev, host in frames:
            k = pd.DataFrame({"device": df[dev].astype(str).to_numpy()})
            k["host"] = df[host].astype(str).to_numpy() if host else k["device"]
            rows.append(k.drop_duplicates())
        t = pd.concat(rows).drop_duplicates().sort_values(["host", "device"]).reset_index(drop=True)
        if self.identity:                       # integer device ids used as they are (one device per group)
            t = t.assign(entity=t.device.astype(np.int64)).sort_values("entity").reset_index(drop=True)
        else:
            t["entity"] = np.arange(len(t))
        t["group"] = pd.factorize(t.host)[0]
        self.table = t
        self._map = {(h, d): e for h, d, e in zip(t.host, t.device, t.entity)}

    def ids(self, df, dev, host) -> np.ndarray:
        d = df[dev].astype(str).to_numpy()
        h = df[host].astype(str).to_numpy() if host else d
        return np.fromiter((self._map[(a, b)] for a, b in zip(h, d)), np.int64, len(d))


def from_tables(*, errors: pd.DataFrame | None = None, uce: pd.DataFrame | None = None, ce=None,
                events: pd.DataFrame | None = None, telemetry: pd.DataFrame | None = None,
                devices: pd.DataFrame | None = None, removed: pd.DataFrame | None = None,
                exclude_train: pd.DataFrame | None = None,
                uce_codes=None, ce_codes=None, device: str | None = None, host: str | None = None,
                time: str | None = None, code: str | None = None, location: dict | None = None,
                ce_cumulative: bool | str = "auto", common_frac: float = 0.05, background_codes=None, keep_codes=None,
                min_devices_for_background: int = 500, max_codes: int = 20,
                min_code_rows: int = 20, warmup_days: int = 0, name: str = "input", shards: int = 1,
                **options) -> UCEData:
    """Build the canonical input from raw tables. Every argument except one UCE source is optional.

    errors      one error log holding several kinds of records; rows whose code is in uce_codes become
                UCE labels, rows in ce_codes become CEs, all other rows become events.
    uce / ce / events / telemetry   separate tables (any subset). `ce` may also be a ShardSource that
                already returns canonical columns (for logs that do not fit in memory).
    devices     optional device inventory (keys + attributes such as model, vendor, slot).
    removed     optional (device keys, time) when a device left service; exclude_train: device keys.
    device / host / time / code   column names, if the aliases do not find them.
    location    optional {canonical level: column} overrides, e.g. {"bank": "BankArray"}.
    background_codes   event codes to treat as background noise; by default codes seen on >= common_frac
                of the devices that never had a UCE (only with >= min_devices_for_background devices).
    keep_codes  event codes to featurise (others -> 'other'); pass a trained model's ``input_spec`` at inference
                so that the feature columns match the training run.
    options     passed to UCEData (episode_gap, storm_per_min, uce_anchor_days, anchor, ...).
    """
    report = {"fields": {}, "modules": []}
    keys = _Keys(device, host, identity=isinstance(ce, ShardSource))
    tables = {}
    if errors is not None:
        cc = code or _find(errors.columns, ALIASES["code"])
        if cc is None or (uce_codes is None and ce_codes is None):
            raise ValueError("errors needs a code / type column and uce_codes and/or ce_codes")
        v = errors[cc]
        norm = lambda xs: {str(x).lower() for x in (xs or [])}           # noqa: E731
        vs = v.astype(str).str.lower()
        is_u, is_c = vs.isin(norm(uce_codes)).to_numpy(), vs.isin(norm(ce_codes)).to_numpy()
        tables["uce"] = errors[is_u]
        if is_c.any():
            tables["ce"] = errors[is_c].drop(columns=[cc])
        if (~is_u & ~is_c).any():
            tables["events"] = errors[~is_u & ~is_c].rename(columns={cc: "__code"})
        report["fields"]["error_code"] = cc
    for k, df in (("uce", uce), ("ce", ce), ("events", events)):
        if df is not None:
            if k == "events" and not isinstance(df, ShardSource):
                cc = code or _find(df.columns, ALIASES["code"])
                df = df.rename(columns={cc: "__code"})
            tables[k] = df
    if "uce" not in tables:
        raise ValueError("a UCE source is required: uce=... or errors=... with uce_codes")

    # ---- keys and time
    frames, cols = [], {}
    for k, df in tables.items():
        if isinstance(df, ShardSource):
            continue
        dev, hst = keys.detect(df)
        tc = time or _find(df.columns, ALIASES["time"])
        if tc is None:
            raise ValueError(f"{k}: no time column among {list(df.columns)}; pass time=...")
        cols[k] = (dev, hst, tc)
        frames.append((df, dev, hst))
    tel_static = None
    if telemetry is not None:
        tc = time or _find(telemetry.columns, ALIASES["time"])
        hst = host or _find(telemetry.columns, ALIASES["host"])
        dev = device or _find(telemetry.columns, ALIASES["device"], exclude=(hst,))
        cols["telemetry"] = (dev, hst, tc)
        if dev is not None:
            frames.append((telemetry, dev, hst))
    if devices is not None:
        dev, hst = keys.detect(devices)
        frames.append((devices, dev, hst))
    lazy = isinstance(tables.get("ce"), ShardSource)     # lazy CE: integer device ids, used as they are
    if frames:
        keys.register(frames, extra_ids=tables["ce"].counts.index if lazy else None)
        report["fields"]["device"] = frames[0][1]
        report["fields"]["host"] = frames[0][2]

    def canon(k, df):
        dev, hst, tc = cols[k]
        out = pd.DataFrame({"entity": keys.ids(df, dev, hst), "t": _to_seconds(df[tc])})
        return out, df

    data = {}
    # ---- UCE
    u, raw = canon("uce", tables["uce"])
    for lvl, col in _locations(raw, set(cols["uce"]) | {"__code"}, location).items():   # UCE addresses, if any
        u[lvl] = _to_int(raw[col])
    data["uce"] = u
    report["fields"]["time"] = cols["uce"][2]
    # ---- CE
    hier, leaf = [], []
    if "ce" in tables:
        ce = tables["ce"]
        if isinstance(ce, ShardSource):
            data["ce"] = ce
            hier, leaf = options.pop("hier", []), tuple(options.pop("leaf", ()))
            report["fields"]["ce_location"] = {k: k for k in [*hier, *leaf]}
            report["fields"]["ce_source"] = "ShardSource (loaded shard by shard)"
        else:
            c, raw = canon("ce", ce)
            used = set(cols["ce"]) | {"__code"}
            loc = _locations(raw, used, location)
            for lvl, col in loc.items():
                c[lvl] = _to_int(raw[col])
            hier = [lvl for lvl, _ in HIER if lvl in loc]
            leaf = tuple(lvl for lvl, _ in LEAF if lvl in loc)
            sc = _find(raw.columns, ALIASES["sub"], exclude=used | set(loc.values()))
            if sc is not None and raw[sc].nunique() <= 8:
                c["sub"] = pd.factorize(raw[sc], sort=True)[0].astype(np.int8) + 1
                report["fields"]["ce_subtype"] = sc
            nc = _find(raw.columns, ALIASES["count"], exclude=used | set(loc.values()))
            if nc is not None:
                n = raw[nc].to_numpy(np.float64)
                c = c.assign(n=n).sort_values(["entity", "t"], kind="stable")
                cum = ce_cumulative
                if cum == "auto":
                    dn = c.groupby("entity").n.diff()
                    cum = bool((dn.dropna() >= 0).mean() > 0.95 and dn.notna().sum() > 0)
                if cum:
                    c["n"] = c.groupby("entity").n.diff().fillna(c.n).clip(lower=0)
                c = c[c.n > 0]
                report["fields"]["ce_count"] = f"{nc} ({'cumulative counter' if cum else 'count per record'})"
            data["ce"] = c
            report["fields"]["ce_location"] = loc
        report["modules"] += ["CE counts (ce__)"] + (["CE spatial / fault modes (sp__)"] if hier or leaf else [])
    # ---- events
    event_codes, common = [], []
    if "events" in tables:
        common = []
        ev, raw = canon("events", tables["events"])
        codes = raw["__code"].astype(str).to_numpy()
        cname = report["fields"].get("error_code") or code or "code"
        if pd.api.types.is_numeric_dtype(raw["__code"]):
            codes = np.array([f"{cname.lower()}{c}" for c in codes])
        codes = np.array([re.sub(r"\W+", "_", c) for c in codes])
        ev["code"] = codes
        n_dev = max(len(keys.table), 1)
        cnt = ev.code.value_counts()
        if keep_codes is not None:                     # code set fixed by a trained model (inference)
            keep = pd.Index([c for c in keep_codes])
        else:
            keep = (ev.groupby("code").entity.nunique().loc[lambda x: cnt.reindex(x.index) >= min_code_rows]
                    .sort_values(ascending=False).index[:max_codes])
        # background = codes common on devices that never had a UCE (ubiquitous noise such as application Xids)
        healthy = np.setdiff1d(keys.table.entity.to_numpy(), data["uce"].entity.unique())
        cov = ev[ev.entity.isin(healthy)].groupby("code").entity.nunique().reindex(keep).fillna(0) / max(len(healthy), 1)
        if background_codes is not None:
            bgn = {re.sub(r"\W+", "_", f"{cname.lower()}{b}" if isinstance(b, (int, np.integer)) else str(b))
                   for b in background_codes}
            common = [c for c in keep if c in bgn]
        elif n_dev >= min_devices_for_background:      # seen on many healthy devices -> background
            common = [c for c in keep if cov[c] >= common_frac]
        ev.loc[~ev.code.isin(keep), "code"] = "other"
        ev["code"] = np.where(ev.code.isin(common), "bg_" + ev.code, ev.code)
        event_codes = [("bg_" + c if c in common else c) for c in keep] + (
            ["other"] if (ev.code == "other").any() or keep_codes is not None else [])
        data["events"] = ev
        report["fields"]["event_codes"] = event_codes
        report["fields"]["background_codes"] = common
        report["modules"].append("discrete events (ev__)")
    # ---- telemetry
    if telemetry is not None:
        dev, hst, tc = cols["telemetry"]
        num = [c for c in telemetry.columns if c not in (dev, hst, tc) and pd.api.types.is_numeric_dtype(telemetry[c])]
        if tc is not None and dev is not None:
            tel = pd.DataFrame({"entity": keys.ids(telemetry, dev, hst), "t": _to_seconds(telemetry[tc])})
            for c in num:
                tel[c] = telemetry[c].to_numpy(np.float32)
            data["telemetry"] = tel
            report["modules"].append("telemetry time series (te__)")
        else:                                              # static baseline, per device or per host
            key = dev or hst
            st = telemetry.groupby(telemetry[key].astype(str))[num].mean()
            lookup = keys.table.device if dev else keys.table.host
            tel_static = st.reindex(lookup.to_numpy()).set_axis(keys.table.entity.to_numpy())
            tel_static["has_telemetry"] = tel_static.notna().all(axis=1).astype(np.float32)
            report["modules"].append("static telemetry baseline (te__)")
        report["fields"]["telemetry_metrics"] = num
    # ---- devices / configuration
    if keys.table is not None:
        ent = keys.table.set_index("entity")[["group", "host", "device"]].copy()
    else:
        ids = data["ce"].counts.index.to_numpy() if isinstance(data.get("ce"), ShardSource) else data["uce"].entity.unique()
        ent = pd.DataFrame({"group": ids}, index=ids)
    cat_cols, num_cols = [], []
    if devices is not None:
        dev, hst = keys.detect(devices)
        attrs = [c for c in devices.columns if c not in (dev, hst)]
        dv = devices.assign(entity=keys.ids(devices, dev, hst)).set_index("entity")[attrs]
        dv = dv[~dv.index.duplicated()]
        for c in attrs:
            ent[c] = dv[c].reindex(ent.index)
            (num_cols if pd.api.types.is_numeric_dtype(dv[c]) else cat_cols).append(c)
        if attrs:
            report["modules"].append("device configuration (cf__)")
    if ent.group.nunique() < len(ent):
        report["modules"].append("same-node context (nb*__)")
    report["modules"].append("UCE history (hi__)")
    # ---- removal / exclusion
    rem = excl = None
    if removed is not None:
        dev, hst = keys.detect(removed)
        tc = time or _find(removed.columns, ALIASES["time"])
        rem = pd.Series(_to_seconds(removed[tc]), index=keys.ids(removed, dev, hst)).groupby(level=0).min()
    if exclude_train is not None:
        dev, hst = keys.detect(exclude_train)
        excl = pd.Series(True, index=np.unique(keys.ids(exclude_train, dev, hst)))
    # ---- time range
    tt = [x.t.to_numpy() for x in data.values() if isinstance(x, pd.DataFrame) and len(x)]
    tt = [a for a in tt if len(a)]
    if tt:
        t0, t1 = int(min(a.min() for a in tt)), int(max(a.max() for a in tt))
    else:                                  # nothing logged yet (e.g. a freshly connected healthy cluster)
        t0 = t1 = int(options.get("t_end", time.time()))
    options.setdefault("t_start", t0 // DAY * DAY + warmup_days * DAY)
    options.setdefault("t_end", t1 + DAY)
    d = UCEData(name=name, entities=ent, uce=data["uce"], ce=data.get("ce"), events=data.get("events"),
                telemetry=data.get("telemetry"), static_tel=tel_static, hier=hier, leaf=tuple(leaf),
                removal=rem, exclude_train=excl, cat_cols=cat_cols, num_cols=num_cols, event_codes=event_codes,
                anchor_codes=[c for c in event_codes if not c.startswith("bg_")] if event_codes else None,
                shards=shards, **options)
    d.report = report
    return d


def describe(d: UCEData) -> dict:
    """detected fields and the feature modules the algorithm will use for this input."""
    r = dict(getattr(d, "report", {}))
    r["devices"] = len(d.entities)
    r["uce_records"] = len(d.uce)
    r["modalities"] = sorted(d.modalities())
    return r
