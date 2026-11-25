"""Feature engine: prediction times, multi-horizon labels and feature groups from a ``UCEData``.

Prediction times (per entity)
  * event-triggered: the end of every 5-minute bin that contains any CE / event / UCE (needed for the
    1 h horizon: in DRAM the median gap from the last CE to the UE is 5 minutes);
  * daily anchors at 00:00 for entities with any event in the last 30 days (anchor='active'), or for
    every entity-day (anchor='all'); this keeps quiet-but-risky entities scored once a day.
Samples after an entity left service, or later than t_end - 1 day (label censoring), are dropped.

Labels: UCEs on one entity separated by <= episode_gap form one episode; for a sample at s the lead is
the time to the next episode start > s (NaN if none). y_H = lead <= H. Episodes are flagged first /
recurrent so that first-UCE prediction can be reported separately.

Every feature uses only records with t < s. Window counts use sorted (entity << 32 | t) codes and
binary search; "distinct components active in a window" uses disjoint coverage intervals per component
(each record covers (t_i, min(t_next, t_i + W)]), so it is also a pair of binary searches.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (DAY, ShardSource, G_CE, G_CF, G_EV, G_HI, G_NBC, G_NBE, G_NBH, G_SP, G_TE, HOUR, UCEData)

SH = 32
MASK = (1 << SH) - 1
BIN = 300
NO_EP = -1


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def enc(e, t) -> np.ndarray:
    return (np.asarray(e, np.int64) << SH) | np.asarray(t, np.int64)


class Query:
    def __init__(self, e: np.ndarray, s: np.ndarray):
        self.e, self.s = e.astype(np.int64), s.astype(np.int64)
        self.c = enc(self.e, self.s)

    def at(self, dt):           # code of s - dt (clipped at 0); dt=None -> start of entity
        return enc(self.e, 0 if dt is None else np.maximum(self.s - dt, 0))


def wcount(codes: np.ndarray, q: Query, W=None) -> np.ndarray:
    """records with s - W <= t < s (W None -> all t < s); `codes` sorted."""
    return (np.searchsorted(codes, q.c, "left") - np.searchsorted(codes, q.at(W), "left")).astype(np.float32)


def wsum(codes: np.ndarray, cw: np.ndarray, q: Query, W=None) -> np.ndarray:
    """weighted wcount: cw = r_[0, cumsum(weights)] aligned with the sorted codes."""
    return (cw[np.searchsorted(codes, q.c, "left")] - cw[np.searchsorted(codes, q.at(W), "left")]).astype(np.float32)


def hours_since(codes: np.ndarray, q: Query) -> np.ndarray:
    if not len(codes):
        return np.full(len(q.e), np.nan, np.float32)
    i = np.searchsorted(codes, q.c, "left") - 1
    ii = np.clip(i, 0, None)
    ok = (i >= 0) & ((codes[ii] >> SH) == q.e)
    return np.where(ok, (q.s - (codes[ii] & MASK)) / HOUR, np.nan).astype(np.float32)


def distinct(e, t, key, q: Query, W=None) -> np.ndarray:
    """number of distinct components (key) with >= 1 record in [s - W, s)."""
    if not len(t):
        return np.zeros(len(q.e), np.float32)
    o = np.lexsort((t, key))
    k, tt, ee = key[o], t[o], e[o]
    if W is None:
        first = np.r_[True, k[1:] != k[:-1]]
        return wcount(np.sort(enc(ee[first], tt[first])), q)
    same = np.r_[k[1:] == k[:-1], False]
    del k
    end = np.where(same, np.minimum(np.r_[tt[1:], 0], tt + W), tt + W)
    del same
    a = np.sort(enc(ee, tt))
    n_lo = np.searchsorted(a, q.c, "left")
    del a, tt
    b = np.sort(enc(ee, end))
    return (n_lo - np.searchsorted(b, q.c, "left")).astype(np.float32)


def running_max(e, t, key, q: Query) -> np.ndarray:
    """max over components of the records a component has accumulated before s.
    e, t must be sorted by (entity, t) (the engine keeps every table in that order)."""
    o = np.lexsort((t, key))
    k = key[o]
    start = np.r_[True, k[1:] != k[:-1]]
    del k
    pos = np.arange(len(o), dtype=np.int64)
    first = np.maximum.accumulate(np.where(start, pos, 0))
    del start
    cnt = np.empty(len(o), np.int64)
    cnt[o] = pos - first + 1
    del o, pos, first
    # running max within each entity: offset entities so one global accumulate never crosses them
    newe = np.r_[True, e[1:] != e[:-1]]
    rank = np.cumsum(newe) - 1
    del newe
    cnt += rank << 32
    cm = (np.maximum.accumulate(cnt) - (rank << 32)).astype(np.float32)
    del cnt, rank
    c = enc(e, t)
    i = np.searchsorted(c, q.c, "left") - 1
    del c
    ii = np.clip(i, 0, None)
    ok = (i >= 0) & (e[ii] == q.e)
    return np.where(ok, cm[ii], 0).astype(np.float32)


def second_child(e, t, pkey, ckey) -> np.ndarray:
    """sorted codes of the times a parent component reaches its 2nd distinct child (fault-mode onset)."""
    o = np.lexsort((t, ckey, pkey))
    p, c = pkey[o], ckey[o]
    f = np.r_[True, (p[1:] != p[:-1]) | (c[1:] != c[:-1])]
    del c
    o = o[f]
    p = p[f]
    del f
    tt = t[o]
    o2 = np.lexsort((tt, p))
    o, p, tt = o[o2], p[o2], tt[o2]
    del o2
    start = np.r_[True, p[1:] != p[:-1]]
    sec = np.r_[False, start[:-1] & ~start[1:]]
    return np.sort(enc(e[o[sec]], tt[sec]))


def repeat_after(e, t, key, gap=DAY) -> np.ndarray:
    """sorted codes of the times a component errs again >= gap after its first record (hard fault)."""
    o = np.lexsort((t, key))
    k = key[o]
    start = np.r_[True, k[1:] != k[:-1]]
    del k
    gid = np.cumsum(start, dtype=np.int32) - 1
    tt = t[o]
    f = tt[start]
    del start
    cand = np.flatnonzero(tt >= f[gid] + gap)
    del f
    g = gid[cand]
    del gid
    idx = cand[np.r_[True, g[1:] != g[:-1]]] if len(cand) else cand
    return np.sort(enc(e[o[idx]], tt[idx]))


class CompKeys:
    """component ids per location level, built on demand by bit-packing (entity, location columns).

    Levels: every prefix of `hier`, each leaf under the full hier path, and 'cell' (all leaves).
    Bit-packed ids keep equality and need no hashing, so memory stays at one int64 per record per level
    in use (a storm server can have > 30 M CE records)."""

    def __init__(self, df: pd.DataFrame, hier: list[str], leaf: tuple[str, ...]):
        cols = ["entity"] + list(hier) + list(leaf)
        self.v, self.bits = {}, {}
        for c in cols:
            x = df[c].to_numpy()
            lo = int(x.min()) if len(x) else 0
            self.v[c] = (x, lo)
            self.bits[c] = max(1, int(int(x.max()) - lo).bit_length()) if len(x) else 1
        self.levels = {h: ["entity"] + hier[: i + 1] for i, h in enumerate(hier)}
        for c in leaf:
            self.levels[c] = ["entity"] + hier + [c]
        if len(leaf) == 2:
            self.levels["cell"] = ["entity"] + hier + list(leaf)
        self.df = df

    def __iter__(self):
        return iter(self.levels)

    def __contains__(self, k):
        return k in self.levels

    def __getitem__(self, lvl) -> np.ndarray:
        cols = self.levels[lvl]
        if sum(self.bits[c] for c in cols) <= 63:
            k = np.zeros(len(self.df), np.int64)
            for c in cols:
                x, lo = self.v[c]
                k <<= self.bits[c]
                k |= (x.astype(np.int64) - lo)
            return k
        return self.df.groupby(cols, sort=False).ngroup().to_numpy(np.int64)


def comp_keys(df: pd.DataFrame, hier: list[str], leaf: tuple[str, ...]) -> CompKeys:
    return CompKeys(df, list(hier), tuple(leaf))


# ----------------------------------------------------------------------------- episodes / sampling
def episodes(uce: pd.DataFrame, gap: int):
    u = uce.sort_values(["entity", "t"], kind="stable")
    e, t = u.entity.to_numpy(np.int64), u.t.to_numpy(np.int64)
    new = np.r_[True, (e[1:] != e[:-1]) | (t[1:] - t[:-1] > gap)] if len(e) else np.zeros(0, bool)
    ep = pd.DataFrame({"entity": e[new], "t": t[new]})
    ep["first"] = np.r_[True, ep.entity.to_numpy()[1:] != ep.entity.to_numpy()[:-1]] if len(ep) else []
    ep["ep"] = np.arange(len(ep))
    return ep


def _sample_times(d: UCEData, srcs: list[pd.DataFrame], tel_days: np.ndarray, ents: np.ndarray, uce=None,
                  tel_grid: np.ndarray | None = None) -> Query:
    """srcs: event tables (trigger samples + activity days); tel_days: (entity<<32|day) codes with telemetry;
    tel_grid: (entity<<32|grid time) codes of regular prediction times (d.grid > 0)."""
    e = np.concatenate([s.entity.to_numpy(np.int64) for s in srcs]) if srcs else np.zeros(0, np.int64)
    t = np.concatenate([s.t.to_numpy(np.int64) for s in srcs]) if srcs else np.zeros(0, np.int64)
    trig = enc(e, (t // BIN + 1) * BIN)
    if d.anchor == "all":
        days = np.arange(d.t_start // DAY, d.t_end // DAY + 1)
        anc = enc(np.repeat(ents, len(days)), np.tile((days + 1) * DAY, len(ents)))
    else:
        if d.anchor_codes is not None:     # only these event codes (plus CE and UCE) open a 30-day anchor window
            keep_ev = np.concatenate([(s.code.isin(d.anchor_codes).to_numpy() if "code" in s else np.ones(len(s), bool))
                                      for s in srcs])
            e, t = e[keep_ev], t[keep_ev]
        ed = np.unique(enc(e, t // DAY))
        de, dd = ed >> SH, ed & MASK
        anc = enc(np.repeat(de, 30), ((dd[:, None] + np.arange(30)[None, :] + 1) * DAY).ravel())
        if len(tel_days):
            anc = np.r_[anc, enc(tel_days >> SH, ((tel_days & MASK) + 1) * DAY)]
        if uce is not None and len(uce) and d.uce_anchor_days > 30:   # longer daily watch after a UCE
            ud = np.unique(enc(uce.entity.to_numpy(np.int64), uce.t.to_numpy(np.int64) // DAY))
            k = d.uce_anchor_days
            anc = np.r_[anc, enc(np.repeat(ud >> SH, k), (((ud & MASK)[:, None] + np.arange(k)[None, :] + 1) * DAY).ravel())]
    c = np.unique(np.r_[trig, anc] if tel_grid is None else np.r_[trig, anc, tel_grid])
    se, ss = c >> SH, c & MASK
    keep = (ss > d.t_start) & (ss <= d.t_end - DAY)
    if d.removal is not None:
        r = d.removal.reindex(se).to_numpy(np.float64)
        keep &= ~(r <= ss)
    return Query(se[keep], ss[keep])


def _slice(df, lo: int, hi: int):
    if df is None:
        return None
    if isinstance(df, ShardSource):
        return df.load(lo, hi)
    e = df.entity.to_numpy()
    a, b = np.searchsorted(e, lo, "left"), np.searchsorted(e, hi, "left")
    return df.iloc[a:b]


# ----------------------------------------------------------------------------- feature groups
def _ce_features(F: dict, ce: pd.DataFrame, q: Query, d: UCEData):
    """CE counts. A column 'n' (CEs per record, e.g. from counters) weights every count; without it each
    record is one CE."""
    e, t = ce.entity.to_numpy(np.int64), ce.t.to_numpy(np.int64)
    codes = enc(e, t)                                  # ce is sorted by (entity, t)
    w = ce["n"].to_numpy(np.float64) if "n" in ce else np.ones(len(ce))
    cnt = lambda c, ww, W: wsum(c, np.r_[0, np.cumsum(ww)], q, W)      # noqa: E731
    for wn, W in [("1h", HOUR), ("6h", 6 * HOUR), ("1d", DAY), ("7d", 7 * DAY), ("30d", 30 * DAY), ("cum", None)]:
        F[f"{G_CE}n_{wn}"] = cnt(codes, w, W)
    F[f"{G_CE}accel_1h_vs_1d"] = F[f"{G_CE}n_1h"] * 24 / (F[f"{G_CE}n_1d"] + 1)
    F[f"{G_CE}h_since_last"] = hours_since(codes, q)
    first = np.r_[True, e[1:] != e[:-1]]
    F[f"{G_CE}h_since_first"] = hours_since(codes[first], q)
    if "sub" in ce:
        sub = ce["sub"].to_numpy()
        for v in np.unique(sub):
            m = sub == v
            F[f"{G_CE}sub{v}_1h"] = cnt(codes[m], w[m], HOUR)
            F[f"{G_CE}sub{v}_1d"] = cnt(codes[m], w[m], DAY)
            F[f"{G_CE}sub{v}_share_1d"] = F[f"{G_CE}sub{v}_1d"] / np.maximum(F[f"{G_CE}n_1d"], 1)
    F[f"{G_CE}active_hours_1d"] = distinct(e, t, enc(e, t // HOUR), q, DAY)
    F[f"{G_CE}active_days_30d"] = distinct(e, t, enc(e, t // DAY), q, 30 * DAY)
    mc = enc(e, t // 60)
    m, inv = np.unique(mc, return_inverse=True)
    per_min = np.bincount(inv, weights=w)
    st = m[per_min >= d.storm_per_min]
    st = np.sort(enc(st >> SH, ((st & MASK) + 1) * 60))
    F[f"{G_CE}storm_min_1d"] = wcount(st, q, DAY)
    F[f"{G_CE}storm_min_30d"] = wcount(st, q, 30 * DAY)


def _sp_features(F: dict, ce: pd.DataFrame, q: Query, d: UCEData):
    e, t = ce.entity.to_numpy(np.int64), ce.t.to_numpy(np.int64)
    keys = comp_keys(ce, d.hier, d.leaf)
    for lvl in keys:
        k = keys[lvl]
        F[f"{G_SP}n_{lvl}_1d"] = distinct(e, t, k, q, DAY)
        F[f"{G_SP}n_{lvl}_cum"] = distinct(e, t, k, q)
        del k
    fine = [x for x in ["cell", *d.leaf, *(d.hier[-1:])] if x in keys]
    for lvl in dict.fromkeys(fine):
        F[f"{G_SP}max_per_{lvl}"] = running_max(e, t, keys[lvl], q)
    pairs = [(d.hier[i], d.hier[i + 1]) for i in range(len(d.hier) - 1)]
    pairs += [(h, c) for h in d.hier for c in d.leaf]
    if "cell" in keys:
        pairs += [(c, "cell") for c in d.leaf]
    for p, c in pairs:
        F[f"{G_SP}{p}_multi_{c}"] = wcount(second_child(e, t, keys[p], keys[c]), q)
    for lvl in [x for x in ("cell", "row", "col") if x in keys]:
        F[f"{G_SP}hard_{lvl}"] = wcount(repeat_after(e, t, keys[lvl]), q)
    if "cell" in keys:   # share of the last day's CEs on the busiest cell (hard vs scattered soft errors)
        F[f"{G_SP}scatter_1d"] = F[f"{G_SP}n_cell_1d"] / np.maximum(F[f"{G_CE}n_1d"], 1) if f"{G_CE}n_1d" in F else \
            F[f"{G_SP}n_cell_1d"]


def _ev_features(F: dict, ev: pd.DataFrame, q: Query, d: UCEData):
    e, t, code = ev.entity.to_numpy(np.int64), ev.t.to_numpy(np.int64), ev.code.to_numpy()
    allc = enc(e, t)
    F[f"{G_EV}all_1h"] = wcount(allc, q, HOUR)
    F[f"{G_EV}all_1d"] = wcount(allc, q, DAY)
    for k in d.event_codes:
        c = allc[code == k]
        for wn, W in [("1h", HOUR), ("1d", DAY), ("7d", 7 * DAY), ("30d", 30 * DAY), ("cum", None)]:
            F[f"{G_EV}{k}_{wn}"] = wcount(c, q, W)
        F[f"{G_EV}{k}_h_since"] = hours_since(c, q)


def _hi_features(F: dict, uce: pd.DataFrame, ep: pd.DataFrame, q: Query, d: UCEData):
    uc = enc(uce.entity.to_numpy(np.int64), uce.t.to_numpy(np.int64))
    epc = enc(ep.entity.to_numpy(np.int64), ep.t.to_numpy(np.int64))
    for wn, W in [("1h", HOUR), ("1d", DAY), ("7d", 7 * DAY), ("30d", 30 * DAY), ("cum", None)]:
        F[f"{G_HI}ep_{wn}"] = wcount(epc, q, W)
    F[f"{G_HI}events_1d"] = wcount(uc, q, DAY)
    F[f"{G_HI}h_since_last"] = hours_since(uc, q)
    locs = [c for c in [*d.hier, *d.leaf] if c in uce]
    if locs and len(uce):
        keys = comp_keys(uce, [h for h in d.hier if h in uce], tuple(c for c in d.leaf if c in uce))
        e, t = uce.entity.to_numpy(np.int64), uce.t.to_numpy(np.int64)
        for lvl in keys:
            F[f"{G_HI}n_{lvl}_cum"] = distinct(e, t, keys[lvl], q)


def _nb_features(F: dict, grp: np.ndarray, q: Query, streams: dict):
    """same-group (node) activity excluding the entity itself."""
    qg = Query(grp[q.e], q.s)
    for pre, (e, t, wins) in streams.items():
        own = np.sort(enc(e, t))
        gc = np.sort(enc(grp[e], t))
        for wn, W in wins:
            F[f"{pre}{wn}"] = wcount(gc, qg, W) - wcount(own, q, W)


def _te_features(F: dict, tel: pd.DataFrame, q: Query):
    e, t = tel.entity.to_numpy(np.int64), tel.t.to_numpy(np.int64)
    codes = enc(e, t)
    for m in [c for c in tel.columns if c not in ("entity", "t")]:
        v = tel[m].to_numpy(np.float64)
        ok = ~np.isnan(v)
        cs = np.r_[0, np.cumsum(np.where(ok, v, 0))]
        cs2 = np.r_[0, np.cumsum(np.where(ok, v * v, 0))]
        cn = np.r_[0, np.cumsum(ok)]
        hi = np.searchsorted(codes, q.c, "left")
        res = {}
        for wn, W in [("1h", HOUR), ("1d", DAY)]:
            lo = np.searchsorted(codes, q.at(W), "left")
            n = cn[hi] - cn[lo]
            mu = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
            res[wn] = mu
            F[f"{G_TE}{m}_mean{wn}"] = mu.astype(np.float32)
            if wn == "1d":
                var = np.where(n > 1, (cs2[hi] - cs2[lo]) / np.maximum(n, 1) - mu ** 2, np.nan)
                F[f"{G_TE}{m}_std1d"] = np.sqrt(np.clip(var, 0, None)).astype(np.float32)
        F[f"{G_TE}{m}_delta1h"] = (res["1h"] - res["1d"]).astype(np.float32)


# ----------------------------------------------------------------------------- driver
def build(d: UCEData, verbose=True, out_dir=None, query: pd.DataFrame | None = None
          ) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """returns (samples, episodes). samples: entity, s, lead, ep, ep_first, excl, <features>.
    With out_dir, every shard is written to out_dir/part-XXX.parquet as soon as it is done (low memory)
    and samples is returned as None. With query (DataFrame entity, s), features are computed exactly at
    those prediction times instead of the built-in sampling (online scoring)."""
    for nm in ("ce", "events", "uce", "telemetry"):
        df = getattr(d, nm)
        if df is not None and not isinstance(df, ShardSource):
            setattr(d, nm, df.sort_values(["entity", "t"], kind="stable").reset_index(drop=True))
    ep_all = episodes(d.uce, d.episode_gap)
    ent = d.entities.sort_index()
    ids = ent.index.to_numpy(np.int64)
    grp_s = ent["group"] if "group" in ent else pd.Series(ids, index=ids)
    grp = np.zeros(ids.max() + 1, np.int64)
    grp[ids] = grp_s.to_numpy(np.int64)
    use_nb = grp_s.nunique() < len(grp_s)
    gstart = np.flatnonzero(np.r_[True, grp[ids][1:] != grp[ids][:-1]])
    # balance shards by number of records (a few storm entities can hold most CEs), cut at group starts
    load = np.ones(len(ids))
    for df in (d.ce, d.events, d.uce):
        if df is not None and len(df):
            vc = df.counts if isinstance(df, ShardSource) else pd.Series(df.entity.to_numpy()).value_counts()
            load += vc.reindex(ids, fill_value=0).to_numpy()
    cum = np.cumsum(load)[gstart] - load[gstart]
    targets = np.linspace(0, load.sum(), d.shards + 1)[:-1]
    cut = np.unique(gstart[np.clip(np.searchsorted(cum, targets, "left"), 0, len(gstart) - 1)])
    cut[0] = 0
    edges = list(ids[cut]) + [ids.max() + 1]
    bounds = list(zip(edges[:-1], edges[1:]))
    out = []
    epc = enc(ep_all.entity.to_numpy(np.int64), ep_all.t.to_numpy(np.int64))
    for si, (lo, hi) in enumerate(bounds):
        part = None if out_dir is None else Path(out_dir) / f"part-{si:03d}.parquet"
        if part is not None and part.exists():
            if verbose:
                log(f"   shard {si + 1}/{len(bounds)}: cached")
            continue
        ce, ev, uce, tel = (_slice(x, lo, hi) for x in (d.ce, d.events, d.uce, d.telemetry))
        ep = ep_all[(ep_all.entity >= lo) & (ep_all.entity < hi)]
        srcs = [x for x in (ce, ev, uce) if x is not None and len(x)]
        tel_days = (np.unique(enc(tel.entity.to_numpy(np.int64), tel.t.to_numpy(np.int64) // DAY))
                    if tel is not None and len(tel) else np.zeros(0, np.int64))
        g = getattr(d, "grid", 0)
        tel_grid = (np.unique(enc(tel.entity.to_numpy(np.int64), (tel.t.to_numpy(np.int64) // g + 1) * g))
                    if g and tel is not None and len(tel) else None)
        if query is not None:
            qq = query[(query.entity >= lo) & (query.entity < hi)].sort_values(["entity", "s"])
            q = Query(qq.entity.to_numpy(np.int64), qq.s.to_numpy(np.int64))
        else:
            q = _sample_times(d, srcs, tel_days, ids[(ids >= lo) & (ids < hi)], uce, tel_grid)
        if not len(q.e):
            continue
        F = {"entity": q.e.astype(np.int32), "s": q.s}
        if len(epc):
            i = np.searchsorted(epc, q.c, "right")
            ii = np.clip(i, 0, len(epc) - 1)
            ok = (i < len(epc)) & ((epc[ii] >> SH) == q.e)
            F["lead"] = np.where(ok, (epc[ii] & MASK) - q.s, np.nan).astype(np.float64)
            F["ep"] = np.where(ok, ii, NO_EP).astype(np.int32)
            F["ep_first"] = np.where(ok, ep_all["first"].to_numpy()[ii], False).astype(np.int8)
        else:                      # no UCE in the input at all (e.g. a healthy live cluster)
            F["lead"] = np.full(len(q.e), np.nan)
            F["ep"] = np.full(len(q.e), NO_EP, np.int32)
            F["ep_first"] = np.zeros(len(q.e), np.int8)
        F["excl"] = (d.exclude_train.reindex(q.e).fillna(False).to_numpy(bool) if d.exclude_train is not None
                     else np.zeros(len(q.e), bool))
        if ce is not None and len(ce):
            _ce_features(F, ce, q, d)
            if d.hier or d.leaf:
                _sp_features(F, ce, q, d)
        if ev is not None and len(ev):
            _ev_features(F, ev, q, d)
        _hi_features(F, uce, ep, q, d)
        if use_nb:
            streams = {f"{G_NBH}ep_": (ep.entity.to_numpy(np.int64), ep.t.to_numpy(np.int64),
                                        [("7d", 7 * DAY), ("30d", 30 * DAY)])}
            if ce is not None and len(ce):
                streams[f"{G_NBC}ce_"] = (ce.entity.to_numpy(np.int64), ce.t.to_numpy(np.int64), [("1d", DAY), ("7d", 7 * DAY)])
            if ev is not None and len(ev):
                streams[f"{G_NBE}ev_"] = (ev.entity.to_numpy(np.int64), ev.t.to_numpy(np.int64), [("1d", DAY), ("7d", 7 * DAY)])
            _nb_features(F, grp, q, streams)
        if tel is not None and len(tel):
            _te_features(F, tel, q)
        del ce, ev, uce, tel
        S = pd.DataFrame(F)
        del F
        for c in S.columns[6:]:
            if S[c].dtype != np.float32:
                S[c] = S[c].astype(np.float32)
        S = _static(S, d, ent)
        if getattr(d, "post", None) is not None:          # input-specific derived features (e.g. bd.derived)
            S = d.post(S)
        if out_dir is not None:
            S.to_parquet(str(part) + ".tmp", index=False, engine="pyarrow")
            Path(str(part) + ".tmp").replace(part)
        else:
            out.append(S)
        if verbose:
            log(f"   shard {si + 1}/{len(bounds)}: {len(S):,} samples, {int((S.lead <= DAY).sum()):,} with UCE within 1 d")
        del S
        import gc
        gc.collect()
    return (pd.concat(out, ignore_index=True) if out_dir is None else None), ep_all


def _static(S: pd.DataFrame, d: UCEData, ent: pd.DataFrame) -> pd.DataFrame:
    add = {}
    if d.static_tel is not None:
        st = d.static_tel.reindex(S.entity.to_numpy())
        for c in st.columns:
            add[f"{G_TE}{c}"] = st[c].to_numpy(np.float32)
    cf = ent.reindex(S.entity.to_numpy())
    for c in d.num_cols:
        add[f"{G_CF}{c}"] = cf[c].to_numpy(np.float32)
    for c in d.cat_cols:
        for v in sorted(ent[c].dropna().unique()):
            add[f"{G_CF}{c}={v}"] = (cf[c].to_numpy() == v).astype(np.float32)
    return pd.concat([S, pd.DataFrame(add, index=S.index)], axis=1) if add else S
