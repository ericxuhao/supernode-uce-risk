"""Canonical input schema for UCE risk prediction.

Every data source is converted by an adapter into one ``UCEData`` object. The three input scenarios
are just different subsets of the optional tables:

    scenario                              tables used
    S1  telemetry + discrete error events  telemetry / static_tel, events, uce
    S2  CE events + statistics             ce, uce
    S3  telemetry + CE                     telemetry / static_tel, ce, uce

``uce`` (the label stream) is always required: past UCEs are known in any deployment that can produce
labels, so UCE history is a feature group in every scenario (reported separately as first vs recurrent).

Times are integer seconds on one monotone axis (Unix UTC, or a synthetic axis for anonymised data).
Entity ids are non-negative ints; entities of the same ``group`` (node / server) must have contiguous ids
so that the feature engine can shard by id range without splitting a group.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HOUR, DAY = 3600, 86400
HORIZONS = {"1h": HOUR, "4h": 4 * HOUR, "1d": DAY}

# feature-group prefixes
G_CE, G_SP, G_EV, G_HI, G_TE, G_CF = "ce__", "sp__", "ev__", "hi__", "te__", "cf__"
G_NBC, G_NBE, G_NBH = "nbc__", "nbe__", "nbh__"          # same-group (node) CE / events / UCE
SCENARIOS = {
    "S1_tel+xid": [G_TE, G_EV, G_HI, G_NBE, G_NBH, G_CF],
    "S2_ce": [G_CE, G_SP, G_HI, G_NBC, G_NBH, G_CF],
    "S3_tel+ce": [G_TE, G_CE, G_SP, G_HI, G_NBC, G_NBH, G_CF],
    "ALL": [G_TE, G_CE, G_SP, G_EV, G_HI, G_NBC, G_NBE, G_NBH, G_CF],
}
PRIMARY = {"S1_tel+xid": [G_TE, G_EV], "S2_ce": [G_CE], "S3_tel+ce": [G_TE, G_CE], "ALL": []}
REQUIRE_ALL = {"S3_tel+ce"}          # S3 needs both telemetry and CE; S1 needs either


class ShardSource:
    """Lazily loaded event table for data that does not fit in memory.

    counts: pd.Series entity -> number of records (used to balance shards);
    load(lo, hi): DataFrame of entities lo <= entity < hi, columns as the in-memory table, sorted by (entity, t).
    """
    def __init__(self, counts: pd.Series, load):
        self.counts, self.load = counts, load

    def __len__(self):
        return int(self.counts.sum())


@dataclass
class UCEData:
    name: str
    entities: pd.DataFrame                    # index = entity id; column 'group' (int); static config columns
    uce: pd.DataFrame                         # entity, t [, location columns]  -- label stream
    t_start: int
    t_end: int
    ce: "pd.DataFrame | ShardSource | None" = None  # entity, t [, sub] [, location columns]
    events: pd.DataFrame | None = None        # entity, t, code   (Xid codes, UEO, ...)
    telemetry: pd.DataFrame | None = None     # entity, t, metric columns (time series)
    static_tel: pd.DataFrame | None = None    # index = entity, metric columns (static baseline)
    hier: list[str] = field(default_factory=list)   # coarse -> fine hierarchical location columns
    leaf: tuple[str, ...] = ()                # leaf coordinates under the finest hier level, e.g. (row, col)
    removal: pd.Series | None = None          # entity -> time it leaves service (no samples after)
    exclude_train: pd.Series | None = None    # entity -> True if its samples must not be used for training
    cat_cols: list[str] = field(default_factory=list)  # categorical columns of `entities`
    num_cols: list[str] = field(default_factory=list)  # numeric config columns of `entities`
    event_codes: list = field(default_factory=list)    # event codes to featurise individually
    anchor: str = "active"                    # daily anchors: 'active' (event in last 30 d) or 'all'
    anchor_codes: list | None = None          # event codes that open 30-day anchor windows (None = all)
    uce_anchor_days: int = 180                # daily anchors keep running this long after a UCE (recurrence)
    episode_gap: int = HOUR                   # UCEs closer than this on one entity form one episode
    storm_per_min: int = 10                   # CE storm: >= this many CEs in one minute (Calchas)
    grid: int = 0                             # > 0: also score every entity with telemetry every `grid` seconds
    post: object = None                       # optional callable(samples) -> samples: derived features
    shards: int = 1

    def modalities(self) -> set[str]:
        m = {G_HI, G_NBH}
        if self.ce is not None and len(self.ce):
            m |= {G_CE, G_NBC}
            if self.hier or self.leaf:
                m.add(G_SP)
        if self.events is not None and len(self.events):
            m |= {G_EV, G_NBE}
        if (self.telemetry is not None and len(self.telemetry)) or self.static_tel is not None:
            m.add(G_TE)
        if self.cat_cols or self.num_cols:
            m.add(G_CF)
        return m


def runnable(scenario: str, present: set[str]) -> bool:
    prim = PRIMARY[scenario]
    if not prim:
        return True
    if scenario in REQUIRE_ALL:
        return all(p in present for p in prim)
    return any(p in present for p in prim)


def scenario_columns(cols, scenario: str, drop_history: bool = False) -> list[str]:
    pre = [p for p in SCENARIOS[scenario] if not (drop_history and p in (G_HI, G_NBH))]
    return [c for c in cols if any(c.startswith(p) for p in pre)]


def to_int(a) -> np.ndarray:
    return np.asarray(a, dtype=np.int64)
