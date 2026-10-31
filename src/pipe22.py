"""v37: v33 + tonnage-noise likelihood + catalog unit compatibility.

In train.csv the record tonnage relates to the catalog tonnage in a very particular way (per vessel): about half
are identical, most of the rest are off by 3-5% (ratios 0.95-0.97 and 1.03-1.05), a few are within 1%, and the
1-3% band is almost empty. A monotone "closer is better" tonnage score cannot express that a candidate 2% away is
much less likely than one 4% away. These features give the ranker the empirical log-likelihood of the observed
ratio (learned from the train rows passed in: the other folds in CV), its rank and gap among the candidates, how
many candidates are tonnage-plausible, and how often the query's (Korean) vessel type goes with the candidate's
catalog size unit (Cars, TEU, cu.m., DWT, ...).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pipe19 import *  # noqa: F401,F403
from pipe19 import Matcher19, FEATURES7 as _F19
import vm

TL_FEATURES = ["ton_ll", "ton_ll_rel", "ton_ll_rank", "ton_n_plaus", "ton_band", "unit_prob"]
FEATURES7 = list(_F19) + TL_FEATURES

LO, HI, W = -0.15, 0.15, 0.005
NB = int(round((HI - LO) / W))            # interior bins; +2 tails
ALPHA = 0.5


def _bins(logr):
    b = np.floor((logr - LO) / W).astype(np.int64) + 1
    return np.clip(b, 0, NB + 1)


class TonTable:
    """log P(ratio band) for true pairs, one table per script (hangul / english)."""

    def __init__(self, ratios: np.ndarray):
        r = ratios[np.isfinite(ratios) & (ratios > 0)]
        exact = np.abs(r - 1.0) < 1e-9
        counts = np.bincount(_bins(np.log(r[~exact])), minlength=NB + 2).astype(float)
        tot = len(r) + ALPHA * (NB + 3)
        self.ll_exact = math.log((exact.sum() + ALPHA) / tot)
        self.ll_bin = np.log((counts + ALPHA) / tot)

    def ll(self, q: float, c: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = q / c
            out = np.where(np.abs(r - 1.0) < 1e-9, self.ll_exact, self.ll_bin[_bins(np.log(np.where(r > 0, r, 1.0)))])
        return np.where(np.isfinite(c) & (c > 0), out, np.nan)


def _band(q: float, c: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.abs(q / c - 1.0)
    return np.select([a < 1e-9, a < 0.01, a < 0.03, a <= 0.051], [0, 1, 2, 3], 4).astype(np.float32)


class Matcher22(Matcher19):
    def __init__(self, *a, train_df=None, **kw):
        super().__init__(*a, **kw)
        cat = self.ix.cat
        self.c_unit = cat["unit"].fillna("__NA__").astype(str).to_numpy()
        self.tables = {}
        self.unit_pair = {}
        if train_df is not None:
            cgt = dict(zip(cat["vessel_id"], cat["gross_tonnage"]))
            cun = dict(zip(cat["vessel_id"], cat["unit"].fillna("__NA__").astype(str)))
            t = train_df.drop_duplicates("target_vessel_id")
            hang = t["vessel_name"].map(vm.is_hangul).to_numpy()
            r = (pd.to_numeric(t["gross_tonnage"], errors="coerce") / t["target_vessel_id"].map(cgt)).to_numpy(float)
            self.tables[True] = TonTable(r[hang])
            self.tables[False] = TonTable(r[~hang])
            ct = pd.crosstab(train_df["vessel_type"].fillna("__NA__"), train_df["target_vessel_id"].map(cun))
            tot = ct.sum(axis=1)
            for qt in ct.index:
                row = ct.loc[qt]
                self.unit_pair[qt] = {k: v / (tot[qt] + 5.0) for k, v in row[row > 0].items()}

    def features(self, row: dict, q=None):
        q = q or self.query_info(row)
        cands, X = super().features(row, q)
        n = len(cands)
        tl = np.full((n, len(TL_FEATURES)), np.nan, np.float32)
        tab = self.tables.get(bool(q["hang"]))
        c = self.ix.gt[cands].astype(float)
        fields = []
        for key in ("gross_tonnage", "international_gross_tonnage"):
            v = vm.pos_float(row.get(key))
            if v and v not in fields:
                fields.append(v)
        if tab is not None and fields and n:
            lls = np.vstack([tab.ll(v, c) for v in fields])
            ll = np.nanmax(np.where(np.isnan(lls), -np.inf, lls), axis=0)
            ll = np.where(np.isfinite(ll), ll, np.nan)
            best = np.nanmax(ll) if np.isfinite(ll).any() else np.nan
            llz = np.nan_to_num(ll, nan=-1e9)
            tl[:, 0] = ll
            tl[:, 1] = ll - best
            tl[:, 2] = (llz[None, :] > llz[:, None]).sum(axis=1)
            tl[:, 3] = float(np.sum(llz >= best - 2.3)) if np.isfinite(best) else 0.0
            bands = np.vstack([_band(v, c) for v in fields]).min(axis=0)
            tl[:, 4] = np.where(np.isfinite(c) & (c > 0), bands, np.nan)
        qt = row.get("vessel_type")
        d = self.unit_pair.get(qt if isinstance(qt, str) else "__NA__")
        if d is not None and n:
            tl[:, 5] = [d.get(u, 0.0) for u in self.c_unit[cands]]
        return cands, np.hstack([X, tl]).astype(np.float32)
