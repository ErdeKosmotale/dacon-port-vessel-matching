"""v23: exclusion-aware candidate generation.

Evaluation vessels are disjoint from train.csv vessels, so train vessels are removed from the
catalog *before* retrieval and feature computation (not after ranking). Every candidate-set
feature (ranks, ties, exact-tonnage counts, score gaps) is then computed among vessels that can
actually be the answer. Training items get the same treatment: for a fold-k query, the vessels
that are train targets of the other folds are removed.
"""
from __future__ import annotations

import numpy as np

from pipe11 import *  # noqa: F401,F403
from pipe11 import Matcher11


class Matcher14(Matcher11):
    def __init__(self, *a, excluded=None, **kw):
        super().__init__(*a, **kw)
        self.excl = None if excluded is None else np.asarray(excluded, bool)

    def set_excluded(self, excluded):
        self.excl = None if excluded is None else np.asarray(excluded, bool)

    def _top(self, arr, k):  # masks excluded vessels in every catalog-wide top-k selection
        if self.excl is not None and len(arr) == self.ix.N:
            arr = np.where(self.excl, -1e18, arr)
        return Matcher11._top(arr, k)

    def candidates(self, q, row):
        out = super().candidates(q, row)
        if self.excl is None:
            return out
        cands = out[0]
        keep = ~self.excl[cands]
        if keep.any():
            cands = cands[keep]
        return (cands,) + tuple(out[1:])
