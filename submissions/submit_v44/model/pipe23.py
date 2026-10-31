"""v42: v35/v38 matcher + typo-robust ('fuzzy') candidate retrieval for hangul names.

The evaluation hangul names carry the organizers' synthetic typos (syllable deleted, duplicated, or swapped with its
neighbour), often without tonnage. Such a typo bends the learned transliteration score (PMI) of the true vessel, so it
can fall out of the short candidate lists although its consonant skeleton / romanisation is still close. This adds,
for hangul rows only, the catalog-wide top-K by
  * skeleton edit similarity (the 'first' score that builds the retrieval pool),
  * romanised-name similarity (fuzz.ratio, i.e. normalised Indel distance),
  * transliteration PMI (a longer list),
  * partial skeleton similarity,
  * skeleton similarity of typo-reversed names (adjacent syllables swapped back, a doubled syllable removed),
to the candidate set. Features are computed exactly as before; the rankers are unchanged. Row-independent: only the
row's own name is used.
"""
from __future__ import annotations

import unicodedata

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from pipe21 import *  # noqa: F401,F403
from pipe21 import Matcher21
import vm

# candidate counts per channel; rows that state a tonnage already retrieve well through the tonnage window, rows
# without one get the wide lists (first / rom / rom_partial / pmi / skp / typo-reversal skeleton)
FZ = {True: dict(first=500, rom=300, romp=0, pmi=200, skp=100, typo=50),
      False: dict(first=500, rom=500, romp=500, pmi=200, skp=100, typo=50)}


def typo_reversals(name: str) -> list[str]:
    """Names that undo one organizer-style syllable typo: swap two adjacent syllables back, or drop one copy of a
    doubled syllable (within a word)."""
    out = []
    words = name.split(" ")
    for wi, w in enumerate(words):
        syl = list(w)
        for i in range(len(syl) - 1):
            if not ("가" <= syl[i] <= "힣" and "가" <= syl[i + 1] <= "힣"):
                continue
            if syl[i] == syl[i + 1]:
                v = syl[:i] + syl[i + 1:]
            else:
                v = syl[:i] + [syl[i + 1], syl[i]] + syl[i + 2:]
            out.append(" ".join(words[:wi] + ["".join(v)] + words[wi + 1:]))
    seen, res = set(), []
    for v in out:
        if v not in seen and v != name:
            seen.add(v)
            res.append(v)
    return res[:12]


class Matcher23(Matcher21):
    def candidates(self, q, row):
        out = super().candidates(q, row)
        if not q["hang"]:
            return out
        cands, nsim, ret, gap, tscore, tinfo = out[0], out[1], out[2], out[3], out[4], out[5]
        ix = self.ix
        sk = ix.skel_sim_all(q["sk"])
        vk = process.cdist([v for v in q["vk"] if v] or [""], ix.vkey, scorer=Levenshtein.normalized_similarity,
                           dtype=np.float32, workers=-1).max(axis=0)
        first = sk + vk
        K = FZ[bool(q["gts"])]
        add = set(self._top(first, K["first"]).tolist())
        if tinfo is not None:
            add.update(self._top(tinfo["romsim"] + 0.01 * first, K["rom"]).tolist())
            add.update(self._top(tinfo["pmi"], K["pmi"]).tolist())
            add.update(self._top(tinfo["skp"] + 0.01 * first, K["skp"]).tolist())
        if K["romp"] and q["romv"]:
            romp = process.cdist(q["romv"], ix.compact, scorer=fuzz.partial_ratio, dtype=np.uint8,
                                 workers=-1).max(axis=0).astype(np.float32) / 100.0
            add.update(self._top(romp + 0.01 * first, K["romp"]).tolist())
        name = unicodedata.normalize("NFC", row.get("vessel_name")) if isinstance(row.get("vessel_name"), str) else ""
        alts = []
        for v in typo_reversals(name):
            alts += [a[0] for a in vm.ko_variants(v)[0][:2]]
        if alts:
            add.update(self._top(ix.skel_sim_all(alts) + 0.01 * first, K["typo"]).tolist())
        sel = set(cands.tolist()) | add
        cands = np.array(sorted(sel, key=lambda i: (-ret[i], i)))
        return (cands,) + tuple(out[1:])
