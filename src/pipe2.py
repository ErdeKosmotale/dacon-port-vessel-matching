"""v2 matcher: v1 features + learned transliteration (P(english|hangul) / LM PMI).

Row independence: `Matcher2.features(row)` only uses that row, the immutable
catalog index and statistics/parameters fitted on the official train.csv.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler, Levenshtein

import vm
import translit as T
from pipe import CatalogIndex, TrainStats, rel_gap  # noqa: F401

PMI_LAMBDA = 0.5
POOL = 3000

FEATURES2 = [
    "is_hangul", "q_len", "c_len",
    "r_ratio", "r_raw_ratio", "r_partial", "r_tsort", "r_tset", "r_jw", "r_dl_dist", "r_lev_dist",
    "exact_compact", "exact_raw", "prefix3", "first_tok_eq",
    "sk_sim", "sk_dist", "sk_len_q", "sk_len_c", "vk_sim", "vk_partial",
    "num_q", "num_c", "num_match", "num_mismatch",
    "gap_gt", "gap_igt", "gap_min", "exact_gt", "exact_igt", "q_ton_missing", "c_gt_missing",
    "log_ratio_gt", "c_log_gt", "c_dwt_gt_ratio",
    "type_prob", "type_tot", "flag_prob", "q_flag_missing", "cat_type_prior",
    "ret_score", "ret_rank", "name_rank", "name_gap_best", "ton_rank",
    "n_exact_ton_in_cands", "n_cands",
    # transliteration (hangul rows only; NaN otherwise)
    "t_pmi_rel", "t_pmi_rank", "t_lp_rel", "t_lp_per_unit", "t_lm", "t_n_units", "t_en_len",
    "t_margin", "t_pmi_minus_2nd",
]


def fit_transducer(td: T.Transducer, train: pd.DataFrame, ix: CatalogIndex, iters: int = 6):
    td.P = td.P0.copy()
    h = train[train["vessel_name"].map(vm.is_hangul)].drop_duplicates("target_vessel_id")
    pairs = []
    for name, tid in zip(h["vessel_name"], h["target_vessel_id"]):
        if tid in ix.id2i:
            pairs.append(([T.text_units(a) for a in T.hangul_text_variants(name)], ix.id2i[tid]))
    td.fit(pairs, iters=iters)
    return td


class Matcher2:
    def __init__(self, ix: CatalogIndex, stats: TrainStats, td: T.Transducer, lmv: np.ndarray,
                 k_comb=40, k_name=15, k_ton=10, k_pmi=30):
        self.ix, self.st, self.td, self.lmv = ix, stats, td, lmv
        self.k_comb, self.k_name, self.k_ton, self.k_pmi = k_comb, k_name, k_ton, k_pmi
        self.vkey = ix.vkey

    # ------------------------------------------------------------------
    def query_info(self, row: dict):
        name = row.get("vessel_name")
        name = name if isinstance(name, str) else ""
        hang = vm.is_hangul(name)
        gts = [x for x in (vm.pos_float(row.get("gross_tonnage")),
                           vm.pos_float(row.get("international_gross_tonnage"))) if x]
        if hang:
            alts, nums = vm.ko_variants(name)
            sk = [a[0] for a in alts]
            vk = [a[1] for a in alts]
            units = [T.text_units(a) for a in T.hangul_text_variants(name)]
            units = [u for u in units if u]
        else:
            sk = vm.en_skeletons(name)
            w, nums = vm.en_split(name)
            vk = ["".join(vm.en_skeleton_word(x, True) for x in w)]
            units = []
        return dict(name=name, hang=hang, gts=gts, sk=sk, vk=vk, nums=tuple(nums),
                    comp=vm.en_compact(name), raw=vm.raw_compact(name),
                    tokens=" ".join(vm.en_split(name)[0]), units=units)

    @staticmethod
    def _top(arr, k):
        k = min(k, len(arr) - 1)
        if k <= 0:
            return np.arange(len(arr))
        idx = np.argpartition(-arr, k)[:k]
        return idx[np.argsort(-arr[idx], kind="stable")]

    def candidates(self, q):
        ix = self.ix
        if q["gts"]:
            gap, tscore = ix.ton_score_all(q["gts"])
        else:
            gap, tscore = np.full(ix.N, np.inf), np.zeros(ix.N)
        tinfo = None
        if q["hang"]:
            sk = ix.skel_sim_all(q["sk"])
            vk = process.cdist([v for v in q["vk"] if v] or [""], self.vkey,
                               scorer=Levenshtein.normalized_similarity, dtype=np.float32, workers=-1).max(axis=0)
            first = sk + vk
            pool = set(self._top(first, POOL).tolist())
            if q["gts"]:
                pool.update(self._top(tscore + 0.01 * first, 300).tolist())
            pool = np.array(sorted(pool))
            lp = None
            for u in q["units"]:
                x = self.td.logprob(u, pool)
                lp = x if lp is None else np.maximum(lp, x)
            if lp is None:
                lp = np.full(len(pool), -1e9)
            pmi = lp - PMI_LAMBDA * self.lmv[pool]
            # map back to full-size arrays for selection
            pmi_full = np.full(ix.N, -1e9)
            pmi_full[pool] = pmi
            lp_full = np.full(ix.N, -1e9)
            lp_full[pool] = lp
            pmax = pmi.max()
            pmi_rel_full = pmi_full - pmax
            # retrieval score: transliteration (scaled) + tonnage
            nsim = np.clip(1 + pmi_rel_full / 10.0, 0, 1).astype(np.float32)
            ret = nsim + 0.6 * tscore
            sel = set(self._top(ret, self.k_comb).tolist())
            sel.update(self._top(pmi_full, self.k_pmi).tolist())
            sel.update(self._top(first, 10).tolist())
            if q["gts"]:
                sel.update(self._top(tscore + 0.01 * nsim, self.k_ton).tolist())
            order_pmi = np.sort(pmi)[::-1]
            tinfo = dict(pmi=pmi_full, lp=lp_full, pmax=pmax, lmax=lp.max(),
                         second=order_pmi[1] if len(order_pmi) > 1 else order_pmi[0],
                         pool_sorted=order_pmi)
            name_sim = nsim
        else:
            nsim = ix.name_sim_all(q["comp"])
            ret = nsim + 0.6 * tscore
            sel = set(self._top(ret, self.k_comb).tolist())
            sel.update(self._top(nsim, self.k_name).tolist())
            if q["gts"]:
                sel.update(self._top(tscore + 0.01 * nsim, self.k_ton).tolist())
            name_sim = nsim
        cands = np.array(sorted(sel, key=lambda i: -ret[i]))
        return cands, name_sim, ret, gap, tscore, tinfo

    # ------------------------------------------------------------------
    def features(self, row: dict, q=None):
        q = q or self.query_info(row)
        cands, nsim, ret, gap, tscore, tinfo = self.candidates(q)
        ix, st = self.ix, self.st
        qt = row.get("vessel_type")
        qf = row.get("flag_name")
        qgt = vm.pos_float(row.get("gross_tonnage"))
        qigt = vm.pos_float(row.get("international_gross_tonnage"))
        cn = nsim[cands]
        name_order = np.argsort(-cn, kind="stable")
        name_rank = np.empty(len(cands)); name_rank[name_order] = np.arange(len(cands))
        tg = gap[cands]
        ton_order = np.argsort(np.nan_to_num(tg, nan=9, posinf=9), kind="stable")
        ton_rank = np.empty(len(cands)); ton_rank[ton_order] = np.arange(len(cands))
        n_exact = int(np.sum(tg == 0))
        best_name = cn.max() if len(cn) else 0
        if tinfo is not None:
            ps = tinfo["pool_sorted"]
            n_units = min(len(u) for u in q["units"]) if q["units"] else 0
            margin = float(tinfo["pmax"] - tinfo["second"])
        rows = []
        qcomp, qraw = q["comp"], q["raw"]
        for r, i in enumerate(cands):
            ccomp = ix.compact[i]
            cgt = ix.gt[i]
            g1 = rel_gap(qgt, cgt); g2 = rel_gap(qigt, cgt)
            if q["hang"]:
                sk_best, sk_d = 0.0, 99
                for a in q["sk"]:
                    for b in ix.sk[i]:
                        s = Levenshtein.normalized_similarity(a, b)
                        if s > sk_best:
                            sk_best = s; sk_d = Levenshtein.distance(a, b)
                vk_sim = max(Levenshtein.normalized_similarity(a, ix.vkey[i]) for a in q["vk"])
                vk_part = max(fuzz.partial_ratio(a, ix.vkey[i]) for a in q["vk"]) / 100
                sk_lq = min(len(a) for a in q["sk"])
                p = tinfo["pmi"][i]
                lpv = tinfo["lp"][i]
                tfe = [p - tinfo["pmax"], float(np.searchsorted(-ps, -p, side="left")),
                       lpv - tinfo["lmax"], lpv / max(n_units, 1), self.lmv[i], n_units,
                       len(self.td.letters[i]), margin, p - tinfo["second"]]
            else:
                sk_best = max((Levenshtein.normalized_similarity(a, b) for a in q["sk"] for b in ix.sk[i]), default=0.0)
                sk_d = min((Levenshtein.distance(a, b) for a in q["sk"] for b in ix.sk[i]), default=99)
                vk_sim = Levenshtein.normalized_similarity(q["vk"][0], ix.vkey[i])
                vk_part = fuzz.partial_ratio(q["vk"][0], ix.vkey[i]) / 100
                sk_lq = len(q["sk"][0]) if q["sk"] else 0
                tfe = [np.nan] * 9
            cnums = ix.nums[i]
            tp = st.type_prob(qt, ix.ctype[i])
            ttot = st.type_pair.get(qt, {}).get("__TOT__", 0.0)
            gmin = np.nanmin([g1, g2]) if not (np.isnan(g1) and np.isnan(g2)) else np.nan
            qg = qgt or qigt
            rows.append([
                float(q["hang"]), len(qcomp), len(ccomp),
                fuzz.ratio(qcomp, ccomp) / 100, fuzz.ratio(qraw, ix.rawc[i]) / 100,
                fuzz.partial_ratio(qcomp, ccomp) / 100 if qcomp and ccomp else 0.0,
                fuzz.token_sort_ratio(q["tokens"], ix.tokens[i]) / 100,
                fuzz.token_set_ratio(q["tokens"], ix.tokens[i]) / 100,
                JaroWinkler.similarity(qcomp, ccomp),
                DamerauLevenshtein.distance(qcomp, ccomp), Levenshtein.distance(qcomp, ccomp),
                float(bool(qcomp) and qcomp == ccomp), float(bool(qraw) and qraw == ix.rawc[i]),
                float(qcomp[:3] == ccomp[:3]) if qcomp else 0.0,
                float(q["tokens"].split(" ")[0] == ix.tokens[i].split(" ")[0]) if q["tokens"] else 0.0,
                sk_best, sk_d, sk_lq, min(len(b) for b in ix.sk[i]) if ix.sk[i] else 0, vk_sim, vk_part,
                len(q["nums"]), len(cnums), float(len(q["nums"]) > 0 and set(q["nums"]) == set(cnums)),
                float(set(q["nums"]) != set(cnums)),
                g1, g2, gmin,
                float(g1 == 0), float(g2 == 0), float(qgt is None and qigt is None), float(not np.isfinite(cgt)),
                math.log(qg / cgt) if qg and np.isfinite(cgt) and cgt > 0 else np.nan,
                math.log1p(cgt) if np.isfinite(cgt) else np.nan,
                ix.dwt[i] / cgt if np.isfinite(cgt) and cgt > 0 and np.isfinite(ix.dwt[i]) else np.nan,
                tp, ttot, st.flag_prob(qf, ix.cflag[i]), float(not isinstance(qf, str)),
                st.cat_type_prior.get(ix.ctype[i], 0.0),
                ret[i], r, name_rank[r], best_name - cn[r], ton_rank[r],
                n_exact, len(cands),
            ] + tfe)
        return cands, np.asarray(rows, np.float32)
