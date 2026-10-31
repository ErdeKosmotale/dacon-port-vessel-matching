"""v32: v31 + token-level phonetic scores from the learned transducer.

A second transducer is built over the distinct English tokens of the catalog and shares the
substitution table learned on train.csv pairs, so each hangul query token can be scored against
each English token of every candidate: log P(token | hangul units) - 0.5 * letter-LM(token).
"""
from __future__ import annotations

import numpy as np

from pipe17 import *  # noqa: F401,F403
from pipe17 import Matcher17, FEATURES7 as _F17, PMI_LAMBDA
import translit as T
import vm

TP_FEATURES = ["tp_mean_rel", "tp_min_rel", "tp_max_unit", "tp_n_best", "tp_mean_rel_rank"]
FEATURES7 = list(_F17) + TP_FEATURES


def _letters_word(w: str) -> str:
    return "".join(ch for ch in w.upper() if "A" <= ch <= "Z")


class Matcher18(Matcher17):
    def __init__(self, ix, stats, td, lmv, **kw):
        super().__init__(ix, stats, td, lmv, **kw)
        tok_id = {}
        self.vtoks = []
        for n in self.ix.cat["vessel_name"].astype(str):
            ids = []
            for w in vm.en_split(n)[0]:
                lw = _letters_word(w)
                if len(lw) < 2:
                    continue
                if lw not in tok_id:
                    tok_id[lw] = len(tok_id)
                ids.append(tok_id[lw])
            self.vtoks.append(ids)
        self.tokens = list(tok_id)
        self.td_tok = T.Transducer(self.tokens, max_len=20)
        lm = T.LetterLM(self.td_tok.letters)
        self.lm_tok = np.array([lm.logprob(s) for s in self.td_tok.letters])
        self.sync_tok()

    def sync_tok(self):
        """Share the substitution table learned by the main transducer (call after fit_transducer)."""
        main = self.td
        m = np.full(self.td_tok.V, main.INVALID, dtype=np.int64)
        for j, ch in enumerate(self.td_tok.inv_vocab):
            m[j] = main.vocab.get(ch, main.INVALID)
        P = main.P[:, m].copy()
        P[:, self.td_tok.INVALID] = 0.0
        self.td_tok.P = P

    def features(self, row: dict, q=None):
        q = q or self.query_info(row)
        cands, X = super().features(row, q)
        n = len(cands)
        extra = np.full((n, len(TP_FEATURES)), np.nan, np.float32)
        if q["hang"] and n:
            qtoks = [t for t in vm.ko_split(q["name"])[0] if sum("가" <= ch <= "힣" for ch in t) >= 1]
            ctok_lists = [self.vtoks[i] for i in cands]
            uniq = np.array(sorted({t for lst in ctok_lists for t in lst}), dtype=np.int64)
            if qtoks and len(uniq):
                pos = {t: k for k, t in enumerate(uniq)}
                S = []          # per query token: PMI over uniq tokens
                U = []
                for a in qtoks:
                    best = np.full(len(uniq), -1e9); nu = 1
                    for v in T.hangul_text_variants(a)[:3]:
                        u = T.text_units(v)
                        if not u:
                            continue
                        g = self.td_tok.logprob(u, uniq)
                        best = np.maximum(best, g); nu = max(nu, len(u))
                    S.append(best - PMI_LAMBDA * self.lm_tok[uniq]); U.append(nu)
                S = np.array(S)                       # (nq, n_uniq)
                smax = S.max(axis=1)                  # best over all candidate tokens per query token
                for r, lst in enumerate(ctok_lists):
                    if not lst:
                        continue
                    cols = [pos[t] for t in lst]
                    b = S[:, cols].max(axis=1)        # best token of this candidate for each query token
                    rel = b - smax
                    extra[r, 0] = rel.mean(); extra[r, 1] = rel.min()
                    extra[r, 2] = float(np.max(b / np.array(U)))
                    extra[r, 3] = float((rel >= -0.5).sum())
                mr = np.nan_to_num(extra[:, 0], nan=-1e9)
                extra[:, 4] = (mr[None, :] > mr[:, None]).sum(axis=1)
        return cands, np.hstack([X, extra]).astype(np.float32)
