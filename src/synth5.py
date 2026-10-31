"""Synthetic *hangul* port-call queries for catalog vessels.

Hangul spellings are composed from a word dictionary aligned on the official
train hangul/english pairs (real spelling conventions), with an English->Hangul
decoder as a fallback for a single unknown word (kept only if the forward
transliteration model still identifies the vessel).
"""
from __future__ import annotations

import collections
import random
import re

import numpy as np
import pandas as pd

import aug
import translit as T
import vm
from gen import KoGenerator
from synth import Synth


def ko_words(n: str) -> list[str]:
    s = re.sub(r"제\s*\d+\s*호?", " ", n)
    s = re.sub(r"\d+\s*호", " ", s)
    s = re.sub(r"\d+", " ", s)
    toks = s.split()
    if toks and toks[-1].endswith("호") and len(toks[-1]) >= 2 and len(toks) + len(toks[-1]) > 3:
        toks[-1] = toks[-1][:-1]
    return [t for t in toks if t]


class KoSynth:
    def __init__(self, train: pd.DataFrame, catalog: pd.DataFrame, td: T.Transducer, ix, lmv: np.ndarray):
        self.cat = catalog
        self.ix = ix
        self.td = td
        self.lmv = lmv
        self.base = Synth(train, catalog)
        c = catalog.set_index("vessel_id")
        h = train[train["vessel_name"].map(vm.is_hangul)].drop_duplicates("target_vessel_id")
        d = collections.defaultdict(collections.Counter)
        for name, tid in zip(h["vessel_name"], h["target_vessel_id"]):
            en = c["vessel_name"].get(tid)
            if not isinstance(en, str):
                continue
            kw, ew = ko_words(name), vm.en_split(en)[0]
            if kw and len(kw) == len(ew):
                for a, b in zip(ew, kw):
                    d[a][b] += 1
        self.dict = {k: v for k, v in d.items()}
        self.gen = KoGenerator(td, h["vessel_name"].tolist())

    def _word(self, w: str, rng: random.Random):
        if w in self.dict:
            opts = self.dict[w]
            ks, ws = zip(*opts.items())
            return rng.choices(ks, weights=ws)[0], True
        if w in vm.ROMAN:
            n = str(vm.ROMAN[w])
            return (rng.choice([n, "투"]) if n == "2" else n), True
        return None, False

    def compose(self, english: str, rng: random.Random, allow_decode: bool = True):
        """-> (hangul, n_decoded) or None."""
        s = T.unicodedata.normalize("NFKC", str(english)).replace("-", " ").replace("/", " ")
        toks = [re.sub(r"[^0-9A-Za-z]", "", t) for t in s.split()]
        toks = [t for t in toks if t]
        out, n_dec, pending_no = [], 0, False
        for tk in toks:
            up = tk.upper()
            if up in ("NO", "NR"):
                pending_no = True
                continue
            if tk.isdigit():
                n = str(int(tk))
                out.append(rng.choice([f"제{n}호", f"{n}호", n, f"제{n}"]) if pending_no else n)
                pending_no = False
                continue
            parts = re.findall(r"[A-Za-z]+|[0-9]+", tk)
            buf = []
            for p in parts:
                if p.isdigit():
                    buf.append(str(int(p)))
                    continue
                w, ok = self._word(p.upper(), rng)
                if not ok:
                    if p.isupper() and len(p) <= 5 and not (len(p) >= 3 and sum(ch in "AEIOU" for ch in p) >= 2):
                        w = "".join(vm.LETTER_KO[ch][0] for ch in p if ch in vm.LETTER_KO)
                    elif allow_decode and n_dec == 0:
                        w = self.gen.decode_word(p)
                        n_dec += 1
                        if w and not self.plausible(w, len(p)):
                            return None
                    else:
                        return None
                    if not w:
                        return None
                buf.append(w)
            out.append("".join(buf))
        if not out or not any(vm.is_hangul(x) for x in out):
            return None
        return " ".join(out), n_dec

    OK_CODA = set(["", "ㄱ", "ㄴ", "ㄹ", "ㅁ", "ㅂ", "ㅇ", "ㅅ"])
    BAD_ONSET = set("ㄸㅃㅉㄲ")

    @staticmethod
    def plausible(word: str, letters: int) -> bool:
        syl = [T.CHO[(ord(ch) - 0xAC00) // 588] + T.JUNG[((ord(ch) - 0xAC00) % 588) // 28] + T.JONG[(ord(ch) - 0xAC00) % 28]
               for ch in word if "가" <= ch <= "힣"]
        if not syl or len(syl) * 3 < letters:
            return False
        for s in syl:
            if s[2:] not in KoSynth.OK_CODA or s[0] in KoSynth.BAD_ONSET:
                return False
            if s[0] == "ㅆ" and s[1] not in "ㅓㅣㅡㅏㅜ":
                return False
        return True

    def consistent(self, hangul: str, cidx: int, pool_k: int = 1500) -> bool:
        """Forward model must rank the true vessel in its top-3 among a skeleton pool."""
        alts, _ = vm.ko_variants(hangul)
        sk = self.ix.skel_sim_all([a[0] for a in alts])
        pool = np.argpartition(-sk, pool_k)[:pool_k]
        if cidx not in set(pool.tolist()):
            pool = np.append(pool, cidx)
        best = np.full(len(pool), -1e9)
        for a in T.hangul_text_variants(hangul):
            u = T.text_units(a)
            if u:
                g, _ = self.td.logprob_both(u, pool)
                best = np.maximum(best, g)
        pmi = best - 0.5 * self.lmv[pool]
        mine = pmi[np.where(pool == cidx)[0][0]]
        return int((pmi > mine).sum()) < 3

    def query(self, crow: dict, rng: random.Random, allow_decode: bool = True):
        res = self.compose(crow["vessel_name"], rng, allow_decode)
        if res is None:
            return None
        hangul, n_dec = res
        if n_dec and not self.consistent(hangul, self.ix.id2i[crow["vessel_id"]]):
            return None
        q = self.base.query_from_catalog(crow, rng)
        if q is None:
            return None
        name = hangul
        u = rng.random()
        if u < 0.3:
            name = name.replace(" ", "")
        if rng.random() < 0.15 and not re.search(r"\d$", name):
            name = name + "호"
        if rng.random() < 0.2:
            name = aug.ko_typo(name, rng)
        q["vessel_name"] = name
        q["sample_id"] = f"SYNKO_{crow['vessel_id']}_{rng.randint(0, 10**6)}"
        return q
