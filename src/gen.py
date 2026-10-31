"""English -> Hangul generator (noisy-channel decode of the learned transducer).

Given the transducer P(chunk | unit) fitted on official train pairs and a unit
bigram prior from the official train hangul names, decode the most probable
jamo-unit sequence for an English word under syllable-structure constraints,
then assemble Hangul.  Used only to synthesise training queries.
"""
from __future__ import annotations

import math
import random
import re

import numpy as np

import translit as T
import vm

CHO = T.CHO
JUNG = T.JUNG
JONG = T.JONG

UNITS = T.UNITS
NU = len(UNITS)
UID = T.UNIT_ID


def _cls(u: str) -> str:
    k = u[0]
    if k == "O" or u == "Rㄹ":
        return "on"
    if k == "V" or u == "Eㅡ":
        return "vo"
    if k == "C":
        return "co"
    return "letter"


CLS = [_cls(u) for u in UNITS]
ALLOWED_NEXT = {"start": {"on", "vo"}, "on": {"vo"}, "vo": {"co", "on", "vo"}, "co": {"on", "vo"}}
LETTER_KO_ONE = {l: kos[0] for l, kos in vm.LETTER_KO.items()}


class KoGenerator:
    def __init__(self, td: T.Transducer, hangul_names: list[str], prior_w: float = 1.0):
        self.td = td
        self.prior_w = prior_w
        # unit bigram prior (with start state index NU)
        cnt = np.full((NU + 1, NU), 0.3)
        for n in hangul_names:
            for a in T.hangul_text_variants(n)[:1]:
                units = T.text_units(a)
                prev = NU
                for u in units:
                    cnt[prev, u] += 1
                    prev = u
        self.logbg = np.log(cnt / cnt.sum(axis=1, keepdims=True))
        self.logP = np.log(np.maximum(td.P, 1e-12))
        self.empty_ok = td.P[:, 0] > 0.02
        # masks per class
        self.cls_mask = {c: np.array([CLS[i] == c for i in range(NU)]) for c in ("on", "vo", "co", "letter")}
        self.gen_units = ~self.cls_mask["letter"]
        self.i_Rl = UID["Rㄹ"]; self.i_Ol = UID["Oㄹ"]; self.i_Cl = UID["Cㄹ"]
        self.i_Eeu = UID["Eㅡ"]

    def _allowed(self, prev: int) -> np.ndarray:
        pc = "start" if prev == NU else CLS[prev]
        m = np.zeros(NU, bool)
        for c in ALLOWED_NEXT[pc]:
            m |= self.cls_mask[c]
        m &= self.gen_units
        # ㄹ onset after coda ㄹ must be Rㄹ; Rㄹ only there
        if prev == self.i_Cl:
            m[self.i_Ol] = False
        else:
            m[self.i_Rl] = False
        return m

    def decode_word(self, word: str) -> str | None:
        s = re.sub(r"[^A-Z]", "", word.upper())
        m = len(s)
        if m == 0:
            return None
        vocab = self.td.vocab
        # chunk ids: cid[pos][l] for chunk s[pos:pos+l]
        cid = [[0] + [vocab.get(s[p:p + l], -1) for l in range(1, T.L + 1)] for p in range(m + 1)]
        NEG = -1e18
        # best[pos][(prev, ne)] = score ; store as dict per pos
        best = [dict() for _ in range(m + 1)]
        back = [dict() for _ in range(m + 1)]
        best[0][(NU, 0)] = 0.0
        allowed_cache = {}
        for pos in range(m + 1):
            if not best[pos]:
                continue
            for (prev, ne), sc in list(best[pos].items()):
                if prev not in allowed_cache:
                    allowed_cache[prev] = self._allowed(prev)
                am = allowed_cache[prev]
                base = sc + self.prior_w * self.logbg[prev]
                for l in range(0, T.L + 1):
                    if pos + l > m:
                        break
                    c = cid[pos][l]
                    if c < 0:
                        continue
                    if l == 0:
                        if ne >= 2:
                            continue
                        mask = am & self.empty_ok
                    else:
                        mask = am
                    if not mask.any():
                        continue
                    tot = base + self.logP[:, c]
                    tot = np.where(mask, tot, NEG)
                    # Eㅡ only word-final
                    if pos + l < m:
                        tot[self.i_Eeu] = NEG
                    ne2 = ne + 1 if l == 0 else 0
                    npos = pos + l
                    d = best[npos]; bd = back[npos]
                    for u in np.flatnonzero(tot > NEG + 1):
                        key = (int(u), ne2)
                        v = float(tot[u])
                        if v > d.get(key, NEG):
                            d[key] = v
                            bd[key] = (pos, prev, ne)
        # final: must end in vowel or coda
        fin = [(sc, k) for k, sc in best[m].items() if k[0] != NU and CLS[k[0]] in ("vo", "co")]
        if not fin:
            return None
        sc, key = max(fin)
        units = []
        pos = m
        while key[0] != NU:
            units.append(key[0])
            pos, prev, ne = back[pos][key]
            key = (prev, ne)
        units.reverse()
        return self.assemble(units)

    @staticmethod
    def assemble(units: list[int]) -> str:
        out = []
        cho, jung, jong = None, None, ""
        def flush():
            nonlocal cho, jung, jong
            if jung is None:
                return
            c = CHO.index(cho) if cho else CHO.index("ㅇ")
            v = JUNG.index(jung)
            j = JONG.index(jong) if jong else 0
            out.append(chr(0xAC00 + (c * 21 + v) * 28 + j))
            cho, jung, jong = None, None, ""
        for u in units:
            name = UNITS[u]
            if name == "Rㄹ":
                flush(); cho = "ㄹ"
            elif name == "Eㅡ":
                jung = "ㅡ"; flush()
            elif name[0] == "O":
                flush(); cho = name[1]
            elif name[0] == "V":
                if jung is not None:
                    flush()
                jung = name[1]
            elif name[0] == "C":
                jong = name[1]; flush()
        flush()
        return "".join(out)

    # -- full names ---------------------------------------------------------
    def generate(self, english_name: str, rng: random.Random | None = None) -> str | None:
        """Catalog english name -> hangul name (digits kept, acronyms as letter names)."""
        rng = rng or random.Random(0)
        s = T.unicodedata.normalize("NFKC", str(english_name))
        s = s.replace("-", " ").replace("/", " ").replace(".", ". ")
        toks = s.split()
        out = []
        pending_no = False
        for tk in toks:
            raw = re.sub(r"[^0-9A-Za-z]", "", tk)
            if not raw:
                continue
            if raw.upper() in ("NO", "NR") and len(raw) <= 2:
                pending_no = True
                continue
            if raw.isdigit():
                n = str(int(raw))
                if pending_no:
                    out.append(rng.choice([f"제{n}호", f"{n}호", n, f"제{n}"]))
                else:
                    out.append(n)
                pending_no = False
                continue
            if raw.upper() in vm.ROMAN and out:
                n = str(vm.ROMAN[raw.upper()])
                out.append(rng.choice([n, n, "투" if n == "2" else n]))
                continue
            if raw.isupper() and len(raw) <= 5 and not (len(raw) >= 3 and sum(c in "AEIOU" for c in raw) >= 2):
                out.append("".join(LETTER_KO_ONE[c] for c in raw if c in LETTER_KO_ONE))
                continue
            # mixed alnum like "Ace1"
            parts = re.findall(r"[A-Za-z]+|[0-9]+", raw)
            buf = []
            for p in parts:
                if p.isdigit():
                    buf.append(str(int(p)))
                else:
                    h = self.decode_word(p)
                    if h is None:
                        return None
                    buf.append(h)
            out.append("".join(buf))
        return " ".join(out) if out else None
