"""English name -> Hangul by pronunciation (CMUdict / g2p_en) + simplified 외래어 표기법 rules.

A *different* transliteration process from the one that produced train.csv hangul names; used to
stress-test the matcher against a style shift (and possibly as a reverse channel).
"""
from __future__ import annotations

import re

import phon
import vm

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
        "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

VOW = {"AA": "ㅏ", "AE": "ㅐ", "AH": "ㅓ", "AO": "ㅗ", "EH": "ㅔ", "ER": "ㅓ", "IH": "ㅣ", "IY": "ㅣ", "OW": "ㅗ",
       "UH": "ㅜ", "UW": "ㅜ"}
DIPH = {"AW": ("ㅏ", "ㅜ"), "AY": ("ㅏ", "ㅣ"), "EY": ("ㅔ", "ㅣ"), "OY": ("ㅗ", "ㅣ")}
Y_V = {"ㅏ": "ㅑ", "ㅐ": "ㅒ", "ㅓ": "ㅕ", "ㅔ": "ㅖ", "ㅗ": "ㅛ", "ㅜ": "ㅠ", "ㅣ": "ㅣ"}
W_V = {"ㅏ": "ㅘ", "ㅐ": "ㅙ", "ㅓ": "ㅝ", "ㅔ": "ㅞ", "ㅗ": "ㅝ", "ㅜ": "ㅜ", "ㅣ": "ㅟ"}
ONSET = {"B": "ㅂ", "CH": "ㅊ", "D": "ㄷ", "DH": "ㄷ", "F": "ㅍ", "G": "ㄱ", "HH": "ㅎ", "JH": "ㅈ", "K": "ㅋ", "L": "ㄹ",
         "M": "ㅁ", "N": "ㄴ", "P": "ㅍ", "R": "ㄹ", "S": "ㅅ", "SH": "ㅅ", "T": "ㅌ", "TH": "ㅅ", "V": "ㅂ", "Z": "ㅈ",
         "ZH": "ㅈ", "NG": "ㅇ"}
SHORT = {"AE", "EH", "IH", "UH", "AH"}
VOWELS = set(VOW) | set(DIPH)
SONOR = {"L", "R", "M", "N", "Y", "W"}


def compose(c, v, j=""):
    return chr(0xAC00 + (CHO.index(c) * 21 + JUNG.index(v)) * 28 + JONG.index(j))


def phones_to_hangul(ph: list[str]) -> str:
    ph = [re.sub(r"\d", "", p) for p in ph]
    fixed = []
    for k, p in enumerate(ph):
        if p == "ER" and k + 1 < len(ph) and ph[k + 1] in VOWELS:
            fixed += ["AH", "R"]
        else:
            fixed.append(p)
    ph = fixed
    syl = []  # list of [onset, vowel, coda]
    n = len(ph)
    i = 0
    pending_glide = None

    def add(on, v, co=""):
        syl.append([on, v, co])

    while i < n:
        p = ph[i]
        nxt = ph[i + 1] if i + 1 < n else None
        if p in VOWELS:
            on = "ㅇ"
            if p in DIPH:
                v1, v2 = DIPH[p]
                if pending_glide == "Y":
                    v1 = Y_V.get(v1, v1)
                elif pending_glide == "W":
                    v1 = W_V.get(v1, v1)
                add(on, v1)
                add("ㅇ", v2)
            else:
                v = VOW[p]
                if pending_glide == "Y":
                    v = Y_V.get(v, v)
                elif pending_glide == "W":
                    v = W_V.get(v, v)
                add(on, v)
            pending_glide = None
            i += 1
            continue
        if p in ("Y", "W"):
            if nxt in VOWELS:
                pending_glide = p
            else:
                add("ㅇ", "ㅣ" if p == "Y" else "ㅜ")
            i += 1
            continue
        # consonant
        if nxt in VOWELS or (nxt in ("Y", "W") and i + 2 < n and ph[i + 2] in VOWELS):
            # onset of the next vowel: emit an empty-onset marker resolved when the vowel arrives
            glide = None
            j = i + 1
            if nxt in ("Y", "W"):
                glide = nxt
                j = i + 2
            v_ph = ph[j]
            if v_ph in DIPH:
                v1, v2 = DIPH[v_ph]
            else:
                v1, v2 = VOW[v_ph], None
            if p == "SH" and glide is None:
                v1 = Y_V.get(v1, v1) if v1 not in ("ㅣ",) else v1
            if glide == "Y":
                v1 = Y_V.get(v1, v1)
            elif glide == "W":
                v1 = W_V.get(v1, v1)
            if p in ("CH", "JH", "SH", "ZH"):
                v1 = {"ㅑ": "ㅏ", "ㅒ": "ㅐ", "ㅕ": "ㅓ", "ㅖ": "ㅔ", "ㅛ": "ㅗ", "ㅠ": "ㅜ"}.get(v1, v1) if p != "SH" else v1
            on = ONSET.get(p, "ㅇ")
            if p == "L" and syl and syl[-1][2] == "" and i > 0 and (ph[i - 1] in VOWELS or syl[-1][1] == "ㅡ"):
                syl[-1][2] = "ㄹ"  # intervocalic l / consonant+l -> ㄹㄹ
            if p == "NG":
                if syl:
                    syl[-1][2] = "ㅇ"
                on = "ㅇ"
            add(on, v1)
            if v2:
                add("ㅇ", v2)
            i = j + 1
            continue
        # consonant not followed by a vowel: coda or 으-syllable
        prev_v = ph[i - 1] if i > 0 else None
        can_coda = bool(syl) and syl[-1][2] == "" and prev_v in VOWELS
        if p in ("M", "N", "NG") and can_coda:
            syl[-1][2] = {"M": "ㅁ", "N": "ㄴ", "NG": "ㅇ"}[p]
        elif p == "L" and can_coda:
            syl[-1][2] = "ㄹ"
        elif p == "R":
            pass  # non-rhotic
        elif p == "HH":
            pass
        elif p in ("P", "T", "K") and can_coda and prev_v in SHORT and (nxt is None or nxt not in SONOR):
            syl[-1][2] = {"P": "ㅂ", "T": "ㅅ", "K": "ㄱ"}[p]
        elif p in ("SH",):
            add("ㅅ", "ㅣ" if nxt is None else "ㅠ")
        elif p in ("CH",):
            add("ㅊ", "ㅣ")
        elif p in ("JH", "ZH"):
            add("ㅈ", "ㅣ")
        else:
            add(ONSET.get(p, "ㅇ"), "ㅡ")
        i += 1
    out = []
    for on, v, co in syl:
        try:
            out.append(compose(on, v, co))
        except ValueError:
            continue
    return "".join(out)


class StdKo:
    def __init__(self, asset_dir: str = "model_assets"):
        self.g2p = phon.G2P(asset_dir)
        self.cache = {}

    def word(self, w: str) -> str:
        key = w.upper()
        if key in self.cache:
            return self.cache[key]
        letters = re.sub(r"[^A-Za-z]", "", w)
        low = letters.lower()
        novowel = sum(c in "AEIOUYaeiouy" for c in letters) == 0
        if not letters:
            r = ""
        elif novowel and len(letters) <= 5:
            r = "".join(vm.LETTER_KO.get(c.upper(), [c])[0] for c in letters)
        elif low in self.g2p.cmu:
            r = phones_to_hangul(self.g2p.cmu[low])
        elif len(letters) <= 3:
            r = "".join(vm.LETTER_KO.get(c.upper(), [c])[0] for c in letters)
        else:
            r = phones_to_hangul(self.g2p.predict(low))
        self.cache[key] = r
        return r

    def name(self, english: str) -> str:
        toks = re.findall(r"[A-Za-z]+|\d+", str(english))
        out = []
        for t in toks:
            if t.isdigit():
                out.append(str(int(t)))
            elif t.upper() in ("NO",):
                continue
            elif t.upper() in vm.ROMAN and out:
                out.append(str(vm.ROMAN[t.upper()]))
            else:
                h = self.word(t)
                if h:
                    out.append(h)
        return " ".join(out)
