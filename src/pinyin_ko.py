"""Chinese pinyin (and Japanese romaji) ship names -> Hangul spellings (rule based, with variants).

Domain rules only (Korean conventions for Chinese / Japanese names); no external data.
"""
from __future__ import annotations

import random
import re

# ---- pinyin --------------------------------------------------------------------
INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w", ""]
FINALS = ["iang", "iong", "uang", "ang", "eng", "ing", "ong", "uai", "iao", "ian", "uan", "ai", "ei", "ao", "ou", "an", "en", "in",
          "un", "ia", "ie", "iu", "ua", "uo", "ui", "ue", "a", "o", "e", "i", "u", "v", "er"]

# initial -> list of (onset jamo index into CHO) variants; first is standard
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
INI_KO = {"b": ["ㅂ"], "p": ["ㅍ"], "m": ["ㅁ"], "f": ["ㅍ"], "d": ["ㄷ"], "t": ["ㅌ"], "n": ["ㄴ"], "l": ["ㄹ"],
          "g": ["ㄱ"], "k": ["ㅋ"], "h": ["ㅎ"], "j": ["ㅈ"], "q": ["ㅊ"], "x": ["ㅅ", "ㅆ"], "zh": ["ㅈ", "ㅉ"],
          "ch": ["ㅊ"], "sh": ["ㅅ"], "r": ["ㄹ"], "z": ["ㅉ", "ㅈ"], "c": ["ㅊ"], "s": ["ㅆ", "ㅅ"], "y": ["ㅇ"],
          "w": ["ㅇ"], "": ["ㅇ"]}
# final -> list of (vowel sequence) variants; each item is a list of (jung, jong) syllable parts after the onset.
FIN_KO = {
    "a": [[("ㅏ", "")]], "o": [[("ㅗ", "")], [("ㅓ", "")]], "e": [[("ㅓ", "")], [("ㅡ", "")]], "er": [[("ㅓ", "ㄹ")]],
    "ai": [[("ㅏ", ""), ("ㅣ", "")], [("ㅐ", "")]], "ei": [[("ㅔ", ""), ("ㅣ", "")], [("ㅔ", "")]],
    "ao": [[("ㅏ", ""), ("ㅗ", "")], [("ㅗ", "")]], "ou": [[("ㅓ", ""), ("ㅜ", "")], [("ㅗ", ""), ("ㅜ", "")], [("ㅗ", "")]],
    "an": [[("ㅏ", "ㄴ")]], "en": [[("ㅓ", "ㄴ")], [("ㅔ", "ㄴ")]], "ang": [[("ㅏ", "ㅇ")]], "eng": [[("ㅓ", "ㅇ")], [("ㅔ", "ㅇ")]],
    "ong": [[("ㅜ", "ㅇ")], [("ㅗ", "ㅇ")]], "i": [[("ㅣ", "")]], "ia": [[("ㅑ", "")]], "ie": [[("ㅖ", "")], [("ㅔ", "")]],
    "iao": [[("ㅑ", ""), ("ㅗ", "")], [("ㅛ", "")]], "iu": [[("ㅠ", "")], [("ㅣ", ""), ("ㅜ", "")]], "ian": [[("ㅖ", "ㄴ")], [("ㅔ", "ㄴ")], [("ㅕ", "ㄴ")]],
    "in": [[("ㅣ", "ㄴ")]], "iang": [[("ㅑ", "ㅇ")]], "ing": [[("ㅣ", "ㅇ")]], "iong": [[("ㅠ", "ㅇ")], [("ㅛ", "ㅇ")]],
    "u": [[("ㅜ", "")]], "ua": [[("ㅘ", "")]], "uo": [[("ㅝ", "")], [("ㅜ", ""), ("ㅗ", "")]], "uai": [[("ㅘ", ""), ("ㅣ", "")]],
    "ui": [[("ㅞ", ""), ("ㅣ", "")], [("ㅟ", "")]], "uan": [[("ㅘ", "ㄴ")], [("ㅜ", ""), ("ㅏ", "ㄴ")]], "un": [[("ㅜ", "ㄴ")], [("ㅝ", "ㄴ")]],
    "uang": [[("ㅘ", "ㅇ")]], "ue": [[("ㅞ", "")], [("ㅟ", ""), ("ㅔ", "")]], "v": [[("ㅟ", "")]],
}
# after j/q/x the 'u' means ü
UMLAUT = {"u": [[("ㅟ", "")], [("ㅠ", "")]], "un": [[("ㅠ", "ㄴ")], [("ㅟ", "ㄴ")]], "uan": [[("ㅝ", "ㄴ")], [("ㅠ", ""), ("ㅏ", "ㄴ")]], "ue": [[("ㅞ", "")], [("ㅠ", ""), ("ㅔ", "")]]}
# y/w initials absorb into the vowel
YW = {("y", "a"): "ia", ("y", "ao"): "iao", ("y", "an"): "ian", ("y", "ang"): "iang", ("y", "e"): "ie", ("y", "i"): "i", ("y", "in"): "in",
      ("y", "ing"): "ing", ("y", "ong"): "iong", ("y", "ou"): "iu", ("y", "u"): "v", ("y", "un"): "un", ("y", "uan"): "uan", ("y", "ue"): "ue",
      ("w", "a"): "ua", ("w", "ai"): "uai", ("w", "an"): "uan", ("w", "ang"): "uang", ("w", "ei"): "ui", ("w", "en"): "un",
      ("w", "eng"): "eng", ("w", "o"): "uo", ("w", "u"): "u"}

VALID = set()
for ini in INITIALS:
    for fin in FINALS:
        VALID.add(ini + fin)
COMMON_BAD = {"zhi", "chi", "shi", "ri", "zi", "ci", "si"}  # handled specially


def _compose(cho, jung, jong=""):
    return chr(0xAC00 + (CHO.index(cho) * 21 + JUNG.index(jung)) * 28 + JONG.index(jong))


def split_initial(syl):
    for ini in INITIALS:
        if ini and syl.startswith(ini):
            return ini, syl[len(ini):]
    return "", syl


def syllable_ko(syl, rng=None, variant=0):
    syl = syl.lower()
    ini, fin = split_initial(syl)
    if syl in COMMON_BAD:  # zhi chi shi ri zi ci si -> 즈 츠 스 르 쯔 츠 쓰
        base = {"zhi": "즈", "chi": "츠", "shi": "스", "ri": "르", "zi": "쯔", "ci": "츠", "si": "쓰"}[syl]
        alt = {"zhi": "지", "chi": "치", "shi": "시", "ri": "리", "zi": "즈", "ci": "츠", "si": "스"}[syl]
        return base if variant == 0 else alt
    if ini in ("y", "w"):
        f2 = YW.get((ini, fin))
        if f2 is not None:
            fin = f2
            ini = ""
    if fin not in FIN_KO:
        return None
    onsets = INI_KO.get(ini, ["ㅇ"])
    fins = UMLAUT.get(fin, FIN_KO[fin]) if ini in ("j", "q", "x") else FIN_KO[fin]
    if rng is not None:
        on = rng.choice(onsets) if rng.random() < 0.3 else onsets[0]
        parts = rng.choice(fins) if rng.random() < 0.35 else fins[0]
    else:
        on = onsets[min(variant, len(onsets) - 1)]
        parts = fins[min(variant, len(fins) - 1)]
    out = []
    for k, (jv, jc) in enumerate(parts):
        c = on if k == 0 else "ㅇ"
        # ㅈ/ㅊ/ㅅ + y-vowels simplify in Korean usage
        if c in "ㅈㅊㅉ" and jv in "ㅑㅕㅛㅠㅖ":
            jv = {"ㅑ": "ㅏ", "ㅕ": "ㅓ", "ㅛ": "ㅗ", "ㅠ": "ㅜ", "ㅖ": "ㅔ"}[jv]
        out.append(_compose(c, jv, jc))
    return "".join(out)


def segment(word):
    """Split a romanised word into valid pinyin syllables (longest-first DP). None if impossible."""
    w = word.lower()
    n = len(w)
    best = [None] * (n + 1)
    best[0] = []
    for i in range(n):
        if best[i] is None:
            continue
        for L in range(6, 0, -1):
            s = w[i:i + L]
            if len(s) == L and (s in VALID or s in COMMON_BAD):
                if best[i + L] is None or len(best[i]) + 1 < len(best[i + L]):
                    best[i + L] = best[i] + [s]
    return best[n]


def is_pinyin_name(name):
    words = [w for w in re.findall(r"[A-Za-z]+", str(name))]
    if not words:
        return False
    return all(segment(w) is not None and len(segment(w)) <= 3 for w in words)


def name_to_hangul(name, rng=None, variant=0, join_prob=0.3):
    toks = re.findall(r"[A-Za-z]+|\d+", str(name))
    out = []
    for t in toks:
        if t.isdigit():
            out.append(str(int(t)))
            continue
        segs = segment(t)
        if segs is None:
            return None
        ko = []
        for s in segs:
            k = syllable_ko(s, rng, variant)
            if k is None:
                return None
            ko.append(k)
        out.append("".join(ko))
    if rng is not None and rng.random() < join_prob:
        return "".join(out)
    return " ".join(out)
