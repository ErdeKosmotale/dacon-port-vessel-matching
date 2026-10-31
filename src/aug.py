"""Training-time perturbations that mimic harder evaluation rows."""
from __future__ import annotations

import random

import numpy as np

import vm

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
SIM_CHO = {"ㄱ": "ㅋㄲ", "ㅋ": "ㄱ", "ㄷ": "ㅌㄸ", "ㅌ": "ㄷ", "ㅂ": "ㅍㅃ", "ㅍ": "ㅂ",
           "ㅈ": "ㅊㅉ", "ㅊ": "ㅈ", "ㅅ": "ㅆ", "ㅆ": "ㅅ", "ㄲ": "ㄱ", "ㄸ": "ㄷ", "ㅃ": "ㅂ", "ㅉ": "ㅈ"}
SIM_JUNG = {"ㅓ": "ㅗㅕ", "ㅗ": "ㅓ", "ㅐ": "ㅔ", "ㅔ": "ㅐ", "ㅕ": "ㅛㅓ", "ㅛ": "ㅕ",
            "ㅜ": "ㅠ", "ㅠ": "ㅜ", "ㅡ": "ㅜ", "ㅣ": "ㅢ", "ㅒ": "ㅖ", "ㅖ": "ㅒ"}


def _decomp(ch):
    code = ord(ch) - 0xAC00
    return code // 588, (code % 588) // 28, code % 28


def _comp(c, v, j):
    return chr(0xAC00 + (c * 21 + v) * 28 + j)


def ko_typo(name: str, rng: random.Random) -> str:
    s = list(name)
    idx = [i for i, c in enumerate(s) if "가" <= c <= "힣"]
    if not idx:
        return name
    op = rng.choice(["del", "dup", "space", "swap", "jamo", "jamo", "jamo"])
    i = rng.choice(idx)
    if op == "del" and len(idx) > 3:
        del s[i]
    elif op == "dup":
        s.insert(i, s[i])
    elif op == "space":
        if " " in s and rng.random() < 0.5:
            s = [c for c in s if c != " "]
        else:
            s = s[:i] + [" "] + s[i:]
    elif op == "swap" and i + 1 < len(s) and "가" <= s[i + 1] <= "힣":
        s[i], s[i + 1] = s[i + 1], s[i]
    else:
        c, v, j = _decomp(s[i])
        if rng.random() < 0.5 and JUNG[v] in SIM_JUNG:
            v = JUNG.index(rng.choice(SIM_JUNG[JUNG[v]]))
        elif CHO[c] in SIM_CHO:
            c = CHO.index(rng.choice(SIM_CHO[CHO[c]]))
        else:
            j = 0 if j else j
        s[i] = _comp(c, v, j)
    return "".join(s)


def en_typo(name: str, rng: random.Random) -> str:
    s = list(name)
    idx = [i for i, c in enumerate(s) if c.isalpha()]
    if len(idx) < 3:
        return name
    i = rng.choice(idx)
    op = rng.choice(["del", "dup", "swap", "space"])
    if op == "del":
        del s[i]
    elif op == "dup":
        s.insert(i, s[i])
    elif op == "swap" and i + 1 < len(s):
        s[i], s[i + 1] = s[i + 1], s[i]
    else:
        if " " in s and rng.random() < 0.5:
            j = s.index(" "); del s[j]
        else:
            s.insert(i, " ")
    return "".join(s)


def drop_ton(r):
    r["gross_tonnage"] = np.nan
    r["international_gross_tonnage"] = np.nan
    return r


def noise_ton(r, rng):
    f = rng.choice([rng.uniform(0.95, 0.995), rng.uniform(1.005, 1.05)])
    for c in ("gross_tonnage", "international_gross_tonnage"):
        v = vm.pos_float(r.get(c))
        if v:
            r[c] = float(round(v * f))
    return r


def perturb(row: dict, rng: random.Random, kind: str) -> dict:
    r = dict(row)
    hang = vm.is_hangul(r.get("vessel_name"))
    if kind == "drop_ton":
        return drop_ton(r)
    if kind == "noise_ton":
        return noise_ton(r, rng)
    if kind == "drop_flag":
        r["flag_name"] = np.nan
        return r
    if kind == "typo":
        r["vessel_name"] = ko_typo(r["vessel_name"], rng) if hang else en_typo(r["vessel_name"], rng)
        return r
    if kind == "mix":
        applied = False
        while not applied:
            if rng.random() < 0.45:
                r = noise_ton(r, rng); applied = True
            elif rng.random() < 0.35:
                r = drop_ton(r); applied = True
            if rng.random() < 0.3:
                r["flag_name"] = np.nan; applied = True
            if rng.random() < 0.45:
                r["vessel_name"] = ko_typo(r["vessel_name"], rng) if hang else en_typo(r["vessel_name"], rng)
                applied = True
        return r
    raise ValueError(kind)


def abbreviate(name: str, rng: random.Random):
    """Drop a leading (company-prefix-like) or trailing word; english may keep an initial."""
    toks = name.split()
    if len(toks) < 2:
        return None
    hang = vm.is_hangul(name)
    r = rng.random()
    if not hang and r < 0.25 and toks[0].isalpha():
        return " ".join([toks[0][0]] + toks[1:])
    if r < 0.7:
        return " ".join(toks[1:])
    return " ".join(toks[:-1])
