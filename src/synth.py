"""Synthetic training queries built only from the official catalog + train statistics.

Used to teach the ranker situations that the official train split almost never
contains (same-name vessels, prefix changes, renamed vessels, truncation).
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

import aug
import vm


class Synth:
    def __init__(self, train: pd.DataFrame, catalog: pd.DataFrame):
        c = catalog.set_index("vessel_id")
        m = train.join(c[["vessel_type", "flag_name"]], on="target_vessel_id", rsuffix="_cat")
        self.flag_map = {}
        for cf, g in m.groupby("flag_name_cat"):
            vc = g["flag_name"].fillna("__NA__").value_counts(normalize=True)
            self.flag_map[cf] = (vc.index.tolist(), vc.values.tolist())
        self.type_map = {}
        for ct, g in m.groupby("vessel_type_cat"):
            vc = g["vessel_type"].value_counts(normalize=True)
            self.type_map[ct] = (vc.index.tolist(), vc.values.tolist())
        en = train[~train["vessel_name"].map(vm.is_hangul)]
        ft = [n.split()[0] for n in en["vessel_name"] if len(n.split()) > 1]
        self.en_first = ft
        hn = train[train["vessel_name"].map(vm.is_hangul)]
        self.ko_first = [n.split()[0] for n in hn["vessel_name"] if len(n.split()) > 1]
        self.ko_names = hn.drop_duplicates("target_vessel_id")[["vessel_name", "target_vessel_id"]].values.tolist()
        self.en_names = en.drop_duplicates("target_vessel_id")[["vessel_name", "target_vessel_id"]].values.tolist()

    @staticmethod
    def _pick(rng, opts):
        vals, probs = opts
        x = rng.random()
        acc = 0.0
        for v, p in zip(vals, probs):
            acc += p
            if x <= acc:
                return v
        return vals[-1]

    def query_from_catalog(self, crow: dict, rng: random.Random):
        """English-style port record for catalog vessel `crow`; None if type unseen in train."""
        ct = crow["vessel_type"]
        if ct not in self.type_map:
            return None
        name = str(crow["vessel_name"]).upper()
        if rng.random() < 0.35:
            name = aug.en_typo(name, rng)
        gt = vm.pos_float(crow["gross_tonnage"])
        u = rng.random()
        if gt is None or u < 0.07:
            gq = np.nan
        elif u < 0.57:
            gq = gt
        else:
            gq = float(round(gt * rng.choice([rng.uniform(0.95, 0.995), rng.uniform(1.005, 1.05)])))
        cf = crow["flag_name"]
        fq = self._pick(rng, self.flag_map[cf]) if cf in self.flag_map else "__NA__"
        return {
            "sample_id": f"SYN_{crow['vessel_id']}_{rng.randint(0, 10**6)}",
            "vessel_name": name,
            "call_sign": f"CS_SYN_{rng.randint(0, 10**9)}",
            "gross_tonnage": gq,
            "international_gross_tonnage": gq,
            "flag_name": np.nan if fq == "__NA__" else fq,
            "vessel_type": self._pick(rng, self.type_map[ct]),
            "arrival_year": rng.randint(2015, 2026),
            "target_vessel_id": crow["vessel_id"],
        }

    # -- name-level perturbations of real rows ---------------------------------
    def prefix_swap(self, name: str, rng: random.Random):
        toks = name.split()
        if len(toks) < 2:
            return None
        pool = self.ko_first if vm.is_hangul(name) else self.en_first
        if not pool:
            return None
        for _ in range(5):
            p = rng.choice(pool)
            if p != toks[0]:
                return " ".join([p] + toks[1:])
        return None

    def renamed(self, row: dict, rng: random.Random):
        pool = self.ko_names if vm.is_hangul(row["vessel_name"]) else self.en_names
        for _ in range(10):
            n, t = rng.choice(pool)
            if t != row["target_vessel_id"]:
                return n
        return None

    @staticmethod
    def truncate(name: str, rng: random.Random):
        if vm.is_hangul(name):
            s = name.replace(" ", "") if rng.random() < 0.5 else name
            if len(s) < 5:
                return None
            return s[: rng.randint(3, len(s) - 1)]
        if len(name) < 12:
            return None
        return name[: rng.randint(8, max(9, len(name) - 3))].rstrip()
