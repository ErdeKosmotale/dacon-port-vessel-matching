"""DACON 항만 선박 매칭 v33 - inference entry point (v31 + rarity-weighted token matching: a matched catalog word counts by its
inverse document frequency over the official catalog, for hangul and english rows).

v6 ranker; vessels already linked in the official train.csv are not chosen as
answers (evaluation vessels are disjoint from train vessels), falling back to
the raw top candidate only if every candidate is a train vessel.

Each test row is processed independently:
  * english names: fuzzy / partial name + tonnage + type/flag retrieval
  * hangul names : learned hangul->english transliteration + Korean romanisation (global + word-skipping
                   local score, EM-fitted on train.csv) + tonnage retrieval
then a LightGBM lambdarank model picks one catalog vessel_id per row.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "model"))

import lightgbm as lgb  # noqa: E402
import pipe19 as pipe4  # noqa: E402
import translit as T  # noqa: E402


def find(rel: str) -> Path:
    for root in (Path.cwd(), BASE):
        for d in ("data", "open"):
            p = root / d / rel
            if p.exists():
                return p
    raise FileNotFoundError(rel)


def main():
    t0 = time.time()
    test = pd.read_csv(find("test.csv"), encoding="utf-8-sig")
    catalog = pd.read_csv(find("vessel_catalog.csv"), encoding="utf-8-sig")
    train = pd.read_csv(BASE / "model" / "train.csv", encoding="utf-8-sig")

    index = pipe4.CatalogIndex(catalog)
    stats = pipe4.TrainStats(train, catalog)
    td = T.Transducer(catalog["vessel_name"].tolist())
    lm = T.LetterLM(td.letters)
    lmv = np.array([lm.logprob(s) for s in td.letters])
    pipe4.fit_transducer(td, train, index)
    matcher = pipe4.Matcher19(index, stats, td, lmv, word_dict=pipe4.build_word_dict(train, catalog))
    model = lgb.Booster(model_file=str(BASE / "model" / "ranker.txt"))
    fnames = json.load(open(BASE / "model" / "features.json"))
    cols = [pipe4.FEATURES7.index(f) for f in fnames]
    print(f"ready in {time.time() - t0:.1f}s", flush=True)

    train_ids = set(train["target_vessel_id"])
    preds = []
    for k, row in enumerate(test.to_dict("records")):
        cands, X = matcher.features(row)
        scores = model.predict(X[:, cols], num_threads=1)
        order = np.argsort(-scores, kind="stable")
        pick = order[0]
        for j in order:
            if index.ids[cands[j]] not in train_ids:
                pick = j
                break
        preds.append(index.ids[cands[pick]])
        if k % 500 == 0:
            print(f"{k}/{len(test)} {time.time() - t0:.1f}s", flush=True)

    out_dir = Path.cwd() / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds}).to_csv(
        out_dir / "submission.csv", index=False, encoding="utf-8")
    print(f"saved {len(preds)} rows in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
