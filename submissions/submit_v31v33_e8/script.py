"""DACON 항만 선박 매칭 v35 - inference entry point (v33 + borrowed-name features: whether the record name equals /
is close to a name already carried by a train.csv vessel; train vessels stay in the candidate set and are only
skipped at the final pick, as in v31/v33) + train-presence features: which candidates are train vessels, whether
the name fits a train vessel better than any eligible one, and each candidate's gaps to the best eligible one).

Each test row is processed independently:
  * english names: fuzzy / partial name + tonnage + type/flag retrieval
  * hangul names : learned hangul->english transliteration + Korean romanisation (global + word-skipping
                   local score, EM-fitted on train.csv) + token-level standard-orthography spellings +
                   tonnage retrieval
then LightGBM lambdarank model(s) pick one catalog vessel_id per row (scores of several rankers, if present,
are averaged per row after standardising them over that row's candidates).
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
import pipe21 as pipe4  # noqa: E402
import translit as T  # noqa: E402


def find(rel: str) -> Path:
    for root in (Path.cwd(), BASE):
        for d in ("data", "open"):
            p = root / d / rel
            if p.exists():
                return p
    raise FileNotFoundError(rel)


def load_rankers():
    """model/ranker*.txt; a name starting with rankerH / rankerE is used only for hangul / non-hangul rows
    (and then replaces the general rankers for those rows)."""
    out = []
    for path in sorted((BASE / "model").glob("ranker*.txt")):
        fj = path.with_name(path.name + ".features.json")
        if not fj.exists():
            fj = BASE / "model" / "features.json"
        fnames = json.load(open(fj))
        scope = "H" if path.name.startswith("rankerH") else "E" if path.name.startswith("rankerE") else "all"
        out.append((lgb.Booster(model_file=str(path)), [pipe4.FEATURES7.index(f) for f in fnames], scope))
    return out


def row_scores(rankers, X, hang: bool):
    own = [r for r in rankers if r[2] == ("H" if hang else "E")]
    use = own or [r for r in rankers if r[2] == "all"]
    if len(use) == 1:
        m, cols, _ = use[0]
        return m.predict(X[:, cols], num_threads=1)
    scores = np.zeros(X.shape[0])
    for m, cols, _ in use:
        s = m.predict(X[:, cols], num_threads=1)
        sd = s.std()
        scores += (s - s.mean()) / (sd if sd > 0 else 1.0)
    return scores


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
    train_ids = set(train["target_vessel_id"])
    matcher = pipe4.Matcher21(index, stats, td, lmv, word_dict=pipe4.build_word_dict(train, catalog),
                              excluded=None, train_names=train["vessel_name"].tolist(),
                              train_mask=np.isin(index.ids, list(train_ids)))
    rankers = load_rankers()
    print(f"ready in {time.time() - t0:.1f}s, {len(rankers)} ranker(s)", flush=True)

    preds = []
    for k, row in enumerate(test.to_dict("records")):
        cands, X = matcher.features(row)
        scores = row_scores(rankers, X, bool(X[0, pipe4.FEATURES7.index("is_hangul")] > 0.5))
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
