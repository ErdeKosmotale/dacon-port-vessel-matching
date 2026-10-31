"""DACON 항만 선박 매칭 v7 - inference entry point (two-stage).

Stage 1 (v6): per-row candidate retrieval (fuzzy/partial name, learned
hangul->english transliteration, tonnage/type/flag) + LightGBM ranker.
Stage 2 (hangul rows only): the top-6 stage-1 candidates are re-scored with a
pretrained Korean causal LM (kanana-nano-2.1b-instruct, public weights) via
log P(hangul spelling | english name); a small LightGBM reranker combines the
LM scores with stage-1 evidence.

Every test row is processed on its own: all statistics come from the official
train.csv / vessel_catalog.csv and fixed model weights; the LM scores a row's
own (english candidate, hangul query) pairs only.
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
import pipe6  # noqa: E402
import translit as T  # noqa: E402
import vm  # noqa: E402

K2 = 6
S2_KEYS = ['gap_min', 'exact_igt', 'exact_gt', 'q_ton_missing', 'type_prob', 'flag_prob', 't_pmi_rel', 't_loc_rel',
           'num_mismatch', 'n_name_ties', 'ret_rank', 'name_rank', 'struct_rank', 't_red_rel']


def find(rel: str) -> Path:
    for root in (Path.cwd(), BASE):
        for d in ("data", "open"):
            p = root / d / rel
            if p.exists():
                return p
    raise FileNotFoundError(rel)


def stage2_features(s1, lp, qname, feats, fn_index):
    s1 = np.asarray(s1, float)
    n = len(s1)
    lpf = np.where(np.isnan(lp), -1e9, lp)
    srt = np.sort(lpf)[::-1]
    llm_rank = (lpf[None, :] > lpf[:, None]).sum(axis=1)
    s1srt = np.sort(s1)[::-1]
    qlen = max(len(qname.replace(" ", "")), 1)
    rows = []
    for i in range(n):
        rows.append([s1[i] - s1[0], i, s1[i] - (s1srt[1] if n > 1 else s1[i]),
                     lpf[i] - srt[0], llm_rank[i], lpf[i] - (srt[1] if n > 1 else lpf[i]), lp[i], lp[i] / qlen, n] +
                    [float(feats[i, fn_index[k]]) for k in S2_KEYS])
    return np.asarray(rows, np.float32)


def main():
    t0 = time.time()
    test = pd.read_csv(find("test.csv"), encoding="utf-8-sig")
    catalog = pd.read_csv(find("vessel_catalog.csv"), encoding="utf-8-sig")
    train = pd.read_csv(BASE / "model" / "train.csv", encoding="utf-8-sig")

    index = pipe6.CatalogIndex(catalog)
    stats = pipe6.TrainStats(train, catalog)
    td = T.Transducer(catalog["vessel_name"].tolist())
    lm = T.LetterLM(td.letters)
    lmv = np.array([lm.logprob(s) for s in td.letters])
    pipe6.fit_transducer(td, train, index)
    matcher = pipe6.Matcher6(index, stats, td, lmv)
    model1 = lgb.Booster(model_file=str(BASE / "model" / "ranker.txt"))
    fnames = json.load(open(BASE / "model" / "features.json"))
    cols = [pipe6.FEATURES6.index(f) for f in fnames]
    fn_index = {f: i for i, f in enumerate(pipe6.FEATURES6)}
    model2 = lgb.Booster(model_file=str(BASE / "model" / "stage2.txt"))
    cat_name = dict(zip(index.ids, catalog["vessel_name"].astype(str)))
    print(f"stage-1 ready in {time.time() - t0:.1f}s", flush=True)

    # ---- pass 1: stage-1 for every row ------------------------------------
    preds = [None] * len(test)
    pending = []  # (row index, qname, cand ids, s1 scores, feats)
    for k, row in enumerate(test.to_dict("records")):
        cs = row.get("call_sign")
        hit = stats.callsign_map.get(cs) if isinstance(cs, str) else None
        if hit is not None and hit in index.id2i:
            preds[k] = hit
            continue
        cands, X = matcher.features(row)
        s = model1.predict(X[:, cols], num_threads=1)
        order = np.argsort(-s, kind="stable")
        preds[k] = index.ids[cands[order[0]]]
        name = row.get("vessel_name")
        if isinstance(name, str) and vm.is_hangul(name) and len(cands) > 1:
            top = order[:K2]
            pending.append((k, name, [index.ids[c] for c in cands[top]], s[top], X[top]))
        if k % 500 == 0:
            print(f"{k}/{len(test)} {time.time() - t0:.1f}s", flush=True)
    print(f"stage-1 done {time.time() - t0:.1f}s; hangul rows for stage-2: {len(pending)}", flush=True)

    # ---- pass 2: LM scoring of each hangul row's own candidate pairs -------
    if pending:
        try:
            from llm_score import Scorer
            scorer = Scorer(str(BASE / "model" / "kanana"))
            print(f"LM loaded on {scorer.device} {time.time() - t0:.1f}s", flush=True)
            pairs = []
            for _, qname, ids, _, _ in pending:
                pairs += [(cat_name[v], qname) for v in ids]
            uniq = list(dict.fromkeys(pairs))
            sc = scorer.score(uniq, batch_size=32 if scorer.device == "cuda" else 8)
            cache = {p: round(float(v), 2) for p, v in zip(uniq, sc)}
            print(f"LM scored {len(uniq)} pairs {time.time() - t0:.1f}s", flush=True)
            # ---- pass 3: stage-2 rerank ------------------------------------
            for k, qname, ids, s1, X in pending:
                lp = np.array([cache.get((cat_name[v], qname), np.nan) for v in ids], float)
                if np.isnan(lp).all():
                    continue
                F = stage2_features(s1, lp, qname, X, fn_index)
                s2 = model2.predict(F, num_threads=1)
                preds[k] = ids[int(np.argmax(s2))]
        except Exception as e:  # stage-2 is an optional refinement; never lose the stage-1 answer
            print(f"stage-2 skipped: {type(e).__name__}: {e}", flush=True)

    out_dir = Path.cwd() / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds}).to_csv(
        out_dir / "submission.csv", index=False, encoding="utf-8")
    print(f"saved {len(preds)} rows in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
