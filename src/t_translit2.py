import pandas as pd, numpy as np, time, sys
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
import vm, pipe, translit as T
from harness import TR, CAT
ix = pipe.CatalogIndex(CAT)
td = T.Transducer(CAT.vessel_name.tolist())
lm = T.LetterLM(td.letters)
lmv = np.array([lm.logprob(s) for s in td.letters])
H = TR[TR.vessel_name.map(vm.is_hangul)].drop_duplicates('target_vessel_id')
def pairs_for(df):
    return [([T.text_units(a) for a in T.hangul_text_variants(r.vessel_name)], ix.id2i[r.target_vessel_id]) for r in df.itertuples()]
allres = []
for K in range(5):
    td.P = td.P0.copy()
    trn = H[H.fold != K]; val = H[H.fold == K]
    td.fit(pairs_for(trn), iters=6)
    for r in val.itertuples():
        q = pipe.Matcher(ix, None).query_info({'vessel_name': r.vessel_name})
        sk = ix.skel_sim_all(q['sk'])
        vk = process.cdist(q['vk'], ix.vkey, scorer=Levenshtein.normalized_similarity, dtype=np.float32, workers=-1).max(axis=0)
        first = sk + vk
        cand = np.argpartition(-first, 3000)[:3000]
        ti = ix.id2i[r.target_vessel_id]
        best = None
        for a in T.hangul_text_variants(r.vessel_name):
            lp = td.logprob(T.text_units(a), cand)
            best = lp if best is None else np.maximum(best, lp)
        row = {'ko': r.vessel_name, 'en': CAT.vessel_name[ti], 'rec': ti in set(cand)}
        for lam in [0, 0.3, 0.5, 0.7, 0.85, 1.0]:
            sc = best - lam * lmv[cand]
            order = cand[np.argsort(-sc)]
            row[f'r{lam}'] = int(np.where(order == ti)[0][0]) if row['rec'] else 9999
        row['first'] = int((first > first[ti]).sum())
        allres.append(row)
d = pd.DataFrame(allres)
print('n', len(d), 'recall', d.rec.mean())
for c in [c for c in d.columns if c.startswith('r') and c != 'rec'] + ['first']:
    print(c, 'top1', round((d[c] == 0).mean(), 4), 'top5', round((d[c] < 5).mean(), 4), 'top30', round((d[c] < 30).mean(), 4))
d.to_csv('t_translit2.csv', index=False)
