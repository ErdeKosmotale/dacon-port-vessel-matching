import pandas as pd, numpy as np, time, sys
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein
import vm, pipe, translit as T
from harness import TR, CAT
t = time.time()
ix = pipe.CatalogIndex(CAT)
td = T.Transducer(CAT.vessel_name.tolist())
print('built', time.time() - t, 'vocab', td.V, flush=True)
H = TR[TR.vessel_name.map(vm.is_hangul)].drop_duplicates('target_vessel_id')
def pairs_for(df):
    out = []
    for r in df.itertuples():
        alts = [T.text_units(a) for a in T.hangul_text_variants(r.vessel_name)]
        out.append((alts, ix.id2i[r.target_vessel_id]))
    return out
K = int(sys.argv[1]) if len(sys.argv) > 1 else 0
trn = H[H.fold != K]; val = H[H.fold == K]
t = time.time()
td.fit(pairs_for(trn), iters=int(sys.argv[2]) if len(sys.argv) > 2 else 6, verbose=True)
print('fit', time.time() - t)
for u in ['Vㅓ', 'Oㅅ', 'Oㅋ', 'Vㅡ', 'Cㅇ', 'Vㅣ', 'Oㄹ', 'Cㄹ', 'Vㅐ']:
    print(u, td.top_chunks(u, 8))
# evaluate on val: name-only ranking (no tonnage)
res = []
t = time.time()
for r in val.itertuples():
    q = pipe.Matcher(ix, None).query_info({'vessel_name': r.vessel_name})
    sk = ix.skel_sim_all(q['sk'])
    vk = process.cdist(q['vk'], ix.vkey, scorer=Levenshtein.normalized_similarity, dtype=np.float32, workers=-1).max(axis=0)
    first = sk + vk
    k = 3000
    cand = np.argpartition(-first, k)[:k]
    ti = ix.id2i[r.target_vessel_id]
    rec = ti in set(cand)
    best = None
    for a in T.hangul_text_variants(r.vessel_name):
        lp = td.logprob(T.text_units(a), cand)
        best = lp if best is None else np.maximum(best, lp)
    order = cand[np.argsort(-best)]
    rank = int(np.where(order == ti)[0][0]) if rec else -1
    # baseline: rank by first-stage only
    frank = int((first > first[ti]).sum())
    res.append((r.vessel_name, ix.cat.vessel_name[ti], rec, rank, frank, ix.cat.vessel_name[order[0]]))
print('score time/q', (time.time() - t) / len(val))
d = pd.DataFrame(res, columns=['ko', 'en', 'rec', 'rank', 'frank', 'top'])
print('n', len(d), 'recall@3000', d.rec.mean(), 'translit top1', (d['rank'] == 0).mean(), 'top5', ((d['rank'] >= 0) & (d['rank'] < 5)).mean(), 'top20', ((d['rank'] >= 0) & (d['rank'] < 20)).mean())
print('first-stage top1', (d.frank == 0).mean(), 'top20', (d.frank < 20).mean())
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 100)
print(d[d['rank'] != 0].head(40))
