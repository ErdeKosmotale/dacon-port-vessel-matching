import numpy as np, pandas as pd, random, json, lightgbm as lgb, sys
import pipe6, translit as T, vm, pinyin_ko as P
from synth import Synth
from harness import TR, CAT
ix = pipe6.CatalogIndex(CAT); td = T.Transducer(CAT.vessel_name.tolist()); lmv = np.load('lmv.npy')
st = pipe6.TrainStats(TR, CAT); pipe6.fit_transducer(td, TR, ix)
M = pipe6.Matcher6(ix, st, td, lmv)
model = lgb.Booster(model_file='ranker_v6.txt'); fn = json.load(open('ranker_v6.txt.features.json')); cols = [pipe6.FEATURES6.index(f) for f in fn]
syn = Synth(TR, CAT); rng = random.Random(7)
train_ids = set(TR.target_vessel_id)
strong = set('xqz')
def chinese(n):
    w = n.lower()
    return P.is_pinyin_name(n) and (any(c in w for c in strong) or any(s in w.split() for s in ['hai', 'hua', 'xin', 'jin', 'hong', 'chang', 'sheng', 'yuan', 'tai', 'feng', 'heng', 'shun', 'xiang']))
pool = CAT[CAT.vessel_name.map(chinese) & CAT.vessel_type.isin(syn.type_map) & ~CAT.vessel_id.isin(train_ids)]
print('chinese catalog pool', len(pool))
rows = pool.sample(int(sys.argv[1]) if len(sys.argv) > 1 else 300, random_state=1).to_dict('records')
res = []
for cr in rows:
    q = syn.query_from_catalog(cr, rng)
    if q is None: continue
    h = P.name_to_hangul(cr['vessel_name'], rng)
    if not h: continue
    q['vessel_name'] = h
    cands, X = M.features(q)
    s = model.predict(X[:, cols], num_threads=2)
    ids = ix.ids[cands]
    order = np.argsort(-s)
    pred = ids[order[0]]
    predx = next((ids[j] for j in order if ids[j] not in train_ids), pred)
    res.append((cr['vessel_name'], h, np.isnan(q['gross_tonnage']), pred == cr['vessel_id'], predx == cr['vessel_id'], cr['vessel_id'] in set(ids), CAT.vessel_name[ix.id2i[pred]]))
d = pd.DataFrame(res, columns=['en', 'ko', 'ton_missing', 'ok', 'ok_excl', 'recall', 'pred'])
print(d.groupby('ton_missing')[['ok', 'ok_excl', 'recall']].mean().round(3), len(d))
pd.set_option('display.width', 200)
print(d[~d.ok].head(25).to_string())
