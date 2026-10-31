import numpy as np, pandas as pd, random, json, lightgbm as lgb, sys
import pipe11 as P6, translit as T, vm, pinyin_ko as P
from synth import Synth
from harness import TR, CAT
ix = P6.CatalogIndex(CAT); td = T.Transducer(CAT.vessel_name.tolist()); lmv = np.load('lmv.npy')
st = P6.TrainStats(TR, CAT); P6.fit_transducer(td, TR, ix)
M = P6.Matcher11(ix, st, td, lmv)
model = lgb.Booster(model_file='ranker_v18.txt'); fn = json.load(open('ranker_v18.txt.features.json')); cols = [P6.FEATURES7.index(f) for f in fn]
syn = Synth(TR, CAT); rng = random.Random(7)
train_ids = set(TR.target_vessel_id)
pool = CAT[CAT.vessel_name.map(P.is_pinyin_name) & CAT.vessel_type.isin(syn.type_map) & ~CAT.vessel_id.isin(train_ids)]
pool = pool[pool.vessel_name.str.split().str.len() >= 2]
print('pinyin catalog pool', len(pool))
rows = pool.sample(400, random_state=1).to_dict('records')
res = []
for cr in rows:
    q = syn.query_from_catalog(cr, rng)
    if q is None: continue
    q['gross_tonnage'] = cr['gross_tonnage']; q['international_gross_tonnage'] = cr['gross_tonnage']
    for mode in ['std', 'var']:
        h = P.name_to_hangul(cr['vessel_name'], None, 0) if mode == 'std' else P.name_to_hangul(cr['vessel_name'], rng)
        if not h: continue
        q2 = dict(q); q2['vessel_name'] = h
        cands, X = M.features(q2); s = model.predict(X[:, cols], num_threads=2)
        ids = ix.ids[cands]; order = np.argsort(-s)
        predx = next((ids[j] for j in order if ids[j] not in train_ids), ids[order[0]])
        res.append((mode, cr['vessel_name'], h, predx == cr['vessel_id'], cr['vessel_id'] in set(ids), CAT.vessel_name[ix.id2i[predx]]))
d = pd.DataFrame(res, columns=['mode', 'en', 'ko', 'ok', 'recall', 'pred'])
print(d.groupby('mode')[['ok', 'recall']].agg(['mean', 'size']).round(3))
pd.set_option('display.width', 200)
print(d[~d.ok].head(30).to_string())
