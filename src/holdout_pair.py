"""Leave-one-pair-out: remove a (korean type, catalog type) pair from TrainStats; evaluate those rows with v18."""
import pandas as pd, numpy as np, json, lightgbm as lgb, random
import pipe11 as P, translit as T, vm
from harness import TR, CAT
ix = P.CatalogIndex(CAT); td = T.Transducer(CAT.vessel_name.tolist()); lmv = np.load('lmv.npy'); P.fit_transducer(td, TR, ix)
model = lgb.Booster(model_file='ranker_v18.txt'); fn = json.load(open('ranker_v18.txt.features.json')); cols = [P.FEATURES7.index(f) for f in fn]
c = CAT.set_index('vessel_id'); m = TR.join(c[['vessel_type', 'flag_name']], on='target_vessel_id', rsuffix='_cat')
m['hang'] = m.vessel_name.map(vm.is_hangul)
pairs = m.groupby(['vessel_type', 'vessel_type_cat']).size().sort_values(ascending=False)
pairs = pairs[(pairs >= 15) & (pairs <= 400)].head(14)
print(pairs.to_string())
def smooth_stats(st, floor, seen_only):
    seen = set()
    for q, d in st.type_pair.items(): seen |= {k for k in d if k != '__TOT__'}
    for q, d in st.type_pair.items():
        for ctn in (seen if seen_only else set(CAT.vessel_type.unique())):
            if d.get(ctn, 0.0) < floor: d[ctn] = floor
    return st
rng = random.Random(3); res = []
for (qt, ctt), n in pairs.items():
    rows = m[(m.vessel_type == qt) & (m.vessel_type_cat == ctt)]
    rows = rows.sample(min(40, len(rows)), random_state=1)
    held = TR[~TR.sample_id.isin(rows.sample_id)]
    # stats WITHOUT any row of this pair (simulate an unseen pair)
    trn_stats = held[~((held.vessel_type == qt) & (c.vessel_type.reindex(held.target_vessel_id).values == ctt))]
    for variant in ['full', 'unseen', 'unseen+floor.02', 'unseen+floor.05']:
        base = held if variant == 'full' else trn_stats
        st = P.TrainStats(base, CAT)
        if 'floor' in variant: st = smooth_stats(st, float(variant.split('floor')[1]), True)
        M = P.Matcher11(ix, st, td, lmv)
        for r in rows.to_dict('records'):
            cands, X = M.features(r); s = model.predict(X[:, cols], num_threads=2)
            res.append(dict(pair=f'{qt}->{ctt}', variant=variant, hang=r['hang'], ok=ix.ids[cands[int(np.argmax(s))]] == r['target_vessel_id']))
    print(qt, ctt, flush=True)
d = pd.DataFrame(res); d.to_pickle('holdout_pair.pkl')
print(d.groupby('variant').ok.mean().round(4))
print(d.pivot_table(index='pair', columns='variant', values='ok', aggfunc='mean').round(3).to_string())
