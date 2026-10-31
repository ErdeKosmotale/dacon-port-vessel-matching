"""Extended: (a) catalog type never seen at all, (b) flag pair unseen; with floors."""
import pandas as pd, numpy as np, json, lightgbm as lgb
import pipe11 as P, translit as T, vm
from harness import TR, CAT
ix = P.CatalogIndex(CAT); td = T.Transducer(CAT.vessel_name.tolist()); lmv = np.load('lmv.npy'); P.fit_transducer(td, TR, ix)
model = lgb.Booster(model_file='ranker_v18.txt'); fn = json.load(open('ranker_v18.txt.features.json')); cols = [P.FEATURES7.index(f) for f in fn]
c = CAT.set_index('vessel_id'); m = TR.join(c[['vessel_type', 'flag_name']], on='target_vessel_id', rsuffix='_cat'); m['hang'] = m.vessel_name.map(vm.is_hangul)
ALL_T = set(CAT.vessel_type.unique()); ALL_F = set(CAT.flag_name.dropna().unique())
def floor_stats(st, ft_seen=0.02, ft_never=0.01, ff_seen=0.02, ff_never=0.01):
    seen = set();
    for q, d in st.type_pair.items(): seen |= {k for k in d if k != '__TOT__'}
    for q, d in st.type_pair.items():
        for t in ALL_T:
            f = ft_seen if t in seen else ft_never
            if d.get(t, 0.0) < f: d[t] = f
    fseen = set()
    for q, d in st.flag_pair.items(): fseen |= {k for k in d if k != '__TOT__'}
    for q, d in st.flag_pair.items():
        for t in ALL_F:
            f = ff_seen if t in fseen else ff_never
            if d.get(t, 0.0) < f: d[t] = f
    return st
res = []
# (a) catalog types: hold out ALL rows whose catalog type is X (type becomes never-seen)
tv = m.vessel_type_cat.value_counts(); tv = tv[(tv >= 20) & (tv <= 400)].head(8)
for ctt, n in tv.items():
    rows = m[m.vessel_type_cat == ctt].sample(min(40, n), random_state=1)
    base = TR[~(c.vessel_type.reindex(TR.target_vessel_id).values == ctt)]
    for variant in ['never', 'never+floor']:
        st = P.TrainStats(base, CAT)
        if 'floor' in variant: st = floor_stats(st)
        M = P.Matcher11(ix, st, td, lmv)
        for r in rows.to_dict('records'):
            cands, X = M.features(r); s = model.predict(X[:, cols], num_threads=2)
            res.append(dict(kind='type', key=ctt, variant=variant, hang=r['hang'], ok=ix.ids[cands[int(np.argmax(s))]] == r['target_vessel_id']))
    print('type', ctt, flush=True)
# (b) flags: hold out rows of (query flag, catalog flag) pair
fp = m.groupby(['flag_name', 'flag_name_cat']).size().sort_values(ascending=False); fp = fp[(fp >= 15) & (fp <= 400)].head(8)
for (qf, cf), n in fp.items():
    rows = m[(m.flag_name == qf) & (m.flag_name_cat == cf)].sample(min(40, n), random_state=1)
    base = TR[~((TR.flag_name == qf) & (c.flag_name.reindex(TR.target_vessel_id).values == cf))]
    for variant in ['unseen', 'unseen+floor']:
        st = P.TrainStats(base, CAT)
        if 'floor' in variant: st = floor_stats(st)
        M = P.Matcher11(ix, st, td, lmv)
        for r in rows.to_dict('records'):
            cands, X = M.features(r); s = model.predict(X[:, cols], num_threads=2)
            res.append(dict(kind='flag', key=f'{qf}->{cf}', variant=variant, hang=r['hang'], ok=ix.ids[cands[int(np.argmax(s))]] == r['target_vessel_id']))
    print('flag', qf, cf, flush=True)
d = pd.DataFrame(res); d.to_pickle('holdout_pair2.pkl')
print(d.groupby(['kind', 'variant']).ok.agg(['mean', 'size']).round(4))
print(d.pivot_table(index=['kind', 'key'], columns='variant', values='ok', aggfunc='mean').round(3).to_string())
