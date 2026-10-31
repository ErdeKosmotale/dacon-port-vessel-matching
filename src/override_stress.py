import pandas as pd, numpy as np, json, lightgbm as lgb
import vm, pipe11 as P, translit as T
from harness import TR, CAT
FN = P.FEATURES7; fn = json.load(open('ranker_v18.txt.features.json')); cols = [FN.index(f) for f in fn]; F = {f: i for i, f in enumerate(FN)}
ix = P.CatalogIndex(CAT); lmv = np.load('lmv.npy'); c = CAT.set_index('vessel_id'); cname = dict(zip(CAT.vessel_id, CAT.vessel_name))
res = []
for k in [0, 3]:
    m = lgb.Booster(model_file=f'ranker_v18_f{k}.txt')
    trk = TR[TR.fold != k]; td = T.Transducer(CAT.vessel_name.tolist()); st = P.TrainStats(trk, CAT); P.fit_transducer(td, trk, ix)
    M = P.Matcher11(ix, st, td, lmv); excl = set(trk.target_vessel_id)
    for row in TR[(TR.fold == k) & TR.vessel_name.map(vm.is_hangul)].drop_duplicates('target_vessel_id').to_dict('records'):
        for tag, f in [('orig', 1.0), ('x1.12', 1.12), ('x1.2', 1.2), ('x0.8', 0.8), ('x1.4', 1.4)]:
            r2 = dict(row); r2['gross_tonnage'] = row['gross_tonnage'] * f; r2['international_gross_tonnage'] = row['gross_tonnage'] * f
            cands, Xq = M.features(r2); s = m.predict(Xq[:, cols], num_threads=1)
            ids = ix.ids[cands]; ok = np.array([x not in excl for x in ids])
            rk = int(np.argmax(np.where(ok, s, -1e9)))
            pmi = np.where(ok, np.nan_to_num(Xq[:, F['t_pmi_rel']], nan=-1e9), -1e9); loc = np.where(ok, np.nan_to_num(Xq[:, F['t_loc_rel']], nan=-1e9), -1e9)
            o = np.argsort(-pmi); b = o[0]; margin = pmi[b] - pmi[o[1]]
            tgt = row['target_vessel_id']
            res.append(dict(tag=tag, ranker_ok=ids[rk] == tgt, name_ok=ids[b] == tgt, margin=margin, loc_agree=int(np.argmax(loc)) == b))
d = pd.DataFrame(res)
for tau in [3, 5, 8]:
    fire = (d.margin >= tau) & d.loc_agree
    d[f'ov{tau}'] = np.where(fire, d.name_ok, d.ranker_ok)
print(d.groupby('tag')[['ranker_ok', 'name_ok', 'ov3', 'ov5', 'ov8']].mean().round(3))
print('margin quantiles when ranker wrong & name right:', d[~d.ranker_ok & d.name_ok].margin.quantile([.1, .25, .5, .75]).round(1).tolist())
