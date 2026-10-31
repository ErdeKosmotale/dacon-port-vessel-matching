"""Field-corruption robustness: clean hangul names, corrupted type / flag / tonnage. Fold rankers, train-vessel exclusion."""
import pandas as pd, numpy as np, json, sys, lightgbm as lgb
import vm, pipe11 as P, translit as T
from harness import TR, CAT
FN = P.FEATURES7; fn = json.load(open('ranker_v18.txt.features.json')); cols = [FN.index(f) for f in fn]
ix = P.CatalogIndex(CAT); lmv = np.load('lmv.npy'); c = CAT.set_index('vessel_id')
cname = dict(zip(CAT.vessel_id, CAT.vessel_name))
def variants(r):
    tid = r['target_vessel_id']; gt = c.gross_tonnage[tid]; dwt = c.deadweight_tonnage[tid]
    out = {'orig': {}}
    out['type_unseen'] = {'vessel_type': '시멘트 운반선'}
    out['type_rare'] = {'vessel_type': '냉동.냉장선'}
    out['type_nan'] = {'vessel_type': np.nan}
    out['type_other'] = {'vessel_type': '기타선'}
    out['flag_unseen'] = {'flag_name': '쿡 제도'}
    out['flag_nan'] = {'flag_name': np.nan}
    for f in (1.08, 1.12, 1.2, 0.92, 0.88, 0.8):
        q = r['gross_tonnage'] * f
        out[f'ton_x{f}'] = {'gross_tonnage': q, 'international_gross_tonnage': q}
    if np.isfinite(dwt): out['ton_dwt'] = {'gross_tonnage': dwt, 'international_gross_tonnage': dwt}
    out['ton_nt'] = {'gross_tonnage': round(gt * 0.45), 'international_gross_tonnage': round(gt * 0.45)}
    out['igt_nan'] = {'international_gross_tonnage': np.nan}
    out['gt_nan_igt_ok'] = {'gross_tonnage': np.nan}
    return out
res = []
for k in [int(x) for x in sys.argv[1].split(',')]:
    m = lgb.Booster(model_file=f'ranker_v18_f{k}.txt')
    trk = TR[TR.fold != k]; td = T.Transducer(CAT.vessel_name.tolist()); st = P.TrainStats(trk, CAT); P.fit_transducer(td, trk, ix)
    M = P.Matcher11(ix, st, td, lmv); excl = set(trk.target_vessel_id)
    rows = TR[(TR.fold == k) & TR.vessel_name.map(vm.is_hangul)].drop_duplicates('target_vessel_id').to_dict('records')
    for row in rows:
        for tag, ch in variants(row).items():
            r2 = dict(row); r2.update(ch)
            cands, Xq = M.features(r2); s = m.predict(Xq[:, cols], num_threads=1)
            ids = ix.ids[cands]; s2 = np.where([x in excl for x in ids], s - 1e6, s); p = int(np.argmax(s2))
            res.append(dict(fold=k, tag=tag, name=row['vessel_name'], en=cname[row['target_vessel_id']], ok=ids[p] == row['target_vessel_id'],
                            rec=row['target_vessel_id'] in set(ids), pred=cname[ids[p]]))
    print('fold', k, len(res), flush=True)
d = pd.DataFrame(res); d.to_pickle(sys.argv[2])
print(d.groupby('tag').agg(acc=('ok', 'mean'), rec=('rec', 'mean'), n=('ok', 'size')).sort_values('acc').round(3).to_string())
