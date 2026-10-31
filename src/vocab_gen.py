"""How well does the hangul matcher generalise to vessels whose name words were never seen in training?"""
import pandas as pd, numpy as np, random, time
import vm, pipe7 as P, translit as T
from harness import TR, CAT
ix = P.CatalogIndex(CAT); td = T.Transducer(CAT.vessel_name.tolist()); lmv = np.load('lmv.npy')
cname = dict(zip(CAT.vessel_id, CAT.vessel_name))
h = TR[TR.vessel_name.map(vm.is_hangul)].drop_duplicates('target_vessel_id')
def words(n): return set(vm.en_split(n)[0])
h = h.assign(w=[words(cname[t]) for t in h.target_vessel_id])
# 4 chunks of validation vessels; for each, train on pairs sharing no word with ANY validation vessel in the chunk
rng = random.Random(0); idx = list(range(len(h))); rng.shuffle(idx)
chunks = [idx[i::4] for i in range(4)]
rows = []
for mode in ['normal', 'word_disjoint']:
    for ci, ch in enumerate(chunks):
        val = h.iloc[ch]; vw = set().union(*val.w)
        trn = h.drop(val.index)
        if mode == 'word_disjoint':
            trn = trn[[not (w & vw) for w in trn.w]]
        st = P.TrainStats(TR[~TR.target_vessel_id.isin(val.target_vessel_id)], CAT)
        P.fit_transducer(td, trn, ix)
        M = P.Matcher7(ix, st, td, lmv)
        for r in val.to_dict('records'):
            q = M.query_info(r); cands, nsim, ret, gap, tscore, tinfo, *_ = M.candidates(q, r)
            t = ix.id2i[r['target_vessel_id']]
            pm = tinfo['pmi']; rk = int((pm > pm[t]).sum())
            rows.append(dict(mode=mode, chunk=ci, ntrain=len(trn), name=r['vessel_name'], en=cname[r['target_vessel_id']],
                             pmi_rank=rk, in_cands=t in set(cands.tolist()), ret_rank=int((ret > ret[t]).sum())))
        print(mode, ci, len(trn), flush=True)
d = pd.DataFrame(rows); d.to_pickle('vocab_gen.pkl')
g = d.groupby('mode').agg(ntrain=('ntrain', 'mean'), pmi_top1=('pmi_rank', lambda s: (s == 0).mean()), pmi_top10=('pmi_rank', lambda s: (s < 10).mean()),
                          in_cands=('in_cands', 'mean'), ret_top1=('ret_rank', lambda s: (s == 0).mean()))
print(g.round(3))
x = d[(d['mode'] == 'word_disjoint') & (d.pmi_rank >= 10)]
print(x[['name', 'en', 'pmi_rank', 'in_cands', 'ret_rank']].head(40).to_string())
