import pandas as pd, numpy as np, random, pickle, sys
import vm, pipe9 as P, translit as T, aug
from synth import Synth
from synth5 import KoSynth
from harness import TR, CAT
k = int(sys.argv[1])
ix = P.CatalogIndex(CAT); td = T.Transducer(CAT.vessel_name.tolist()); lmv = np.load('lmv.npy')
rng = random.Random(777 + k)
trn = TR[TR.fold != k]
st = P.TrainStats(trn, CAT); syn = Synth(trn, CAT); P.fit_transducer(td, trn, ix)
M = P.Matcher7(ix, st, td, lmv)
id2i = {v: i for i, v in enumerate(ix.ids)}
res = []
rows = [r for r in TR[TR.fold == k].to_dict('records') if vm.is_hangul(r['vessel_name']) and vm.pos_float(r.get('gross_tonnage'))]
for row in rows:
    vs = []
    a = aug.abbreviate(row['vessel_name'], rng); vs.append(('abbr', a))
    vs.append(('prefix', syn.prefix_swap(row['vessel_name'], rng)))
    vs.append(('trunc', syn.truncate(row['vessel_name'], rng)))
    tk = row['vessel_name'].split()
    if len(tk) >= 2: vs.append(('swap', ' '.join(reversed(tk))))
    for tag, nm in vs:
        if not nm: continue
        r2 = dict(row); r2['vessel_name'] = nm
        q = M.query_info(r2)
        if not q["hang"]: continue
        cands, *_ = M.candidates(q, r2)
        L = M._last; ti = L['tinfo']; t = id2i[row['target_vessel_id']]
        def rk(arr): return int((arr > arr[t]).sum())
        inwin = L['gap'] <= 0.06
        def rkw(arr): return int(((arr > arr[t]) & inwin).sum())
        res.append(dict(tag=tag, name=nm, orig=row['vessel_name'], in_sel=t in set(cands.tolist()), in_pool=t in set(L['pool'].tolist()),
                        gap=float(L['gap'][t]), r_pmi=rk(ti['pmi']), r_pmil=rk(ti['pmil']), r_pmir=rk(ti['pmir']), r_ret=rk(L['ret']),
                        w_pmi=rkw(ti['pmi']), w_pmil=rkw(ti['pmil']), w_pmir=rkw(ti['pmir']), w_romsim=rkw(ti['romsim']), w_skp=rkw(ti['skp']), ncand=len(cands)))
d = pd.DataFrame(res); d.to_pickle(f'diag_recall_{k}.pkl')
print(d.groupby('tag')[['in_sel', 'in_pool']].mean())
m = d[~d.in_sel]
print(len(m), 'misses'); print(m.drop(columns=['orig']).describe().T[['50%', 'mean']])
print(m[['tag', 'name', 'orig', 'gap', 'w_pmi', 'w_pmil', 'w_pmir', 'w_romsim', 'w_skp']].head(40).to_string())
