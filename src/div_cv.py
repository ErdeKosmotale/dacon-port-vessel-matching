"""Fold-k OOF check of the extra rankers (exclusion-aware, as at test time): base v37A0_f{k} (lambdarank, v33 recipe)
alone, each extra fold model alone, and z-averaged ensembles. usage: div_cv.py <fold> <cfg> [<cfg> ...]"""
import sys, pickle, numpy as np, pandas as pd, lightgbm as lgb
import pipe22 as P
from harness import TR
k = int(sys.argv[1]); cfgs = sys.argv[2:]
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in P.FEATURES7 if f not in drop]
TL = set(P.TL_FEATURES)
fc = [i for i, f in enumerate(fn) if f not in TL]
X = np.load('v37_X.npy', mmap_mode='r'); y = np.load('v37_y.npy')
meta = pickle.load(open('v37_meta.pkl', 'rb')); cids = pickle.load(open('v37_cids.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)])
models = {'base': lgb.Booster(model_file=f'v37A0_f{k}.txt')}
for c in cfgs:
    models[c] = lgb.Booster(model_file=f'div_{c}_f{k}.txt')
excl = set(TR[TR.fold != k].target_vessel_id)
z = lambda s: (s - s.mean()) / (s.std() or 1.0)
combos = {n: [n] for n in models}
combos['base+all'] = list(models)
for c in cfgs:
    combos[f'base+{c}'] = ['base', c]
out = []
for j in np.flatnonzero(np.array([m[1] for m in meta]) == k):
    mt = meta[j]
    if mt[4] or mt[3].startswith('synko'):
        continue
    Xj = np.asarray(X[gs[j]:gs[j + 1]])[:, fc]; yj = y[gs[j]:gs[j + 1]]
    ok = np.array([c not in excl for c in cids[j]])
    sc = {n: z(m.predict(Xj, num_threads=2)) for n, m in models.items()}
    r = dict(hang=bool(mt[2]), real=not mt[3].startswith('x'), tag=mt[3])
    for cn, mem in combos.items():
        s = sum(sc[n] for n in mem)
        r[cn] = bool(yj[np.argmax(np.where(ok, s, -1e9))])
    out.append(r)
d = pd.DataFrame(out)
for nm, msk in [('HANGUL real', d.hang & d.real), ('ENGLISH real', ~d.hang & d.real), ('HANGUL renamed+x', d.hang & ~d.real)]:
    s = d[msk]
    print(f'{nm:18s} n={len(s):5d} ' + ' '.join(f'{c}={s[c].sum()}' for c in combos), flush=True)
