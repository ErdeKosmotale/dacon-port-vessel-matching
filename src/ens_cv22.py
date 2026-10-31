"""OOF accuracy of score-averaged rankers on one memmap (z-scored per item, exclusion-aware pick).
usage: ens_cv.py <tag> <folds> <spec> [<spec> ...]   spec = name:model_prefix:variant  (variant decides the columns;
       'v31' columns = first n features of the model)"""
import sys, pickle, numpy as np, pandas as pd, lightgbm as lgb
sys.argv += []
import importlib
P = importlib.import_module('pipe22')
from harness import TR
tag = sys.argv[1]; folds = [int(x) for x in sys.argv[2].split(',')]; specs = [s.split(':') for s in sys.argv[3:]]
FN = P.FEATURES7; drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}; fn = [f for f in FN if f not in drop]
TN = set(); TPF = set(P.TL_FEATURES)
def cols(v):
    v = v.rstrip('m')
    d = TN | TPF if v.endswith('0') else TPF if v.endswith('n') else TN if v.endswith('p') else set()
    return [i for i, f in enumerate(fn) if f not in d]
X = np.load(tag + '_X.npy', mmap_mode='r'); y = np.load(tag + '_y.npy'); meta = pickle.load(open(tag + '_meta.pkl', 'rb')); cids = pickle.load(open(tag + '_cids.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)])
res = []
for k in folds:
    ms = []
    for name, pref, v in specs:
        m = lgb.Booster(model_file=f'{pref}{k}.txt')
        c = list(range(m.num_feature())) if v == 'prefix' else cols(v)
        ms.append((name, m, c))
    excl = set(TR[TR.fold != k].target_vessel_id)
    for j in np.flatnonzero(np.array([m[1] for m in meta]) == k):
        Xj = np.asarray(X[gs[j]:gs[j + 1]]); yj = y[gs[j]:gs[j + 1]]
        ok = np.array([c not in excl for c in cids[j]])
        r = dict(sid=meta[j][0], fold=k, hang=meta[j][2], tag=meta[j][3], ton_missing=meta[j][4])
        zs = []
        for name, m, c in ms:
            s = m.predict(Xj[:, c], num_threads=2)
            r[name] = bool(yj[int(np.argmax(np.where(ok, s, s - 1e6)))])
            sd = s.std(); zs.append((s - s.mean()) / (sd if sd > 0 else 1))
        z = np.mean(zs, axis=0)
        r['ens'] = bool(yj[int(np.argmax(np.where(ok, z, z - 1e6)))])
        res.append(r)
    print('fold', k, flush=True)
d = pd.DataFrame(res)
out = f'ens_{tag}_' + '_'.join(s[0] for s in specs) + '.csv'
d.to_csv(out, index=False); print(out)
for col in [s[0] for s in specs] + ['ens']:
    d.assign(new_ex=d[col]).to_csv(f'cvres/ens_{col}.csv', index=False)
