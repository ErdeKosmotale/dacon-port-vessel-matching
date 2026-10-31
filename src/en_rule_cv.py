"""English rows: would trusting a much better name match help? (exclusion-aware OOF, v37A0 fold models)

Rule N_d: if the eligible candidate with the highest name similarity (best_name - name_gap_best) beats the ranker's
pick by at least d in name similarity, and its tonnage is within tau of the record (or the record has none),
switch to it. Prints fixed / broken counts per d on real and synthetic english items (tonnage present or not).
"""
import pickle, numpy as np, pandas as pd, lightgbm as lgb
import pipe22 as P
from harness import TR
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in P.FEATURES7 if f not in drop]
TL = set(P.TL_FEATURES)
cols = [i for i, f in enumerate(fn) if f not in TL]
iBN, iNG, iG = fn.index('best_name'), fn.index('name_gap_best'), fn.index('gap_min')
X = np.load('v37_X.npy', mmap_mode='r'); y = np.load('v37_y.npy')
meta = pickle.load(open('v37_meta.pkl', 'rb')); cids = pickle.load(open('v37_cids.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)])
ifold = np.array([m[1] for m in meta])
DS = [0.05, 0.1, 0.2, 0.3]
TAUS = [0.06, 1.0]
out = []
for k in range(5):
    m = lgb.Booster(model_file=f'v37A0_f{k}.txt')
    excl = set(TR[TR.fold != k].target_vessel_id)
    for j in np.flatnonzero(ifold == k):
        mt = meta[j]
        if mt[2]:
            continue
        Xj = np.asarray(X[gs[j]:gs[j + 1]]); yj = y[gs[j]:gs[j + 1]]
        s = m.predict(Xj[:, cols], num_threads=2)
        ok = np.array([c not in excl for c in cids[j]])
        if not ok.any():
            continue
        top = int(np.argmax(np.where(ok, s, -1e9)))
        nsim = np.nan_to_num(Xj[:, iBN] - Xj[:, iNG], nan=-1)
        gap = Xj[:, iG]
        r = dict(tag=mt[3], tmiss=bool(mt[4]), base=bool(yj[top]))
        for tau in TAUS:
            okt = ok & ~(np.isfinite(gap) & (gap > tau))
            nb = int(np.argmax(np.where(okt, nsim + 1e-6 * s, -1e9))) if okt.any() else top
            for d in DS:
                pick = nb if nsim[nb] >= nsim[top] + d else top
                r[f'{d}_{tau}'] = bool(yj[pick]); r[f'sw{d}_{tau}'] = pick != top
        out.append(r)
    print('fold', k, flush=True)
d = pd.DataFrame(out)
real = ~d.tag.str.startswith('x')
for nm, msk in [('ENGLISH real (A items)', real), ('ENGLISH x-items', ~real)]:
    s = d[msk]
    line = f'{nm:24s} n={len(s)} base={s.base.sum()}'
    for tau in TAUS:
        for dd in DS:
            c = f'{dd}_{tau}'
            line += f' | d{dd}/t{tau}: sw {int(s["sw" + c].sum())} fix {int((s[c] & ~s.base).sum())} brk {int((~s[c] & s.base).sum())}'
    print(line)
