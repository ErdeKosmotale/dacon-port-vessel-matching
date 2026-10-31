"""CV effect of simple top-3 override rules on hangul rows (stage-1 = v37A0 fold models, exclusion-aware OOF).

R1 'trust exact tonnage': if the pick's tonnage is not an exact match and some other top-3 candidate's is,
   switch to the best-scored such candidate.
R2 'trust the name'    : if another top-3 candidate has a clearly higher name similarity (+delta) and its tonnage is
   within 5%, switch to it.
"""
import sys, pickle, numpy as np, pandas as pd, lightgbm as lgb
import pipe22 as P
from harness import TR
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in P.FEATURES7 if f not in drop]
TL = set(P.TL_FEATURES)
cols = [i for i, f in enumerate(fn) if f not in TL]
iBN, iNG, iEG, iEI, iGM = (fn.index(f) for f in ('best_name', 'name_gap_best', 'exact_gt', 'exact_igt', 'gap_min'))
X = np.load('v37_X.npy', mmap_mode='r'); y = np.load('v37_y.npy')
meta = pickle.load(open('v37_meta.pkl', 'rb')); cids = pickle.load(open('v37_cids.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)])
ifold = np.array([m[1] for m in meta])
DELTAS = [0.05, 0.1, 0.2]
out = []
for k in range(5):
    m = lgb.Booster(model_file=f'v37A0_f{k}.txt')
    excl = set(TR[TR.fold != k].target_vessel_id)
    for j in np.flatnonzero(ifold == k):
        mt = meta[j]
        if not mt[2] or mt[4]:
            continue
        Xj = np.asarray(X[gs[j]:gs[j + 1]]); yj = y[gs[j]:gs[j + 1]]
        ok = np.array([c not in excl for c in cids[j]])
        s = np.where(ok, m.predict(Xj[:, cols], num_threads=2), -1e9)
        o = [i for i in np.argsort(-s, kind='stable')[:3] if ok[i]]
        top = o[0]
        nsim = Xj[:, iBN] - Xj[:, iNG]
        exact = (np.nan_to_num(Xj[:, iEG]) + np.nan_to_num(Xj[:, iEI])) > 0
        gapm = np.nan_to_num(Xj[:, iGM], nan=9.0)
        r = dict(tag=mt[3], base=bool(yj[top]))
        # R1
        pick = top
        if not exact[top]:
            alt = [i for i in o[1:] if exact[i]]
            if alt:
                pick = alt[0]
        r['R1'] = bool(yj[pick]); r['R1_sw'] = pick != top
        for d in DELTAS:
            pick = top
            alt = [i for i in o[1:] if nsim[i] >= nsim[top] + d and gapm[i] <= 0.05]
            if alt:
                pick = max(alt, key=lambda i: nsim[i])
            r[f'R2_{d}'] = bool(yj[pick]); r[f'R2_{d}_sw'] = pick != top
        out.append(r)
    print('fold', k, flush=True)
d = pd.DataFrame(out)
real = ~d.tag.str.startswith('synko') & ~d.tag.str.startswith('x')
for nm, msk in [('hangul real w/ton', real), ('hangul all w/ton', d.index == d.index)]:
    s = d[msk]
    line = f'{nm}: n={len(s)} base={s.base.mean():.4f}'
    for c in ['R1'] + [f'R2_{x}' for x in DELTAS]:
        line += f' | {c}={s[c].mean():.4f} (sw {int(s[c + "_sw"].sum())}, fixed {int((s[c] & ~s.base).sum())}, broke {int((~s[c] & s.base).sum())})'
    print(line)
