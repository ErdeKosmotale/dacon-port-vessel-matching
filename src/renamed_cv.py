"""Renamed queries in CV: does the ranker fail when the vessel that owns the borrowed name is still eligible?

Synthetic renamed queries borrow a train vessel's name; in CV that vessel is excluded at the final pick when it is in
another fold (~80%) and eligible when it is in the held-out fold (~20%). In the real test the owner of a borrowed
name would always be eligible if names come from non-train catalog vessels. Split accuracy / truth rank by that.
"""
import pickle, numpy as np, pandas as pd, lightgbm as lgb
import pipe22 as P
from harness import TR
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in P.FEATURES7 if f not in drop]
TL = set(P.TL_FEATURES)
cols = [i for i, f in enumerate(fn) if f not in TL]
I = {f: fn.index(f) for f in fn}
X = np.load('v37_X.npy', mmap_mode='r'); y = np.load('v37_y.npy')
meta = pickle.load(open('v37_meta.pkl', 'rb')); cids = pickle.load(open('v37_cids.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)])
ifold = np.array([m[1] for m in meta])
out = []
for k in range(5):
    m = lgb.Booster(model_file=f'v37A0_f{k}.txt')
    excl = set(TR[TR.fold != k].target_vessel_id)
    for j in np.flatnonzero(ifold == k):
        mt = meta[j]
        if mt[4] or mt[3] not in ('renamed', 'xren', 'orig'):
            continue
        Xj = np.asarray(X[gs[j]:gs[j + 1]]); yj = y[gs[j]:gs[j + 1]]
        s = m.predict(Xj[:, cols], num_threads=2)
        ok = np.array([c not in excl for c in cids[j]])
        order = [i for i in np.argsort(-s, kind='stable') if ok[i]]
        pos = [r + 1 for r, i in enumerate(order) if yj[i]]
        nsim = np.nan_to_num(Xj[:, I['best_name']] - Xj[:, I['name_gap_best']], nan=-1)
        b = int(np.argmax(nsim))
        strong = nsim[b] >= 0.9 and not yj[b]
        state = 'no strong wrong-name match' if not strong else ('wrong-name owner ELIGIBLE' if ok[b] else 'wrong-name owner excluded')
        out.append(dict(hang=bool(mt[2]), tag=mt[3], state=state, rank=pos[0] if pos else 0,
                        pick_is_owner=bool(strong and order and order[0] == b)))
    print('fold', k, flush=True)
d = pd.DataFrame(out)
for (h, t, st), s in d.groupby(['hang', 'tag', 'state']):
    r = s['rank']
    print(f'{"HANG" if h else "ENG ":4s} {t:8s} {st:28s} n={len(s):5d} acc={(r == 1).mean():.3f} '
          f'r2={(r == 2).mean():.3f} r3={(r == 3).mean():.3f} deep={((r == 0) | (r > 3)).mean():.3f} '
          f'pick=name-owner {s.pick_is_owner.mean():.3f}')
