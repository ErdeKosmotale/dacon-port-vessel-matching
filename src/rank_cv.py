"""Where does the true vessel sit when the ranker's pick is wrong? (exclusion-aware OOF, per tag)

Stage-1 = per-fold z-averaged v37A0_f{k} (v33 recipe) and v31_f{k} when present, i.e. a small v31+v33 ensemble.
For every held-out item: eligible order (train vessels of the other folds removed, as at the final pick), rank of
the true vessel in it (0 = not among the candidates). Prints, per script/tag, accuracy and how the errors split
into truth at rank 2 / 3 / 4-5 / 6+ / not retrieved.
"""
import os, pickle, numpy as np, pandas as pd, lightgbm as lgb
import pipe22 as P
from harness import TR
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in P.FEATURES7 if f not in drop]
TL = set(P.TL_FEATURES)
cols37 = [i for i, f in enumerate(fn) if f not in TL]
X = np.load('v37_X.npy', mmap_mode='r'); y = np.load('v37_y.npy')
meta = pickle.load(open('v37_meta.pkl', 'rb')); cids = pickle.load(open('v37_cids.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)])
ifold = np.array([m[1] for m in meta])


def z(s):
    sd = s.std()
    return (s - s.mean()) / (sd if sd > 0 else 1.0)


out = []
for k in range(5):
    models = [(lgb.Booster(model_file=f'v37A0_f{k}.txt'), cols37)]
    if os.path.exists(f'v31_f{k}.txt'):
        m31 = lgb.Booster(model_file=f'v31_f{k}.txt')
        models.append((m31, list(range(m31.num_feature()))))
    excl = set(TR[TR.fold != k].target_vessel_id)
    for j in np.flatnonzero(ifold == k):
        mt = meta[j]
        Xj = np.asarray(X[gs[j]:gs[j + 1]]); yj = y[gs[j]:gs[j + 1]]
        s = sum(z(m.predict(Xj[:, c], num_threads=2)) for m, c in models)
        ok = np.array([c not in excl for c in cids[j]])
        order = [i for i in np.argsort(-s, kind='stable') if ok[i]]
        pos = [r + 1 for r, i in enumerate(order) if yj[i]]
        out.append(dict(hang=bool(mt[2]), tag=mt[3], tmiss=bool(mt[4]), rank=pos[0] if pos else 0,
                        margin=float(s[order[0]] - s[order[1]]) if len(order) > 1 else 9.0))
    print('fold', k, len(models), 'models', flush=True)
d = pd.DataFrame(out)
d.to_pickle('rank_cv.pkl')


def line(nm, s):
    n = len(s); err = s[s['rank'] != 1]; ne = len(err)
    if not n:
        return
    b = lambda q: int(q.sum())
    print(f'{nm:28s} n={n:6d} acc={(s["rank"] == 1).mean():.4f} errors={ne:5d} | '
          f'r2 {b(err["rank"] == 2)} r3 {b(err["rank"] == 3)} r4-5 {b(err["rank"].between(4, 5))} '
          f'r6+ {b(err["rank"] >= 6)} miss {b(err["rank"] == 0)}'
          + (f' | share r2 {b(err["rank"] == 2) / ne:.2f} r3 {b(err["rank"] == 3) / ne:.2f} miss {b(err["rank"] == 0) / ne:.2f}' if ne else ''))


for h in (True, False):
    for tm in (False, True):
        s = d[(d.hang == h) & (d.tmiss == tm)]
        line(f'{"HANGUL" if h else "ENGLISH"} ton_missing={tm}', s)
        for t, st in s.groupby('tag'):
            line(f'   {t}', st)
