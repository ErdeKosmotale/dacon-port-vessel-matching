"""Top-2 'duel' model on top of the stage-1 ranker (exclusion-aware OOF).

For every item: stage-1 fold model (trained without the item's fold) ranks the candidates; the two best eligible
candidates c1 (stage-1 pick) and c2 are described side by side (their features, their differences, the stage-1
margin). A binary LightGBM learns whether c2 is the true vessel when one of the two is. Evaluated by folds:
the duel model for fold k is trained on the other folds' items only.

usage: duel_cv.py <tag> <stage1 prefix e.g. v37A0_f> <stage1 variant e.g. A0>
"""
import sys, pickle, numpy as np, pandas as pd, lightgbm as lgb
import pipe22 as P
from harness import TR

tag, s1pref, s1var = sys.argv[1], sys.argv[2], sys.argv[3]
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in P.FEATURES7 if f not in drop]
TL = set(P.TL_FEATURES)
s1cols = [i for i, f in enumerate(fn) if not (s1var.endswith('0') and f in TL)]
X = np.load(tag + '_X.npy', mmap_mode='r'); y = np.load(tag + '_y.npy')
meta = pickle.load(open(tag + '_meta.pkl', 'rb')); cids = pickle.load(open(tag + '_cids.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)])
ifold = np.array([m[1] for m in meta])

rows = []      # per item: fold, hang, tag, tm, x1, x2, s1, s2, y1, y2, base_ok
for k in range(5):
    m = lgb.Booster(model_file=f'{s1pref}{k}.txt')
    excl = set(TR[TR.fold != k].target_vessel_id)
    for j in np.flatnonzero(ifold == k):
        Xj = np.asarray(X[gs[j]:gs[j + 1]]); yj = y[gs[j]:gs[j + 1]]
        ok = np.array([c not in excl for c in cids[j]])
        s = m.predict(Xj[:, s1cols], num_threads=2)
        s = np.where(ok, s, s - 1e6)
        o = np.argsort(-s, kind='stable')
        if len(o) < 2 or not ok[o[1]]:
            o = np.r_[o[:1], o[:1]]
        a, b = o[0], o[1]
        rows.append((k, meta[j][2], meta[j][3], meta[j][4], Xj[a], Xj[b], s[a], s[b], int(yj[a]), int(yj[b])))
    print('stage1 fold', k, flush=True)

fold = np.array([r[0] for r in rows]); hang = np.array([r[1] for r in rows]); tags = np.array([r[2] for r in rows]); tm = np.array([r[3] for r in rows])
x1 = np.vstack([r[4] for r in rows]); x2 = np.vstack([r[5] for r in rows])
s1 = np.array([r[6] for r in rows]); s2 = np.array([r[7] for r in rows])
y1 = np.array([r[8] for r in rows]); y2 = np.array([r[9] for r in rows])
F = np.hstack([x1, x2, x1 - x2, (s1 - s2)[:, None]]).astype(np.float32)
lab = y2                       # 1: the runner-up is the true vessel
usable = (y1 + y2) == 1
params = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2, seed=7)
p = np.zeros(len(rows))
for k in range(5):
    tr = usable & (fold != k)
    md = lgb.train(params, lgb.Dataset(F[tr], lab[tr]), 400)
    p[fold == k] = md.predict(F[fold == k], num_threads=2)
    print('duel fold', k, int(tr.sum()), 'train pairs', flush=True)
res = pd.DataFrame(dict(fold=fold, hang=hang, tag=tags, tm=tm, base=y1.astype(bool), y2=y2.astype(bool), p=p, margin=s1 - s2))
res.to_pickle(f'duel_{tag}.pkl')
real = ~res.tag.str.startswith('synko') & ~res.tag.str.startswith('x')
for thr in (0.5, 0.6, 0.7, 0.8):
    flip = res.p > thr
    new = np.where(flip, res.y2, res.base)
    out = []
    for nm, msk in [('hangul real w/ton', real & res.hang & ~res.tm), ('hangul all w/ton', res.hang & ~res.tm), ('english real', real & ~res.hang)]:
        out.append(f'{nm}: {res.base[msk].mean():.4f} -> {new[msk].mean():.4f} (flips {int(flip[msk].sum())})')
    print(f'thr {thr}: ' + ' | '.join(out))
