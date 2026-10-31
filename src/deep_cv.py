"""Which signal finds the true vessel when it is NOT in the ranker's top 3? (exclusion-aware OOF, v37A0 fold models)

For every held-out item with tonnage: eligible order by the stage-1 ranker; the true vessel's rank; and the picks of
simple alternative predictors over the eligible candidates outside the top 3 (what a public probe would answer):
  S_a  struct_score (tonnage + type + flag, as used by retrieval), ties by ranker score
  S_b  2*exact tonnage + type_prob + flag_prob - 3*min(gap_min, 0.3), ties by ranker score
  N_a  candidate name similarity (best_name - name_gap_best), ties by ranker score
  R4   the ranker's own 4th candidate
Prints, per script / tag: errors, deep errors (truth outside top 3) and how many of those each predictor recovers.
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
        if mt[4]:                       # tonnage missing: not in the public test (hangul) / rare
            continue
        Xj = np.asarray(X[gs[j]:gs[j + 1]]); yj = y[gs[j]:gs[j + 1]]
        s = m.predict(Xj[:, cols], num_threads=2)
        ok = np.array([c not in excl for c in cids[j]])
        order = [i for i in np.argsort(-s, kind='stable') if ok[i]]
        pos = [r + 1 for r, i in enumerate(order) if yj[i]]
        rank = pos[0] if pos else 0
        rest = np.array(order[3:], dtype=int)
        z = (s - s.mean()) / (s.std() or 1.0)
        ex = np.maximum(np.nan_to_num(Xj[:, I['exact_gt']]), np.nan_to_num(Xj[:, I['exact_igt']]))
        preds = {}
        if len(rest):
            sa = np.nan_to_num(Xj[:, I['struct_score']], nan=-9) + 1e-4 * z
            sb = (2 * ex + np.nan_to_num(Xj[:, I['type_prob']]) + np.nan_to_num(Xj[:, I['flag_prob']])
                  - 3 * np.minimum(np.nan_to_num(Xj[:, I['gap_min']], nan=1.0), 0.3) + 1e-4 * z)
            na = np.nan_to_num(Xj[:, I['best_name']] - Xj[:, I['name_gap_best']], nan=-9) + 1e-4 * z
            for nm, sc in (('S_a', sa), ('S_b', sb), ('N_a', na)):
                preds[nm] = bool(yj[rest[np.argmax(sc[rest])]])
            preds['R4'] = bool(yj[rest[0]])
        out.append(dict(hang=bool(mt[2]), tag=mt[3], rank=rank, **{p: preds.get(p, False) for p in ('S_a', 'S_b', 'N_a', 'R4')}))
    print('fold', k, flush=True)
d = pd.DataFrame(out)
d.to_pickle('deep_cv.pkl')


def line(nm, s):
    err = s[s['rank'] != 1]; deep = s[(s['rank'] == 0) | (s['rank'] > 3)]
    r = f'{nm:22s} n={len(s):6d} err={len(err):4d} deep={len(deep):4d} (miss {int((s["rank"] == 0).sum())})'
    if len(deep):
        r += ' | recovered: ' + ' '.join(f'{p} {int(deep[p].sum())} ({deep[p].mean():.2f})' for p in ('S_a', 'S_b', 'N_a', 'R4'))
    print(r)


for h in (True, False):
    s = d[d.hang == h]
    line('HANGUL' if h else 'ENGLISH', s)
    for t, st in s.groupby('tag'):
        line('   ' + t, st)
