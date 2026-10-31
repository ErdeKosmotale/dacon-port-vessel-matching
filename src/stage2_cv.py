import pandas as pd, numpy as np, pickle, gc, json, lightgbm as lgb
import pipe11 as P
from stage2_win import s2_feats, K2, fn, F
from harness import TR
X = np.load('v18_X.npy', mmap_mode='r'); y = np.load('v18_y.npy'); meta = pickle.load(open('v18_meta.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gstart = np.concatenate([[0], np.cumsum(g)]); fold = np.repeat([m[1] for m in meta], g)
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
p2 = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2, seed=7)
TM = F['q_ton_missing']; GAP = F['gap_min']
# stage-1 OOF scores for all hangul-with-ton items, per fold (stage-1 model trained on other folds)
s1 = {}
for k in range(5):
    tr = fold != k; gi = np.array([m[1] != k for m in meta])
    m = lgb.train(params, lgb.Dataset(np.ascontiguousarray(X[tr]), y[tr], group=g[gi], free_raw_data=True), 500)
    for j in np.flatnonzero(~gi):
        if not meta[j][2] or meta[j][4]: continue
        s1[j] = m.predict(np.ascontiguousarray(X[gstart[j]:gstart[j + 1]]), num_threads=2)
    del m; gc.collect(); print('stage1 fold', k, flush=True)
pickle.dump(s1, open('stage2_s1.pkl', 'wb'))
# build stage-2 items: top-K2 in-window (gap<=0.07) by stage-1 score; fall back to top-K2 overall
excl = {k: set(TR[TR.fold != k].target_vessel_id) for k in range(5)}
items = []
for j, s in s1.items():
    Xj = X[gstart[j]:gstart[j + 1]]; yj = y[gstart[j]:gstart[j + 1]]
    gap = np.nan_to_num(Xj[:, GAP], nan=9.0); inw = gap <= 0.07
    order = np.argsort(-np.where(inw, s, s - 1e6)) if inw.any() else np.argsort(-s)
    top = order[:K2]
    items.append((j, meta[j][1], meta[j][3], top, s[top], s2_feats(s[top], Xj[top]), yj[top]))
print('stage2 items', len(items), 'target in top-K2', np.mean([it[6].any() for it in items]).round(4), flush=True)
rows = []
for k in range(5):
    trn = [it for it in items if it[1] != k]
    m2 = lgb.train(p2, lgb.Dataset(np.concatenate([it[5] for it in trn]), np.concatenate([it[6] for it in trn]), group=[len(it[6]) for it in trn]), 300)
    for it in [it for it in items if it[1] == k]:
        j, _, tag, top, s, Fx, yy = it
        s2 = m2.predict(Fx, num_threads=2)
        rows.append(dict(tag=tag, s1=bool(yy[0]), s2=bool(yy[int(np.argmax(s2))]), s12=bool(yy[int(np.argmax(s2 + 0.5 * (s - s[0])))])))
    print('stage2 fold', k, flush=True)
d = pd.DataFrame(rows); d['clean'] = d.tag.isin(['orig', 'synko', 'synko_dup'])
print('hangul with-ton: stage1', d.s1.mean().round(4), 'stage2', d.s2.mean().round(4), 'blend', d.s12.mean().round(4))
print('H (0.5 clean + 0.5 damaged): s1', (0.5 * d[d.clean].s1.mean() + 0.5 * d[~d.clean].s1.mean()).round(4), 's2', (0.5 * d[d.clean].s2.mean() + 0.5 * d[~d.clean].s2.mean()).round(4), 'blend', (0.5 * d[d.clean].s12.mean() + 0.5 * d[~d.clean].s12.mean()).round(4))
print(d.groupby('tag')[['s1', 's2', 's12']].mean().round(3))
