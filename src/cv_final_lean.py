"""Memory-lean CV + final fit: load pickles once, keep only selected columns."""
import pandas as pd, numpy as np, pickle, sys, gc, json, lightgbm as lgb
import pipe11 as P
FN = P.FEATURES7; drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in FN if f not in drop]; cols = [FN.index(f) for f in fn]
tm = FN.index('q_ton_missing')
meta, Xs, ys = [], [], []
for f in sys.argv[1].split(','):
    for it in pickle.load(open(f, 'rb')):
        meta.append((it[0], it[1], it[3], it[4], bool(it[6][0, tm]))); Xs.append(it[6][:, cols].astype(np.float32)); ys.append(it[7])
    gc.collect()
X = np.concatenate(Xs); y = np.concatenate(ys); g = np.array([len(a) for a in ys]); del Xs, ys; gc.collect()
fold = np.repeat([m[1] for m in meta], g); gstart = np.concatenate([[0], np.cumsum(g)])
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
out = sys.argv[2]; res = []
HW = float(sys.argv[4]) if len(sys.argv) > 4 else None
hang_q = np.array([m[2] for m in meta], bool)
if HW is not None:
    # per-query weight so that total weight hangul : english = HW : (1-HW)
    nh, ne = hang_q.sum(), (~hang_q).sum()
    wq = np.where(hang_q, HW / nh, (1 - HW) / ne) * (nh + ne)
    W = np.repeat(wq, g).astype(np.float32); print('weights hangul', round(float(wq[hang_q][0]), 3), 'english', round(float(wq[~hang_q][0]), 3), flush=True)
else:
    W = None
if len(sys.argv) > 3 and sys.argv[3] == 'cv':
    for k in range(5):
        tr = fold != k; gi = np.array([m[1] != k for m in meta])
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr], group=g[gi], weight=None if W is None else W[tr]), 500)
        for j in np.flatnonzero(~gi):
            s = m.predict(X[gstart[j]:gstart[j + 1]], num_threads=2)
            res.append((meta[j][0], meta[j][2], meta[j][3], bool(y[gstart[j] + int(np.argmax(s))]), meta[j][4]))
        del m; gc.collect(); print('fold', k, flush=True)
    pd.DataFrame(res, columns=['sid', 'hang', 'tag', 'ok', 'ton_missing']).to_csv(out + '.oof.csv', index=False)
else:
    m = lgb.train(params, lgb.Dataset(X, y, group=g, weight=W), 500)
    m.save_model(out); json.dump(fn, open(out + '.features.json', 'w')); print('saved', out, X.shape)
