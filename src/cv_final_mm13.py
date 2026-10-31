"""Memory-mapped CV + final fit for large feature sets. Pass 1 writes X.npy/y.npy/meta; pass 2 trains from the memmap."""
import pandas as pd, numpy as np, pickle, sys, gc, json, os, lightgbm as lgb
import pipe13 as P
FN = P.FEATURES7; drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in FN if f not in drop]; cols = [FN.index(f) for f in fn]; tm = FN.index('q_ton_missing')
files = sys.argv[1].split(','); out = sys.argv[2]; mode = sys.argv[3]; tag = sys.argv[4]
if not os.path.exists(tag + '_X.npy'):
    n = 0; metas = []
    for f in files:
        items = pickle.load(open(f, 'rb'))
        for it in items: metas.append((it[0], it[1], it[3], it[4], bool(it[6][0, tm]), len(it[7]))); n += len(it[7])
        del items; gc.collect()
    X = np.lib.format.open_memmap(tag + '_X.npy', mode='w+', dtype=np.float32, shape=(n, len(cols)))
    y = np.zeros(n, np.int8); pos = 0
    for f in files:
        items = pickle.load(open(f, 'rb'))
        for it in items:
            k = len(it[7]); X[pos:pos + k] = it[6][:, cols]; y[pos:pos + k] = it[7]; pos += k
        del items; gc.collect()
    X.flush(); del X; np.save(tag + '_y.npy', y); pickle.dump(metas, open(tag + '_meta.pkl', 'wb')); print('materialised', n, flush=True)
X = np.load(tag + '_X.npy', mmap_mode='r'); y = np.load(tag + '_y.npy'); meta = pickle.load(open(tag + '_meta.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gstart = np.concatenate([[0], np.cumsum(g)]); fold = np.repeat([m[1] for m in meta], g)
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
if mode == 'cv':
    res = []
    for k in range(5):
        tr = fold != k; gi = np.array([m[1] != k for m in meta])
        Xt = np.ascontiguousarray(X[tr]); ds = lgb.Dataset(Xt, y[tr], group=g[gi], free_raw_data=True)
        m = lgb.train(params, ds, 500); del Xt, ds; gc.collect()
        for j in np.flatnonzero(~gi):
            s = m.predict(np.ascontiguousarray(X[gstart[j]:gstart[j + 1]]), num_threads=2)
            res.append((meta[j][0], meta[j][2], meta[j][3], bool(y[gstart[j] + int(np.argmax(s))]), meta[j][4]))
        del m; gc.collect(); print('fold', k, flush=True)
    pd.DataFrame(res, columns=['sid', 'hang', 'tag', 'ok', 'ton_missing']).to_csv(out + '.oof.csv', index=False)
else:
    Xa = np.ascontiguousarray(X); m = lgb.train(params, lgb.Dataset(Xa, y, group=g, free_raw_data=True), 500)
    m.save_model(out); json.dump(fn, open(out + '.features.json', 'w')); print('saved', out, Xa.shape)
