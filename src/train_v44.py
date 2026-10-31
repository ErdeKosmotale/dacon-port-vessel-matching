"""Materialise the v44 fold pickles (hangul-only, pipe23 candidates) into a memmap and train rankers on the v33
feature set (94 columns, by name) with the v33 lambdarank recipe.

usage: train_v44.py mat                      -> v44_X.npy / v44_y.npy / v44_meta.pkl / v44_cids.pkl (deletes the pickles)
       train_v44.py fit <out> [seed] [holdout fold] [variant]
         variant A  : every item     |  variant S : without the orgtypo items (sanity: old recipe on new candidates)
"""
import sys, os, gc, json, pickle, numpy as np, lightgbm as lgb
import pipe23 as P
FN = P.FEATURES7
V33 = json.load(open('ranker_v33.txt.features.json'))
cols = [FN.index(f) for f in V33]
mode = sys.argv[1]
if mode == 'mat':
    files = [f'v44_f{k}.pkl' for k in range(5) if os.path.exists(f'v44_f{k}.pkl')]
    n = 0; metas = []; cids = []
    for f in files:
        items = pickle.load(open(f, 'rb'))
        tm = FN.index('q_ton_missing')
        for it in items:
            metas.append((it[0], it[1], it[3], it[4], bool(it[6][0, tm]), len(it[7]), it[2])); cids.append(it[5]); n += len(it[7])
        del items; gc.collect()
    X = np.lib.format.open_memmap('v44_X.npy', mode='w+', dtype=np.float32, shape=(n, len(cols))); y = np.zeros(n, np.int8); pos = 0
    for f in files:
        items = pickle.load(open(f, 'rb'))
        for it in items:
            k = len(it[7]); X[pos:pos + k] = it[6][:, cols]; y[pos:pos + k] = it[7]; pos += k
        del items; gc.collect(); X.flush(); os.remove(f)
    del X; np.save('v44_y.npy', y); pickle.dump(metas, open('v44_meta.pkl', 'wb')); pickle.dump(cids, open('v44_cids.pkl', 'wb'))
    print('materialised', n, 'rows from', files, flush=True)
    sys.exit(0)
out = sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
hold = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != '-' else None
variant = sys.argv[5] if len(sys.argv) > 5 else 'A'
y = np.load('v44_y.npy'); meta = pickle.load(open('v44_meta.pkl', 'rb'))
g = np.array([m[5] for m in meta])
item = np.ones(len(meta), bool)
if hold is not None:
    item &= np.array([m[1] != hold for m in meta])
if variant == 'S':
    item &= np.array([not m[3].startswith('orgtypo') for m in meta])
rows = np.repeat(item, g)
Xm = np.load('v44_X.npy', mmap_mode='r')
Xa = np.ascontiguousarray(Xm[rows]); del Xm; gc.collect()
p = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=seed)
ds = lgb.Dataset(Xa, y[rows], group=g[item], params=p, free_raw_data=True); ds.construct(); del Xa; gc.collect()
m = lgb.train(p, ds, 500)
m.save_model(out); json.dump(V33, open(out + '.features.json', 'w'))
print('saved', out, 'items', int(item.sum()), 'rows', int(rows.sum()), 'variant', variant, 'holdout', hold, flush=True)
