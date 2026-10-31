"""v44 rankers: hangul-only training data with the pipe23 (fuzzy) candidate lists and organizer-style syllable typos.

The lists are long (~1,300 candidates per item), so training rows are subsampled per item: the true vessel, the
first KEEP_HEAD candidates in retrieval order and KEEP_RAND random others (seeded). Features: the v33 set (94 cols).

usage: train_v44.py mat <tag> <fold pickles...>          -> <tag>_X.npy / _y.npy / _meta.pkl (pickles are kept)
       train_v44.py fit <out> <tags comma-separated> [seed] [holdout fold or -] [variant A|S]
         variant S leaves the orgtypo items out (sanity check: old recipe on the new candidate lists)
"""
import sys, os, gc, json, pickle, numpy as np, lightgbm as lgb
import pipe23 as P
FN = P.FEATURES7
V33 = json.load(open('ranker_v33.txt.features.json'))
cols = [FN.index(f) for f in V33]
KEEP_HEAD, KEEP_RAND = 120, 180
mode = sys.argv[1]
if mode == 'mat':
    tag = sys.argv[2]; files = sys.argv[3:]
    rng = np.random.default_rng(1234)
    tm = FN.index('q_ton_missing')
    Xs, ys, metas = [], [], []
    for f in files:
        items = pickle.load(open(f, 'rb'))
        for it in items:
            X, y = it[6], it[7]
            n = len(y)
            keep = np.zeros(n, bool); keep[:KEEP_HEAD] = True; keep |= y > 0
            rest = np.flatnonzero(~keep)
            if len(rest) > KEEP_RAND:
                keep[rng.choice(rest, KEEP_RAND, replace=False)] = True
            else:
                keep[rest] = True
            Xs.append(np.ascontiguousarray(X[keep][:, cols])); ys.append(y[keep])
            metas.append((it[0], it[1], it[3], it[4], bool(X[0, tm]), int(keep.sum()), it[2], n))
        del items; gc.collect()
        print('read', f, len(metas), 'items', flush=True)
    X = np.concatenate(Xs); y = np.concatenate(ys); del Xs, ys; gc.collect()
    np.save(f'{tag}_X.npy', X); np.save(f'{tag}_y.npy', y); pickle.dump(metas, open(f'{tag}_meta.pkl', 'wb'))
    print('materialised', tag, X.shape, 'items', len(metas), 'mean list', round(float(np.mean([m[7] for m in metas])), 1), '->', round(float(np.mean([m[5] for m in metas])), 1), flush=True)
    sys.exit(0)
out = sys.argv[2]; tags = sys.argv[3].split(',')
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
hold = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != '-' else None
variant = sys.argv[6] if len(sys.argv) > 6 else 'A'
Xs, ys, gs = [], [], []
nitems = 0
for tag in tags:
    X = np.load(f'{tag}_X.npy', mmap_mode='r'); y = np.load(f'{tag}_y.npy'); meta = pickle.load(open(f'{tag}_meta.pkl', 'rb'))
    g = np.array([m[5] for m in meta])
    item = np.ones(len(meta), bool)
    if hold is not None:
        item &= np.array([m[1] != hold for m in meta])
    if variant == 'S':
        item &= np.array([not m[3].startswith('orgtypo') for m in meta])
    rows = np.repeat(item, g)
    Xs.append(np.ascontiguousarray(X[rows])); ys.append(y[rows]); gs.append(g[item]); nitems += int(item.sum())
    del X; gc.collect()
Xa = np.concatenate(Xs); ya = np.concatenate(ys); ga = np.concatenate(gs); del Xs, ys; gc.collect()
p = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=seed)
ds = lgb.Dataset(Xa, ya, group=ga, params=p, free_raw_data=True); ds.construct(); del Xa; gc.collect()
m = lgb.train(p, ds, 500)
m.save_model(out); json.dump(V33, open(out + '.features.json', 'w'))
print('saved', out, 'items', nitems, 'rows', int(ga.sum()), 'variant', variant, 'holdout', hold, 'seed', seed, flush=True)
