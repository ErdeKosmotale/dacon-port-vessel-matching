"""Train one extra full-data ranker on the v37 memmap (seed-3000 synthetic items, variant A, v33 feature set) with
a different objective / capacity, for ensemble diversity.

usage: train_div.py <config> <out ranker file> [held-out fold: train without that fold's items, for CV]
configs:
  xendcg  rank_xendcg objective (cross-entropy NDCG), 63 leaves, lr 0.05, 600 rounds
  bin     pointwise binary logloss, 63 leaves, lr 0.05, 600 rounds
  big     lambdarank, 255 leaves, min_data 50, lr 0.05, 600 rounds, feature_fraction 0.6, truncation 20
  enbig   'big' on english items only (a non-hangul specialist)
  habig   'big' on hangul items only (a hangul specialist)
Peak memory: one raw copy of the selected rows + LightGBM's binned data (the cgroup allows ~6 GB).
"""
import sys, gc, json, time, pickle, numpy as np, lightgbm as lgb
import pipe22 as P

cfg, out = sys.argv[1], sys.argv[2]
hold = int(sys.argv[3]) if len(sys.argv) > 3 else None
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in P.FEATURES7 if f not in drop]
TL = set(P.TL_FEATURES)
fc = [i for i, f in enumerate(fn) if f not in TL]                 # = the 94 v33 features
meta = pickle.load(open('v37_meta.pkl', 'rb')); y = np.load('v37_y.npy')
g = np.array([m[5] for m in meta])
item = np.array([not m[3].startswith('x') for m in meta])          # variant A
hang = np.array([bool(m[2]) for m in meta])
if cfg == 'enbig':
    item &= ~hang
elif cfg == 'habig':
    item &= hang
if hold is not None:
    item &= np.array([m[1] != hold for m in meta])
rows = np.repeat(item, g)
base = dict(metric='None', verbose=-1, num_threads=2, seed=42, bagging_freq=1, feature_fraction=0.8, bagging_fraction=0.8,
            min_data_in_leaf=20, num_leaves=63, learning_rate=0.05)
if cfg == 'xendcg':
    p = dict(base, objective='rank_xendcg'); rounds = 600
elif cfg == 'bin':
    p = dict(base, objective='binary'); rounds = 600
elif cfg in ('big', 'enbig', 'habig'):
    p = dict(base, objective='lambdarank', num_leaves=255, min_data_in_leaf=50, feature_fraction=0.6,
             bagging_fraction=0.7, lambdarank_truncation_level=20); rounds = 600
else:
    raise SystemExit('unknown config ' + cfg)
t0 = time.time()
Xm = np.load('v37_X.npy', mmap_mode='r')
Xa = np.ascontiguousarray(Xm[rows][:, fc]); del Xm; gc.collect()
ya = y[rows]; ga = g[item]
ds = lgb.Dataset(Xa, ya, group=None if p['objective'] == 'binary' else ga, params=p, free_raw_data=True)
ds.construct(); del Xa; gc.collect()
print(cfg, 'items', int(item.sum()), 'rows', int(rows.sum()), f'binned in {time.time() - t0:.0f}s', flush=True)
m = lgb.train(p, ds, rounds)
m.save_model(out)
json.dump([fn[i] for i in fc], open(out + '.features.json', 'w'))
print('saved', out, f'{time.time() - t0:.0f}s', flush=True)
