"""v26 = v18 ranker features + catalog prior columns (per-fold OOF prior from prior_cv.npy); final fit."""
import numpy as np, pickle, json, gc, lightgbm as lgb
import pipe11 as P
from harness import CAT
FN = [f for f in P.FEATURES7 if f not in {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}]
X = np.load('v18_X.npy', mmap_mode='r'); y = np.load('v18_y.npy'); meta = pickle.load(open('v18_meta.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)])
cids = [it[5] for f in ['feats_v12_a.pkl', 'feats_v12_b.pkl'] for it in pickle.load(open(f, 'rb'))]
PRI = np.load('prior_cv.npy'); id2i = {v: i for i, v in enumerate(CAT.vessel_id)}
P3 = np.zeros((len(y), 3), np.float32)
for j, m in enumerate(meta):
    p = PRI[m[1]][[id2i[c] for c in cids[j]]]
    P3[gs[j]:gs[j + 1], 0] = p; P3[gs[j]:gs[j + 1], 1] = p - p.max(); P3[gs[j]:gs[j + 1], 2] = (p[None, :] > p[:, None]).sum(1)
del cids; gc.collect()
Xa = np.hstack([np.ascontiguousarray(X), P3]); del P3; gc.collect()
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
m = lgb.train(params, lgb.Dataset(Xa, y, group=g, free_raw_data=True), 500)
m.save_model('ranker_v26.txt'); json.dump(FN + ['prior', 'prior_rel', 'prior_rank'], open('ranker_v26.txt.features.json', 'w'))
print('saved', Xa.shape)
