import numpy as np, pickle, lightgbm as lgb, json
import pipe7
from harness import CAT
FN = pipe7.FEATURES7
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in FN if f not in drop]; cols = [FN.index(f) for f in fn]
PRI = np.load('prior_cv.npy'); id2i = {v: i for i, v in enumerate(CAT.vessel_id)}
feats = pickle.load(open('feats_v7_a.pkl', 'rb')) + pickle.load(open('feats_v7_b.pkl', 'rb'))
def addp(it):
    p = PRI[it[1]][[id2i[c] for c in it[5]]]
    return np.column_stack([it[6][:, cols], p, p - p.max(), (p[None, :] > p[:, None]).sum(1)]).astype(np.float32)
X = np.concatenate([addp(it) for it in feats]); y = np.concatenate([it[7] for it in feats]); g = [len(it[7]) for it in feats]
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
m = lgb.train(params, lgb.Dataset(X, y, group=g), 500)
m.save_model('ranker_v14.txt'); json.dump(fn, open('ranker_v14.features.json', 'w'))
print('saved', X.shape)
