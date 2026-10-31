import numpy as np, pickle, lightgbm as lgb, json
import pipe7
FN = pipe7.FEATURES7
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in FN if f not in drop]; cols = [FN.index(f) for f in fn]
KEEP = {'orig', 'mix', 'drop_ton', 'dup', 'synko', 'synko_dup'}
feats = [it for f in ['feats_v7_a.pkl', 'feats_v7_b.pkl'] for it in pickle.load(open(f, 'rb')) if it[4] in KEEP]
X = np.concatenate([it[6][:, cols] for it in feats]); y = np.concatenate([it[7] for it in feats]); g = [len(it[7]) for it in feats]
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=1, seed=42)
m = lgb.train(params, lgb.Dataset(X, y, group=g), 500)
m.save_model('ranker_v16b.txt'); json.dump(fn, open('ranker_v16b.features.json', 'w')); print('saved', X.shape, len(feats))
