import numpy as np, pickle, sys, lightgbm as lgb, json
import pipe3
feats = []
for f in sys.argv[1].split(','):
    feats += pickle.load(open(f, 'rb'))
out = sys.argv[2]; rounds = int(sys.argv[3])
FN = pipe3.FEATURES2
X = np.concatenate([it[6] for it in feats]); y = np.concatenate([it[7] for it in feats]); g = [len(it[7]) for it in feats]
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
m = lgb.train(params, lgb.Dataset(X, y, group=g), num_boost_round=rounds)
m.save_model(out); json.dump(FN, open(out + '.features.json', 'w'))
print('saved', out, X.shape)
