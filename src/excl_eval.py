import pandas as pd, numpy as np, pickle, lightgbm as lgb
import pipe3
from harness import TR
feats = pickle.load(open('feats_v31_a.pkl', 'rb')) + pickle.load(open('feats_v31_b.pkl', 'rb'))
FN = pipe3.FEATURES2
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
res = []
for k in range(5):
    trn = [it for it in feats if it[1] != k]
    val = [it for it in feats if it[1] == k]
    excl = set(TR[TR.fold != k].target_vessel_id)
    X = np.concatenate([it[6] for it in trn]); y = np.concatenate([it[7] for it in trn]); g = [len(it[7]) for it in trn]
    m = lgb.train(params, lgb.Dataset(X, y, group=g), num_boost_round=500)
    for it in val:
        s = m.predict(it[6], num_threads=2)
        order = np.argsort(-s)
        pred = it[5][order[0]]
        keep = [j for j in order if it[5][j] not in excl]
        pred_x = it[5][keep[0]] if keep else pred
        res.append((it[3], it[4], pred == it[2], pred in excl, pred_x == it[2]))
r = pd.DataFrame(res, columns=['hang', 'tag', 'ok', 'pred_in_train', 'ok_excl'])
print(r.groupby(['tag', 'hang']).agg(n=('ok', 'size'), acc=('ok', 'mean'), wrong_pred_in_train=('pred_in_train', lambda x: x[~r.loc[x.index, 'ok']].mean()), acc_excl=('ok_excl', 'mean')).round(4))
