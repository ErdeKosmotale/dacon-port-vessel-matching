import pandas as pd, numpy as np, pickle, sys, lightgbm as lgb, json
import pipe2
feats = pickle.load(open(sys.argv[1], 'rb'))
FN = pipe2.FEATURES2
use_aug = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 500
drop = set(sys.argv[4].split(',')) if len(sys.argv) > 4 and sys.argv[4] else set()
fn = [f for f in FN if f not in drop]
cols = [FN.index(f) for f in fn]
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
res = []
imps = np.zeros(len(fn))
for k in range(5):
    trn = [it for it in feats if it[1] != k and (use_aug or it[4] == 'orig')]
    val = [it for it in feats if it[1] == k]
    X = np.concatenate([it[6][:, cols] for it in trn]); y = np.concatenate([it[7] for it in trn])
    g = [len(it[7]) for it in trn]
    m = lgb.train(params, lgb.Dataset(X, y, group=g), num_boost_round=rounds)
    imps += m.feature_importance('gain')
    for it in val:
        s = m.predict(it[6][:, cols], num_threads=2)
        res.append((it[0], it[3], it[4], it[5][int(np.argmax(s))] == it[2], it[7].any()))
r = pd.DataFrame(res, columns=['sid', 'hang', 'tag', 'ok', 'recall'])
o = r[r.tag == 'orig']
h, e = o[o.hang].ok.mean(), o[~o.hang].ok.mean()
print(f'ORIG  hang {h:.4f} en {e:.4f} weighted {0.7*h+0.3*e:.4f} | recall hang {o[o.hang].recall.mean():.4f} en {o[~o.hang].recall.mean():.4f}')
for tag in ['drop_ton', 'mix']:
    for hg in [True, False]:
        x = r[(r.tag == tag) & (r.hang == hg)]
        if len(x):
            print(f'{tag:8s} {"hang" if hg else "en  "} n={len(x)} acc {x.ok.mean():.4f} recall {x.recall.mean():.4f}')
print(pd.Series(imps, fn).sort_values(ascending=False).head(20).round(0).to_string())
r.to_csv(sys.argv[1] + f'.oof{use_aug}.csv', index=False)
