import pandas as pd, numpy as np, pickle, sys, lightgbm as lgb
import pipe4
feats = []
for f in sys.argv[1].split(','):
    feats += pickle.load(open(f, 'rb'))
FN = pipe4.FEATURES4
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 500
excl_tags = set(sys.argv[3].split(',')) if len(sys.argv) > 3 and sys.argv[3] else set()
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
res = []; imps = np.zeros(len(FN))
for k in range(5):
    trn = [it for it in feats if it[1] != k and it[4] not in excl_tags]
    val = [it for it in feats if it[1] == k]
    X = np.concatenate([it[6] for it in trn]); y = np.concatenate([it[7] for it in trn]); g = [len(it[7]) for it in trn]
    m = lgb.train(params, lgb.Dataset(X, y, group=g), num_boost_round=rounds)
    imps += m.feature_importance('gain')
    for it in val:
        s = m.predict(it[6], num_threads=2)
        res.append((it[0], it[3], it[4], it[5][int(np.argmax(s))] == it[2], it[7].any(), bool(it[6][0, FN.index('q_ton_missing')])))
r = pd.DataFrame(res, columns=['sid', 'hang', 'tag', 'ok', 'recall', 'ton_missing'])
o = r[r.tag == 'orig']
h, e = o[o.hang].ok.mean(), o[~o.hang].ok.mean()
print(f'ORIG  hang {h:.4f} en {e:.4f} weighted {0.7*h+0.3*e:.4f}')
print(r[r.tag != 'orig'].groupby(['tag', 'hang', 'ton_missing']).agg(n=('ok', 'size'), acc=('ok', 'mean'), recall=('recall', 'mean')).round(4).to_string())
print(pd.Series(imps, FN).sort_values(ascending=False).head(20).round(0).to_string())
r.to_csv(sys.argv[1].split(',')[0] + '.oof.csv', index=False)
