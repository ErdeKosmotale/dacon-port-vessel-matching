import pandas as pd, numpy as np, pickle, sys, lightgbm as lgb
import pipe6 as pipe5
feats = []
for f in sys.argv[1].split(','):
    feats += pickle.load(open(f, 'rb'))
FN = pipe5.FEATURES6
rounds = int(sys.argv[2]); drop = set(sys.argv[3].split(',')) if len(sys.argv) > 3 and sys.argv[3] else set()
fn = [f for f in FN if f not in drop]; cols = [FN.index(f) for f in fn]
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=10, verbose=-1, num_threads=1, seed=42)
res = []
for k in range(5):
    trn = [it for it in feats if it[1] != k]
    val = [it for it in feats if it[1] == k]
    X = np.concatenate([it[6][:, cols] for it in trn]); y = np.concatenate([it[7] for it in trn]); g = [len(it[7]) for it in trn]
    m = lgb.train(params, lgb.Dataset(X, y, group=g), num_boost_round=rounds)
    for it in val:
        s = m.predict(it[6][:, cols], num_threads=1)
        res.append((it[0], it[3], it[4], it[5][int(np.argmax(s))] == it[2], bool(it[6][0, FN.index('q_ton_missing')])))
r = pd.DataFrame(res, columns=['sid', 'hang', 'tag', 'ok', 'ton_missing'])
o = r[r.tag == 'orig']
print(f'drop={sorted(drop)} | ORIG hang {o[o.hang].ok.mean():.4f} en {o[~o.hang].ok.mean():.4f}')
print(r[r.tag != 'orig'].groupby(['tag', 'hang', 'ton_missing']).ok.mean().round(4).to_string())
r.to_csv('cv_v5d_' + ('all' if len(drop) > 2 else 'rank') + '.oof.csv', index=False)
