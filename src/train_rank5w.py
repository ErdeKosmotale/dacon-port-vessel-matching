import pandas as pd, numpy as np, pickle, sys, lightgbm as lgb
import pipe5
feats = []
for f in sys.argv[1].split(','):
    feats += pickle.load(open(f, 'rb'))
FN = pipe5.FEATURES5
rounds = int(sys.argv[2]); wsyn = float(sys.argv[3]); wreal_h = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
def w(it):
    if it[4].startswith('synko'): return wsyn
    if it[4] == 'dup': return wsyn
    if it[3]: return wreal_h
    return 1.0
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=10, verbose=-1, num_threads=1, seed=42)
res = []
for k in range(5):
    trn = [it for it in feats if it[1] != k]
    val = [it for it in feats if it[1] == k]
    X = np.concatenate([it[6] for it in trn]); y = np.concatenate([it[7] for it in trn]); g = [len(it[7]) for it in trn]
    wt = np.concatenate([np.full(len(it[7]), w(it)) for it in trn])
    m = lgb.train(params, lgb.Dataset(X, y, group=g, weight=wt), num_boost_round=rounds)
    for it in val:
        s = m.predict(it[6], num_threads=1)
        res.append((it[0], it[3], it[4], it[5][int(np.argmax(s))] == it[2], bool(it[6][0, FN.index('q_ton_missing')])))
r = pd.DataFrame(res, columns=['sid', 'hang', 'tag', 'ok', 'ton_missing'])
o = r[r.tag == 'orig']
print(f'rounds={rounds} wsyn={wsyn} wreal_h={wreal_h} | ORIG hang {o[o.hang].ok.mean():.4f} en {o[~o.hang].ok.mean():.4f}')
g = r[r.tag != 'orig'].groupby(['tag', 'hang', 'ton_missing']).ok.mean().round(4)
print(g.to_string())
