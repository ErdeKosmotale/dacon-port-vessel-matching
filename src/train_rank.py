import pandas as pd, numpy as np, pickle, sys, lightgbm as lgb
import pipe
feats = pickle.load(open(sys.argv[1],'rb'))
DROP = {'train_name_hit','train_cs_hit'}
FN = [f for f in pipe.FEATURES if f not in DROP]
cols = [pipe.FEATURES.index(f) for f in FN]
hw = float(sys.argv[2]) if len(sys.argv)>2 else 1.0
def stack(items):
    X = np.concatenate([it[5][:,cols] for it in items]); y = np.concatenate([it[6] for it in items])
    g = [len(it[6]) for it in items]
    w = np.array([hw if it[3] else 1.0 for it in items])
    return X,y,g,w
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2)
res=[]; imps=np.zeros(len(FN))
for k in range(5):
    trn=[it for it in feats if it[1]!=k]; val=[it for it in feats if it[1]==k]
    X,y,g,w = stack(trn)
    ds = lgb.Dataset(X,y,group=g, weight=None)
    # lambdarank weights are per-group
    ds.set_weight(None)
    ds = lgb.Dataset(X,y,group=g)
    ds.set_group(g)
    gw = w
    ds = lgb.Dataset(X, y, group=g, weight=np.repeat(gw, g))
    m = lgb.train(params, ds, num_boost_round=int(sys.argv[3]) if len(sys.argv)>3 else 400)
    imps += m.feature_importance('gain')
    for it in val:
        s = m.predict(it[5][:,cols], num_threads=2)
        pred = it[4][int(np.argmax(s))]
        res.append((it[0], it[3], pred==it[2], it[6].any()))
r = pd.DataFrame(res, columns=['sid','hang','ok','recall'])
h = r[r.hang].ok.mean(); e = r[~r.hang].ok.mean()
print(f'hang {h:.4f} en {e:.4f} weighted {0.7*h+0.3*e:.4f} | recall hang {r[r.hang].recall.mean():.4f} en {r[~r.hang].recall.mean():.4f}')
print(pd.Series(imps, FN).sort_values(ascending=False).head(25).round(0))
r.to_csv(sys.argv[1]+'.oof.csv', index=False)
