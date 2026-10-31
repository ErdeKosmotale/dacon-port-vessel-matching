import pandas as pd, numpy as np, pickle, sys, lightgbm as lgb
from catboost import CatBoostRanker, Pool
import pipe7
FN = pipe7.FEATURES7
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
cols = [FN.index(f) for f in FN if f not in drop]
feats = pickle.load(open('feats_v7_a.pkl', 'rb')) + pickle.load(open('feats_v7_b.pkl', 'rb'))
which = sys.argv[1]
res = []
for k in range(5):
    trn = [it for it in feats if it[1] != k]; val = [it for it in feats if it[1] == k]
    X = np.concatenate([it[6][:, cols] for it in trn]); y = np.concatenate([it[7] for it in trn])
    gid = np.concatenate([np.full(len(it[7]), i) for i, it in enumerate(trn)])
    if which == 'cat':
        m = CatBoostRanker(loss_function='YetiRank', iterations=1500, learning_rate=0.08, depth=7, verbose=0, thread_count=2, random_seed=1)
        m.fit(Pool(X, y, group_id=gid))
        pr = lambda Z: m.predict(Z)
    elif which == 'lgb_bin':
        m = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8,
                           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2, seed=42), lgb.Dataset(X, y), 600)
        pr = lambda Z: m.predict(Z, num_threads=2)
    for it in val:
        s = pr(it[6][:, cols])
        res.append((it[4], it[3], bool(it[6][0, FN.index('q_ton_missing')]), bool(it[7][int(np.argmax(s))])))
    print(which, 'fold', k, flush=True)
d = pd.DataFrame(res, columns=['tag', 'hang', 'tm', 'ok'])
d.to_pickle(f'cmp_{which}.pkl')
print(d.groupby(['tag', 'hang', 'tm']).ok.mean().round(4).to_string())
