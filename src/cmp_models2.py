import pandas as pd, numpy as np, pickle, sys, time, lightgbm as lgb
from catboost import CatBoost, Pool
import pipe7
FN = pipe7.FEATURES7
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
cols = [FN.index(f) for f in FN if f not in drop]
feats = pickle.load(open('feats_v7_a.pkl', 'rb'))   # folds 0,1,2
K = 0
trn = [it for it in feats if it[1] != K]; val = [it for it in feats if it[1] == K]
X = np.concatenate([it[6][:, cols] for it in trn]); y = np.concatenate([it[7] for it in trn])
g = [len(it[7]) for it in trn]; gid = np.concatenate([np.full(n, i) for i, n in enumerate(g)])
def evaluate(name, pr):
    res = [(it[4], it[3], bool(it[6][0, FN.index('q_ton_missing')]), bool(it[7][int(np.argmax(pr(it[6][:, cols])))])) for it in val]
    d = pd.DataFrame(res, columns=['tag', 'hang', 'tm', 'ok']); d.to_pickle(f'cmp2_{name}.pkl')
    print(name, 'overall', round(d.ok.mean(), 4), flush=True)
t = time.time()
m = lgb.train(dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42),
              lgb.Dataset(X, y, group=g), 500)
evaluate('lgb_rank', lambda Z: m.predict(Z, num_threads=2)); print(round(time.time() - t), flush=True)
m2 = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8,
                    bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2, seed=42), lgb.Dataset(X, y), 600)
evaluate('lgb_bin', lambda Z: m2.predict(Z, num_threads=2)); print(round(time.time() - t), flush=True)
m3 = CatBoost(dict(loss_function='QuerySoftMax', iterations=800, learning_rate=0.1, depth=6, verbose=0, thread_count=2, random_seed=1))
m3.fit(Pool(X, y, group_id=gid))
evaluate('cat_qsm', lambda Z: m3.predict(Z)); print(round(time.time() - t), flush=True)
print('DONE')
