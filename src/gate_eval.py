import pandas as pd, numpy as np, pickle, lightgbm as lgb
import pipe7
FN = pipe7.FEATURES7
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
cols = [FN.index(f) for f in FN if f not in drop]
GAP, TM = FN.index('gap_min'), FN.index('q_ton_missing')
feats = pickle.load(open('feats_v7_a.pkl', 'rb')) + pickle.load(open('feats_v7_b.pkl', 'rb'))
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
TAUS = [None, 0.04, 0.055, 0.06, 0.07, 0.08, 0.10, 0.15]
rows = []
for k in range(5):
    trn = [it for it in feats if it[1] != k]
    X = np.concatenate([it[6][:, cols] for it in trn]); y = np.concatenate([it[7] for it in trn]); g = [len(it[7]) for it in trn]
    m = lgb.train(params, lgb.Dataset(X, y, group=g), 500)
    for it in [it for it in feats if it[1] == k]:
        if it[6][0, TM] == 1:
            continue
        s = m.predict(it[6][:, cols], num_threads=2)
        gap = np.nan_to_num(it[6][:, GAP], nan=9.0)
        r = dict(tag=it[4], hang=it[3], clean=it[4] in ('orig', 'synko'))
        for t in TAUS:
            if t is None:
                j = int(np.argmax(s))
            else:
                ok = gap <= t
                j = int(np.argmax(np.where(ok, s, -1e18))) if ok.any() else int(np.argmax(s))
            r[str(t)] = bool(it[7][j])
        rows.append(r)
    print('fold', k, flush=True)
d = pd.DataFrame(rows); d.to_pickle('gate_eval.pkl')
def tl(d):
    a = lambda h, c, col: d[(d.hang == h) & (d.clean == c)][col].mean()
    return {str(t): round(0.7 * (0.5 * a(True, True, str(t)) + 0.5 * a(True, False, str(t))) + 0.3 * (0.78 * a(False, True, str(t)) + 0.22 * a(False, False, str(t))), 4) for t in TAUS}
print('test-like weighted:', tl(d))
print(d.groupby(['tag', 'hang'])[[str(t) for t in TAUS]].mean().round(3).to_string())
