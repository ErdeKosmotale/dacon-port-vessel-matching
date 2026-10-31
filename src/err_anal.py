import pandas as pd, numpy as np, pickle, lightgbm as lgb
import pipe7
FN = pipe7.FEATURES7
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in FN if f not in drop]; cols = [FN.index(f) for f in fn]
feats = pickle.load(open('feats_v7_a.pkl', 'rb')) + pickle.load(open('feats_v7_b.pkl', 'rb'))
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
F = lambda X, j, f: float(X[j, FN.index(f)])
rows = []
for k in range(5):
    trn = [it for it in feats if it[1] != k]
    X = np.concatenate([it[6][:, cols] for it in trn]); y = np.concatenate([it[7] for it in trn]); g = [len(it[7]) for it in trn]
    m = lgb.train(params, lgb.Dataset(X, y, group=g), num_boost_round=500)
    for it in [it for it in feats if it[1] == k]:
        Xa = it[6]; s = m.predict(Xa[:, cols], num_threads=2); p = int(np.argmax(s)); ti = np.where(it[7])[0]
        r = dict(sid=it[0], fold=k, tag=it[4], hang=it[3], ok=bool(it[7][p]), recall=len(ti) > 0,
                 ton_missing=F(Xa, 0, 'q_ton_missing') == 1, n_cands=len(it[7]), pred=it[5][p], target=it[2])
        if len(ti) and not it[7][p]:
            t = ti[0]
            r.update(t_rank=int((s > s[t]).sum()),
                     same_name=F(Xa, p, 'exact_compact') == 1 and F(Xa, t, 'exact_compact') == 1,
                     name_gap_pred=F(Xa, p, 'name_gap_best'), name_gap_tgt=F(Xa, t, 'name_gap_best'),
                     gap_pred=F(Xa, p, 'gap_min'), gap_tgt=F(Xa, t, 'gap_min'),
                     type_pred=F(Xa, p, 'type_prob'), type_tgt=F(Xa, t, 'type_prob'),
                     tpmi_pred=F(Xa, p, 't_pmi_rel'), tpmi_tgt=F(Xa, t, 't_pmi_rel'),
                     tloc_pred=F(Xa, p, 't_loc_rel'), tloc_tgt=F(Xa, t, 't_loc_rel'),
                     num_mm_pred=F(Xa, p, 'num_mismatch'), num_mm_tgt=F(Xa, t, 'num_mismatch'))
        rows.append(r)
    print('fold', k, flush=True)
d = pd.DataFrame(rows); d.to_pickle('err_v10.pkl')
