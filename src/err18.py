import pandas as pd, numpy as np, pickle, gc, lightgbm as lgb
import pipe11 as P
from harness import TR, CAT
FN = P.FEATURES7; drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
cols = [FN.index(f) for f in FN if f not in drop]; F = {f: i for i, f in enumerate(FN)}
feats = pickle.load(open('feats_v12_a.pkl', 'rb'))
k = 0
trn = [it for it in feats if it[1] != k]
X = np.concatenate([it[6][:, cols] for it in trn]); y = np.concatenate([it[7] for it in trn]); g = [len(it[7]) for it in trn]
del trn; gc.collect()
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
m = lgb.train(params, lgb.Dataset(X, y, group=g), 500); del X, y; gc.collect()
excl = set(TR[TR.fold != k].target_vessel_id); cname = dict(zip(CAT.vessel_id, CAT.vessel_name))
tr_name = dict(zip(TR.sample_id, TR.vessel_name))
rows = []
for it in feats:
    if it[1] != k or not it[3] or it[6][0, F['q_ton_missing']] == 1: continue
    s = m.predict(it[6][:, cols], num_threads=2)
    ok_ex = np.array([c not in excl for c in it[5]]); s2 = np.where(ok_ex, s, s - 1e6)
    o = np.argsort(-s2); p = o[0]; t = int(np.argmax(it[7])) if it[7].any() else -1
    X = it[6]
    r = dict(sid=it[0], tag=it[4], recall=t >= 0, ok=t >= 0 and p == t, pred=cname[it[5][p]], tgt=cname[it[2]], n=len(it[5]),
             t_rank=int((s2 > s2[t]).sum()) if t >= 0 else -1, margin=float(s2[p] - s2[t]) if t >= 0 else np.nan)
    for f in ['gap_min', 't_pmi_rel', 't_loc_rel', 't_red_rel', 'type_prob', 'flag_prob', 'sk_sim', 'vk_sim', 'num_mismatch', 'r_partial', 'in_win' if 'in_win' in F else 'ret_rank']:
        r[f + '_p'] = float(X[p, F[f]]); r[f + '_t'] = float(X[t, F[f]]) if t >= 0 else np.nan
    rows.append(r)
d = pd.DataFrame(rows); d.to_pickle('err18_f0.pkl')
print('hangul with-ton fold0 items', len(d), 'acc', d.ok.mean().round(3), 'recall', d.recall.mean().round(3))
print(d.groupby('tag').agg(n=('ok', 'size'), acc=('ok', 'mean'), recall=('recall', 'mean')).round(3))
w = d[~d.ok & d.recall]
print('rank errors', len(w), '| t_rank==1:', (w.t_rank == 1).mean().round(2), '| same exact gt:', ((w.gap_min_p == 0) & (w.gap_min_t == 0)).mean().round(2),
      '| pred closer ton:', (w.gap_min_p < w.gap_min_t).mean().round(2), '| tgt closer:', (w.gap_min_t < w.gap_min_p).mean().round(2))
print('tgt better pmi:', (w.t_pmi_rel_t > w.t_pmi_rel_p).mean().round(2), 'tgt better loc:', (w.t_loc_rel_t > w.t_loc_rel_p).mean().round(2),
      'tgt better type:', (w.type_prob_t > w.type_prob_p).mean().round(2), 'pred better type:', (w.type_prob_p > w.type_prob_t).mean().round(2),
      'tgt better flag:', (w.flag_prob_t > w.flag_prob_p).mean().round(2), 'pred better flag:', (w.flag_prob_p > w.flag_prob_t).mean().round(2))
print('margin quantiles', w.margin.quantile([.25, .5, .75]).round(2).tolist())
pd.set_option('display.width', 250)
print(w[['tag', 'tgt', 'pred', 'gap_min_t', 'gap_min_p', 't_pmi_rel_t', 't_pmi_rel_p', 't_loc_rel_t', 't_loc_rel_p', 'type_prob_t', 'type_prob_p', 'flag_prob_t', 'flag_prob_p', 'margin']].assign(q=[tr_name.get(s, '') for s in w.sid]).sample(min(40, len(w)), random_state=1).round(2).to_string())
