"""Stage-2 reranker over stage-1 top-K with LLM features; OOF evaluation + final model."""
import pickle, sys, numpy as np, pandas as pd, lightgbm as lgb, json
import pipe6
from harness import CAT
FN = pipe6.FEATURES6
KEYS = ['gap_min', 'exact_igt', 'exact_gt', 'q_ton_missing', 'type_prob', 'flag_prob', 't_pmi_rel', 't_loc_rel', 'num_mismatch', 'n_name_ties', 'ret_rank', 'name_rank', 'struct_rank', 't_red_rel']
S2_FEATURES = ['s1_rel', 's1_rank', 's1_margin', 'llm_rel', 'llm_rank', 'llm_margin', 'llm_lp', 'llm_lp_per_char', 'n_cands'] + KEYS

def build(o, cache, cat):
    s1 = np.asarray(o['s1']); n = len(s1)
    lp = np.array([cache.get((cat[v], o['qname']), np.nan) for v in o['cands']])
    if np.isnan(lp).all():
        return None
    lpf = np.where(np.isnan(lp), -1e9, lp)
    srt = np.sort(lpf)[::-1]
    llm_rank = (lpf[None, :] > lpf[:, None]).sum(axis=1)
    s1srt = np.sort(s1)[::-1]
    rows = []
    for i in range(n):
        rows.append([s1[i] - s1[0], i, s1[i] - (s1srt[1] if n > 1 else s1[i]),
                     lpf[i] - srt[0], llm_rank[i], lpf[i] - (srt[1] if n > 1 else lpf[i]), lp[i],
                     lp[i] / max(len(o['qname'].replace(' ', '')), 1), n] +
                    [float(o['feats'][i, FN.index(k)]) for k in KEYS])
    return np.asarray(rows, np.float32)

if __name__ == '__main__':
    items = pickle.load(open('stage1_hangul.pkl', 'rb'))
    cache = pickle.load(open('llm_cache.pkl', 'rb'))
    cat = CAT.set_index('vessel_id').vessel_name
    data = []
    for o in items:
        X = build(o, cache, cat)
        if X is not None:
            data.append((o, X))
    print('items with llm feats', len(data), 'of', len(items))
    params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=15,
                  min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2, seed=1)
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    res = []
    for k in range(5):
        trn = [(o, X) for o, X in data if o['fold'] != k]; val = [(o, X) for o, X in data if o['fold'] == k]
        Xt = np.concatenate([X for _, X in trn]); yt = np.concatenate([o['y'] for o, _ in trn]); g = [len(o['y']) for o, _ in trn]
        m = lgb.train(params, lgb.Dataset(Xt, yt, group=g), num_boost_round=rounds)
        for o, X in val:
            s = m.predict(X, num_threads=2)
            j = int(np.argmax(s))
            res.append((o['tag'], bool(o['feats'][0, FN.index('q_ton_missing')]), o['y'][0] == 1, o['y'][j] == 1, o['y'][int(np.argmax(X[:, 3]))] == 1))
    d = pd.DataFrame(res, columns=['tag', 'ton_missing', 'stage1', 'stage2', 'llm_only'])
    print(d.groupby(['tag', 'ton_missing']).agg(n=('stage1', 'size'), stage1=('stage1', 'mean'), stage2=('stage2', 'mean'), llm_only=('llm_only', 'mean')).round(4).to_string())
    print('ALL', d[['stage1', 'stage2', 'llm_only']].mean().round(4).to_dict())
    if len(sys.argv) > 2:
        Xa = np.concatenate([X for _, X in data]); ya = np.concatenate([o['y'] for o, _ in data]); ga = [len(o['y']) for o, _ in data]
        m = lgb.train(params, lgb.Dataset(Xa, ya, group=ga), num_boost_round=rounds)
        m.save_model(sys.argv[2]); json.dump(S2_FEATURES, open(sys.argv[2] + '.features.json', 'w')); print('saved', sys.argv[2])
