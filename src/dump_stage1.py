"""Stage-1 (v6) OOF top-K candidates for real hangul items (+ regenerated query names)."""
import pandas as pd, numpy as np, pickle, random, lightgbm as lgb
import vm, pipe6, aug
from synth import Synth
from harness import TR, CAT
K = 6
feats = pickle.load(open('feats_v6_a.pkl', 'rb')) + pickle.load(open('feats_v6_b.pkl', 'rb'))
FN = pipe6.FEATURES6
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in FN if f not in drop]; cols = [FN.index(f) for f in fn]
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
KEEP = {'orig', 'drop_ton', 'mix', 'abbr', 'prefix', 'trunc'}
# regenerate perturbed names (same rng protocol as build_feats6)
names = {}
for k in range(5):
    rng = random.Random(3000 + k)
    trn = TR[TR.fold != k]
    syn = Synth(trn, CAT)
    for row in TR[TR.fold == k].to_dict('records'):
        hang = vm.is_hangul(row['vessel_name'])
        v = [('orig', row)]
        def named(tag, newname, drop_=False):
            if newname:
                r2 = dict(row); r2['vessel_name'] = newname
                if drop_:
                    r2 = aug.drop_ton(r2)
                v.append((tag, r2))
        if hang:
            v.append(('drop_ton', aug.perturb(row, rng, 'drop_ton')))
            v.append(('mix', aug.perturb(row, rng, 'mix')))
            named('abbr', aug.abbreviate(row['vessel_name'], rng), rng.random() < 0.4)
            if rng.random() < 0.5:
                named('prefix', syn.prefix_swap(row['vessel_name'], rng), rng.random() < 0.2)
            if rng.random() < 0.15:
                named('renamed', syn.renamed(row, rng))
            if rng.random() < 0.3:
                named('trunc', syn.truncate(row['vessel_name'], rng), rng.random() < 0.2)
        else:
            if rng.random() < 0.25:
                v.append(('mix', aug.perturb(row, rng, 'mix')))
            if rng.random() < 0.08:
                named('abbr', aug.abbreviate(row['vessel_name'], rng))
            if rng.random() < 0.06:
                named('prefix', syn.prefix_swap(row['vessel_name'], rng))
            if rng.random() < 0.03:
                named('renamed', syn.renamed(row, rng))
            if rng.random() < 0.04:
                named('trunc', syn.truncate(row['vessel_name'], rng))
        for tag, r in v:
            names[(row['sample_id'], tag)] = r
out = []
for k in range(5):
    trn = [it for it in feats if it[1] != k]
    val = [it for it in feats if it[1] == k and it[3] and it[4] in KEEP]
    X = np.concatenate([it[6][:, cols] for it in trn]); y = np.concatenate([it[7] for it in trn]); g = [len(it[7]) for it in trn]
    m = lgb.train(params, lgb.Dataset(X, y, group=g), num_boost_round=500)
    m.save_model(f'ranker_v6_fold{k}.txt')
    for it in val:
        s = m.predict(it[6][:, cols], num_threads=2)
        order = np.argsort(-s)[:K]
        r = names[(it[0], it[4])]
        out.append(dict(sid=it[0], fold=k, tag=it[4], target=it[2], qname=r['vessel_name'], row=r,
                        cands=it[5][order], s1=s[order], feats=it[6][order], y=it[7][order], hit=bool(it[7][order].any()), hit_all=bool(it[7].any())))
    print('fold', k, 'items', len(val), flush=True)
pickle.dump(out, open('stage1_hangul.pkl', 'wb'))
d = pd.DataFrame([(o['tag'], o['y'][0] == 1, o['hit'], o['hit_all']) for o in out], columns=['tag', 'top1', 'inK', 'inAll'])
print(d.groupby('tag').mean().round(4)); print('items', len(out))
