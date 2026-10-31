"""Fold models + perturbation stress tests on hangul / english validation rows."""
import pandas as pd, numpy as np, pickle, sys, random, lightgbm as lgb
from sklearn.model_selection import GroupKFold
import vm, pipe

TR = pd.read_csv('/home/claude/data/open/train.csv', encoding='utf-8-sig')
CAT = pd.read_csv('/home/claude/data/open/vessel_catalog.csv', encoding='utf-8-sig')
gkf = GroupKFold(n_splits=5)
FOLD = np.zeros(len(TR), int)
for k, (a, b) in enumerate(gkf.split(TR, groups=TR.target_vessel_id)):
    FOLD[b] = k
TR['fold'] = FOLD
DROP = {'train_name_hit', 'train_cs_hit'}
PARAMS = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)


def fold_models(feats, fn, rounds=400):
    cols = [pipe.FEATURES.index(f) for f in fn]
    models = []
    for k in range(5):
        trn = [it for it in feats if it[1] != k]
        X = np.concatenate([it[5][:, cols] for it in trn]); y = np.concatenate([it[6] for it in trn])
        g = [len(it[6]) for it in trn]
        models.append(lgb.train(PARAMS, lgb.Dataset(X, y, group=g), num_boost_round=rounds))
    return models


def evaluate(models, fn, ix, rows_by_fold, perturb=None, seed=0):
    cols = [pipe.FEATURES.index(f) for f in fn]
    rng = random.Random(seed)
    res = []
    for k in range(5):
        st = pipe.TrainStats(TR[TR.fold != k], CAT)
        M = pipe.Matcher(ix, st)
        for row in rows_by_fold[k]:
            r = dict(row)
            if perturb:
                r = perturb(r, rng)
            cands, X = M.features(r)
            s = models[k].predict(X[:, cols], num_threads=2)
            pred = ix.ids[cands[int(np.argmax(s))]]
            res.append((row['sample_id'], vm.is_hangul(row['vessel_name']), pred == row['target_vessel_id'],
                        row['target_vessel_id'] in set(ix.ids[cands])))
    return pd.DataFrame(res, columns=['sid', 'hang', 'ok', 'recall'])
