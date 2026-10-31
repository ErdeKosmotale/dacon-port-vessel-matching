"""Catalog-only prior: P(vessel appears in Korean port-call records | catalog attributes).
Every catalog vessel gets an out-of-fold prediction (it never scores itself)."""
import numpy as np, pandas as pd, lightgbm as lgb, zlib
import vm

def cat_matrix(cat):
    X = pd.DataFrame(index=cat.index)
    for c in ['vessel_type', 'flag_name', 'unit']:
        X[c] = pd.Categorical(cat[c].astype(str))
    for c in ['size', 'deadweight_tonnage', 'gross_tonnage']:
        X[c] = pd.to_numeric(cat[c], errors='coerce')
    X['ratio'] = X.deadweight_tonnage / X.gross_tonnage
    nm = cat.vessel_name.astype(str)
    X['nlen'] = nm.str.len(); X['nw'] = nm.str.split().str.len()
    X['hasnum'] = nm.str.contains(r'\d').astype(int)
    X['upper'] = nm.map(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
    first = nm.map(lambda n: (vm.en_split(n)[0] or [''])[0])
    X['first_cnt'] = first.map(first.value_counts()).astype(float)
    return X

PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=63, verbose=-1, min_data_in_leaf=50, seed=1, num_threads=2, deterministic=True, force_row_wise=True)

def fit_prior(cat, pos_ids, fold_of, n_rounds=400):
    """fold_of: array of fold ids per catalog row. Returns OOF logit per catalog row."""
    X = cat_matrix(cat); y = cat.vessel_id.isin(pos_ids).astype(int).values
    out = np.zeros(len(cat))
    for k in np.unique(fold_of):
        tr = fold_of != k
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), n_rounds)
        p = np.clip(m.predict(X[~tr]), 1e-6, 1 - 1e-6)
        out[~tr] = np.log(p / (1 - p))
    return out

def crc_fold(ids, n=5):
    return np.array([zlib.crc32(v.encode()) % n for v in ids])
