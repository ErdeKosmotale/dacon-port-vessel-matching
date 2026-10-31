import pandas as pd, numpy as np, lightgbm as lgb, re
from sklearn.metrics import roc_auc_score
import vm
cat = pd.read_csv('/home/claude/data/open/vessel_catalog.csv', encoding='utf-8-sig')
tr = pd.read_csv('/home/claude/data/open/train.csv', encoding='utf-8-sig')
pos = set(tr.target_vessel_id)
cat['y'] = cat.vessel_id.isin(pos).astype(int)
print(cat.head(3).to_string()); print(cat.dtypes)
X = pd.DataFrame(index=cat.index)
for c in ['vessel_type', 'flag_name', 'unit']:
    X[c] = cat[c].astype('category')
for c in ['size', 'deadweight_tonnage', 'gross_tonnage']:
    X[c] = pd.to_numeric(cat[c], errors='coerce')
X['ratio'] = X.deadweight_tonnage / X.gross_tonnage
X['nlen'] = cat.vessel_name.astype(str).str.len()
X['nw'] = cat.vessel_name.astype(str).str.split().str.len()
X['hasnum'] = cat.vessel_name.astype(str).str.contains(r'\d').astype(int)
X['upper'] = cat.vessel_name.astype(str).map(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
X['idnum'] = cat.vessel_id.str.extract(r'(\d+)')[0].astype(int)
first = cat.vessel_name.astype(str).map(lambda n: (vm.en_split(n)[0] or [''])[0])
X['first_cnt'] = first.map(first.value_counts())
X['gt_round'] = (X.gross_tonnage % 100 == 0).astype(int)
rng = np.random.RandomState(0); fold = rng.randint(0, 5, len(cat))
oof = np.zeros(len(cat))
for k in range(5):
    m = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=63, verbose=-1, min_data_in_leaf=50), lgb.Dataset(X[fold != k], cat.y[fold != k]), 400)
    oof[fold == k] = m.predict(X[fold == k])
print('AUC all', roc_auc_score(cat.y, oof))
imp = pd.Series(m.feature_importance('gain'), X.columns).sort_values(ascending=False); print(imp.round(0))
for drop in [['idnum'], ['idnum', 'flag_name'], ['idnum', 'first_cnt']]:
    Xd = X.drop(columns=drop); o2 = np.zeros(len(cat))
    for k in range(5):
        mm = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=63, verbose=-1, min_data_in_leaf=50), lgb.Dataset(Xd[fold != k], cat.y[fold != k]), 400)
        o2[fold == k] = mm.predict(Xd[fold == k])
    print('AUC without', drop, roc_auc_score(cat.y, o2))
