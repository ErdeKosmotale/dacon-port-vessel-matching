import pandas as pd, numpy as np, time, sys, pickle
from sklearn.model_selection import GroupKFold
import vm, pipe
tr = pd.read_csv('/home/claude/data/open/train.csv', encoding='utf-8-sig')
cat = pd.read_csv('/home/claude/data/open/vessel_catalog.csv', encoding='utf-8-sig')
ix = pipe.CatalogIndex(cat)
gkf = GroupKFold(n_splits=5)
fold = np.zeros(len(tr), int)
for k,(a,b) in enumerate(gkf.split(tr, groups=tr.target_vessel_id)): fold[b]=k
tr['fold']=fold
out = []
t=time.time()
for k in range(5):
    st = pipe.TrainStats(tr[tr.fold!=k], cat)
    M = pipe.Matcher(ix, st)
    sub = tr[tr.fold==k]
    for row in sub.to_dict('records'):
        cands, X = M.features(row)
        y = (ix.ids[cands] == row['target_vessel_id']).astype(np.int8)
        out.append((row['sample_id'], k, row['target_vessel_id'], vm.is_hangul(row['vessel_name']), ix.ids[cands], X, y))
    print('fold', k, time.time()-t, flush=True)
pickle.dump(out, open(sys.argv[1], 'wb'))
