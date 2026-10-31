import pandas as pd, numpy as np, pickle, lightgbm as lgb, os
import pipe11 as P
from harness import TR
FN = [f for f in P.FEATURES7 if f not in {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}]; F = {f: i for i, f in enumerate(FN)}
X = np.load('v18_X.npy', mmap_mode='r'); y = np.load('v18_y.npy'); meta = pickle.load(open('v18_meta.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)])
cids = [it[5] for f in ['feats_v12_a.pkl', 'feats_v12_b.pkl'] for it in pickle.load(open(f, 'rb'))]
rows = []
for k in range(5):
    if not os.path.exists(f'ranker_v18_f{k}.txt'): continue
    m = lgb.Booster(model_file=f'ranker_v18_f{k}.txt'); excl = set(TR[TR.fold != k].target_vessel_id)
    for j in range(len(meta)):
        if meta[j][1] != k or meta[j][2]: continue   # english items of fold k
        Xj = np.ascontiguousarray(X[gs[j]:gs[j + 1]]); yj = y[gs[j]:gs[j + 1]]
        s = m.predict(Xj, num_threads=2); ok = np.array([c not in excl for c in cids[j]])
        rk = int(np.argmax(np.where(ok, s, -1e9)))
        ex = (Xj[:, F['exact_compact']] == 1) & ok
        fire = ex.sum() == 1 and not ex[rk]
        b = int(np.argmax(ex)) if ex.sum() == 1 else rk
        rows.append(dict(fold=k, tag=meta[j][3], ranker_ok=bool(yj[rk]), new_ok=bool(yj[b]) if fire else bool(yj[rk]), fire=fire))
    print('fold', k, flush=True)
d = pd.DataFrame(rows)
print('english items', len(d), '| ranker', d.ranker_ok.mean().round(4), '-> override', d.new_ok.mean().round(4), '| fires', int(d.fire.sum()),
      'right', int((d.fire & d.new_ok).sum()), 'wrong', int((d.fire & ~d.new_ok).sum()))
print(d.groupby('tag')[['ranker_ok', 'new_ok', 'fire']].agg(['mean']).round(3))
