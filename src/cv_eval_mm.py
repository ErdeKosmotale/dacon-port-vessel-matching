"""Materialise features as memmap (+cand ids), fit final ranker, or run exclusion-aware OOF vs a baseline's fold rankers.
usage: cv_eval_mm.py <feat files> <tag> final <out ranker>
       cv_eval_mm.py <feat files> <tag> cv <base_prefix e.g. ranker_v18_f> <folds e.g. 0,1,3,4>"""
import pandas as pd, numpy as np, pickle, sys, gc, json, os, lightgbm as lgb
import pipe11 as P
from harness import TR
FN = P.FEATURES7; drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in FN if f not in drop]; cols = [FN.index(f) for f in fn]; tm = FN.index('q_ton_missing')
files = sys.argv[1].split(','); tag = sys.argv[2]; mode = sys.argv[3]
if not os.path.exists(tag + '_X.npy'):
    n = 0; metas = []; cids = []
    for f in files:
        items = pickle.load(open(f, 'rb'))
        for it in items: metas.append((it[0], it[1], it[3], it[4], bool(it[6][0, tm]), len(it[7]), it[2])); cids.append(it[5]); n += len(it[7])
        del items; gc.collect()
    X = np.lib.format.open_memmap(tag + '_X.npy', mode='w+', dtype=np.float32, shape=(n, len(cols))); y = np.zeros(n, np.int8); pos = 0
    for f in files:
        items = pickle.load(open(f, 'rb'))
        for it in items:
            k = len(it[7]); X[pos:pos + k] = it[6][:, cols]; y[pos:pos + k] = it[7]; pos += k
        del items; gc.collect()
    X.flush(); del X; np.save(tag + '_y.npy', y); pickle.dump(metas, open(tag + '_meta.pkl', 'wb')); pickle.dump(cids, open(tag + '_cids.pkl', 'wb'))
    print('materialised', n, flush=True)
X = np.load(tag + '_X.npy', mmap_mode='r'); y = np.load(tag + '_y.npy'); meta = pickle.load(open(tag + '_meta.pkl', 'rb')); cids = pickle.load(open(tag + '_cids.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)]); fold = np.repeat([m[1] for m in meta], g)
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
if mode == 'final':
    out = sys.argv[4]; Xa = np.ascontiguousarray(X)
    m = lgb.train(params, lgb.Dataset(Xa, y, group=g, free_raw_data=True), 500)
    m.save_model(out); json.dump(fn, open(out + '.features.json', 'w')); print('saved', out, Xa.shape, flush=True)
else:
    base = sys.argv[4]; folds = [int(x) for x in sys.argv[5].split(',')]; res = []
    for k in folds:
        tr = fold != k; gi = np.array([m_[1] != k for m_ in meta])
        if os.path.exists(f'{tag}_f{k}.txt'):
            m = lgb.Booster(model_file=f'{tag}_f{k}.txt')
        else:
            m = lgb.train(params, lgb.Dataset(np.ascontiguousarray(X[tr]), y[tr], group=g[gi], free_raw_data=True), 500)
            m.save_model(f'{tag}_f{k}.txt'); gc.collect()
        mb = lgb.Booster(model_file=f'{base}{k}.txt') if os.path.exists(f'{base}{k}.txt') else None
        excl = set(TR[TR.fold != k].target_vessel_id)
        for j in np.flatnonzero(~gi):
            Xj = np.ascontiguousarray(X[gs[j]:gs[j + 1]]); yj = y[gs[j]:gs[j + 1]]
            ok = np.array([c not in excl for c in cids[j]])
            r = dict(sid=meta[j][0], fold=k, hang=meta[j][2], tag=meta[j][3], ton_missing=meta[j][4])
            for nm, mm in (('new', m), ('base', mb)):
                if mm is None: continue
                s = mm.predict(Xj[:, :mm.num_feature()], num_threads=2)
                r[nm] = bool(yj[int(np.argmax(s))]); r[nm + '_ex'] = bool(yj[int(np.argmax(np.where(ok, s, s - 1e6)))])
            res.append(r)
        del m; gc.collect(); print('fold', k, flush=True)
        pd.DataFrame(res).to_csv(f'cv_{tag}.oof.csv', index=False)
