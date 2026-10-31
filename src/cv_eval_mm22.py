"""[v35: pipe21 features] Materialise features as memmap (+cand ids), fit final ranker(s), or run exclusion-aware OOF vs a baseline's fold rankers.

usage: cv_eval_mm20.py <feat files> <tag> final <out ranker> [variant] [seed]
       cv_eval_mm20.py <feat files> <tag> cv <base_prefix e.g. v31_f> <folds e.g. 0,1,3,4> [variant]

variant (which items train the ranker; evaluation always uses every held-out item):
  A  the v33 recipe (extra 'x*' items left out)
  B  A without hangul items that lack tonnage, plus the tonnage-keeping hangul damage extras (xabbr*, xprefix*)
  C  B plus the extra renamed queries (xren)
  D  A plus the extra renamed queries (xren)
  E  C with the renamed queries (renamed, xren) weighted x3
  H<x> hangul items of variant <x> only (a hangul-specialised ranker)
a trailing 'm' (e.g. Cm): borrowed-name features masked on half of the renamed queries (see mask_tn)
a trailing '0' (e.g. A0) trains without the borrowed-name features (q_train_*).
"""
import pandas as pd, numpy as np, pickle, sys, gc, json, os, lightgbm as lgb
import pipe22 as P
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
        if mode == 'matdel':        # free the disk as we go (the memmap holds everything)
            X.flush(); os.remove(f)
    X.flush(); del X; np.save(tag + '_y.npy', y); pickle.dump(metas, open(tag + '_meta.pkl', 'wb')); pickle.dump(cids, open(tag + '_cids.pkl', 'wb'))
    print('materialised', n, flush=True)
if mode in ('mat', 'matdel'):
    sys.exit(0)
y = np.load(tag + '_y.npy'); meta = pickle.load(open(tag + '_meta.pkl', 'rb')); cids = pickle.load(open(tag + '_cids.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)])
ifold = np.array([m[1] for m in meta]); fold = np.repeat(ifold, g)


def use_item(m, variant):
    if 'S' in variant:                     # S: leave out the synthetic catalog-name hangul queries (synko*)
        if m[3].startswith('synko'):
            return False
        variant = variant.replace('S', '')
    variant = variant.rstrip('m').rstrip('0npv')
    t, hang, tmiss = m[3], m[2], m[4]
    if variant.startswith('H'):          # hangul-only ranker on the items of variant H<x>
        return hang and use_item(m, variant[1:] or 'A')
    if variant == 'A':
        return not t.startswith('x')
    if variant == 'D':
        return (not t.startswith('x')) or t == 'xren'
    if hang and tmiss:
        return False
    if variant == 'B':
        return t != 'xren'
    if variant in ('C', 'E'):
        return True
    raise ValueError(variant)


def item_weight(m, variant):
    """E = C with the renamed queries (renamed, xren) counted three times."""
    return 3.0 if variant.rstrip('m').rstrip('0npv').endswith('E') and m[3] in ('renamed', 'xren') else 1.0


def mask_tn(Xa, item_mask, variant, seed=7):
    """variant ending in 'm': for half of the renamed queries, overwrite the borrowed-name features with the values
    of a random clean query of the same script, so renamed handling is also learned without that signal."""
    if not variant.endswith('m'):
        return
    rng = np.random.default_rng(seed)
    sel = np.flatnonzero(item_mask); gg = g[sel]; st = np.concatenate([[0], np.cumsum(gg)])[:-1]
    fc = feat_cols(variant)
    tn = [fc.index(fn.index(f)) for f in ('q_train_exact', 'q_train_near', 'q_train_near_tok') if f in fn and fn.index(f) in fc]
    if not tn:
        return
    pool = {h: Xa[st[[j for j, i in enumerate(sel) if meta[i][3] == 'orig' and meta[i][2] == h]]][:, tn] for h in (True, False)}
    n = 0
    for j, i in enumerate(sel):
        if meta[i][3] in ('renamed', 'xren') and rng.random() < 0.5:
            v = pool[meta[i][2]][rng.integers(len(pool[meta[i][2]]))]
            Xa[st[j]:st[j] + gg[j], tn] = v; n += 1
    print('masked borrowed-name features on', n, 'renamed queries', flush=True)


def params(seed=42):
    return dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=seed)


TN = set(); TPF = set(P.TL_FEATURES)   # v37: suffix '0' = without the tonnage-likelihood / unit features (= v33)


import pipe17 as _P17
N31 = len([f for f in _P17.FEATURES7 if f not in drop])   # the v31 feature set is a prefix of every later one


def feat_cols(variant):
    """suffix '0': neither borrowed-name (TN) nor train-presence (TP) features; 'n': TN only; 'p': TP only;
    'v': the v31 columns only (no IDF / TN / TP)."""
    if variant.rstrip('m').endswith('v'):
        return list(range(N31))
    v = variant.rstrip('m')
    drop_ = TN | TPF if v.endswith('0') else TPF if v.endswith('n') else TN if v.endswith('p') else set()
    return [i for i, f in enumerate(fn) if f not in drop_]


def fit(item_mask, seed=42, variant='A'):
    """Copies the selected rows out of the memmap (then drops the mapping), bins them, frees the raw copy, trains:
    peak memory is one raw copy plus LightGBM's binned data (the bash cgroup is limited to ~6 GB)."""
    rows = np.repeat(item_mask, g); fc = feat_cols(variant)
    Xm = np.load(tag + '_X.npy', mmap_mode='r')
    if fc == list(range(len(fc))):
        Xa = np.ascontiguousarray(Xm[rows, :len(fc)])
    else:                                   # column subset: copy in chunks so only one copy is ever held
        Xa = np.empty((int(rows.sum()), len(fc)), np.float32); pos = 0; CH = 500000
        for a in range(0, len(rows), CH):
            r = rows[a:a + CH]; k = int(r.sum())
            if k:
                Xa[pos:pos + k] = Xm[a:a + CH][r][:, fc]; pos += k
    del Xm; gc.collect()
    ya = y[rows]; ga = g[item_mask]
    mask_tn(Xa, item_mask, variant)
    wi = np.array([item_weight(m_, variant) for m_ in meta])[item_mask]
    w = np.repeat(wi, ga) if (wi != 1).any() else None
    p = params(seed)
    ds = lgb.Dataset(Xa, ya, group=ga, weight=w, params=p, free_raw_data=True)
    ds.construct()
    del Xa; gc.collect()
    m = lgb.train(p, ds, 500)
    del ds; gc.collect()
    return m


if mode == 'final':
    out = sys.argv[4]; variant = sys.argv[5] if len(sys.argv) > 5 else 'A'; seed = int(sys.argv[6]) if len(sys.argv) > 6 else 42
    mask = np.array([use_item(m, variant) for m in meta])
    m = fit(mask, seed, variant)
    m.save_model(out); json.dump([fn[i] for i in feat_cols(variant)], open(out + '.features.json', 'w')); print('saved', out, variant, int(mask.sum()), 'items', flush=True)
else:
    base = sys.argv[4]; folds = [int(x) for x in sys.argv[5].split(',')]; variant = sys.argv[6] if len(sys.argv) > 6 else 'A'; res = []
    usable = np.array([use_item(m, variant) for m in meta])
    for k in folds:
        name = f'{tag}{variant}_f{k}.txt'
        if os.path.exists(name):
            m = lgb.Booster(model_file=name)
        else:
            m = fit(usable & (ifold != k), 42, variant); m.save_model(name); gc.collect()
        fc = feat_cols(variant)
        mb = lgb.Booster(model_file=f'{base}{k}.txt') if os.path.exists(f'{base}{k}.txt') else None
        X = np.load(tag + '_X.npy', mmap_mode='r')
        excl = set(TR[TR.fold != k].target_vessel_id)
        for j in np.flatnonzero(ifold == k):
            Xj = np.ascontiguousarray(X[gs[j]:gs[j + 1]]); yj = y[gs[j]:gs[j + 1]]
            ok = np.array([c not in excl for c in cids[j]])
            r = dict(sid=meta[j][0], fold=k, hang=meta[j][2], tag=meta[j][3], ton_missing=meta[j][4], recall=bool(yj.any()))
            for nm, mm in (('new', m), ('base', mb)):
                if mm is None: continue
                s = mm.predict(Xj[:, fc] if nm == 'new' else Xj[:, :mm.num_feature()], num_threads=2)
                r[nm] = bool(yj[int(np.argmax(s))]); r[nm + '_ex'] = bool(yj[int(np.argmax(np.where(ok, s, s - 1e6)))])
            res.append(r)
        del m, X; gc.collect(); print('fold', k, flush=True)
        pd.DataFrame(res).to_csv(f'cv_{tag}{variant}.oof.csv', index=False)
