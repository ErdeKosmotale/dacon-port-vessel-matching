"""Is a catalog-side 'visits Korean ports' prior a new signal for name-less (renamed) matching?

Simulates renamed rows: the record name is ignored. For each train row of fold k: eligible catalog vessels within
10% of its gross / international tonnage (other folds' train vessels removed, as at test time), ranked by
  A  structure: 2*exact tonnage + P(catalog type | record type) + P(catalog flag | record flag) - 3*gap
  B  A + w * log prior, prior = P(vessel is a train vessel | catalog fields) from a LightGBM classifier trained
     with the other folds' train vessels as positives and everything else (fold k included, like test vessels)
     as negatives.
Prints top-1 accuracy and the true vessel's mean reciprocal rank, by script.
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb, unicodedata
from collections import defaultdict
sys.path.insert(0, 'submit_probeH/model')
import vm
from harness import TR, CAT

cat = CAT.copy()
cid = {v: i for i, v in enumerate(cat.vessel_id)}
cgt = pd.to_numeric(cat.gross_tonnage, errors='coerce').to_numpy(float)
name = cat.vessel_name.fillna('').str.upper()
toks = name.str.split()
feat = pd.DataFrame({
    'type': cat.vessel_type.fillna('NA').astype('category'),
    'flag': cat.flag_name.fillna('NA').astype('category'),
    'unit': cat.unit.fillna('NA').astype('category'),
    'lgt': np.log1p(cgt), 'ldwt': np.log1p(pd.to_numeric(cat.deadweight_tonnage, errors='coerce')),
    'lsize': np.log1p(pd.to_numeric(cat['size'], errors='coerce')),
    'ntok': toks.str.len(), 'nlen': name.str.len(), 'digit': name.str.contains(r'\d').astype(int),
})
for nm, s in (('tok1', toks.str[0]), ('tokL', toks.str[-1])):
    vc = s.value_counts()
    feat[nm] = s.where(s.map(vc) >= 15, 'RARE').fillna('NA').astype('category')
ctype = cat.vessel_type.fillna('NA').to_numpy(); cflag = cat.flag_name.fillna('NA').to_numpy()
TR = TR.copy()
TR['hang'] = TR.vessel_name.map(lambda s: vm.is_hangul(unicodedata.normalize('NFC', s)) if isinstance(s, str) else False)
WS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.8]
res = []
for k in range(5):
    pos = set(TR[TR.fold != k].target_vessel_id)
    y = cat.vessel_id.isin(pos).astype(int).to_numpy()
    m = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=30, feature_fraction=0.8,
                       bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2, seed=1), lgb.Dataset(feat, y), 300)
    prior = m.predict(feat, num_threads=2)
    lp = np.log(np.clip(prior, 1e-4, 1))
    trk = TR[TR.fold != k]
    ptype = defaultdict(dict); pflag = defaultdict(dict)
    for qt, ct in zip(trk.vessel_type.fillna('NA'), trk.target_vessel_id.map(lambda v: ctype[cid[v]])):
        ptype[qt][ct] = ptype[qt].get(ct, 0) + 1
    for qf, cf in zip(trk.flag_name.fillna('NA'), trk.target_vessel_id.map(lambda v: cflag[cid[v]])):
        pflag[qf][cf] = pflag[qf].get(cf, 0) + 1
    norm = lambda d: {a: {b: c / (sum(v.values()) + 5.0) for b, c in v.items()} for a, v in d.items()}
    ptype, pflag = norm(ptype), norm(pflag)
    excl = np.isin(cat.vessel_id, list(pos))
    for r in TR[TR.fold == k].itertuples():
        q = [x for x in (pd.to_numeric(r.gross_tonnage, errors='coerce'), pd.to_numeric(r.international_gross_tonnage, errors='coerce')) if x and np.isfinite(x) and x > 0]
        if not q:
            continue
        with np.errstate(invalid='ignore', divide='ignore'):
            gap = np.nanmin(np.vstack([np.abs(cgt - x) / np.maximum(cgt, x) for x in q]), axis=0)
        w = np.flatnonzero((gap <= 0.10) & ~excl)
        t = cid[r.target_vessel_id]
        if t not in set(w.tolist()):
            continue
        tp = ptype.get(r.vessel_type if isinstance(r.vessel_type, str) else 'NA', {})
        fp = pflag.get(r.flag_name if isinstance(r.flag_name, str) else 'NA', {})
        base = (2 * (gap[w] == 0) + np.array([tp.get(c, 0.0) for c in ctype[w]]) + np.array([fp.get(c, 0.0) for c in cflag[w]])
                - 3 * gap[w])
        out = dict(fold=k, hang=r.hang, nwin=len(w))
        ti = int(np.flatnonzero(w == t)[0])
        for wt in WS:
            s = base + wt * lp[w]
            better = int((s > s[ti]).sum()); ties = int((s == s[ti]).sum())   # ties include the truth
            out[f'acc{wt}'] = (better == 0) / ties                               # expected, random tie-break
            out[f'rr{wt}'] = 1.0 / (1 + better + (ties - 1) / 2)
        res.append(out)
    print('fold', k, 'prior AUC on fold-k vessels vs non-train catalog:',
          round(float(pd.Series(prior[~excl]).pipe(lambda p: __import__('sklearn.metrics', fromlist=['x']).roc_auc_score(cat.vessel_id[~excl].isin(set(TR[TR.fold == k].target_vessel_id)), p))), 3), flush=True)
d = pd.DataFrame(res)
for h in (True, False):
    s = d[d.hang == h]
    print(('HANGUL ' if h else 'ENGLISH'), 'rows', len(s), 'median window', int(s.nwin.median()), '| ' +
          ' | '.join(f'w={wt}: acc {s[f"acc{wt}"].mean():.3f} mrr {s[f"rr{wt}"].mean():.3f}' for wt in WS))
