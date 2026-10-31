"""Tonnage hard filter at the final pick (exclusion-aware OOF, v37A0 fold models).

In train.csv the true vessel's catalog gross tonnage is ALWAYS within 10% of the record's gross_tonnage or
international_gross_tonnage (hangul: 0/1191 rows beyond 10%, english: 0/10238). So for a row with tonnage, a
candidate more than tau away from both fields is (almost) never the answer. Rule: if the ranker's best eligible
candidate is beyond tau, take the best eligible candidate within tau (or with unknown catalog tonnage) instead.
Reports, per script / tag, how often the rule fires, fixes and breaks.
"""
import pickle, numpy as np, pandas as pd, lightgbm as lgb
import pipe22 as P
from harness import TR
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in P.FEATURES7 if f not in drop]
TL = set(P.TL_FEATURES)
cols = [i for i, f in enumerate(fn) if f not in TL]
iG = fn.index('gap_min')
X = np.load('v37_X.npy', mmap_mode='r'); y = np.load('v37_y.npy')
meta = pickle.load(open('v37_meta.pkl', 'rb')); cids = pickle.load(open('v37_cids.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)])
ifold = np.array([m[1] for m in meta])
TAUS = [0.06, 0.08, 0.10, 0.15, 0.25]
out = []
for k in range(5):
    m = lgb.Booster(model_file=f'v37A0_f{k}.txt')
    excl = set(TR[TR.fold != k].target_vessel_id)
    for j in np.flatnonzero(ifold == k):
        mt = meta[j]
        if mt[4]:
            continue
        Xj = np.asarray(X[gs[j]:gs[j + 1]]); yj = y[gs[j]:gs[j + 1]]
        s = m.predict(Xj[:, cols], num_threads=2)
        ok = np.array([c not in excl for c in cids[j]])
        order = [i for i in np.argsort(-s, kind='stable') if ok[i]]
        if not order:
            continue
        gap = Xj[:, iG]
        r = dict(hang=bool(mt[2]), tag=mt[3], base=bool(yj[order[0]]), gap1=float(gap[order[0]]),
                 tgap=float(np.nanmin(np.where(yj > 0, gap, np.nan))) if yj.any() else np.nan)
        for t in TAUS:
            pick = order[0]
            if np.isfinite(gap[pick]) and gap[pick] > t:
                inside = [i for i in order if not (np.isfinite(gap[i]) and gap[i] > t)]
                if inside:
                    pick = inside[0]
            r[f'f{t}'] = bool(yj[pick]); r[f'sw{t}'] = pick != order[0]
        out.append(r)
    print('fold', k, flush=True)
d = pd.DataFrame(out)
d.to_pickle('gapfilter_cv.pkl')
print('true vessel gap > tau (all items with tonnage, truth retrieved):',
      {t: int((d.tgap > t).sum()) for t in TAUS}, 'of', int(d.tgap.notna().sum()))


def line(nm, s):
    r = f'{nm:20s} n={len(s):6d} base={s.base.mean():.4f}'
    for t in TAUS:
        sw = s[f'sw{t}']
        r += f' | {t}: sw {int(sw.sum())} fix {int((s[f"f{t}"] & ~s.base).sum())} brk {int((~s[f"f{t}"] & s.base).sum())}'
    print(r)


for h in (True, False):
    s = d[d.hang == h]
    line('HANGUL' if h else 'ENGLISH', s)
    for t, st in s.groupby('tag'):
        line('   ' + t, st)
