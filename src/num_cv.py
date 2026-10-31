"""How much do numbers in names matter for the remaining errors? (exclusion-aware OOF, v37A0 fold models)

For each held-out item: is the pick wrong, and how do the record's numbers relate to the pick's and the truth's
(num_q = record has numbers, num_match / num_mismatch per candidate). Error classes:
  ignored  record has a number, the truth matches it, the pick does not  -> the model under-used number evidence
  missing  record has no number, the truth's name has one                -> numbered sisters, nothing to go on
  other
"""
import pickle, numpy as np, pandas as pd, lightgbm as lgb
import pipe22 as P
from harness import TR
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in P.FEATURES7 if f not in drop]
TL = set(P.TL_FEATURES)
cols = [i for i, f in enumerate(fn) if f not in TL]
I = {f: fn.index(f) for f in ('num_q', 'num_c', 'num_match', 'num_mismatch')}
X = np.load('v37_X.npy', mmap_mode='r'); y = np.load('v37_y.npy')
meta = pickle.load(open('v37_meta.pkl', 'rb')); cids = pickle.load(open('v37_cids.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)])
ifold = np.array([m[1] for m in meta])
out = []
for k in range(5):
    m = lgb.Booster(model_file=f'v37A0_f{k}.txt')
    excl = set(TR[TR.fold != k].target_vessel_id)
    for j in np.flatnonzero(ifold == k):
        mt = meta[j]
        if mt[4] or mt[3].startswith('synko'):
            continue
        Xj = np.asarray(X[gs[j]:gs[j + 1]]); yj = y[gs[j]:gs[j + 1]]
        if not yj.any():
            continue
        s = m.predict(Xj[:, cols], num_threads=2)
        ok = np.array([c not in excl for c in cids[j]])
        top = int(np.argmax(np.where(ok, s, -1e9))); t = int(np.argmax(yj))
        nq = np.nan_to_num(Xj[t, I['num_q']]); nct = np.nan_to_num(Xj[t, I['num_c']])
        mt_t = np.nan_to_num(Xj[t, I['num_match']]); mm_p = np.nan_to_num(Xj[top, I['num_mismatch']])
        mt_p = np.nan_to_num(Xj[top, I['num_match']])
        cls = 'ok' if top == t else ('ignored' if nq > 0 and mt_t > 0 and mt_p == 0 else ('missing' if nq == 0 and nct > 0 else 'other'))
        out.append(dict(hang=bool(mt[2]), real=not mt[3].startswith('x'), tag=mt[3], cls=cls, has_num=nq > 0, truth_num=nct > 0))
    print('fold', k, flush=True)
d = pd.DataFrame(out)
for nm, msk in [('HANGUL real', d.hang & d.real), ('HANGUL renamed/x', d.hang & ~d.real), ('ENGLISH real', ~d.hang & d.real)]:
    s = d[msk]
    e = s[s.cls != 'ok']
    print(f'{nm:18s} n={len(s)} record-has-number {s.has_num.mean():.3f} truth-has-number {s.truth_num.mean():.3f} | errors {len(e)}: '
          + ' '.join(f'{c} {int((e.cls == c).sum())}' for c in ('ignored', 'missing', 'other')))
    print('   accuracy when truth name has a number:', round(float((s[s.truth_num].cls == 'ok').mean()), 4), 'n', int(s.truth_num.sum()),
          '| without:', round(float((s[~s.truth_num].cls == 'ok').mean()), 4))
