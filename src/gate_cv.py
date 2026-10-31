"""Per-item table for gating experiments (exclusion-aware OOF, v37A0 fold models), hangul + english, with tonnage.

For each held-out item: the ranker's top-1 / top-2 / structure-best (S_a over all eligible candidates) / name-best,
whether each is the true vessel, and row-level gate features (top-1's name similarity, tonnage exactness and gap,
type/flag fit, the same for the structure-best candidate, score margins). Saved to gate_cv.pkl so gates can be
evaluated in pandas without re-running the models.
"""
import pickle, numpy as np, pandas as pd, lightgbm as lgb
import pipe22 as P
from harness import TR
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in P.FEATURES7 if f not in drop]
TL = set(P.TL_FEATURES)
cols = [i for i, f in enumerate(fn) if f not in TL]
I = {f: fn.index(f) for f in fn}
X = np.load('v37_X.npy', mmap_mode='r'); y = np.load('v37_y.npy')
meta = pickle.load(open('v37_meta.pkl', 'rb')); cids = pickle.load(open('v37_cids.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)])
ifold = np.array([m[1] for m in meta])
F = ['exact_gt', 'exact_igt', 'gap_min', 'type_prob', 'flag_prob', 'struct_score', 'n_exact_ton_in_cands', 't_pmi_rel',
     'tk_mean_q', 'unit_prob', 'ton_ll']
out = []
for k in range(5):
    m = lgb.Booster(model_file=f'v37A0_f{k}.txt')
    excl = set(TR[TR.fold != k].target_vessel_id)
    for j in np.flatnonzero(ifold == k):
        mt = meta[j]
        if mt[4] or mt[3].startswith('synko'):
            continue
        Xj = np.asarray(X[gs[j]:gs[j + 1]]); yj = y[gs[j]:gs[j + 1]]
        s = m.predict(Xj[:, cols], num_threads=2)
        ok = np.array([c not in excl for c in cids[j]])
        el = np.flatnonzero(ok)
        if len(el) < 2:
            continue
        z = (s - s.mean()) / (s.std() or 1.0)
        order = el[np.argsort(-z[el], kind='stable')]
        nsim = np.nan_to_num(Xj[:, I['best_name']] - Xj[:, I['name_gap_best']], nan=-1)
        sa = np.nan_to_num(Xj[:, I['struct_score']], nan=-9) + 1e-4 * z
        sbest = el[np.argmax(sa[el])]
        nbest = el[np.argmax(nsim[el] + 1e-4 * z[el])]
        t1, t2 = order[0], order[1]
        pos = [r + 1 for r, i in enumerate(order) if yj[i]]
        r = dict(fold=k, hang=bool(mt[2]), tag=mt[3], rank=pos[0] if pos else 0,
                 y1=bool(yj[t1]), y2=bool(yj[t2]), yS=bool(yj[sbest]), yN=bool(yj[nbest]),
                 S_is_top1=bool(sbest == t1), S_rank=int(np.flatnonzero(order == sbest)[0]) + 1,
                 m12=float(z[t1] - z[t2]), m1S=float(z[t1] - z[sbest]), nsim1=float(nsim[t1]), nsimS=float(nsim[sbest]),
                 nsim_max=float(nsim[el].max()), n_el=len(el))
        for f in F:
            v = Xj[:, I[f]]
            r[f + '1'] = float(v[t1]); r[f + 'S'] = float(v[sbest])
        out.append(r)
    print('fold', k, flush=True)
pd.DataFrame(out).to_pickle('gate_cv.pkl')
print('saved', len(out))
