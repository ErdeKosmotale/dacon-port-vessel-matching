import numpy as np, pandas as pd, pickle
from scipy.optimize import minimize
import pipe7
from harness import TR
FN = pipe7.FEATURES7; F = {f: FN.index(f) for f in FN}
use = ['t_pmi_rel', 't_loc_rel', 't_red_rel', 'gap_min', 'type_prob', 'flag_prob', 'num_mismatch', 'rom_sim', 'sk_sim', 'vk_sim', 'q_flag_missing']
items = []
for f in ['feats_v7_a.pkl', 'feats_v7_b.pkl']:
    for it in pickle.load(open(f, 'rb')):
        if it[3] and it[6][0, F['q_ton_missing']] == 0:
            items.append((it[1], it[4], it[5], it[7], it[6][:, [F[u] for u in use]].astype(np.float64)))
print('hangul with-ton items', len(items), flush=True)
excl = {k: set(TR[TR.fold != k].target_vessel_id) for k in range(5)}
EX = [np.array([c in excl[k] for c in cid]) for k, _, cid, _, _ in items]
CLEAN = np.array([t in ('orig', 'synko') for _, t, _, _, _ in items])
TAGS = np.array([t for _, t, _, _, _ in items])
U = {u: i for i, u in enumerate(use)}
def score(X, th):
    a1, a2, wt, we, wty, wf, wn, wr, ws = th
    nm = np.nan_to_num(X[:, U['t_pmi_rel']], nan=-50)
    nm = np.maximum(nm, np.nan_to_num(X[:, U['t_loc_rel']], nan=-50) - a1)
    nm = np.maximum(nm, np.nan_to_num(X[:, U['t_red_rel']], nan=-50) - a2)
    g = np.nan_to_num(X[:, U['gap_min']], nan=1.0)
    ton = -wt * np.minimum(g, 0.3) / 0.01 + we * (g == 0) - 20 * (g > 0.07)
    ty = wty * np.log(X[:, U['type_prob']] + 1e-3)
    fl = wf * np.log(np.nan_to_num(X[:, U['flag_prob']], nan=0.0) + 1e-3)
    return nm + ton + ty + fl - wn * X[:, U['num_mismatch']] + wr * X[:, U['rom_sim']] + ws * (X[:, U['sk_sim']] + X[:, U['vk_sim']])
def acc(th, mask=None):
    ok = np.empty(len(items), bool)
    for n, (k, t, cid, y, X) in enumerate(items):
        s = score(X, th) - 1e6 * EX[n]
        ok[n] = y[int(np.argmax(s))] == 1
    return ok
def obj(th):
    ok = acc(th)
    return -(0.5 * ok[CLEAN].mean() + 0.5 * ok[~CLEAN].mean())
th0 = np.array([1.0, 2.0, 0.3, 1.0, 1.0, 0.5, 2.0, 2.0, 2.0])
best = (obj(th0), th0); print('init', best[0], flush=True)
rng = np.random.RandomState(0)
for i in range(150):
    th = best[1] * np.exp(rng.randn(9) * 0.4) if i > 20 else np.exp(rng.randn(9)) * th0
    v = obj(th)
    if v < best[0]:
        best = (v, th); print(i, round(-v, 4), np.round(th, 3), flush=True)
r = minimize(obj, best[1], method='Nelder-Mead', options=dict(maxiter=300, xatol=1e-3, fatol=1e-5))
if r.fun < best[0]: best = (r.fun, r.x)
print('BEST', -best[0], list(np.round(best[1], 4)))
ok = acc(best[1])
d = pd.DataFrame(dict(tag=TAGS, gen=ok))
b = pd.read_pickle('prior_cv.pkl'); b = b[b.hang & ~b.tm]
print(pd.concat([d.groupby('tag').gen.mean(), b.groupby('tag').base_ex.mean()], axis=1).round(3))
print('v10 ranker test-like H:', 0.5 * b[b.clean].base_ex.mean() + 0.5 * b[~b.clean].base_ex.mean())
np.save('gen_theta.npy', best[1])
