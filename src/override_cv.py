"""Calibrate a name-trust override on v18 OOF (hangul, with tonnage): pick the transliteration-best eligible
candidate when its margin over the 2nd best is large, else the ranker's pick (both with train-vessel exclusion)."""
import pandas as pd, numpy as np, pickle
import pipe11 as P
from harness import TR
FN = [f for f in P.FEATURES7 if f not in {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}]; F = {f: i for i, f in enumerate(FN)}
X = np.load('v18_X.npy', mmap_mode='r'); y = np.load('v18_y.npy'); meta = pickle.load(open('v18_meta.pkl', 'rb'))
g = np.array([m[5] for m in meta]); gs = np.concatenate([[0], np.cumsum(g)])
s1 = pickle.load(open('stage2_s1.pkl', 'rb'))
cids = [it[5] for f in ['feats_v12_a.pkl', 'feats_v12_b.pkl'] for it in pickle.load(open(f, 'rb'))]
excl = {k: set(TR[TR.fold != k].target_vessel_id) for k in range(5)}
rows = []
for j, s in s1.items():
    k = meta[j][1]; Xj = X[gs[j]:gs[j + 1]]; yj = y[gs[j]:gs[j + 1]]
    ok = np.array([c not in excl[k] for c in cids[j]])
    if not ok.any(): continue
    pr = np.where(ok, s, -1e9); rk = int(np.argmax(pr))
    pmi = np.where(ok, np.nan_to_num(Xj[:, F['t_pmi_rel']], nan=-1e9), -1e9)
    loc = np.where(ok, np.nan_to_num(Xj[:, F['t_loc_rel']], nan=-1e9), -1e9)
    o = np.argsort(-pmi); b, b2 = o[0], o[1] if len(o) > 1 else o[0]
    margin = pmi[b] - pmi[b2]
    lo = np.argsort(-loc); lmargin = loc[lo[0]] - (loc[lo[1]] if len(lo) > 1 else loc[lo[0]])
    rows.append(dict(tag=meta[j][3], ranker_ok=bool(yj[rk]), name_ok=bool(yj[b]), same=b == rk, margin=float(margin),
                     loc_agree=lo[0] == b, lmargin=float(lmargin), gap_b=float(np.nan_to_num(Xj[b, F['gap_min']], nan=9)),
                     pmi_abs=float(Xj[b, F['t_lp_per_unit']]) if np.isfinite(Xj[b, F['t_lp_per_unit']]) else -99))
d = pd.DataFrame(rows); d.to_pickle('override_cv.pkl')
print('items', len(d), 'ranker acc', d.ranker_ok.mean().round(4))
for tau in [3, 5, 8, 12]:
    for need_loc in [False, True]:
        fire = (d.margin >= tau) & (~d.same) & ((d.loc_agree) if need_loc else True)
        new_ok = np.where(fire, d.name_ok, d.ranker_ok)
        by = d.assign(new=new_ok, fire=fire).groupby('tag').agg(n=('new', 'size'), ranker=('ranker_ok', 'mean'), new=('new', 'mean'), fires=('fire', 'sum'))
        print(f'tau={tau} need_loc={need_loc}: fires {int(fire.sum())} | acc {d.ranker_ok.mean():.4f} -> {new_ok.mean():.4f} | fired-and-right {int((fire & d.name_ok).sum())} fired-and-wrong {int((fire & ~d.name_ok).sum())}')
        if tau in (5, 8) and need_loc: print(by.round(3).to_string())
