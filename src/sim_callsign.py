"""OFFLINE ONLY (never packaged for submission): how much would grouping rows by call_sign raise accuracy
if name damage were applied to each row independently? Held-out fold of train, v18 fold ranker."""
import pandas as pd, numpy as np, random, json, lightgbm as lgb, sys
import vm, pipe11 as P, translit as T, aug, aug2
from synth import Synth
from harness import TR, CAT
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
FN = P.FEATURES7; fn = json.load(open('ranker_v18.txt.features.json')); cols = [FN.index(f) for f in fn]
ix = P.CatalogIndex(CAT); lmv = np.load('lmv.npy')
trk = TR[TR.fold != k]; td = T.Transducer(CAT.vessel_name.tolist()); st = P.TrainStats(trk, CAT); P.fit_transducer(td, trk, ix)
M = P.Matcher11(ix, st, td, lmv); m = lgb.Booster(model_file=f'ranker_v18_f{k}.txt'); syn = Synth(trk, CAT); excl = set(trk.target_vessel_id)
rng = random.Random(42)
def damage(name, hang):
    for _ in range(5):
        if hang:
            op = rng.choice(['abbr', 'prefix', 'trunc', 'typo', 'swap', 'abbr_typo'])
            tk = name.split()
            out = {'abbr': lambda: aug.abbreviate(name, rng), 'prefix': lambda: syn.prefix_swap(name, rng), 'trunc': lambda: syn.truncate(name, rng),
                   'typo': lambda: aug2.ko_noise(name, rng), 'swap': lambda: ' '.join(reversed(tk)) if len(tk) >= 2 else None,
                   'abbr_typo': lambda: (lambda a: aug2.ko_noise(a, rng) if a else None)(aug.abbreviate(name, rng))}[op]()
        else:
            op = rng.choice(['abbr', 'prefix', 'trunc', 'typo2', 'swap'])
            tk = name.split()
            out = {'abbr': lambda: aug.abbreviate(name, rng), 'prefix': lambda: syn.prefix_swap(name, rng), 'trunc': lambda: syn.truncate(name, rng),
                   'typo2': lambda: aug.en_typo(aug.en_typo(name, rng), rng), 'swap': lambda: ' '.join(tk[1:] + tk[:1]) if len(tk) >= 2 else None}[op]()
        if out and out != name: return out
    return name
res = []
for row in TR[TR.fold == k].to_dict('records'):
    hang = vm.is_hangul(row['vessel_name'])
    r2 = dict(row)
    if rng.random() < (0.5 if hang else 0.22): r2['vessel_name'] = damage(row['vessel_name'], hang)
    cands, X = M.features(r2); s = m.predict(X[:, cols], num_threads=1); ids = ix.ids[cands]
    se = np.where([x in excl for x in ids], s - 1e6, s); p = int(np.argmax(se))
    res.append(dict(cs=row['call_sign'], hang=hang, tgt=row['target_vessel_id'], pred=ids[p], conf=float(np.sort(se)[-1] - (np.sort(se)[-2] if len(se) > 1 else 0))))
d = pd.DataFrame(res); d['ok'] = d.pred == d.tgt
best = d.loc[d.groupby('cs').conf.idxmax(), ['cs', 'pred']].set_index('cs').pred
d['ok_vote'] = d.cs.map(best) == d.tgt
for c in ['ok', 'ok_vote']:
    H, E = d[d.hang][c].mean(), d[~d.hang][c].mean()
    print(f"{'row-independent' if c == 'ok' else 'grouped by call_sign (FORBIDDEN)'}: hangul {H:.3f} english {E:.3f} -> score {0.7 * H + 0.3 * E:.4f}")
