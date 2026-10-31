import pandas as pd, numpy as np, pickle, random, sys
import vm, pipe
from harness import *
feats = pickle.load(open('feats_v1.pkl', 'rb'))
FN = [f for f in pipe.FEATURES if f not in DROP]
models = fold_models(feats, FN)
pickle.dump(models, open('fold_models_v1.pkl', 'wb'))
ix = pipe.CatalogIndex(CAT)
# one row per vessel to save time
h = TR[TR.vessel_name.map(vm.is_hangul)].drop_duplicates('target_vessel_id')
e = TR[~TR.vessel_name.map(vm.is_hangul)].drop_duplicates('target_vessel_id').sample(600, random_state=0)
sel = pd.concat([h, e])
rows_by_fold = {k: sel[sel.fold == k].to_dict('records') for k in range(5)}

def drop_ton(r, rng):
    r['gross_tonnage'] = np.nan; r['international_gross_tonnage'] = np.nan; return r
def noise_ton(r, rng):
    for c in ('gross_tonnage', 'international_gross_tonnage'):
        v = vm.pos_float(r[c])
        if v: r[c] = round(v * rng.choice([rng.uniform(0.95, 0.99), rng.uniform(1.01, 1.05)]))
    return r
def drop_flag_type(r, rng):
    r['flag_name'] = np.nan; return r
def ko_typo(r, rng):
    n = r['vessel_name']
    if not vm.is_hangul(n): return r
    s = list(n)
    op = rng.choice(['del', 'dup', 'space', 'swap'])
    idx = [i for i, c in enumerate(s) if '가' <= c <= '힣']
    i = rng.choice(idx)
    if op == 'del' and len(idx) > 2: del s[i]
    elif op == 'dup': s.insert(i, s[i])
    elif op == 'space': s = [c for c in s if c != ' '] if ' ' in s else s[:i] + [' '] + s[i:]
    elif op == 'swap' and i + 1 < len(s): s[i], s[i + 1] = s[i + 1], s[i]
    r['vessel_name'] = ''.join(s); return r

for name, pf in [('none', None), ('drop_ton', drop_ton), ('noise_ton', noise_ton), ('drop_flag', drop_flag_type), ('ko_typo', ko_typo)]:
    r = evaluate(models, FN, ix, rows_by_fold, pf)
    hh = r[r.hang]; ee = r[~r.hang]
    print(f'{name:10s} hang acc {hh.ok.mean():.4f} recall {hh.recall.mean():.4f} | en acc {ee.ok.mean():.4f} recall {ee.recall.mean():.4f}', flush=True)
