"""v3 OOF features for a subset of folds (original + augmented rows)."""
import pandas as pd, numpy as np, time, sys, pickle, random
import vm, pipe3, translit as T, aug
from harness import TR, CAT
folds = [int(x) for x in sys.argv[2].split(',')]
ix = pipe3.CatalogIndex(CAT)
td = T.Transducer(CAT.vessel_name.tolist())
lmv = np.load('lmv.npy')
out = []
t = time.time()
for k in folds:
    rng = random.Random(1000 + k)
    trn = TR[TR.fold != k]
    st = pipe3.TrainStats(trn, CAT)
    pipe3.fit_transducer(td, trn, ix)
    M = pipe3.Matcher3(ix, st, td, lmv)
    sub = TR[TR.fold == k]
    for n, row in enumerate(sub.to_dict('records')):
        hang = vm.is_hangul(row['vessel_name'])
        variants = [('orig', row)]
        if hang:
            variants.append(('drop_ton', aug.perturb(row, rng, 'drop_ton')))
            variants.append(('mix', aug.perturb(row, rng, 'mix')))
            ab = aug.abbreviate(row['vessel_name'], rng)
            if ab:
                r2 = dict(row); r2['vessel_name'] = ab
                if rng.random() < 0.4:
                    r2 = aug.drop_ton(r2)
                variants.append(('abbr', r2))
        else:
            if rng.random() < 0.25:
                variants.append(('mix', aug.perturb(row, rng, 'mix')))
            if rng.random() < 0.08:
                ab = aug.abbreviate(row['vessel_name'], rng)
                if ab:
                    r2 = dict(row); r2['vessel_name'] = ab
                    variants.append(('abbr', r2))
        for tag, r in variants:
            cands, X = M.features(r)
            y = (ix.ids[cands] == row['target_vessel_id']).astype(np.int8)
            out.append((row['sample_id'], k, row['target_vessel_id'], hang, tag, ix.ids[cands], X, y))
        if n % 500 == 0:
            print('fold', k, n, len(sub), round(time.time() - t), flush=True)
    print('fold done', k, round(time.time() - t), flush=True)
pickle.dump(out, open(sys.argv[1], 'wb'))
print('saved', len(out), flush=True)
