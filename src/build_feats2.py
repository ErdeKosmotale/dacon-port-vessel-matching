"""OOF features (fold-specific stats + transducer) for original + augmented rows."""
import pandas as pd, numpy as np, time, sys, pickle, random
import vm, pipe2, translit as T, aug
from harness import TR, CAT
ix = pipe2.CatalogIndex(CAT)
td = T.Transducer(CAT.vessel_name.tolist())
lm = T.LetterLM(td.letters)
lmv = np.array([lm.logprob(s) for s in td.letters])
np.save('lmv.npy', lmv)
rng = random.Random(123)
out = []
t = time.time()
for k in range(5):
    trn = TR[TR.fold != k]
    st = pipe2.TrainStats(trn, CAT)
    pipe2.fit_transducer(td, trn, ix)
    M = pipe2.Matcher2(ix, st, td, lmv)
    sub = TR[TR.fold == k]
    for n, row in enumerate(sub.to_dict('records')):
        hang = vm.is_hangul(row['vessel_name'])
        variants = [('orig', row)]
        if hang:
            variants.append(('drop_ton', aug.perturb(row, rng, 'drop_ton')))
            variants.append(('mix', aug.perturb(row, rng, 'mix')))
        elif rng.random() < 0.25:
            variants.append(('mix', aug.perturb(row, rng, 'mix')))
        for tag, r in variants:
            cands, X = M.features(r)
            y = (ix.ids[cands] == row['target_vessel_id']).astype(np.int8)
            out.append((row['sample_id'], k, row['target_vessel_id'], hang, tag, ix.ids[cands], X, y))
        if n % 500 == 0:
            print('fold', k, n, len(sub), round(time.time() - t), flush=True)
    print('fold done', k, round(time.time() - t), flush=True)
pickle.dump(out, open(sys.argv[1], 'wb'))
print('saved', len(out))
