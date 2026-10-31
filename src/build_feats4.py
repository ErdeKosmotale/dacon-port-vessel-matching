"""v4 OOF features: real rows (+ hard-case variants) and synthetic same-name queries."""
import pandas as pd, numpy as np, time, sys, pickle, random, zlib
import vm, pipe4, translit as T, aug
from synth import Synth
from harness import TR, CAT
folds = [int(x) for x in sys.argv[2].split(',')]
ix = pipe4.CatalogIndex(CAT)
td = T.Transducer(CAT.vessel_name.tolist())
lmv = np.load('lmv.npy')
# synthetic same-name queries: catalog vessels whose compact name is shared
comp = CAT.vessel_name.map(vm.en_compact)
vc = comp.value_counts()
dup = CAT[(comp.map(vc) > 1)].copy()
dup['fold'] = [zlib.crc32(v.encode()) % 5 for v in dup.vessel_id]
out = []
t = time.time()
for k in folds:
    rng = random.Random(2000 + k)
    trn = TR[TR.fold != k]
    st = pipe4.TrainStats(trn, CAT)
    syn = Synth(trn, CAT)
    pipe4.fit_transducer(td, trn, ix)
    M = pipe4.Matcher4(ix, st, td, lmv)
    jobs = []
    for row in TR[TR.fold == k].to_dict('records'):
        hang = vm.is_hangul(row['vessel_name'])
        v = [('orig', row)]
        def named(tag, newname, drop=False):
            if newname:
                r2 = dict(row); r2['vessel_name'] = newname
                if drop:
                    r2 = aug.drop_ton(r2)
                v.append((tag, r2))
        if hang:
            v.append(('drop_ton', aug.perturb(row, rng, 'drop_ton')))
            v.append(('mix', aug.perturb(row, rng, 'mix')))
            named('abbr', aug.abbreviate(row['vessel_name'], rng), rng.random() < 0.4)
            if rng.random() < 0.5:
                named('prefix', syn.prefix_swap(row['vessel_name'], rng), rng.random() < 0.2)
            if rng.random() < 0.15:
                named('renamed', syn.renamed(row, rng))
            if rng.random() < 0.3:
                named('trunc', syn.truncate(row['vessel_name'], rng), rng.random() < 0.2)
        else:
            if rng.random() < 0.25:
                v.append(('mix', aug.perturb(row, rng, 'mix')))
            if rng.random() < 0.08:
                named('abbr', aug.abbreviate(row['vessel_name'], rng))
            if rng.random() < 0.06:
                named('prefix', syn.prefix_swap(row['vessel_name'], rng))
            if rng.random() < 0.03:
                named('renamed', syn.renamed(row, rng))
            if rng.random() < 0.04:
                named('trunc', syn.truncate(row['vessel_name'], rng))
        jobs += [(tag, r, row['target_vessel_id'], hang, row['sample_id']) for tag, r in v]
    for crow in dup[dup.fold == k].to_dict('records'):
        for rep in range(2):
            q = syn.query_from_catalog(crow, rng)
            if q is not None:
                jobs.append(('dup', q, crow['vessel_id'], False, q['sample_id']))
    for n, (tag, r, tgt, hang, sid) in enumerate(jobs):
        cands, X = M.features(r)
        y = (ix.ids[cands] == tgt).astype(np.int8)
        out.append((sid, k, tgt, hang, tag, ix.ids[cands], X, y))
        if n % 1000 == 0:
            print('fold', k, n, len(jobs), round(time.time() - t), flush=True)
    print('fold done', k, round(time.time() - t), flush=True)
pickle.dump(out, open(sys.argv[1], 'wb'))
print('saved', len(out), flush=True)
