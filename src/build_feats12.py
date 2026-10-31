"""v5 OOF features: real rows (+hard variants), synthetic english dup queries, synthetic hangul queries."""
import pandas as pd, numpy as np, time, sys, pickle, random, zlib
import vm, pipe11 as pipe5, translit as T, aug
from synth import Synth
from synth5 import KoSynth
from harness import TR, CAT
folds = [int(x) for x in sys.argv[2].split(',')]
N_SYNKO = int(sys.argv[3]) if len(sys.argv) > 3 else 1400   # per fold
ix = pipe5.CatalogIndex(CAT)
td = T.Transducer(CAT.vessel_name.tolist())
lmv = np.load('lmv.npy')
comp = CAT.vessel_name.map(vm.en_compact)
vc = comp.value_counts()
first = CAT.vessel_name.map(lambda n: (vm.en_split(n)[0] or [''])[0])
fc = first.value_counts()
CATX = CAT.copy()
CATX['dup'] = comp.map(vc) > 1
CATX['family'] = first.map(fc) >= 3
tfold = dict(zip(TR.target_vessel_id, TR.fold))
CATX['fold'] = [tfold.get(v, zlib.crc32(v.encode()) % 5) for v in CATX.vessel_id]
out = []
t = time.time()
for k in folds:
    rng = random.Random(3000 + k)
    trn = TR[TR.fold != k]
    st = pipe5.TrainStats(trn, CAT)
    syn = Synth(trn, CAT)
    pipe5.fit_transducer(td, trn, ix)
    ks = KoSynth(trn, CAT, td, ix, lmv)
    M = pipe5.Matcher11(ix, st, td, lmv)
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
            tk = row['vessel_name'].split()
            if len(tk) >= 2 and rng.random() < 0.35:
                named('swap', ' '.join(tk[1:] + tk[:1]) if rng.random() < 0.5 else ' '.join(reversed(tk)), rng.random() < 0.2)
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
            tk = row['vessel_name'].split()
            if len(tk) >= 2 and rng.random() < 0.05:
                named('swap', ' '.join(tk[1:] + tk[:1]))
            if len(tk) >= 2 and tk[0].isalpha() and rng.random() < 0.05:
                named('init', ' '.join([tk[0][0] + ('.' if rng.random() < 0.5 else '')] + tk[1:]))
        jobs += [(tag, r, row['target_vessel_id'], hang, row['sample_id']) for tag, r in v]
    # synthetic english same-name queries
    for crow in CATX[(CATX.fold == k) & CATX.dup].to_dict('records'):
        for rep in range(2):
            q = syn.query_from_catalog(crow, rng)
            if q is not None:
                jobs.append(('dup', q, crow['vessel_id'], False, q['sample_id']))
    # synthetic hangul queries: prioritise twins / families / hangul-typical types
    pool = CATX[(CATX.fold == k) & CATX.vessel_type.isin(ks.base.type_map)].copy()
    pri = pool.dup.astype(int) * 3 + pool.family.astype(int) + pool.vessel_type.isin(['PCC', 'Container', 'Chemical Parcel', 'Chem & Oil', 'GCargo', 'LPG']).astype(int)
    pool = pool.assign(pri=pri + np.array([rng.random() for _ in range(len(pool))])).sort_values('pri', ascending=False)
    n_ko = 0
    for crow in pool.to_dict('records'):
        if n_ko >= N_SYNKO:
            break
        reps = 2 if crow['dup'] else 1
        for rep in range(reps):
            q = ks.query(crow, rng)
            if q is None:
                break
            tag = 'synko_dup' if crow['dup'] else 'synko'
            if rng.random() < 0.15:
                ab = aug.abbreviate(q['vessel_name'], rng)
                if ab:
                    q['vessel_name'] = ab; tag = 'synko_abbr'
            jobs.append((tag, q, crow['vessel_id'], True, q['sample_id']))
            n_ko += 1
    print('fold', k, 'jobs', len(jobs), 'synko', n_ko, round(time.time() - t), flush=True)
    for n, (tag, r, tgt, hang, sid) in enumerate(jobs):
        cands, X = M.features(r)
        y = (ix.ids[cands] == tgt).astype(np.int8)
        out.append((sid, k, tgt, hang, tag, ix.ids[cands], X, y))
        if n % 1000 == 0:
            print('fold', k, n, len(jobs), round(time.time() - t), flush=True)
    print('fold done', k, round(time.time() - t), flush=True)
pickle.dump(out, open(sys.argv[1], 'wb'))
print('saved', len(out), flush=True)
