"""v44 training build: HANGUL items only, with the fuzzy (pipe23) candidate lists and organizer-style syllable typos.

Items per fold (rng seeded as before so the v33-recipe items repeat the old names):
  orig / drop_ton / mix / abbr(_typo) / prefix / renamed / trunc(_typo) / swap(_typo) / typo2_sdel   (the v33 recipe)
  synko / synko_dup / synko_abbr(_typo)                                                               (the v33 recipe)
  + orgtypo1 / orgtypo2: one / two organizer-style edits (syllable deleted 35%, doubled 35%, swapped with its right
    neighbour 30%), tonnage dropped with probability 0.5; for every train hangul row and every synko item.
usage: build_feats26.py <out prefix> <folds> [N_SYNKO=1400] [SEED=3000]   -> <prefix>_f{k}.pkl per fold
"""
import pandas as pd, numpy as np, time, sys, pickle, random, gc
import vm, pipe23 as pipe5, translit as T, aug, aug2
from synth import Synth
from synth5 import KoSynth
from harness import TR, CAT
folds = [int(x) for x in sys.argv[2].split(',')]
N_SYNKO = int(sys.argv[3]) if len(sys.argv) > 3 else 1400
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
ix = pipe5.CatalogIndex(CAT)
td = T.Transducer(CAT.vessel_name.tolist())
lmv = np.load('lmv.npy')
comp = CAT.vessel_name.map(vm.en_compact); vc = comp.value_counts()
first = CAT.vessel_name.map(lambda n: (vm.en_split(n)[0] or [''])[0]); fc = first.value_counts()
CATX = CAT.copy(); CATX['dup'] = comp.map(vc) > 1; CATX['family'] = first.map(fc) >= 3
import zlib
tfold = dict(zip(TR.target_vessel_id, TR.fold))
CATX['fold'] = [tfold.get(v, zlib.crc32(v.encode()) % 5) for v in CATX.vessel_id]


def org_edit(name, rng):
    pos = [i for i, ch in enumerate(name) if '가' <= ch <= '힣']
    if len(pos) < 3:
        return None
    i = rng.choice(pos); r = rng.random()
    if r < 0.35:
        return name[:i] + name[i + 1:]
    if r < 0.70:
        return name[:i + 1] + name[i] + name[i + 1:]
    if i + 1 < len(name) and '가' <= name[i + 1] <= '힣':
        return name[:i] + name[i + 1] + name[i] + name[i + 2:]
    if i - 1 >= 0 and '가' <= name[i - 1] <= '힣':
        return name[:i - 1] + name[i] + name[i - 1] + name[i + 1:]
    return None


t = time.time()
for k in folds:
    rng = random.Random(SEED + k)
    trn = TR[TR.fold != k]
    st = pipe5.TrainStats(trn, CAT); syn = Synth(trn, CAT)
    pipe5.fit_transducer(td, trn, ix)
    ks = KoSynth(trn, CAT, td, ix, lmv)
    train_ids = set(trn.target_vessel_id)
    M = pipe5.Matcher23(ix, st, td, lmv, word_dict=pipe5.build_word_dict(trn, CAT), excluded=None,
                        train_names=trn['vessel_name'].tolist(), train_mask=np.isin(ix.ids, list(train_ids)))
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
            ab = aug.abbreviate(row['vessel_name'], rng)
            if ab and rng.random() < 0.45:
                named('abbr_typo', aug2.ko_noise(ab, rng), rng.random() < 0.2)
            else:
                named('abbr', ab, rng.random() < 0.4)
            if rng.random() < 0.5:
                named('prefix', syn.prefix_swap(row['vessel_name'], rng), rng.random() < 0.2)
            if rng.random() < 0.15:
                named('renamed', syn.renamed(row, rng))
            if rng.random() < 0.3:
                tr_ = syn.truncate(row['vessel_name'], rng)
                if tr_ and rng.random() < 0.4:
                    named('trunc_typo', aug2.ko_noise(tr_, rng), rng.random() < 0.2)
                else:
                    named('trunc', tr_, rng.random() < 0.2)
            tk = row['vessel_name'].split()
            if len(tk) >= 2 and rng.random() < 0.35:
                sw = ' '.join(tk[1:] + tk[:1]) if rng.random() < 0.5 else ' '.join(reversed(tk))
                if rng.random() < 0.4:
                    named('swap_typo', aug2.ko_noise(sw, rng), rng.random() < 0.2)
                else:
                    named('swap', sw, rng.random() < 0.2)
            if rng.random() < 0.15:
                named('typo2_sdel', aug2.syl_del(aug2.kbd_typo(row['vessel_name'], rng, 2), rng, 1))
            jobs += [(tag, r, row['target_vessel_id'], True, row['sample_id']) for tag, r in v]
        else:
            # keep the english rows' rng draws so the hangul items above stay the same as in the old builds
            if rng.random() < 0.25: aug.perturb(row, rng, 'mix')
            if rng.random() < 0.08:
                ab = aug.abbreviate(row['vessel_name'], rng)
                if ab and rng.random() < 0.4: aug.en_typo(ab, rng)
            if rng.random() < 0.06:
                pf = syn.prefix_swap(row['vessel_name'], rng)
                if pf and rng.random() < 0.4: aug.en_typo(pf, rng)
            if rng.random() < 0.03: syn.renamed(row, rng)
            if rng.random() < 0.04:
                tr_ = syn.truncate(row['vessel_name'], rng)
                if tr_ and rng.random() < 0.4: aug.en_typo(tr_, rng)
            tk = row['vessel_name'].split()
            if len(tk) >= 2 and rng.random() < 0.05: pass
            if len(tk) >= 2 and tk[0].isalpha() and rng.random() < 0.05: rng.random()
    for crow in CATX[(CATX.fold == k) & CATX.dup].to_dict('records'):
        for rep in range(2):
            syn.query_from_catalog(crow, rng)
    pool = CATX[(CATX.fold == k) & CATX.vessel_type.isin(ks.base.type_map)].copy()
    pri = pool.dup.astype(int) * 3 + pool.family.astype(int) + pool.vessel_type.isin(['PCC', 'Container', 'Chemical Parcel', 'Chem & Oil', 'GCargo', 'LPG']).astype(int)
    pool = pool.assign(pri=pri + np.array([rng.random() for _ in range(len(pool))])).sort_values('pri', ascending=False)
    n_ko = 0
    synko_items = []
    for crow in pool.to_dict('records'):
        if n_ko >= N_SYNKO:
            break
        reps = 2 if crow['dup'] else 1
        for rep in range(reps):
            q = ks.query(crow, rng)
            if q is None:
                break
            tag = 'synko_dup' if crow['dup'] else 'synko'
            clean = dict(q)
            if rng.random() < 0.15:
                ab = aug.abbreviate(q['vessel_name'], rng)
                if ab:
                    if rng.random() < 0.4:
                        q['vessel_name'] = aug2.ko_noise(ab, rng); tag = 'synko_abbr_typo'
                    else:
                        q['vessel_name'] = ab; tag = 'synko_abbr'
            jobs.append((tag, q, crow['vessel_id'], True, q['sample_id']))
            synko_items.append((clean, crow['vessel_id']))
            n_ko += 1
    # organizer-style typos (own rng)
    rng3 = random.Random(SEED + 8000 + k)
    def orgjobs(base, tgt, sid):
        e1 = org_edit(base['vessel_name'], rng3)
        if not e1:
            return
        for tag, nm in (('orgtypo1', e1), ('orgtypo2', org_edit(e1, rng3))):
            if not nm:
                continue
            r2 = dict(base); r2['vessel_name'] = nm
            if rng3.random() < 0.5:
                r2['gross_tonnage'] = None; r2['international_gross_tonnage'] = None
            jobs.append((tag, r2, tgt, True, sid))
    for row in TR[(TR.fold == k)].to_dict('records'):
        if vm.is_hangul(row['vessel_name']):
            orgjobs(row, row['target_vessel_id'], row['sample_id'])
    for clean, tgt in synko_items:
        orgjobs(clean, tgt, clean['sample_id'])
    print('fold', k, 'jobs', len(jobs), 'synko', n_ko, round(time.time() - t), flush=True)
    out = []
    for n, (tag, r, tgt, hang, sid) in enumerate(jobs):
        cands, X = M.features(r)
        y = (ix.ids[cands] == tgt).astype(np.int8)
        out.append((sid, k, tgt, hang, tag, ix.ids[cands], X, y))
        if n % 500 == 0:
            print('fold', k, n, len(jobs), round(time.time() - t), flush=True)
    pickle.dump(out, open(f'{sys.argv[1]}_f{k}.pkl', 'wb'))
    print('saved fold', k, len(out), round(time.time() - t), flush=True)
    del out; gc.collect()
