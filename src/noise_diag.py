"""For the noise-set rows a package got wrong: is the true vessel among the package's candidates, at what ensemble
rank, and which retrieval channel would have found it? Runs the package's own matcher on those rows only.
usage: noise_diag.py <package dir e.g. nz_fuzzyK500> <noise dir>
"""
import sys, json, time, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
pkg, nd = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(pkg / 'model'))
import lightgbm as lgb
import pipe23 as P4
import translit as T
import vm
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein

cat = pd.read_csv('/home/claude/data/open/vessel_catalog.csv', encoding='utf-8-sig')
train = pd.read_csv(pkg / 'model' / 'train.csv', encoding='utf-8-sig')
test = pd.read_csv(nd / 'test.csv', encoding='utf-8-sig'); truth = pd.read_csv(nd / 'truth.csv')
pred = pd.read_csv(pkg / 'output' / 'submission.csv')
m = truth.merge(pred, on='sample_id', suffixes=('', '_p'))
wrong = m[m.target_vessel_id != m.target_vessel_id_p]
print('wrong rows', len(wrong), flush=True)
index = P4.CatalogIndex(cat); stats = P4.TrainStats(train, cat)
td = T.Transducer(cat['vessel_name'].tolist()); lm = T.LetterLM(td.letters)
lmv = np.array([lm.logprob(s) for s in td.letters]); P4.fit_transducer(td, train, index)
train_ids = set(train.target_vessel_id) - (set(json.load(open(pkg / 'model' / 'eval_vessels.json'))) if (pkg / 'model' / 'eval_vessels.json').exists() else set())
matcher = P4.Matcher23(index, stats, td, lmv, word_dict=P4.build_word_dict(train, cat), excluded=None,
                       train_names=train['vessel_name'].tolist(), train_mask=np.isin(index.ids, list(train_ids)))
rankers = []
for p in sorted((pkg / 'model').glob('ranker*.txt')):
    fn = json.load(open(str(p) + '.features.json'))
    rankers.append((lgb.Booster(model_file=str(p)), [P4.FEATURES7.index(f) for f in fn]))
cid = {v: i for i, v in enumerate(index.ids)}
rows = test.set_index('sample_id')
F = P4.FEATURES7
iBN, iNG = F.index('best_name'), F.index('name_gap_best')
out = []
for r in wrong.itertuples():
    row = rows.loc[r.sample_id].to_dict(); row['sample_id'] = r.sample_id
    q = matcher.query_info(row)
    cands, X = matcher.features(row, q)
    sc = np.zeros(len(cands))
    for mdl, cols in rankers:
        s = mdl.predict(X[:, cols], num_threads=2); sc += (s - s.mean()) / (s.std() or 1.0)
    elig = [i for i in np.argsort(-sc, kind='stable') if index.ids[cands[i]] not in train_ids]
    t = cid[r.target_vessel_id]
    pos = [k + 1 for k, i in enumerate(elig) if cands[i] == t]
    # channel scores of the truth over the whole catalog
    ix = index
    sk = ix.skel_sim_all(q['sk']); vk = process.cdist([v for v in q['vk'] if v] or [''], ix.vkey, scorer=Levenshtein.normalized_similarity, dtype=np.float32, workers=-1).max(axis=0)
    first = sk + vk
    romv = q['romv']
    rom = process.cdist(romv, ix.compact, scorer=fuzz.ratio, dtype=np.uint8, workers=-1).max(axis=0).astype(np.float32) / 100 if romv else np.zeros(ix.N, np.float32)
    romp = process.cdist(romv, ix.compact, scorer=fuzz.partial_ratio, dtype=np.uint8, workers=-1).max(axis=0).astype(np.float32) / 100 if romv else np.zeros(ix.N, np.float32)
    rank = lambda a: int((a > a[t]).sum()) + 1
    nsim = X[:, iBN] - X[:, iNG]
    out.append(dict(sid=r.sample_id, kind=r.kind, ton=r.ton, name=row['vessel_name'], truth=cat.vessel_name[t],
                    pick=cat.vessel_name[cid[r.target_vessel_id_p]], n_cands=len(cands), in_cands=bool(pos), rank=pos[0] if pos else 0,
                    nsim_truth=float(nsim[list(cands).index(t)]) if pos else np.nan, nsim_pick=float(nsim[elig[0]]),
                    r_first=rank(first), r_rom=rank(rom), r_romp=rank(romp), r_sk=rank(sk)))
    print(out[-1], flush=True)
d = pd.DataFrame(out)
d.to_csv(f'noise_diag_{pkg.name}.csv', index=False)
print('\nin candidates:', int(d.in_cands.sum()), 'of', len(d), '| rank 2-3:', int(d['rank'].between(2, 3).sum()), '| rank 4+:', int((d['rank'] >= 4).sum()))
print('truth catalog-wide rank by channel (median / <=500 count): first', d.r_first.median(), int((d.r_first <= 500).sum()),
      '| rom', d.r_rom.median(), int((d.r_rom <= 500).sum()), '| rom partial', d.r_romp.median(), int((d.r_romp <= 500).sum()), '| sk', d.r_sk.median(), int((d.r_sk <= 500).sum()))
print('rows not in candidates but within 2000 by some channel:', int((~d.in_cands & (d[['r_first', 'r_rom', 'r_romp', 'r_sk']].min(axis=1) <= 2000)).sum()))
