"""Organizer-style noise on train hangul names, for testing candidate retrieval locally.

Per sampled vessel (one train row each): a clean copy, a 1-edit copy and a 2-edit copy. An edit hits one hangul
syllable: delete it (35%), double it (35%) or swap it with its right neighbour in the same word (30%). Typo copies
lose both tonnage fields with probability 0.5. Writes <out>/test.csv, <out>/truth.csv, <out>/eval_vessels.json.
usage: make_noise.py <out dir> [n vessels] [seed]
"""
import sys, json, random, unicodedata, pathlib
import pandas as pd
import vm

out = pathlib.Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
fold = int(sys.argv[4]) if len(sys.argv) > 4 else None
rng = random.Random(seed)
tr = pd.read_csv('/home/claude/data/open/train.csv', encoding='utf-8-sig')
tr['vessel_name'] = tr.vessel_name.map(lambda s: unicodedata.normalize('NFC', s) if isinstance(s, str) else s)
from harness import TR as _TR
tr['fold'] = _TR.fold.values
h = tr[tr.vessel_name.map(lambda s: vm.is_hangul(s) if isinstance(s, str) else False)]
if fold is not None:
    h = h[h.fold == fold]
h = h.drop_duplicates('target_vessel_id').sample(n=min(n, h.target_vessel_id.nunique()), random_state=seed)


def edit(name):
    pos = [i for i, ch in enumerate(name) if '가' <= ch <= '힣']
    if len(pos) < 3:
        return None
    i = rng.choice(pos)
    r = rng.random()
    if r < 0.35:
        return name[:i] + name[i + 1:]
    if r < 0.70:
        return name[:i + 1] + name[i] + name[i + 1:]
    if i + 1 < len(name) and '가' <= name[i + 1] <= '힣':
        return name[:i] + name[i + 1] + name[i] + name[i + 2:]
    j = i - 1
    if j >= 0 and '가' <= name[j] <= '힣':
        return name[:j] + name[i] + name[j] + name[i + 1:]
    return None


rows, truth = [], []
for r in h.itertuples():
    base = r._asdict(); base.pop('Index', None)
    variants = [('clean', r.vessel_name, False)]
    e1 = edit(r.vessel_name)
    if e1:
        variants.append(('typo1', e1, rng.random() < 0.5))
        e2 = edit(e1)
        if e2:
            variants.append(('typo2', e2, rng.random() < 0.5))
    for kind, nm, drop in variants:
        d = dict(base)
        d['vessel_name'] = nm
        if drop:
            d['gross_tonnage'] = None; d['international_gross_tonnage'] = None
        sid = f'NOISE_{len(rows):05d}'
        d['sample_id'] = sid
        truth.append(dict(sample_id=sid, target_vessel_id=d.pop('target_vessel_id'), kind=kind, ton=not drop))
        d.pop('fold', None)
        rows.append(d)
out.mkdir(parents=True, exist_ok=True)
cols = ['sample_id', 'vessel_name', 'call_sign', 'gross_tonnage', 'international_gross_tonnage', 'flag_name', 'vessel_type', 'arrival_year']
pd.DataFrame(rows)[cols].to_csv(out / 'test.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(truth).to_csv(out / 'truth.csv', index=False)
json.dump(sorted(set(h.target_vessel_id)), open(out / 'eval_vessels.json', 'w'))
t = pd.DataFrame(truth)
print('rows', len(t), t.groupby(['kind', 'ton']).size().to_dict())
