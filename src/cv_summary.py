"""Summarise exclusion-aware OOF files on the same items.

usage: cv_summary.py <oof csv>[:label] ...
Items are joined on (sid, fold, tag); only the items present in every file are compared.
Test-like score: hangul rows keep their tonnage (as in train.csv); hangul = 0.5 clean + 0.1 renamed + 0.4 damaged,
english = 0.78 clean + 0.22 damaged (damaged/renamed shares from the leaderboard diagnostics)."""
import sys
import pandas as pd

CLEAN = {'orig', 'synko', 'synko_dup', 'dup', 'mix', 'drop_ton'}
REN = {'renamed', 'xren'}
frames = []
for arg in sys.argv[1:]:
    path, _, label = arg.partition(':')
    d = pd.read_csv(path)
    d = d.drop_duplicates(['sid', 'fold', 'tag'])
    frames.append((label or path, d.set_index(['sid', 'fold', 'tag'])))
common = frames[0][1].index
for _, d in frames[1:]:
    common = common.intersection(d.index)
print('common items', len(common))
rows = []
for label, d in frames:
    d = d.loc[common].reset_index()
    d['tag0'] = d.tag.str.lstrip('x').where(~d.tag.isin(['xren']), 'renamed')
    h = d[d.hang & ~d.ton_missing]
    e = d[~d.hang]
    hc = h[h.tag0.isin(CLEAN)].new_ex.mean(); hr = h[h.tag0 == 'renamed'].new_ex.mean()
    hd = h[~h.tag0.isin(CLEAN) & (h.tag0 != 'renamed')].new_ex.mean()
    ec = e[e.tag0.isin(CLEAN)].new_ex.mean(); ed = e[~e.tag0.isin(CLEAN)].new_ex.mean()
    H = 0.5 * hc + 0.1 * hr + 0.4 * hd; E = 0.78 * ec + 0.22 * ed
    rows.append(dict(model=label, H_clean=hc, H_ren=hr, H_dmg=hd, E_clean=ec, E_dmg=ed, H=H, E=E, score=0.7 * H + 0.3 * E))
    if len(sys.argv) > 1 and label == frames[-1][0]:
        pass
print(pd.DataFrame(rows).round(4).to_string(index=False))
print()
by = []
for label, d in frames:
    d = d.loc[common].reset_index()
    t = d[d.hang & ~d.ton_missing].groupby('tag').new_ex.mean().rename(label)
    by.append(t)
print('hangul with tonnage, by tag')
print(pd.concat(by, axis=1).round(3).to_string())
