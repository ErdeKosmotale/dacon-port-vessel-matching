import pandas as pd, numpy as np, random, lightgbm as lgb, re
import vm, pipe7 as P, translit as T
from harness import TR, CAT
k = 0
FN = P.FEATURES7; drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
cols = [FN.index(f) for f in FN if f not in drop]
m = lgb.Booster(model_file='stress_fold0.txt')
ix = P.CatalogIndex(CAT); td = T.Transducer(CAT.vessel_name.tolist()); lmv = np.load('lmv.npy')
tr = TR[TR.fold != k]; st = P.TrainStats(tr, CAT); P.fit_transducer(td, tr, ix); M = P.Matcher7(ix, st, td, lmv)
cname = dict(zip(CAT.vessel_id, CAT.vessel_name))
CHO, JUNG, JONG = T.CHO, T.JUNG, T.JONG
def dec(c):
    o = ord(c) - 0xAC00; return o // 588, (o % 588) // 28, o % 28
def enc(a, b, c): return chr(0xAC00 + a * 588 + b * 28 + c)
VOW = {'ㅐ': 'ㅔ', 'ㅔ': 'ㅐ', 'ㅓ': 'ㅗ', 'ㅗ': 'ㅓ', 'ㅡ': 'ㅜ', 'ㅜ': 'ㅡ', 'ㅑ': 'ㅏ', 'ㅕ': 'ㅔ', 'ㅖ': 'ㅔ', 'ㅟ': 'ㅜ', 'ㅢ': 'ㅣ'}
ONS = {'ㅅ': 'ㅆ', 'ㅆ': 'ㅅ', 'ㅈ': 'ㅊ', 'ㅊ': 'ㅈ', 'ㅌ': 'ㄷ', 'ㄷ': 'ㅌ', 'ㅍ': 'ㅂ', 'ㅂ': 'ㅍ', 'ㅋ': 'ㄱ', 'ㄱ': 'ㅋ', 'ㄹ': 'ㄴ'}
def perturb(name, rng, n):
    s = list(name); idx = [i for i, c in enumerate(s) if '가' <= c <= '힣']
    rng.shuffle(idx); done = 0
    for i in idx:
        if done >= n: break
        a, b, c = dec(s[i]); op = rng.random()
        if op < 0.45 and JUNG[b] in VOW: b = JUNG.index(VOW[JUNG[b]])
        elif op < 0.75 and CHO[a] in ONS: a = CHO.index(ONS[CHO[a]])
        elif op < 0.9 and c != 0: c = 0                      # drop final consonant
        elif c == 0: c = JONG.index(rng.choice(['ㄹ', 'ㅇ', 'ㄴ']))  # add final consonant
        else: continue
        s[i] = enc(a, b, c); done += 1
    return ''.join(s)
rng = random.Random(11); res = []
rows = TR[(TR.fold == k) & TR.vessel_name.map(vm.is_hangul)].drop_duplicates('vessel_name').to_dict('records')
for row in rows:
    for n in [0, 1, 2, 3]:
        nm = row['vessel_name'] if n == 0 else perturb(row['vessel_name'], rng, n)
        r2 = dict(row); r2['vessel_name'] = nm
        cands, X = M.features(r2); s = m.predict(X[:, cols], num_threads=2)
        tgt = row['target_vessel_id']
        res.append(dict(n=n, name=nm, en=cname[tgt], ok=ix.ids[cands[int(np.argmax(s))]] == tgt, rec=tgt in set(ix.ids[cands]), pred=cname[ix.ids[cands[int(np.argmax(s))]]]))
d = pd.DataFrame(res); d.to_pickle('stress_conv.pkl')
print(d.groupby('n')[['ok', 'rec']].agg(['mean', 'size']).round(3))
for _, r in d[(d.n >= 2) & ~d.ok].head(25).iterrows(): print(r.n, '|', r['name'], '|', r.en, '| pred:', r.pred)
