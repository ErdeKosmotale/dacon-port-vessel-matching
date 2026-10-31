import pandas as pd, numpy as np, random, pickle, sys, lightgbm as lgb, re
import vm, pipe7 as P, translit as T
from harness import TR, CAT
k = 0
FN = P.FEATURES7; drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
cols = [FN.index(f) for f in FN if f not in drop]
feats = pickle.load(open('feats_v7_a.pkl', 'rb'))
params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
feats += pickle.load(open('feats_v7_b.pkl', 'rb'))
trn = [it for it in feats if it[1] != k]; del feats
m = lgb.train(params, lgb.Dataset(np.concatenate([it[6][:, cols] for it in trn]), np.concatenate([it[7] for it in trn]), group=[len(it[7]) for it in trn]), 500)
del trn
m.save_model('stress_fold0.txt')
ix = P.CatalogIndex(CAT); td = T.Transducer(CAT.vessel_name.tolist()); lmv = np.load('lmv.npy')
tr = TR[TR.fold != k]
st = P.TrainStats(tr, CAT); P.fit_transducer(td, tr, ix)
M = P.Matcher7(ix, st, td, lmv)
cname = dict(zip(CAT.vessel_id, CAT.vessel_name))
NUMKO = {'1': '원', '2': '투', '3': '쓰리', '5': '파이브', '7': '세븐', '8': '에잇', '9': '나인'}
def variants(row, rng):
    n = row['vessel_name']; tk = n.split(); en = str(cname[row['target_vessel_id']]).split()
    out = {}
    out['nospace'] = n.replace(' ', '')
    out['ho'] = n + '호' if not n.endswith('호') else n[:-1]
    w = n.replace(' ', ''); p = rng.randrange(1, len(w)) if len(w) > 2 else 1
    out['badspace'] = w[:p] + ' ' + w[p:]
    out['mv'] = rng.choice(['엠브이 ', 'M/V ', '엠/브이 ']) + n
    if len(tk) == len(en) and len(tk) >= 2:
        j = rng.randrange(len(tk)); t2 = list(tk); t2[j] = en[j].upper(); out['mixed'] = ' '.join(t2)
    if len(tk) == len(en) and len(tk) >= 2:
        j = rng.randrange(len(tk)); t2 = list(tk); t2[j] = en[j]; out['mixed_lc'] = ' '.join(t2).lower()
    if re.search(r'\d', n):
        out['numko'] = re.sub(r'\d', lambda mm: NUMKO.get(mm.group(), mm.group()), n)
    elif tk:
        pass
    # jamo-level shortening: drop last syllable of each word
    if all(len(t) >= 3 for t in tk):
        out['shortsyl'] = ' '.join(t[:-1] for t in tk)
    # double vowel / spelling variants
    rep = [('케','캐'),('세','새'),('에','애'),('레','래'),('테','태'),('베','배'),('페','패'),('메','매'),('네','내'),('스','즈'),('트','드'),('프','쁘')]
    s2 = n
    for a, b in rng.sample(rep, len(rep)):
        if a in s2:
            s2 = s2.replace(a, b, 1); break
    if s2 != n: out['spellvar'] = s2
    return out
rng = random.Random(5)
res = []
rows = [r for r in TR[TR.fold == k].to_dict('records') if vm.is_hangul(r['vessel_name'])]
seen = set()
for row in rows:
    if row['vessel_name'] in seen: continue
    seen.add(row['vessel_name'])
    for tag, nm in [('orig', row['vessel_name'])] + list(variants(row, rng).items()):
        r2 = dict(row); r2['vessel_name'] = nm
        cands, X = M.features(r2)
        s = m.predict(X[:, cols], num_threads=2)
        o = np.argsort(-s)
        tgt = row['target_vessel_id']
        res.append(dict(tag=tag, name=nm, en=cname[tgt], ok=ix.ids[cands[o[0]]] == tgt, rec=tgt in set(ix.ids[cands]), pred=cname[ix.ids[cands[o[0]]]]))
d = pd.DataFrame(res); d.to_pickle('stress_new.pkl')
print(d.groupby('tag')[['ok', 'rec']].agg(['mean', 'size']).round(3))
for t in d.tag.unique():
    x = d[(d.tag == t) & ~d.ok].head(6)
    for _, r in x.iterrows(): print(t, '|', r['name'], '|', r.en, '| pred:', r.pred)
