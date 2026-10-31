import pandas as pd, numpy as np, json, random, lightgbm as lgb
import vm, pipe11 as P, translit as T
from synth import Synth
from pinyin_ko import is_pinyin_name
from harness import TR, CAT
ix = P.CatalogIndex(CAT); td = T.Transducer(CAT.vessel_name.tolist()); lmv = np.load('lmv.npy'); P.fit_transducer(td, TR, ix)
st = P.TrainStats(TR, CAT); M = P.Matcher11(ix, st, td, lmv)
model = lgb.Booster(model_file='ranker_v18.txt'); fn = json.load(open('ranker_v18.txt.features.json')); cols = [P.FEATURES7.index(f) for f in fn]
syn = Synth(TR, CAT); excl = set(TR.target_vessel_id); c = CAT.set_index('vessel_id'); rng = random.Random(9)
names = json.load(open('llm_style_names.json')); res = []
for vid, (en, ko) in names.items():
    cr = c.loc[vid].to_dict(); cr['vessel_id'] = vid
    q = syn.query_from_catalog(cr, rng)
    if q is None: continue
    q['gross_tonnage'] = cr['gross_tonnage']; q['international_gross_tonnage'] = cr['gross_tonnage']; q['vessel_name'] = ko
    cands, X = M.features(q); s = model.predict(X[:, cols], num_threads=2); ids = ix.ids[cands]
    p = int(np.argmax(np.where([x in excl for x in ids], s - 1e6, s)))
    res.append(dict(en=en, ko=ko, hang=vm.is_hangul(ko), pinyin=is_pinyin_name(en), ok=ids[p] == vid, rec=vid in set(ids), pred=CAT.vessel_name.iloc[ix.id2i[ids[p]]]))
d = pd.DataFrame(res); d.to_pickle('eval_llm_style.pkl')
print('all', len(d), 'acc', round(d.ok.mean(), 3))
print(d.groupby(['hang', 'pinyin']).agg(n=('ok', 'size'), acc=('ok', 'mean'), rec=('rec', 'mean')).round(3))
pd.set_option('display.width', 200)
print(d[~d.ok][['en', 'ko', 'pinyin', 'pred']].to_string())
