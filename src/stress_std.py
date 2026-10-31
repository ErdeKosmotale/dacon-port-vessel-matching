"""Fresh (non-train) catalog vessels, record-style queries, hangul names from a DIFFERENT transliteration process.
Submission setting: v18 ranker, transducer/stats fit on all train, train-vessel exclusion."""
import pandas as pd, numpy as np, random, json, sys, lightgbm as lgb
import vm, pipe11 as P, translit as T, gen
from synth import Synth
from stdko import StdKo
from harness import TR, CAT
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
ix = P.CatalogIndex(CAT); td = T.Transducer(CAT.vessel_name.tolist()); lmv = np.load('lmv.npy'); P.fit_transducer(td, TR, ix)
st = P.TrainStats(TR, CAT); M = P.Matcher11(ix, st, td, lmv)
model = lgb.Booster(model_file='ranker_v18.txt'); fn = json.load(open('ranker_v18.txt.features.json')); cols = [P.FEATURES7.index(f) for f in fn]
syn = Synth(TR, CAT); S = StdKo('model_assets')
G = gen.KoGenerator(td, TR[TR.vessel_name.map(vm.is_hangul)].vessel_name.tolist())
excl = set(TR.target_vessel_id)
# hangul-typical population: types/flags seen with hangul rows in train
c = CAT.set_index('vessel_id'); ht = TR[TR.vessel_name.map(vm.is_hangul)]
types = set(c.vessel_type.reindex(ht.target_vessel_id)); 
pool = CAT[~CAT.vessel_id.isin(excl) & CAT.vessel_type.isin(types) & CAT.vessel_type.isin(syn.type_map) & CAT.gross_tonnage.notna()]
rows = pool.sample(N, random_state=7).to_dict('records')
rng = random.Random(9); res = []
for cr in rows:
    q = syn.query_from_catalog(cr, rng)
    if q is None: continue
    q['gross_tonnage'] = cr['gross_tonnage']; q['international_gross_tonnage'] = cr['gross_tonnage']
    names = {'gen_trainstyle': G.generate(cr['vessel_name']), 'std': S.name(cr['vessel_name'])}
    for tag, nm in names.items():
        if not nm or not vm.is_hangul(nm): continue
        q2 = dict(q); q2['vessel_name'] = nm
        cands, X = M.features(q2); s = model.predict(X[:, cols], num_threads=2)
        ids = ix.ids[cands]; s2 = np.where([x in excl for x in ids], s - 1e6, s); p = int(np.argmax(s2))
        res.append(dict(tag=tag, en=cr['vessel_name'], ko=nm, ok=ids[p] == cr['vessel_id'], rec=cr['vessel_id'] in set(ids), pred=CAT.vessel_name.iloc[ix.id2i[ids[p]]]))
d = pd.DataFrame(res); d.to_pickle('stress_std.pkl')
print(d.groupby('tag').agg(acc=('ok', 'mean'), rec=('rec', 'mean'), n=('ok', 'size')).round(3))
pd.set_option('display.width', 220)
print(d[(d.tag == 'std') & ~d.ok].head(30).to_string())
