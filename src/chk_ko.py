import pandas as pd, numpy as np, sys
from rapidfuzz.distance import Levenshtein
import vm
tr = pd.read_csv('/home/claude/data/open/train.csv', encoding='utf-8-sig')
cat = pd.read_csv('/home/claude/data/open/vessel_catalog.csv', encoding='utf-8-sig')
c = cat.set_index('vessel_id')
h = tr[tr.vessel_name.map(vm.is_hangul)].drop_duplicates('target_vessel_id').copy()
h['en'] = h.target_vessel_id.map(c.vessel_name)
rows=[]
for _,r in h.iterrows():
    ks,_ = vm.ko_variants(r.vessel_name)
    es = vm.en_skeletons(r.en)
    best = max(Levenshtein.normalized_similarity(a[0], b) for a in ks for b in es)
    rows.append((r.vessel_name, r.en, [k[0] for k in ks][:3], es, round(best,2)))
df = pd.DataFrame(rows, columns=['ko','en','kosk','ensk','sim'])
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',50); pd.set_option('display.max_rows',200)
print(df.sim.describe(), (df.sim==1).mean())
print(df.sort_values('sim').head(int(sys.argv[1]) if len(sys.argv)>1 else 40))
