import pandas as pd, numpy as np, time, sys
import vm, pipe
tr = pd.read_csv('/home/claude/data/open/train.csv', encoding='utf-8-sig')
cat = pd.read_csv('/home/claude/data/open/vessel_catalog.csv', encoding='utf-8-sig')
t=time.time(); ix = pipe.CatalogIndex(cat); print('index', time.time()-t)
st = pipe.TrainStats(tr, cat)
M = pipe.Matcher(ix, st)
tr['hang']=tr.vessel_name.map(vm.is_hangul)
samp = pd.concat([tr[tr.hang].drop_duplicates('target_vessel_id'), tr[~tr.hang].drop_duplicates('target_vessel_id').sample(400, random_state=0)])
res=[]
t=time.time()
for row in samp.to_dict('records'):
    q = M.query_info(row)
    cands, nsim, ret, gap, ts = M.candidates(q)
    ti = ix.id2i[row['target_vessel_id']]
    pos = np.where(cands==ti)[0]
    rr = int((ret > ret[ti]).sum())
    res.append((row['hang'], len(cands), pos[0] if len(pos) else -1, rr, nsim[ti]))
print('time/row', (time.time()-t)/len(samp))
r = pd.DataFrame(res, columns=['hang','nc','pos','retrank','nsim'])
for h,g in r.groupby('hang'):
    print('hang' if h else 'en', 'recall', (g.pos>=0).mean(), 'top1 by ret', (g.retrank==0).mean(), 'ncand', g.nc.mean(), 'retrank<5', (g.retrank<5).mean())
