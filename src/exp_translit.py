"""Name-only hangul ranking experiments (5-fold on 382 real hangul vessels)."""
import sys, importlib, numpy as np, pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
import vm, pipe
from harness import TR, CAT
modname = sys.argv[1]; tau = float(sys.argv[2]); iters = int(sys.argv[3]); lam = float(sys.argv[4])
ctx_final = sys.argv[5] == '1' if len(sys.argv) > 5 else True; ctx_init = sys.argv[6] == '1' if len(sys.argv) > 6 else False
T = importlib.import_module(modname)
if hasattr(T, 'CTX_FINAL'):
    T.CTX_FINAL = ctx_final; T.CTX_INITIAL = ctx_init
ix = pipe.CatalogIndex(CAT)
td = T.Transducer(CAT.vessel_name.tolist())
lm = T.LetterLM(td.letters); lmv = np.array([lm.logprob(s) for s in td.letters])
H = TR[TR.vessel_name.map(vm.is_hangul)].drop_duplicates('target_vessel_id')
def pairs_for(df):
    return [([T.text_units(a) for a in T.hangul_text_variants(r.vessel_name)], ix.id2i[r.target_vessel_id]) for r in df.itertuples()]
res = []
for K in range(5):
    td.P = td.P0.copy()
    trn = H[H.fold != K]; val = H[H.fold == K]
    td.fit(pairs_for(trn), iters=iters, tau=tau)
    for r in val.itertuples():
        q = pipe.Matcher(ix, None).query_info({'vessel_name': r.vessel_name})
        sk = ix.skel_sim_all(q['sk'])
        vk = process.cdist(q['vk'], ix.vkey, scorer=Levenshtein.normalized_similarity, dtype=np.float32, workers=-1).max(axis=0)
        first = sk + vk
        cand = np.argpartition(-first, 3000)[:3000]
        ti = ix.id2i[r.target_vessel_id]
        if ti not in set(cand.tolist()):
            cand = np.append(cand, ti)
        best = np.full(len(cand), -1e9); bestl = np.full(len(cand), -1e9)
        for a in T.hangul_text_variants(r.vessel_name):
            u = T.text_units(a)
            if not u: continue
            g, l = td.logprob_both(u, cand)
            best = np.maximum(best, g); bestl = np.maximum(bestl, l)
        pmi = best - lam * lmv[cand]
        pmil = bestl - lam * lmv[cand]
        rk = int((pmi > pmi[np.where(cand == ti)[0][0]]).sum())
        rkl = int((pmil > pmil[np.where(cand == ti)[0][0]]).sum())
        comb = np.maximum(pmi, pmil - 1.0)
        rkc = int((comb > comb[np.where(cand == ti)[0][0]]).sum())
        res.append((rk, rkl, rkc))
d = pd.DataFrame(res, columns=['g', 'l', 'c'])
print(f'{modname} tau={tau} iters={iters} lam={lam} ctxF={ctx_final} ctxI={ctx_init} | global top1 {(d.g==0).mean():.4f} top3 {(d.g<3).mean():.4f} | local top1 {(d.l==0).mean():.4f} | comb top1 {(d.c==0).mean():.4f} top3 {(d.c<3).mean():.4f}')
