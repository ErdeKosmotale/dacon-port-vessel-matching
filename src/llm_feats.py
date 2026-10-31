"""Compute LLM log P(hangul | english) for stage-1 candidate pairs (cached, resumable)."""
import pickle, time, sys, os, numpy as np, pandas as pd
from llm_score import Scorer
from harness import CAT
items = pickle.load(open('stage1_hangul.pkl', 'rb'))
cat = CAT.set_index('vessel_id').vessel_name
cache_path = 'llm_cache.pkl'
cache = pickle.load(open(cache_path, 'rb')) if os.path.exists(cache_path) else {}
order = {'orig': 0, 'drop_ton': 1, 'mix': 2, 'abbr': 3, 'prefix': 4, 'trunc': 5}
items.sort(key=lambda o: order.get(o['tag'], 9))
pairs = []
for o in items:
    for vid in o['cands']:
        p = (cat[vid], o['qname'])
        if p not in cache:
            pairs.append(p)
pairs = list(dict.fromkeys(pairs))
print('items', len(items), 'pairs to score', len(pairs), 'cached', len(cache), flush=True)
S = Scorer('/home/claude/models/kanana')
t = time.time()
B = 200
for s in range(0, len(pairs), B):
    chunk = pairs[s:s + B]
    sc = S.score(chunk, batch_size=8)
    for p, v in zip(chunk, sc):
        cache[p] = float(v)
    pickle.dump(cache, open(cache_path + '.tmp', 'wb')); os.replace(cache_path + '.tmp', cache_path)
    done = s + len(chunk)
    el = time.time() - t
    print(f'{done}/{len(pairs)} {el/60:.1f} min, eta {el/done*(len(pairs)-done)/60:.1f} min', flush=True)
print('DONE', flush=True)
