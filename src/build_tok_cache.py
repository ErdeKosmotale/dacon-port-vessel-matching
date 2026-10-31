"""Hangul spelling (standard-orthography engine) for every distinct English token in the catalog -> tok_ko.json"""
import json, time, pandas as pd
import vm
from stdko import StdKo
from harness import CAT
S = StdKo('model_assets')
toks = set()
for n in CAT.vessel_name.astype(str):
    toks.update(vm.en_split(n)[0])
toks = sorted(toks); print('unique tokens', len(toks), flush=True)
out = {}; t = time.time()
for i, w in enumerate(toks):
    try:
        out[w] = S.word(w)
    except Exception:
        out[w] = ''
    if i % 10000 == 0: print(i, round(time.time() - t), flush=True)
json.dump(out, open('tok_ko.json', 'w'), ensure_ascii=False)
print('saved', len(out), round(time.time() - t))
