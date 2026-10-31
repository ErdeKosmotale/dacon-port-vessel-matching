"""Generate hangul names for fresh (non-train) catalog vessels with a public Korean LLM (zero-shot), CPU."""
import pandas as pd, numpy as np, torch, time, json, sys, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from harness import TR, CAT
from synth import Synth
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained('/home/claude/models/kanana'); tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained('/home/claude/models/kanana', torch_dtype=torch.bfloat16).eval()
syn = Synth(TR, CAT)
c = CAT.set_index('vessel_id'); import vm
types = set(c.vessel_type.reindex(TR[TR.vessel_name.map(vm.is_hangul)].target_vessel_id))
pool = CAT[~CAT.vessel_id.isin(set(TR.target_vessel_id)) & CAT.vessel_type.isin(types) & CAT.vessel_type.isin(syn.type_map) & CAT.gross_tonnage.notna()]
rows = pool.sample(N, random_state=11)
def prompt(n):
    msgs = [{"role": "user", "content": f"다음 영어 선박 이름을 한국어 발음대로 한글로 적어 주세요. 한글 이름만 답하세요.\n선박 이름: {n}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
out = {}; t = time.time(); names = rows.vessel_name.tolist(); ids = rows.vessel_id.tolist()
for s in range(0, len(names), 8):
    batch = names[s:s + 8]
    enc = tok([prompt(n) for n in batch], return_tensors='pt', padding=True, add_special_tokens=False)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
    dec = tok.batch_decode(g[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
    for vid, n, d in zip(ids[s:s + 8], batch, dec):
        d = d.strip().split('\n')[0].strip().strip('"').strip("'").strip()
        out[vid] = (n, d)
    print(s + len(batch), round(time.time() - t), list(out.values())[-1], flush=True)
json.dump(out, open('llm_style_names.json', 'w'), ensure_ascii=False)
print('done', len(out))
