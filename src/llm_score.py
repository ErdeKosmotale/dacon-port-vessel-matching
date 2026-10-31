"""Discriminative LLM scoring: log P(hangul spelling | english name) with a causal LM."""
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

class Scorer:
    def __init__(self, path, device=None, dtype=None, threads=2):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = dtype or (torch.float16 if self.device == 'cuda' else torch.bfloat16)
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, low_cpu_mem_usage=True).to(self.device).eval()
        self.prefix = '선박 제원 카탈로그의 영문 선박명과 항만 입출항 기록에 적힌 한글 선박명입니다.\n'

    def pair_ids(self, en, ko):
        a = self.tok(self.prefix + f'영문: {en}\n한글:', add_special_tokens=True)['input_ids']
        b = self.tok(f' {ko}\n', add_special_tokens=False)['input_ids']
        return a, b

    @torch.no_grad()
    def score(self, pairs, batch_size=16, fixed_len=None):
        """pairs: list of (en, ko) -> np.array of sum log P(ko | en).

        Sequences are right-padded to a fixed length so that a pair's score does
        not depend on which other pairs share its batch."""
        out = np.zeros(len(pairs))
        items = [(i,) + self.pair_ids(en, ko) for i, (en, ko) in enumerate(pairs)]
        items.sort(key=lambda x: len(x[1]) + len(x[2]))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        for s in range(0, len(items), batch_size):
            chunk = items[s:s + batch_size]
            L = max(len(a) + len(b) for _, a, b in chunk)
            if fixed_len:
                L = max(L, fixed_len)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            spans = []
            for r, (_, a, b) in enumerate(chunk):
                seq = a + b
                ids[r, :len(seq)] = torch.tensor(seq); att[r, :len(seq)] = 1
                spans.append((len(a), len(seq)))
            logits = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device)).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = ids[:, 1:].to(self.device)
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu().numpy()
            for r, (i, a, b) in enumerate(chunk):
                st, en_ = spans[r]
                out[i] = tok_lp[r, st - 1:en_ - 1].sum()
        return out
