"""English grapheme -> phoneme (CMUdict lookup + the g2p_en neural model, Apache-2.0).

Vendored so that no network / nltk data is needed at inference time.  Phonemes
(ARPAbet, stress stripped) are mapped to single symbols so the transducer code
can treat a pronunciation as a short string.
"""
from __future__ import annotations

import os
import re

import numpy as np

import vm

SYM = {"AA": "a", "AE": "@", "AH": "^", "AO": "c", "AW": "W", "AY": "Y", "B": "b", "CH": "C", "D": "d",
       "DH": "D", "EH": "e", "ER": "R", "EY": "A", "F": "f", "G": "g", "HH": "h", "IH": "i", "IY": "I",
       "JH": "J", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "N", "OW": "o", "OY": "O", "P": "p",
       "R": "r", "S": "s", "SH": "S", "T": "t", "TH": "T", "UH": "u", "UW": "U", "V": "v", "W": "w",
       "Y": "y", "Z": "z", "ZH": "Z"}

GRAPHEMES = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
PHONEMES = ["<pad>", "<unk>", "<s>", "</s>"] + [
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2',
    'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
    'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1',
    'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2',
    'V', 'W', 'Y', 'Z', 'ZH']


class G2P:
    def __init__(self, asset_dir: str):
        v = np.load(os.path.join(asset_dir, "checkpoint20.npz"))
        self.enc_emb, self.enc_w_ih, self.enc_w_hh = v["enc_emb"], v["enc_w_ih"], v["enc_w_hh"]
        self.enc_b_ih, self.enc_b_hh = v["enc_b_ih"], v["enc_b_hh"]
        self.dec_emb, self.dec_w_ih, self.dec_w_hh = v["dec_emb"], v["dec_w_ih"], v["dec_w_hh"]
        self.dec_b_ih, self.dec_b_hh, self.fc_w, self.fc_b = v["dec_b_ih"], v["dec_b_hh"], v["fc_w"], v["fc_b"]
        self.g2idx = {g: i for i, g in enumerate(GRAPHEMES)}
        self.cmu = {}
        with open(os.path.join(asset_dir, "cmudict.dict"), encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                w = re.sub(r"\(\d+\)$", "", parts[0])
                if w not in self.cmu:
                    self.cmu[w] = parts[1:]
        self.cache = {}

    @staticmethod
    def _sig(x):
        return 1 / (1 + np.exp(-x))

    def _cell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = np.matmul(x, w_ih.T) + b_ih
        rzn_hh = np.matmul(h, w_hh.T) + b_hh
        k = rzn_ih.shape[-1] * 2 // 3
        rz = self._sig(rzn_ih[:, :k] + rzn_hh[:, :k])
        r, z = np.split(rz, 2, -1)
        n = np.tanh(rzn_ih[:, k:] + r * rzn_hh[:, k:])
        return (1 - z) * n + z * h

    def predict(self, word: str) -> list[str]:
        x = [self.g2idx.get(c, 1) for c in list(word) + ["</s>"]]
        x = np.take(self.enc_emb, np.expand_dims(x, 0), axis=0)
        h = np.zeros((1, self.enc_w_hh.shape[1]), np.float32)
        for t in range(x.shape[1]):
            h = self._cell(x[:, t, :], h, self.enc_w_ih, self.enc_w_hh, self.enc_b_ih, self.enc_b_hh)
        dec = np.take(self.dec_emb, [2], axis=0)
        preds = []
        for _ in range(20):
            h = self._cell(dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh)
            logits = np.matmul(h, self.fc_w.T) + self.fc_b
            p = int(logits.argmax())
            if p == 3:
                break
            preds.append(PHONEMES[p])
            dec = np.take(self.dec_emb, [p], axis=0)
        return preds

    def word(self, w: str) -> str:
        """lower-case letters-only word -> symbol string."""
        w = re.sub(r"[^a-z]", "", w.lower())
        if not w:
            return ""
        if w in self.cache:
            return self.cache[w]
        ph = self.cmu.get(w) or self.predict(w)
        s = "".join(SYM.get(re.sub(r"\d", "", p), "") for p in ph)
        self.cache[w] = s
        return s

    def letters(self, w: str) -> str:
        return "".join(self.word(c) for c in w.lower())

    def name(self, name) -> str:
        """Catalog english name -> concatenated symbol string (words only; acronyms as letter names)."""
        words, _ = vm.en_split(name)
        out = []
        for tok in str(name).split():
            pass
        for w in words:
            if len(w) <= 5 and sum(c in "AEIOU" for c in w) == 0:
                out.append(self.letters(w))
            elif w.isupper() and len(w) <= 4 and str(name).find(w) >= 0 and not (len(w) >= 3 and sum(c in "AEIOU" for c in w) >= 2):
                out.append(self.letters(w))
            else:
                out.append(self.word(w))
        return "".join(out)
