"""Stage-2: hangul-only reranker over the stage-1 top-K candidates inside the tonnage window.
Stage-1 = v18 ranker (per fold for CV). Stage-2 features = stage-1 score context + a compact subset of
candidate features, relative to the best candidate in the window."""
import pandas as pd, numpy as np, pickle, gc, json, sys, lightgbm as lgb
import pipe11 as P
FN = P.FEATURES7; drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
fn = [f for f in FN if f not in drop]; cols = [FN.index(f) for f in fn]
F = {f: i for i, f in enumerate(fn)}
K2 = 8
S2 = ['gap_min', 'exact_gt', 'exact_igt', 'type_prob', 'flag_prob', 't_pmi_rel', 't_loc_rel', 't_red_rel', 't_pmi_rank', 't_loc_rank',
      'sk_sim', 'vk_sim', 'sk_partial', 'num_mismatch', 'num_match', 'n_name_ties', 'name_rank', 'ret_rank', 'struct_rank', 'c_len', 'q_len',
      't_margin', 't_pmi_minus_2nd', 'best_name', 'name_gap_best']
def s2_feats(s1, X):
    """s1: stage-1 scores of the K2 candidates (sorted desc), X: their feature rows (fn columns)."""
    n = len(s1); rows = []
    for i in range(n):
        base = [s1[i] - s1[0], i, s1[i] - (s1[1] if n > 1 else s1[0]), n]
        rel = [X[i, F[f]] for f in S2]
        rel2 = [X[i, F[f]] - X[0, F[f]] for f in ['gap_min', 'type_prob', 'flag_prob', 't_pmi_rel', 't_loc_rel', 'sk_sim', 'vk_sim']]
        rows.append(base + rel + rel2)
    return np.asarray(rows, np.float32)
X = np.load('v18_X.npy', mmap_mode='r') if __import__('os').path.exists('v18_X.npy') else None
