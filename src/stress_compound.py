"""Broad recoverable-damage stress test for v18, 5 folds, held-out hangul vessels, train-vessel exclusion."""
import pandas as pd, numpy as np, random, pickle, gc, json, sys, lightgbm as lgb
import vm, pipe11 as P, translit as T
from harness import TR, CAT
FN = P.FEATURES7; fn = json.load(open('ranker_v18.txt.features.json')); cols = [FN.index(f) for f in fn]

params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=2, seed=42)
CHO, JUNG, JONG = T.CHO, T.JUNG, T.JONG
def dec(c): o = ord(c) - 0xAC00; return o // 588, (o % 588) // 28, o % 28
def enc(a, b, c): return chr(0xAC00 + a * 588 + b * 28 + c)
def ish(c): return '가' <= c <= '힣'
R1 = list('ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔ'); R2 = list('ㅁㄴㅇㄹㅎㅗㅓㅏㅣ'); R3 = list('ㅋㅌㅊㅍㅠㅜㅡ'); ROWS = [R1, R2, R3]
NB = {}
for ri, row in enumerate(ROWS):
    for ci, k in enumerate(row):
        s = set()
        for dc in (-1, 1):
            if 0 <= ci + dc < len(row): s.add(row[ci + dc])
        for dr in (-1, 1):
            if 0 <= ri + dr < 3:
                for dc in (0, 1, -1):
                    if 0 <= ci + dc < len(ROWS[ri + dr]): s.add(ROWS[ri + dr][ci + dc])
        NB[k] = s
def sub_jamo(ch, rng, mode):
    a, b, c = dec(ch); w = rng.random()
    if w < 0.45:   # onset
        cand = [x for x in (NB[CHO[a]] if mode == 'kbd' and CHO[a] in NB else CHO) if x in CHO and x != CHO[a]]
        if cand: a = CHO.index(rng.choice(cand))
    elif w < 0.9:  # vowel
        cand = [x for x in (NB[JUNG[b]] if mode == 'kbd' and JUNG[b] in NB else JUNG) if x in JUNG and x != JUNG[b]]
        if cand: b = JUNG.index(rng.choice(cand))
    else:          # coda
        cur = JONG[c]
        cand = [x for x in (NB[cur] if (mode == 'kbd' and cur in NB) else JONG[1:]) if x in JONG and x != cur] if cur else ['ㄴ', 'ㄹ', 'ㅇ', 'ㅁ']
        if cand: c = JONG.index(rng.choice(cand))
    return enc(a, b, c)
def typo(name, rng, k, mode):
    s = list(name); idx = [i for i, ch in enumerate(s) if ish(ch)]
    for i in rng.sample(idx, min(k, len(idx))): s[i] = sub_jamo(s[i], rng, mode)
    return ''.join(s)
def syl_del(name, rng, k):
    s = list(name); idx = [i for i, ch in enumerate(s) if ish(ch)]
    if len(idx) < k + 3: return None
    for i in sorted(rng.sample(idx, k), reverse=True): del s[i]
    return ''.join(s)
def syl_ins(name, rng, k):
    s = list(name)
    for _ in range(k):
        idx = [i for i, ch in enumerate(s) if ish(ch)]; i = rng.choice(idx)
        s.insert(i + 1, enc(rng.randrange(19), rng.randrange(21), rng.choice([0, 0, 4, 8, 21])))
    return ''.join(s)
def syl_swap(name, rng):
    s = list(name); idx = [i for i in range(len(s) - 1) if ish(s[i]) and ish(s[i + 1])]
    if not idx: return None
    i = rng.choice(idx); s[i], s[i + 1] = s[i + 1], s[i]; return ''.join(s)
def drop_mid(name, rng):
    t = name.split(); return ' '.join(t[:1] + t[2:]) if len(t) >= 3 else None
def trunc_end(name, frac):
    s = name; n = sum(ish(c) for c in s)
    if n < 4: return None
    keep = max(2, int(round(n * frac))); out, cnt = [], 0
    for ch in s:
        if ish(ch): cnt += 1
        if cnt > keep: break
        out.append(ch)
    return ''.join(out).strip()
def trunc_front(name, frac):
    r = trunc_end(name[::-1], frac); return r[::-1].strip() if r else None
TRANS = {'OCEAN': '대양', 'SEA': '바다', 'STAR': '별', 'SUN': '태양', 'MOON': '달', 'PACIFIC': '태평양', 'ATLANTIC': '대서양', 'GLORY': '영광',
         'QUEEN': '여왕', 'KING': '왕', 'PRINCESS': '공주', 'PRINCE': '왕자', 'BRIDGE': '다리', 'PIONEER': '개척자', 'EXPRESS': '급행', 'LEADER': '지도자',
         'DREAM': '꿈', 'HOPE': '희망', 'VICTORY': '승리', 'SPIRIT': '정신', 'MORNING': '아침', 'GOLDEN': '황금', 'GOLD': '금', 'SILVER': '은',
         'PEARL': '진주', 'CRYSTAL': '수정', 'EAGLE': '독수리', 'LION': '사자', 'TIGER': '호랑이', 'DRAGON': '용', 'ISLAND': '섬', 'RIVER': '강',
         'HARMONY': '조화', 'FORTUNE': '행운', 'GRACE': '은혜', 'JOY': '기쁨', 'PEACE': '평화', 'LIBERTY': '자유', 'COURAGE': '용기', 'TREASURE': '보물',
         'NAVIGATOR': '항해사', 'TRADER': '무역상', 'EXPLORER': '탐험가', 'VOYAGER': '항해자', 'WIND': '바람', 'SKY': '하늘', 'BLUE': '푸른', 'GREEN': '녹색',
         'GRAND': '대', 'GREAT': '위대한', 'NEW': '새', 'HAPPY': '행복', 'BRIGHT': '밝은', 'GLOBAL': '세계', 'WORLD': '세계', 'EAST': '동', 'WEST': '서',
         'NORTH': '북', 'SOUTH': '남', 'FUTURE': '미래', 'FLOWER': '꽃', 'ROSE': '장미', 'GALAXY': '은하', 'HORIZON': '수평선', 'SUNSHINE': '햇살',
         'SUNNY': '화창한', 'LUCKY': '행운', 'RICH': '부유한', 'UNITY': '단결', 'WISDOM': '지혜', 'TRUST': '신뢰', 'SAFETY': '안전', 'SUPERIOR': '우수한',
         'CHAMPION': '챔피언', 'CHALLENGE': '도전', 'COSMOS': '우주', 'DIAMOND': '다이아몬드', 'JADE': '옥', 'ACE': '에이스', 'CROWN': '왕관', 'BIRD': '새', 'SPRING': '봄'}
COMP = ['현대', '팬오션', '에이치엠엠', '고려', '흥아', '장금', '대한', '한진']
def translate(name, en):
    tk = name.split(); ew = vm.en_split(en)[0]
    if len(tk) != len(ew) or len(tk) < 2: return None
    js = [j for j, w in enumerate(ew) if w.upper() in TRANS]
    if not js: return None
    j = js[0]; t2 = list(tk); t2[j] = TRANS[ew[j].upper()]; return ' '.join(t2)
def variants_single(name, en, rng):
    v = {'orig': name}
    for k in (1, 2, 3):
        v[f'kbd{k}'] = typo(name, rng, k, 'kbd'); v[f'any{k}'] = typo(name, rng, k, 'any')
    v['sdel1'] = syl_del(name, rng, 1); v['sdel2'] = syl_del(name, rng, 2); v['sins1'] = syl_ins(name, rng, 1); v['sins2'] = syl_ins(name, rng, 2)
    v['sswap'] = syl_swap(name, rng); v['dropmid'] = drop_mid(name, rng)
    v['tr_end80'] = trunc_end(name, 0.8); v['tr_end60'] = trunc_end(name, 0.6); v['tr_front80'] = trunc_front(name, 0.8)
    v['translate'] = translate(name, en); v['add_comp'] = rng.choice(COMP) + ' ' + name
    v['kbd1_sdel1'] = syl_del(typo(name, rng, 1, 'kbd'), rng, 1)
    return {k: x for k, x in v.items() if x and (k == 'orig' or x != name)}
import aug
from synth import Synth
def variants(name, en, rng):
    v = {'orig': name}
    ab = aug.abbreviate(name, rng); tr_ = trunc_end(name, 0.7); tk = name.split()
    v['abbr'] = ab
    v['abbr+kbd1'] = typo(ab, rng, 1, 'kbd') if ab else None
    v['abbr+kbd2'] = typo(ab, rng, 2, 'kbd') if ab else None
    v['trunc70'] = tr_
    v['trunc70+kbd1'] = typo(tr_, rng, 1, 'kbd') if tr_ else None
    v['swap'] = ' '.join(reversed(tk)) if len(tk) >= 2 else None
    v['swap+kbd1'] = typo(v['swap'], rng, 1, 'kbd') if v['swap'] else None
    v['kbd2+sdel1'] = syl_del(typo(name, rng, 2, 'kbd'), rng, 1)
    v['abbr+sdel1'] = syl_del(ab, rng, 1) if ab else None
    v['nospace+kbd2'] = typo(name.replace(' ', ''), rng, 2, 'kbd')
    return {k: x for k, x in v.items() if x and (k == 'orig' or x != name)}
cname = dict(zip(CAT.vessel_id, CAT.vessel_name)); ix = P.CatalogIndex(CAT); lmv = np.load('lmv.npy')
res = []
FOLDS = [int(x) for x in sys.argv[1].split(',')]; OUT = sys.argv[2]
for k in FOLDS:
    m = lgb.Booster(model_file=f'ranker_v18_f{k}.txt')
    trk = TR[TR.fold != k]; td = T.Transducer(CAT.vessel_name.tolist()); st = P.TrainStats(trk, CAT); P.fit_transducer(td, trk, ix)
    M = P.Matcher11(ix, st, td, lmv); excl = set(trk.target_vessel_id)
    rng = random.Random(100 + k)
    rows = TR[(TR.fold == k) & TR.vessel_name.map(vm.is_hangul)].drop_duplicates('target_vessel_id').to_dict('records')
    for row in rows:
        for tag, nm in variants(row['vessel_name'], cname[row['target_vessel_id']], rng).items():
            r2 = dict(row); r2['vessel_name'] = nm
            cands, Xq = M.features(r2); s = m.predict(Xq[:, cols], num_threads=2)
            ids = ix.ids[cands]; s2 = np.where([c in excl for c in ids], s - 1e6, s); p = int(np.argmax(s2))
            res.append(dict(fold=k, tag=tag, name=nm, orig=row['vessel_name'], en=cname[row['target_vessel_id']], ok=ids[p] == row['target_vessel_id'],
                            rec=row['target_vessel_id'] in set(ids), pred=cname[ids[p]]))
    print('fold', k, len(res), flush=True)
    pd.DataFrame(res).to_pickle(OUT)
d = pd.DataFrame(res)
print(d.groupby('tag').agg(acc=('ok', 'mean'), rec=('rec', 'mean'), n=('ok', 'size')).sort_values('acc').round(3).to_string())
