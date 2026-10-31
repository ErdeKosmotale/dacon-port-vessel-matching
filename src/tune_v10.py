import pandas as pd, numpy as np, pickle, time, json, sys, optuna, lightgbm as lgb
import pipe7
FN = pipe7.FEATURES7
drop = {'rom_rank', 'rom_gap_best', 'rom_sim', 'rom_partial'}
cols = [FN.index(f) for f in FN if f not in drop]
feats = pickle.load(open('feats_v7_a.pkl', 'rb')) + pickle.load(open('feats_v7_b.pkl', 'rb'))
TM = FN.index('q_ton_missing')
def split(k):
    trn = [it for it in feats if it[1] != k]; val = [it for it in feats if it[1] == k]
    X = np.concatenate([it[6][:, cols] for it in trn]); y = np.concatenate([it[7] for it in trn]); g = [len(it[7]) for it in trn]
    return X, y, g, val
def score(val, pr):
    rows = []
    for it in val:
        tm = bool(it[6][0, TM])
        if it[3] and tm:      # test hangul rows always have tonnage
            continue
        rows.append((it[3], it[4] in ('orig', 'synko'), bool(it[7][int(np.argmax(pr(it[6][:, cols])))])))
    d = pd.DataFrame(rows, columns=['h', 'clean', 'ok'])
    a = lambda h, c: d[(d.h == h) & (d.clean == c)].ok.mean()
    H = 0.5 * a(True, True) + 0.5 * a(True, False)
    E = 0.78 * a(False, True) + 0.22 * a(False, False)
    return 0.7 * H + 0.3 * E, H, E
DATA = {}
def get(k):
    if k not in DATA: DATA[k] = split(k)
    return DATA[k]
def run(p, k):
    X, y, g, val = get(k)
    params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], verbose=-1, num_threads=2, seed=42,
                  learning_rate=p['lr'], num_leaves=p['leaves'], min_data_in_leaf=p['mdl'], feature_fraction=p['ff'],
                  bagging_fraction=p['bf'], bagging_freq=1, lambda_l2=p['l2'], lambdarank_truncation_level=p['trunc'],
                  max_depth=p['depth'], min_sum_hessian_in_leaf=1e-3)
    m = lgb.train(params, lgb.Dataset(X, y, group=g), num_boost_round=p['rounds'])
    return score(val, lambda Z: m.predict(Z, num_threads=2))
base = dict(lr=0.05, leaves=63, mdl=20, ff=0.8, bf=0.8, l2=0.0, trunc=10, depth=-1, rounds=500)
log = open('tune_v10.log', 'a')
t0 = time.time()
s = run(base, 0); print('BASE fold0', [round(x, 4) for x in s], round(time.time() - t0), file=log, flush=True)
def objective(trial):
    p = dict(lr=trial.suggest_float('lr', 0.02, 0.15, log=True), leaves=trial.suggest_int('leaves', 15, 255, log=True),
             mdl=trial.suggest_int('mdl', 5, 200, log=True), ff=trial.suggest_float('ff', 0.4, 1.0), bf=trial.suggest_float('bf', 0.5, 1.0),
             l2=trial.suggest_float('l2', 1e-3, 30, log=True), trunc=trial.suggest_int('trunc', 3, 40),
             depth=trial.suggest_categorical('depth', [-1, 6, 8, 12]), rounds=trial.suggest_int('rounds', 200, 1200, step=100))
    s = run(p, 0)
    print(trial.number, [round(x, 4) for x in s], json.dumps(p), round(time.time() - t0), file=log, flush=True)
    return s[0]
st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
st.enqueue_trial(dict(lr=0.05, leaves=63, mdl=20, ff=0.8, bf=0.8, l2=1e-3, trunc=10, depth=-1, rounds=500))
st.optimize(objective, n_trials=int(sys.argv[1]), timeout=float(sys.argv[2]))
json.dump(st.best_params, open('tune_v10_best.json', 'w'))
print('BEST', st.best_value, st.best_params, file=log, flush=True)
# confirm on fold 1 & 2 vs base
bp = st.best_params
for k in (1, 2):
    print('CONFIRM fold', k, 'base', [round(x, 4) for x in run(base, k)], 'best', [round(x, 4) for x in run(bp, k)], file=log, flush=True)
print('DONE', file=log, flush=True)
