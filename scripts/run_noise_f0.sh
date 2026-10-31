#!/bin/bash
# held-out evaluation: train.csv without fold 0 (transducer / stats / exclusion list), fold-0 rankers, fold-0 noise rows
# usage: run_noise_f0.sh <template dir> <tag>
set -e
cd /home/claude/work
T=$1; N=$2; D=noise_f0; S=nzf0_$N
rm -rf $S && cp -r $T $S && rm -f $S/model/ranker*.txt $S/model/*.features.json $S/model/eval_vessels.json
cp v31_f0.txt $S/model/ranker_01_HE.txt; cp ranker_v31.txt.features.json $S/model/ranker_01_HE.txt.features.json
cp v37A0_f0.txt $S/model/ranker_02_HE.txt; cp ranker_v33.txt.features.json $S/model/ranker_02_HE.txt.features.json
cp train_nofold0.csv $S/model/train.csv
mkdir -p $S/data && cp /home/claude/data/open/vessel_catalog.csv $S/data/ && cp $D/test.csv $S/data/test.csv
cd $S && python3 script.py > run.log 2>&1; tail -n 1 run.log
cd /home/claude/work && python3 - $S $D <<'PY'
import sys, pandas as pd
S, D = sys.argv[1], sys.argv[2]
p = pd.read_csv(f'{S}/output/submission.csv'); t = pd.read_csv(f'{D}/truth.csv')
m = t.merge(p, on='sample_id', suffixes=('', '_p')); m['ok'] = m.target_vessel_id == m.target_vessel_id_p
print(S, 'acc', round(m.ok.mean(), 4), '|', ' '.join(f'{k}/{"ton" if tn else "noton"}={g.ok.sum()}/{len(g)}' for (k, tn), g in m.groupby(['kind', 'ton'])))
PY
