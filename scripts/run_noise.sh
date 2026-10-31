#!/bin/bash
# usage: run_noise.sh <template dir> <tag> <noise dir>  -> runs the template on the noise set with e8 rankers (HE), eval vessels made eligible
set -e
cd /home/claude/work
T=$1; N=$2; D=$3
S=nz_$N
rm -rf $S && cp -r $T $S && rm -f $S/model/ranker*.txt $S/model/*.features.json
i=1
for R in ranker_v31.txt ranker_v31_s5.txt ranker_v31_s7.txt ranker_v31_s9.txt ranker_v33.txt ranker_v33_s5.txt ranker_v33_s7.txt ranker_v33_s9.txt; do
  cp $R $S/model/ranker_0${i}_HE.txt; cp $R.features.json $S/model/ranker_0${i}_HE.txt.features.json; i=$((i+1))
done
cp $D/eval_vessels.json $S/model/
python3 - "$S/script.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = '    train_ids = set(train["target_vessel_id"])\n'
assert s.count(old) == 1
s = s.replace(old, old + '    _ev = BASE / "model" / "eval_vessels.json"\n    if _ev.exists():\n        train_ids -= set(json.load(open(_ev)))\n')
open(p, 'w').write(s)
PY
mkdir -p $S/data && cp /home/claude/data/open/vessel_catalog.csv $S/data/ && cp $D/test.csv $S/data/test.csv
cd $S && python3 script.py > run.log 2>&1; tail -n 1 run.log
cd /home/claude/work && python3 - $S $D <<'PY'
import sys, pandas as pd
S, D = sys.argv[1], sys.argv[2]
p = pd.read_csv(f'{S}/output/submission.csv'); t = pd.read_csv(f'{D}/truth.csv')
m = t.merge(p, on='sample_id', suffixes=('', '_p')); m['ok'] = m.target_vessel_id == m.target_vessel_id_p
print(S, 'acc', round(m.ok.mean(), 4), '|', ' '.join(f'{k}/{"ton" if tn else "noton"}={g.ok.sum()}/{len(g)}' for (k, tn), g in m.groupby(['kind', 'ton'])))
PY
