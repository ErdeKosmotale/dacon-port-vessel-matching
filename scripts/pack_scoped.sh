#!/bin/bash
# usage: pack_scoped.sh <template dir> <name> <full|smoke> <src ranker>:<dest name> ...   (dest e.g. ranker_01_HE.txt)
set -e
cd /home/claude/work
T=$1; N=$2; MODE=$3; shift 3
rm -rf submit_$N && cp -r $T submit_$N && rm -f submit_$N/model/ranker*.txt submit_$N/model/*.features.json submit_$N/model/features.json
for P in "$@"; do
  S=${P%%:*}; D=${P##*:}
  cp $S submit_$N/model/$D && cp $S.features.json submit_$N/model/$D.features.json
done
find submit_$N -name "__pycache__" -prune -exec rm -rf {} \;
rm -rf sim_$N && mkdir -p sim_$N/data && cp -r submit_$N/* sim_$N/ && cp /home/claude/data/open/vessel_catalog.csv sim_$N/data/
cd sim_$N
if [ "$MODE" = full ]; then
  cp /home/claude/work/sim4_test_real.csv data/test.csv; python3 script.py > ../sim_${N}_run.log 2>&1; cp output/submission.csv ../sim_${N}_full.csv
  for v in shuf one sub; do cp /home/claude/work/sim4_$v.csv data/test.csv; python3 script.py > /dev/null 2>&1; cp output/submission.csv /home/claude/work/out_${N}_$v.csv; done
  cd /home/claude/work
  python3 -c "
import pandas as pd
f=pd.read_csv('sim_${N}_full.csv').set_index('sample_id').target_vessel_id
for v in ['shuf','one','sub']:
    o=pd.read_csv(f'out_${N}_{v}.csv').set_index('sample_id').target_vessel_id
    print(v, len(o), 'identical:', bool((f.loc[o.index]==o).all()))
" > check_${N}.log 2>&1
else
  head -61 /home/claude/work/sim4_test_real.csv > data/test.csv; python3 script.py > ../sim_${N}_run.log 2>&1; cp output/submission.csv ../sim_${N}_smoke.csv
  cd /home/claude/work; echo "smoke rows: $(($(wc -l < sim_${N}_smoke.csv)-1))" > check_${N}.log
fi
cd /home/claude/work/submit_$N && mkdir -p ../out && rm -f ../out/submit_$N.zip && zip -r -q ../out/submit_$N.zip script.py requirements.txt model
echo ALLDONE >> /home/claude/work/check_${N}.log
