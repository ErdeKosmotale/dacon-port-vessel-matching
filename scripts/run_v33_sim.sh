#!/bin/bash
cd /home/claude/work
rm -rf sim33 && mkdir -p sim33/data && cp -r submit_v33/* sim33/ && cp /home/claude/data/open/vessel_catalog.csv sim33/data/
cd sim33
cp /home/claude/work/sim4_test_real.csv data/test.csv; python3 script.py > ../sim33_run.log 2>&1; cp output/submission.csv ../sim33_full.csv
for v in shuf one sub; do cp /home/claude/work/sim4_$v.csv data/test.csv; python3 script.py > /dev/null 2>&1; cp output/submission.csv /home/claude/work/out33_$v.csv; done
cd /home/claude/work
python3 -c "
import pandas as pd
f=pd.read_csv('sim33_full.csv').set_index('sample_id').target_vessel_id
for v in ['shuf','one','sub']:
    o=pd.read_csv(f'out33_{v}.csv').set_index('sample_id').target_vessel_id
    print(v, len(o), 'identical:', bool((f.loc[o.index]==o).all()))
b=pd.read_csv('sim10_full.csv').set_index('sample_id').target_vessel_id if __import__('os').path.exists('sim10_full.csv') else None
if b is not None: print('agree with v10', (b.loc[f.index]==f).mean())
" > check33.log 2>&1
cd submit_v33 && find . -name "__pycache__" -prune -exec rm -rf {} \; ; mkdir -p ../out; rm -f ../out/submit_v33.zip && zip -r -q ../out/submit_v33.zip script.py requirements.txt model
echo ALLDONE >> /home/claude/work/check33.log
