#!/bin/bash
cd /home/claude/work
rm -rf sim14 && mkdir -p sim14/data && cp -r submit_v14/* sim14/ && cp /home/claude/data/open/vessel_catalog.csv sim14/data/
cd sim14
cp /home/claude/work/sim4_test_real.csv data/test.csv; python3 script.py > ../sim14_run.log 2>&1; cp output/submission.csv ../sim14_full.csv
for v in shuf one sub; do cp /home/claude/work/sim4_$v.csv data/test.csv; python3 script.py > /dev/null 2>&1; cp output/submission.csv /home/claude/work/out14_$v.csv; done
cd /home/claude/work
python3 -c "
import pandas as pd
f=pd.read_csv('sim14_full.csv').set_index('sample_id').target_vessel_id
for v in ['shuf','one','sub']:
    o=pd.read_csv(f'out14_{v}.csv').set_index('sample_id').target_vessel_id
    print(v, len(o), 'identical:', bool((f.loc[o.index]==o).all()))
b=pd.read_csv('sim10_full.csv').set_index('sample_id').target_vessel_id if __import__('os').path.exists('sim10_full.csv') else None
if b is not None: print('agree with v10', (b.loc[f.index]==f).mean())
" > check14.log 2>&1
cd submit_v14 && find . -name "__pycache__" -prune -exec rm -rf {} \; ; mkdir -p ../out; rm -f ../out/submit_v14.zip && zip -r -q ../out/submit_v14.zip script.py requirements.txt model
echo ALLDONE >> /home/claude/work/check14.log
