#!/bin/bash
cd /home/claude/work
./run_v10f.sh
python3 train_final7d.py feats_v10_a.pkl,feats_v10_b.pkl ranker_v13.txt 500 rom_rank,rom_gap_best,rom_sim,rom_partial > final_v13.log 2>&1
rm -rf submit_v13 && cp -r submit_v10 submit_v13 && cp ranker_v13.txt submit_v13/model/ranker.txt && cp ranker_v13.txt.features.json submit_v13/model/features.json
sed -i 's/v10 - inference entry point./v13 - inference entry point./' submit_v13/script.py
rm -rf sim13 && mkdir -p sim13/data && cp -r submit_v13/* sim13/ && cp /home/claude/data/open/vessel_catalog.csv sim13/data/
cd sim13
cp /home/claude/work/sim4_test_real.csv data/test.csv; python3 script.py > ../sim13_run.log 2>&1; cp output/submission.csv ../sim13_full.csv
for v in shuf one sub; do cp /home/claude/work/sim4_$v.csv data/test.csv; python3 script.py > /dev/null 2>&1; cp output/submission.csv /home/claude/work/out13_$v.csv; done
cd /home/claude/work
python3 -c "
import pandas as pd
f=pd.read_csv('sim13_full.csv').set_index('sample_id').target_vessel_id
for v in ['shuf','one','sub']:
    o=pd.read_csv(f'out13_{v}.csv').set_index('sample_id').target_vessel_id
    print(v, len(o), 'identical:', bool((f.loc[o.index]==o).all()))
" > check13.log 2>&1
cd submit_v13 && find . -name "__pycache__" -prune -exec rm -rf {} \; ; rm -f ../out/submit_v13.zip && zip -r -q ../out/submit_v13.zip script.py requirements.txt model
echo ALLDONE >> /home/claude/work/check13.log
