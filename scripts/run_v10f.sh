#!/bin/bash
cd /home/claude/work
python3 build_feats10.py feats_v10_a.pkl 0,1,2 1000 > build_v10a.log 2>&1 &
P1=$!
python3 build_feats10.py feats_v10_b.pkl 3,4 1000 > build_v10b.log 2>&1 &
P2=$!
wait $P1 $P2
python3 train_rank7d.py feats_v10_a.pkl,feats_v10_b.pkl 500 rom_rank,rom_gap_best,rom_sim,rom_partial > cv_v10f.log 2>&1
echo DONE >> cv_v10f.log
