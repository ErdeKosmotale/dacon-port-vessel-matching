#!/bin/bash
cd /home/claude/work
python3 build_feats7.py feats_v7_a.pkl 0,1,2 1000 > build_v7a.log 2>&1 &
P1=$!
python3 build_feats7.py feats_v7_b.pkl 3,4 1000 > build_v7b.log 2>&1 &
P2=$!
wait $P1 $P2
python3 train_rank7d.py feats_v7_a.pkl,feats_v7_b.pkl 500 rom_rank,rom_gap_best,rom_sim,rom_partial > cv_v7f.log 2>&1
echo DONE >> cv_v7f.log
