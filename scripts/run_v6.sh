#!/bin/bash
cd /home/claude/work
python3 build_feats6.py feats_v6_a.pkl 0,1,2 1000 > build_v6a.log 2>&1 &
P1=$!
python3 build_feats6.py feats_v6_b.pkl 3,4 1000 > build_v6b.log 2>&1 &
P2=$!
wait $P1 $P2
python3 train_rank6d.py feats_v6_a.pkl,feats_v6_b.pkl 500 rom_rank,rom_gap_best,rom_sim,rom_partial > cv_v6.log 2>&1
echo DONE >> cv_v6.log
