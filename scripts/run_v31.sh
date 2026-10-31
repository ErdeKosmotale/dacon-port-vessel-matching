#!/bin/bash
cd /home/claude/work
python3 build_feats3.py feats_v31_a.pkl 0,1,2 > build_v31a.log 2>&1 &
P1=$!
python3 build_feats3.py feats_v31_b.pkl 3,4 > build_v31b.log 2>&1 &
P2=$!
wait $P1 $P2
python3 train_rank3.py feats_v31_a.pkl,feats_v31_b.pkl 1 500 > cv_v31.log 2>&1
echo DONE >> cv_v31.log
