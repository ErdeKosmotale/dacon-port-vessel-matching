#!/bin/bash
cd /home/claude/work
python3 build_feats5.py feats_v5_a.pkl 0,1,2 1000 > build_v5a.log 2>&1 &
P1=$!
python3 build_feats5.py feats_v5_b.pkl 3,4 1000 > build_v5b.log 2>&1 &
P2=$!
wait $P1 $P2
python3 train_rank5.py feats_v5_a.pkl,feats_v5_b.pkl 500 > cv_v5.log 2>&1
echo DONE >> cv_v5.log
