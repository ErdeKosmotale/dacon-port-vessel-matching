#!/bin/bash
cd /home/claude/work
python3 build_feats4.py feats_v4_a.pkl 0,1,2 > build_v4a.log 2>&1 &
P1=$!
python3 build_feats4.py feats_v4_b.pkl 3,4 > build_v4b.log 2>&1 &
P2=$!
wait $P1 $P2
python3 train_rank4.py feats_v4_a.pkl,feats_v4_b.pkl 500 > cv_v4.log 2>&1
echo DONE >> cv_v4.log
