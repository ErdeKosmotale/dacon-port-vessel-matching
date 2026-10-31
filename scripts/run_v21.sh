#!/bin/bash
cd /home/claude/work
python3 build_feats14.py feats_v14_a.pkl 0,1,2 1000 > bf14_a.log 2>&1 &
python3 build_feats14.py feats_v14_b.pkl 3,4 1000 > bf14_b.log 2>&1
wait
python3 cv_final_lean.py feats_v14_a.pkl,feats_v14_b.pkl ranker_v21.txt final > final_v21.log 2>&1
python3 cv_final_lean.py feats_v14_a.pkl,feats_v14_b.pkl cv_v21 cv > cv_v21.log 2>&1
echo V21DONE >> final_v21.log
