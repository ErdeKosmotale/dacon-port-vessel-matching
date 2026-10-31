#!/bin/bash
# v40 = v38 (hangul: e8, non-hangul: e12) + three rankers trained with other objectives, used for both scripts
cd /home/claude/work
./pack_scoped.sh tpl_v40 v40_div full \
  ranker_v31.txt:ranker_01_HE.txt ranker_v31_s5.txt:ranker_02_HE.txt ranker_v31_s7.txt:ranker_03_HE.txt ranker_v31_s9.txt:ranker_04_HE.txt \
  ranker_v31_s11000.txt:ranker_05_E.txt ranker_v31_s13000.txt:ranker_06_E.txt \
  ranker_v33.txt:ranker_07_HE.txt ranker_v33_s5.txt:ranker_08_HE.txt ranker_v33_s7.txt:ranker_09_HE.txt ranker_v33_s9.txt:ranker_10_HE.txt \
  ranker_v33_s11000.txt:ranker_11_E.txt ranker_v33_s13000.txt:ranker_12_E.txt \
  ranker_v37xendcg.txt:ranker_13_HE.txt ranker_v37bin.txt:ranker_14_HE.txt ranker_v37big.txt:ranker_15_HE.txt > pack_v40.log 2>&1
python3 - <<'PY'
import pandas as pd, unicodedata, sys
sys.path.insert(0, 'submit_probeH/model')
import vm
t = pd.read_csv('sim4_test_real.csv', encoding='utf-8-sig')
h = t.vessel_name.map(lambda s: vm.is_hangul(unicodedata.normalize('NFC', s)) if isinstance(s, str) else False).values
a = pd.read_csv('sim_v38_e8H_e12E_full.csv').target_vessel_id.values
b = pd.read_csv('sim_v40_div_full.csv').target_vessel_id.values
print('v40 vs v38 differing picks: hangul', int((a[h] != b[h]).sum()), 'of', int(h.sum()), '| english', int((a[~h] != b[~h]).sum()), 'of', int((~h).sum()))
PY
cat check_v40_div.log; tail -n 1 sim_v40_div_run.log
