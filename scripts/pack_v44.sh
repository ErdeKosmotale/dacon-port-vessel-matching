#!/bin/bash
# v44 : hangul -> the three v44 rankers only; non-hangul -> the e12 set (as v38/v42b)
# v44p: hangul -> v44 rankers + the e8 set; non-hangul -> e12
set -e
cd /home/claude/work
E12="ranker_v31_s11000.txt:ranker_05_E.txt ranker_v31_s13000.txt:ranker_06_E.txt ranker_v33_s11000.txt:ranker_11_E.txt ranker_v33_s13000.txt:ranker_12_E.txt"
V44="ranker_v44_s42.txt:ranker_13_H.txt ranker_v44_s7.txt:ranker_14_H.txt ranker_v44_s2026.txt:ranker_15_H.txt"
E8E="ranker_v31.txt:ranker_01_E.txt ranker_v31_s5.txt:ranker_02_E.txt ranker_v31_s7.txt:ranker_03_E.txt ranker_v31_s9.txt:ranker_04_E.txt ranker_v33.txt:ranker_07_E.txt ranker_v33_s5.txt:ranker_08_E.txt ranker_v33_s7.txt:ranker_09_E.txt ranker_v33_s9.txt:ranker_10_E.txt"
E8HE="ranker_v31.txt:ranker_01_HE.txt ranker_v31_s5.txt:ranker_02_HE.txt ranker_v31_s7.txt:ranker_03_HE.txt ranker_v31_s9.txt:ranker_04_HE.txt ranker_v33.txt:ranker_07_HE.txt ranker_v33_s5.txt:ranker_08_HE.txt ranker_v33_s7.txt:ranker_09_HE.txt ranker_v33_s9.txt:ranker_10_HE.txt"
./pack_scoped.sh tpl_v44 v44 full $E8E $E12 $V44 > pack_v44_a.log 2>&1
./pack_scoped.sh tpl_v44 v44p full $E8HE $E12 $V44 > pack_v44_b.log 2>&1
python3 - <<'PY'
import pandas as pd, unicodedata, sys
sys.path.insert(0, 'submit_probeH/model')
import vm
t = pd.read_csv('sim4_test_real.csv', encoding='utf-8-sig')
h = t.vessel_name.map(lambda s: vm.is_hangul(unicodedata.normalize('NFC', s)) if isinstance(s, str) else False).values
b = pd.read_csv('sim_v42b_fuzzy500_full.csv').target_vessel_id.values
for n in ('v44', 'v44p'):
    a = pd.read_csv(f'sim_{n}_full.csv').target_vessel_id.values
    print(n, 'vs v42b: hangul differs', int((a[h] != b[h]).sum()), 'of', int(h.sum()), '| english differs', int((a[~h] != b[~h]).sum()), '(must be 0)')
PY
cat check_v44.log check_v44p.log; tail -n 1 sim_v44_run.log sim_v44p_run.log; ls -la out/submit_v44.zip out/submit_v44p.zip
