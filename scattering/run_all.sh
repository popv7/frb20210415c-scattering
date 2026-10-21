#!/usr/bin/env bash
# Regenerate every number and figure of the paper, in order.
# Requirements: see ../requirements.txt; the Catalog 2 dynamic spectrum in
# ../data/ (see ../data/README.md). Steps 1 and the DM grid need the
# per-channel arrival-time pipeline in pipe_instr/ (Chamma et al. 2025);
# without it the stored drift slope is used and those steps are skipped.
set -euo pipefail
cd "$(dirname "$0")"
if [ -f pipe_instr/run_toas.py ]; then
    echo "== 1. per-channel arrival times"
    ( cd pipe_instr && python3 run_toas.py > run.log 2>&1 && grep "dtdnu =" run.log )
fi
echo "== 2. core analysis + Figures 1, 2"
python3 final_analysis.py --figures
python3 derived_quantities.py > derived_quantities.log
echo "== 3. injection-recovery (Figure 3), Gaussian-envelope variant, negative control, width-law variants"
python3 inject_final.py main
python3 inject_final.py gauss
python3 inject_final.py null
python3 inject_final.py width_nu1
python3 inject_final.py width_measured
python3 -c "import json, figures; figures.figure3(json.load(open('results.json'))['free'])"
echo "== 4. appendix tests"
python3 appendix_tests.py selection
if [ -d pipe_instr ]; then python3 appendix_tests.py dm; fi
python3 appendix_tests.py foreground
python3 width_tests.py grid
python3 width_tests.py inject
python3 pbf_tests.py
echo "== 5. joint eight-profile fit (slow) and tau at 600 MHz"
python3 joint_600.py               # -> joint_fit_scaled_rerun.json, joint_600.json
echo "done"
