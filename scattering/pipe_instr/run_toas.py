#!/usr/bin/env python3
"""run_toas.py -- per-channel arrival times and the drift slope dt/dnu
(Sec. 2.2, Fig. 1b,d) with the arrival-time pipeline. Writes
out/instr_subdf.csv (arrival times) and out/instr_geom.json (slope, window
geometry), which final_analysis.py and figures.py read.

    python run_toas.py > run.log
"""
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import arrivaltimes_stacked_v56_9 as A
import make_input

A.logdebug = False
CFG = {'downfactors': (155, 1), 'model_runs': [{'model': 'EMG', 'subband_plot': 8, 'constrain_index': False,
       'ignore_subindex': [(1)], 'constrain_sigma': [1.05]}]}


def main():
    if not os.path.exists(make_input.OUT):
        make_input.main()
    os.makedirs('out', exist_ok=True)
    for kw in A.expand_model_runs(CFG):
        kw = dict(kw)
        df = kw.pop('downfactors')
        A.measureburst(make_input.OUT, downfactors=df, outdir='out/', show=False, show_components=False,
                       save=False, targetDM=float(np.load(make_input.OUT)['DM']), **kw)


if __name__ == "__main__":
    main()
