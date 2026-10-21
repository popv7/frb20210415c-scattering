#!/usr/bin/env python3
"""
joint_600.py -- rerun the paper's joint eight-profile MCMC (appendix_tests.joint,
unchanged: same walkers, steps, burn-in, moves and seed) and keep the chain.
From the posterior samples it reports
tau at 600 MHz, tau(600) = exp(ln tau_ref + alpha ln(600/nu_ref)), evaluated
sample by sample so the tau_ref-alpha covariance is carried automatically.

Methods and sources: emcee (Foreman-Mackey et al. 2013); see appendix_tests.py.
"""
import json
import os

import emcee
import numpy as np

import appendix_tests as T
import final_analysis as F

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = json.load(open(os.path.join(HERE, "results_paper.json")))
F.drift_slope = lambda: (PAPER["dtdnu"], PAPER["dtdnu_err"])

_samplers = []
_Orig = emcee.EnsembleSampler


class KeepSampler(_Orig):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _samplers.append(self)


emcee.EnsembleSampler = KeepSampler


def main():
    T.joint(noise_mode="scaled", nstep=T.JOINT_STEPS, burn=T.JOINT_BURN, tag="_rerun")
    s = _samplers[-1]
    ch = s.get_chain(discard=T.JOINT_BURN, thin=10, flat=True)
    np.save(os.path.join(HERE, "joint_chain_lntauref_alpha.npy"), ch[:, :2])
    nu_ref = PAPER["nu_ref"]
    x = np.log(600.0 / nu_ref)
    t600 = np.exp(ch[:, 0] + ch[:, 1] * x)
    q = np.percentile(t600, [16, 50, 84])
    corr = np.corrcoef(ch[:, 0], ch[:, 1])[0, 1]
    out = dict(tau600=[q[1], q[2] - q[1], q[1] - q[0]], corr_lntauref_alpha=float(corr),
               n_samples=int(len(t600)), nu_ref=nu_ref)
    json.dump(out, open(os.path.join(HERE, "joint_600.json"), "w"), indent=1)
    print(f"joint: tau(600) = {q[1]:.4f} +{q[2]-q[1]:.4f}/-{q[1]-q[0]:.4f} ms; corr(ln tau_ref, alpha) = {corr:+.3f}")


if __name__ == "__main__":
    main()
