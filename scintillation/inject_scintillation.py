#!/usr/bin/env python
"""
inject_scintillation.py -- injection-recovery test for the ACF scintillation
search in acf_scintillation.py.

What is injected
----------------
Diffractive scintillation from a thin screen with an exponential
pulse-broadening function, tau = 1 / (2 pi nu_s) (C1 = 1):

  * a complex field E(nu) is built as the Fourier transform of complex
    Gaussian white noise whose variance over delay t is exp(-t / tau);
    |<E(nu) E*(nu + dnu)>|^2 is then a Lorentzian with HWHM nu_s, and
    I = |E|^2 has unit mean, unit variance (exponential statistics, m = 1);
  * partial modulation is G = 1 + m (I - 1), so m is the intrinsic
    modulation index before channelisation;
  * the field is generated OSF times finer than a CHIME channel and
    box-car averaged into 24.414-kHz channels, so sub-channel scintillation
    is washed out by the channelisation exactly as it would be in the data.

How it is injected
------------------
Into the real waterfall, through the CHIME fitburst model of this burst:

    data_inj = data + (G(nu) - 1) * model(nu, t)

so the burst signal is scintillated, while the real radiometer noise, the
real unmodelled spectral structure and the real channel mask are kept.
The injected waterfall then goes through prepare() and analyse_block() from
acf_scintillation.py unchanged -- the same estimator used for the paper.

Bandwidths
----------
  * "YMW16"  : nu_s from YMW16 at each block centre (nu^4 scaling, C1 = 1)
  * "NE2001" : nu_s from NE2001 at each block centre
  * a fixed grid of nu_s, the same in every block

Outputs per trial and block: formal detection flag (the script's criterion),
modulation significance, recovered m and nu_s, fit type, and the modulation
index actually present in the channelised data.

Usage:
    python inject_scintillation.py [--ntrial 100] [--nproc 2]

Sources: the burst model scintillated here is the catalogue's fitburst model
(Fonseca et al. 2024; CHIME/FRB Collaboration 2026); nu_s-tau relation as in
acf_scintillation.py (Cordes & Rickett 1998).
"""

import argparse
import itertools
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

import acf_scintillation as A

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "inject_results.npz")

OSF = 32                       # fine bins per CHIME channel
NUS_GRID_KHZ = [4, 8, 12, 18, 25, 35, 50, 70, 100, 150]
M_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
MODELS = ["YMW16", "NE2001"]
SEED = 20261017

# ------------------------------------------------------------------ data ---
FREQS, OK, DATA, MODEL, ATTRS = A.load(A.FILE)
RES = ATTRS["res_freq"]
BLOCKS = A.find_blocks(OK, FREQS, A.FMIN, A.FMAX, A.MIN_RUN)
NLAG = int(A.MAX_LAG_MHZ / RES)
FC = [0.5 * (FREQS[a] + FREQS[b]) for a, b in BLOCKS]


def scint_gain(nchan, nus_khz, m, rng, osf=OSF):
    """
    Channel-averaged multiplicative gain for one block.

    The field is generated over twice the block span and the first half kept,
    so the circular FFT does not correlate the two block edges.
    """
    nfine = 2 * nchan * osf
    dnu_khz = RES * 1e3 / osf
    tau_ms = 1.0 / (2.0 * np.pi * nus_khz)          # 1/kHz = ms
    t_ms = np.arange(nfine) / (nfine * dnu_khz)
    w = np.exp(-t_ms / tau_ms)
    a = (rng.standard_normal(nfine) + 1j * rng.standard_normal(nfine)) \
        * np.sqrt(w / 2.0)
    intensity = np.abs(np.fft.fft(a)) ** 2 / w.sum()
    g = 1.0 + m * (intensity - 1.0)
    return g[: nchan * osf].reshape(nchan, osf).mean(axis=1)


def block_nus(label, j):
    if label in MODELS:
        return A.nu_s_pred_khz(label, FC[j])
    return float(label)


def run_config(job):
    """All trials for one (bandwidth label, m). Returns a structured array."""
    warnings.filterwarnings("ignore")
    label, m, ntrial, cfg_id = job
    rows = []
    for k in range(ntrial):
        rng = np.random.default_rng([SEED, cfg_id, k])
        for j, (a, b) in enumerate(BLOCKS):
            sl = slice(a, b + 1)
            if m == 0.0:
                nus, g = np.nan, np.ones(b - a + 1)
            else:
                nus = block_nus(label, j)
                g = scint_gain(b - a + 1, nus, m, rng)
            okb = OK[sl]
            m_chan = np.std(g[okb])
            d_inj = DATA[sl] + (g - 1.0)[:, None] * MODEL[sl]
            spec, sigma, ctrl, _, _ = A.prepare(d_inj, ATTRS)
            r = A.analyse_block(spec, sigma, ctrl, okb, RES, FC[j], NLAG)
            if r is None:
                rows.append((label, m, k, j, nus, m_chan, False, np.nan,
                             np.nan, np.nan, np.nan, "failed", np.nan))
                continue
            rows.append((label, m, k, j, nus, m_chan, r["detected"], r["m"],
                         r["sig_m"], r["nus"] * 1e3, r["sig_nus"],
                         r["kind"].split(" ")[0], r["m_e"]))
    return rows


DTYPE = [("label", "U8"), ("m_inj", "f8"), ("trial", "i4"), ("block", "i4"),
         ("nus_inj", "f8"), ("m_chan", "f8"), ("detected", "?"),
         ("m_rec", "f8"), ("sig_m", "f8"), ("nus_rec", "f8"),
         ("sig_nus", "f8"), ("kind", "U8"), ("m_err", "f8")]


def self_test():
    """Check the generator: unit mean, variance m^2, HWHM = nu_s (fine grid)."""
    rng = np.random.default_rng(1)
    nchan, osf = 4000, OSF
    for nus in (8.0, 50.0):
        hw = []
        for _ in range(20):
            nfine = 2 * nchan * osf
            dnu = RES * 1e3 / osf
            tau = 1.0 / (2 * np.pi * nus)
            t = np.arange(nfine) / (nfine * dnu)
            w = np.exp(-t / tau)
            a = (rng.standard_normal(nfine) + 1j * rng.standard_normal(nfine)) \
                * np.sqrt(w / 2)
            i = np.abs(np.fft.fft(a)) ** 2 / w.sum()
            x = i - i.mean()
            f = np.fft.rfft(x, 2 * x.size)
            acf = np.fft.irfft(np.abs(f) ** 2)[: x.size] / np.arange(x.size, 0, -1)
            acf /= acf[0]
            hw.append(np.argmax(acf < 0.5) * dnu)
            mean, var = i.mean(), i.var()
        print(f"  nu_s={nus:5.1f} kHz: mean={mean:.3f} var={var:.3f} "
              f"HWHM={np.median(hw):.2f} kHz")
    for nus in (4, 8, 18, 50, 150):
        mc = np.mean([np.std(scint_gain(600, nus, 1.0, rng)) for _ in range(50)])
        print(f"  channelised m for m=1, nu_s={nus:4d} kHz: {mc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntrial", type=int, default=100)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    print("blocks:", [f"{FREQS[a]:.1f}-{FREQS[b]:.1f}" for a, b in BLOCKS])
    for lab in MODELS:
        print(f"  {lab} nu_s per block (kHz):",
              [round(A.nu_s_pred_khz(lab, f), 1) for f in FC])
    if args.self_test:
        self_test()
        return 0

    labels = MODELS + [str(x) for x in NUS_GRID_KHZ]
    jobs = [(lab, m, args.ntrial, i)
            for i, (lab, m) in enumerate(itertools.product(labels, M_GRID))]
    # m = 0: the real data, no injection (deterministic, one trial)
    jobs.append(("none", 0.0, 1, len(jobs)))

    t = time.time()
    with Pool(args.nproc) as p:
        out = p.map(run_config, jobs, chunksize=1)
    rows = [r for chunk in out for r in chunk]
    arr = np.array([tuple(r) for r in rows], dtype=DTYPE)
    np.savez(OUT, results=arr, blocks=np.array(BLOCKS), fc=np.array(FC),
             osf=OSF, seed=SEED, ntrial=args.ntrial)
    print(f"{len(arr)} block-fits in {time.time() - t:.0f} s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
