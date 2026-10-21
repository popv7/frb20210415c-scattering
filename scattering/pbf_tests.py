#!/usr/bin/env python3
"""
pbf_tests.py -- alternative pulse-broadening-function (PBF) tests
(Appendix A.4, last two paragraphs).

Synthetic bursts
----------------
* Same Catalog 2 grid as the data: native 24.4-kHz channels, catalogue mask,
  162 samples of 0.983 ms.
* Each unmasked channel i carries
      F_i * [ G(t - mu; sigma_i) * P(t; tau_i) ]   averaged over each bin,
  with fluence F_i proportional to the Catalog 2 spectral model S(nu_i),
  normalised so the summed on-pulse fluence equals the data's;
  mu = full-band EMG onset of the data.
* Intrinsic width  sigma_i = sigma_0 (nu_i / nu_ref)^-1,  sigma_0 = 1.12 ms
  (the fixed-width value), nu_ref = 582.9 MHz: the same nu^-1 width law as
  the width-recovery injections.
* PBF timescale    tau_i = tau_d (nu_i / nu_ref)^-4.
* White noise with each channel's measured off-pulse rms; 50 realisations
  per PBF.

PBFs (Williamson 1972, in the forms of Bhat et al. 2004), unit area, t >= 0:
    exponential (thin screen):  exp(-t/tau) / tau
    thick screen:               (pi tau / 4 t^3)^{1/2} exp(-pi^2 tau / 16 t) / 2
    uniform medium:             (pi^5 tau^3 / 8 t^5)^{1/2} exp(-pi^2 tau / 4 t) / sqrt(2)

Calibration: for each PBF, tau_d is scaled so that the width-free analysis of
a calibration realisation returns tau_ref = 7.98 ms (the measured value), so
every injected burst has the observed tail strength. The PBFs differ only in
shape.

Analysis: the synthetic dynamic spectra go through final_analysis.analyse()
unchanged (same sub-bands, dedrift, bin-integrated EMG, width-free fits,
weighted power law), exactly as the data.

Comparison with the data: for each PBF, the mean recovered width-free
sub-band widths sigma_rec,j are compared with the observed widths of the
eight retained sub-bands,
    chi^2 = sum_j [(sigma_obs,j - c sigma_rec,j) / e_j]^2,
with the normalisation c fitted freely and e_j the observed width
uncertainties (7 degrees of freedom).

    python pbf_tests.py [--nreal 50]      -> pbf_tests.json

Sources: PBF shapes from Williamson (1972, MNRAS 157, 55) in the forms given
by Bhat et al. (2004, ApJ 605, 759); spectral model from the Second CHIME/FRB Catalog (CHIME/FRB Collaboration 2026).
"""
import argparse
import json
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np

import scatlaw as L
import final_analysis as F

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = json.load(open(os.path.join(HERE, "results_paper.json")))
F.drift_slope = lambda: (PAPER["dtdnu"], PAPER["dtdnu_err"])   # slope used in the paper

SIGMA0 = PAPER["sigma_fixed"]          # 1.12 ms
NU_REF = PAPER["nu_ref"]               # 582.9 MHz
TAU_TARGET = PAPER["free"]["ref"]      # 7.98 ms
MU = PAPER["fullband"]["mu"]           # full-band onset, ms
OS = 32                                # fine samples per bin
PAD = 3                                # fine grid spans PAD x the window
SEED = 20261020
WIDTH_INDEX = -1.0                     # intrinsic width law sigma ~ nu^WIDTH_INDEX
PBFS = ("exponential", "thick", "uniform")


def pbf(kind, t, tau):
    """Unit-area PBF on t >= 0 (t, tau broadcastable; tau per row)."""
    pos = t > 0
    tt = np.where(pos, t, 1.0)
    if kind == "exponential":
        v = np.exp(-tt / tau) / tau
    elif kind == "thick":
        v = np.sqrt(np.pi * tau / (4 * tt ** 3)) * np.exp(-np.pi ** 2 * tau / (16 * tt)) / 2.0
    elif kind == "uniform":
        v = np.sqrt(np.pi ** 5 * tau ** 3 / (8 * tt ** 5)) * np.exp(-np.pi ** 2 * tau / (4 * tt)) / np.sqrt(2.0)
    else:
        raise ValueError(kind)
    return np.where(pos, v, 0.0)


def unit_model(burst, kind, tau_d, chunk=256):
    """Noise-free unit-fluence profiles (nchan_good, ntime), bin-averaged."""
    dt = burst.dt
    n = burst.ntime
    nf = n * OS * PAD
    dtf = dt / OS
    # fine-grid centres: bin k covers [k dt - dt/2, k dt + dt/2) (scatlaw convention)
    tf = (np.arange(nf) + 0.5) * dtf - 0.5 * dt
    tlag = (np.arange(nf) + 0.5) * dtf                 # PBF lag axis, t > 0
    nu = burst.freqs[burst.good]
    sig = SIGMA0 * (nu / NU_REF) ** WIDTH_INDEX
    tau = tau_d * (nu / NU_REF) ** -4.0
    nfft = 2 * nf
    out = np.empty((len(nu), n))
    for a in range(0, len(nu), chunk):
        s, tt = sig[a:a + chunk, None], tau[a:a + chunk, None]
        g = np.exp(-0.5 * ((tf[None, :] - MU) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
        p = pbf(kind, tlag[None, :], tt)
        conv = np.fft.irfft(np.fft.rfft(g, nfft, axis=1) * np.fft.rfft(p, nfft, axis=1),
                            nfft, axis=1)[:, :nf] * dtf
        out[a:a + chunk] = conv[:, : n * OS].reshape(-1, n, OS).mean(axis=2)
    return out


def data_reference(burst):
    """On-pulse fluence to match, and each good channel's off-pulse rms."""
    d = burst.data[burst.good]
    off = d[:, : burst.on_region[0] - 5]
    base = off.mean(axis=1, keepdims=True)
    fluence = float((d - base).sum())
    return fluence, off.std(axis=1)


def synth(burst, prof, scale, rms, rng):
    """Full dynamic spectrum (masked rows zero) = scaled model + noise."""
    nu = burst.freqs[burst.good]
    syn = np.zeros_like(burst.data)
    syn[burst.good] = scale * burst.S(nu)[:, None] * prof + rng.normal(size=prof.shape) * rms[:, None]
    return syn


def analyse(syn, burst):
    r = F.analyse(syn, burst)
    return dict(alpha=r["pl_free"]["index"], tau_ref=r["pl_free"]["ref"],
                sigma=r["sigma"].tolist(), ok=[bool(x["ok"]) for x in r["free"]],
                nu=r["nu"].tolist())


def build(kind, burst, fluence, rms, tau_d):
    prof = unit_model(burst, kind, tau_d)
    nu = burst.freqs[burst.good]
    scale = fluence / float((burst.S(nu)[:, None] * prof).sum())
    return prof, scale


def calibrate(kind, burst, fluence, rms):
    """Scale tau_d until the width-free analysis returns tau_ref = 7.98 ms."""
    tau_d = TAU_TARGET
    for _ in range(4):
        prof, scale = build(kind, burst, fluence, rms, tau_d)
        rng = np.random.default_rng([SEED, PBFS.index(kind), 999])
        got = np.mean([analyse(synth(burst, prof, scale, rms, rng), burst)["tau_ref"] for _ in range(3)])
        if abs(got / TAU_TARGET - 1) < 0.003:
            break
        tau_d *= TAU_TARGET / got
    return tau_d


def run_kind(args):
    kind, nreal = args
    warnings.filterwarnings("ignore")
    burst = L.Burst(F.H5)
    fluence, rms = data_reference(burst)
    tau_d = calibrate(kind, burst, fluence, rms)
    prof, scale = build(kind, burst, fluence, rms, tau_d)
    rng = np.random.default_rng([SEED, PBFS.index(kind)])
    trials = [analyse(synth(burst, prof, scale, rms, rng), burst) for _ in range(nreal)]
    return kind, tau_d, trials


def compare(trials, obs_sigma, obs_err, keep):
    sig = np.array([t["sigma"] for t in trials])
    mean = sig.mean(axis=0)
    sem = sig.std(axis=0, ddof=1) / np.sqrt(len(sig))
    p, o, e = mean[keep], obs_sigma[keep], obs_err[keep]
    w = 1.0 / e ** 2
    c = float((w * o * p).sum() / (w * p * p).sum())
    chi2 = float((w * (o - c * p) ** 2).sum())
    alpha = np.array([t["alpha"] for t in trials])
    tref = np.array([t["tau_ref"] for t in trials])
    return dict(mean_sigma=mean.tolist(), sem_sigma=sem.tolist(), norm=c, chi2=chi2,
                dof=int(keep.sum()) - 1, alpha_mean=float(alpha.mean()),
                alpha_sd=float(alpha.std(ddof=1)), tau_ref_mean=float(tref.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nreal", type=int, default=50)
    ap.add_argument("--nproc", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    with Pool(args.nproc) as pool:
        res = pool.map(run_kind, [(k, args.nreal) for k in PBFS])
    sub = PAPER["subbands"]
    obs = np.array([s["sigma"] for s in sub])
    err = np.array([s["sigma_err"] for s in sub])
    keep = np.array([s["retained"] for s in sub])
    out = dict(sigma0=SIGMA0, nu_ref=NU_REF, tau_target=TAU_TARGET, nreal=args.nreal,
               observed_sigma=obs.tolist(), observed_err=err.tolist(), retained=keep.tolist(),
               nu_eff=[s["nu_flux"] for s in sub], pbf={})
    for kind, tau_d, trials in res:
        out["pbf"][kind] = dict(tau_d=tau_d, **compare(trials, obs, err, keep))
        r = out["pbf"][kind]
        print(f"{kind:12s} tau_d {tau_d:6.3f} ms | chi2 {r['chi2']:6.1f} ({r['dof']} dof), norm {r['norm']:.3f} | "
              f"alpha {r['alpha_mean']:+.3f} (sd {r['alpha_sd']:.3f}), tau_ref {r['tau_ref_mean']:.2f} ms")
        print("   mean widths:", " ".join(f"{x:.2f}" for x in r["mean_sigma"]))
    print("   observed   :", " ".join(f"{x:.2f}" for x in obs))
    json.dump(out, open(os.path.join(HERE, "pbf_tests.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f} s -> pbf_tests.json")


if __name__ == "__main__":
    main()
