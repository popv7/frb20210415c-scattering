#!/usr/bin/env python3
"""
inject_final.py -- injection-recovery tests (App. A.1, A.3):
Figure 3, the effective-frequency comparison, the Gaussian-envelope variant
and the negative control.

From the paper: synthetic bursts with known (tau_ref, alpha) on the
observed time-frequency grid, "using the measured spectral envelope, channel
mask, sampling, and sub-band construction", "processed with the same
dedrifting and EMG-fitting procedure as the data"; 100 trials per injected
alpha over -4.3..-3.7; observed 162-sample window vs an extended window that
contains the full injected tail; flux-weighted vs arithmetic-mean nu.

ASSUMED (not stated in the manuscript):
  * injected spectrum: Catalog 2 S(nu) per unmasked native channel, scaled
    so the total injected fluence equals the full-band EMG fluence;
  * scattering law tau(nu) = tau_ref (nu/nu_ref)^alpha with tau_ref = the
    measured width-free value and nu_ref = 582.9 MHz (band midpoint), both
    read from results.json;
  * Gaussian (intrinsic) width sigma(nu) = SIGMA_0 (nu/nu_ref)^WIDTH_INDEX
    with SIGMA_0 = the fixed-width value from results.json (median of the
    eight retained sigma_int,j, 1.12 ms) and WIDTH_INDEX = 0, i.e. the same
    1.12-ms width at every frequency; the width reference frequency is the
    same nu_ref. Variants: WIDTH_INDEX = -1 ("width_nu1") and the measured
    per-sub-band widths ("width_measured"). Every run writes the values it
    used to inject_out/injection_config_<tag>.json;
  * onset mu = full-band mu; no intrinsic drift (the measured-slope dedrift
    is still applied, so channel i is shifted by -dtdnu (nu_i - nu_eff,j)
    exactly as for the data);
  * noise: Gaussian, white, with each channel's off-pulse rms, generated at
    sub-band level (sum of independent channel noises); the observed window
    is the first 162 samples of the extended-window realisation;
  * extended window: 600 samples (>12 e-folds at 400 MHz for alpha=-4.3);
  * negative control: frequency-independent tau = tau_ref.

Sources: injected/plotted spectral envelope S(nu) is the running-power-law
model of the Second CHIME/FRB Catalog (CHIME/FRB Collaboration 2026), in the convention of fitburst (Fonseca et al. 2024, ApJS; arXiv:2311.05829).
"""
import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import scatlaw as L
import final_analysis as F

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "inject_out")

ALPHAS = [-4.3, -4.15, -4.0, -3.85, -3.7]
N_TRIALS = 100
N_EXT = 600
# Injected Gaussian width: sigma(nu) = SIGMA_0 * (nu / nu_ref)**WIDTH_INDEX, with
# SIGMA_0 = results.json "sigma_fixed" and nu_ref = results.json "nu_ref".
WIDTH_MODES = {"fixed": 0.0, "nu1": -1.0, "measured": None}   # None = per-sub-band measured widths
_G = {}


def setup(envelope="catalog", width="fixed"):
    burst = L.Burst(F.H5)
    core = json.load(open(os.path.join(HERE, "results.json")))
    sbs = L.build_subbands(burst, F.BAND[0], F.BAND[1], F.N_SUB)
    if envelope == "gaussian":
        # Gaussian fitted to the time-integrated spectrum of unmasked channels
        t0 = burst.on_region[0] - 5
        base = burst.data[:, :t0].mean(1, keepdims=True)
        spec = (burst.data[:, t0:] - base).sum(1)
        g = burst.good
        gfun = lambda x, a, m, s: a * np.exp(-0.5 * ((x - m) / s) ** 2)
        p, _ = curve_fit(gfun, burst.freqs[g], spec[g], p0=[spec[g].max(), 470, 110])
        env = lambda nu: np.exp(-0.5 * ((nu - p[1]) / p[2]) ** 2)
        for sb in sbs:                       # nu_eff weights follow the same envelope
            w = env(sb["nu"])
            sb["nu_flux"] = float(np.sum(w * sb["nu"]) / np.sum(w))
    else:
        env = burst.S
    noise_rms = burst.data[:, : burst.on_region[0] - 5].std(axis=1)
    fb = core["fullband"]
    # fluence normalisation: sum over all unmasked channels = full-band EMG area
    A_total = F.fullband(burst)["emg"]["A"]
    amp = env(burst.freqs) * burst.good
    amp *= A_total / amp.sum()
    sigma0, nu_ref, w_idx = core["sigma_fixed"], core["nu_ref"], WIDTH_MODES[width]
    if w_idx is None:                    # each channel gets its sub-band's measured width
        sig_sb = [np.full(len(sb["chans"]), b["sigma"]) for sb, b in zip(sbs, core["subbands"])]
    else:
        sig_sb = [sigma0 * (sb["nu"] / nu_ref) ** w_idx for sb in sbs]
    _G.update(burst=burst, sbs=sbs, amp=amp, noise_rms=noise_rms, mu=fb["mu"],
              sigma0=sigma0, width_index=w_idx, width_mode=width, sig_sb=sig_sb, envelope=envelope,
              tau_ref=core["free"]["ref"], nu_ref=nu_ref, A_total=float(A_total),
              dtdnu=F.drift_slope()[0],
              sb_noise=np.array([np.sqrt(np.sum(noise_rms[sb["chans"]] ** 2)) for sb in sbs]))
    return _G


def clean_profiles(alpha, n_samp):
    """Noise-free dedrifted sub-band profiles for one injected alpha."""
    b, sbs, amp = _G["burst"], _G["sbs"], _G["amp"]
    t = np.arange(n_samp) * b.dt
    prof = np.zeros((len(sbs), n_samp))
    for j, sb in enumerate(sbs):
        nu = sb["nu"]
        tau = _G["tau_ref"] * (nu / _G["nu_ref"]) ** alpha
        mu_i = _G["mu"] - _G["dtdnu"] * (nu - sb["nu_flux"])       # after the dedrift shift
        # every channel individually (vectorised over channels in chunks)
        tg = (t[:, None] + L._OFFS[None, :] * b.dt).ravel()
        a_ch = amp[sb["chans"]]
        sig = _G["sig_sb"][j]
        for i in range(0, len(nu), 64):
            sl = slice(i, i + 64)
            pdf = np.array([L.emg_pdf(tg, m, sg, ta) for m, sg, ta in zip(mu_i[sl], sig[sl], tau[sl])])
            prof[j] += (a_ch[sl, None] * pdf.reshape(len(pdf), n_samp, L.OVERSAMP).mean(2)).sum(0)
    return prof


def fit_trial(args):
    alpha, trial, clean_obs, clean_ext = args
    b = _G["burst"]
    rng = np.random.default_rng(int(1e6 * (alpha + 10)) + trial)
    noise = rng.normal(size=(len(_G["sbs"]), N_EXT)) * _G["sb_noise"][:, None]
    out = {}
    for name, clean in (("obs", clean_obs), ("ext", clean_ext)):
        n = clean.shape[1]
        prof = clean + noise[:, :n]
        t = L.subband_time_axis(n, b.dt)
        taus, errs, ok = [], [], []
        for y in prof:
            nz, _ = L.offpulse_rms(y, b)
            try:
                r = L.fit_emg(y, t, b.dt, nz, p0=[np.sum(y) * b.dt, _G["mu"], 1.0, 10.0, 0.0])
                good = np.isfinite(r["tau"]) and not r["at_floor"] and r["chi2_red"] <= F.CHI2_CUT
                taus.append(r["tau"]); errs.append(r["perr_scaled"][3]); ok.append(good)
            except Exception:                                     # noqa: BLE001
                taus.append(np.nan); errs.append(np.nan); ok.append(False)
        out[name] = (np.array(taus), np.array(errs), np.array(ok))
    return alpha, trial, out


def alpha_fit(taus, errs, ok, nu_key):
    keep = (np.arange(len(_G["sbs"])) < F.N_SUB - F.N_EXCLUDE_TOP) & ok
    nu = np.array([sb[nu_key] for sb in _G["sbs"]])
    return L.powerlaw_wls(nu[keep], taus[keep], errs[keep], _G["nu_ref"])["index"]


def write_config(alphas, n_trials, tag):
    """Record exactly what was injected, for quoting in the text."""
    g = _G
    cfg = dict(tag=tag, alphas=list(alphas), n_trials=n_trials, n_samples_observed=162, n_samples_extended=N_EXT,
               scattering_law="tau(nu) = tau_ref (nu/nu_ref)^alpha", tau_ref_ms=g["tau_ref"], nu_ref_MHz=g["nu_ref"],
               width_law="sigma(nu) = sigma_0 (nu/nu_ref)^width_index" if g["width_index"] is not None
               else "per-sub-band measured sigma_int,j", width_mode=g["width_mode"],
               sigma_0_ms=g["sigma0"], width_index=g["width_index"], width_reference_MHz=g["nu_ref"],
               sigma_range_ms=[float(min(v.min() for v in g["sig_sb"])), float(max(v.max() for v in g["sig_sb"]))],
               onset_mu_ms=g["mu"], dedrift_slope_ms_per_MHz=g["dtdnu"], spectral_envelope=g["envelope"],
               total_fluence=g["A_total"], time_axis="exact")
    os.makedirs(OUT, exist_ok=True)
    json.dump(cfg, open(os.path.join(OUT, f"injection_config_{tag}.json"), "w"), indent=1, default=float)


def run(alphas, n_trials, envelope="catalog", tag="main", width="fixed"):
    setup(envelope, width)
    write_config(alphas, n_trials, tag)
    jobs = []
    for a in alphas:
        co, ce = clean_profiles(a, 162), clean_profiles(a, N_EXT)
        jobs += [(a, k, co, ce) for k in range(n_trials)]
    with Pool(2, initializer=setup, initargs=(envelope, width)) as p:
        res = p.map(fit_trial, jobs, chunksize=4)
    rows, taus_m4 = [], {"obs": [], "ext": []}
    for a, k, out in res:
        row = dict(alpha_inj=a, trial=k)
        for w in ("obs", "ext"):
            ta, er, ok = out[w]
            row[f"flux_{w}"] = alpha_fit(ta, er, ok, "nu_flux")
            row[f"arith_{w}"] = alpha_fit(ta, er, ok, "nu_arith")
            row[f"n_ok_{w}"] = int(ok.sum())
            if abs(a + 4.0) < 1e-9:
                taus_m4[w].append(ta)
        rows.append(row)
    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"trials_{tag}.csv"), index=False)
    if taus_m4["obs"]:
        np.savez(os.path.join(OUT, f"taus_m4_{tag}.npz"), obs=np.array(taus_m4["obs"]),
                 ext=np.array(taus_m4["ext"]),
                 nu_flux=np.array([sb["nu_flux"] for sb in _G["sbs"]]),
                 nu_arith=np.array([sb["nu_arith"] for sb in _G["sbs"]]),
                 tau_ref=_G["tau_ref"], nu_ref=_G["nu_ref"])
    return df


def summarise(df, label):
    print(f"\n=== {label} ===")
    g = df.groupby("alpha_inj")
    for a, d in g:
        diff = d.flux_obs - d.flux_ext
        print(f" inj {a:+.2f}: flux ext {d.flux_ext.mean():+.3f}±{d.flux_ext.std():.3f} | flux obs "
              f"{d.flux_obs.mean():+.3f}±{d.flux_obs.std():.3f} | arith obs {d.arith_obs.mean():+.3f}±"
              f"{d.arith_obs.std():.3f} | mean(obs-ext) {diff.mean():+.4f} | arith-flux "
              f"{(d.arith_obs - d.flux_obs).mean():+.3f} | min n_ok {d.n_ok_obs.min()}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "main"
    if what == "main":
        summarise(run(ALPHAS, N_TRIALS, tag="main"), "Figure 3 injections (Catalog 2 envelope)")
    elif what == "gauss":
        summarise(run([-4.0], N_TRIALS, envelope="gaussian", tag="gauss"),
                  "Gaussian-envelope variant (alpha = -4)")
    elif what == "null":
        summarise(run([0.0], N_TRIALS, tag="null"), "Negative control (frequency-independent tail)")
    elif what == "width_nu1":
        summarise(run([-4.0], N_TRIALS, tag="width_nu1", width="nu1"),
                  "Width-law variant: sigma = 1.12 ms (nu/nu_ref)^-1 (alpha = -4)")
    elif what == "width_measured":
        summarise(run([-4.0], N_TRIALS, tag="width_measured", width="measured"),
                  "Width-law variant: measured per-sub-band widths (alpha = -4)")
