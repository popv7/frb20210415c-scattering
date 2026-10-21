#!/usr/bin/env python3
"""
width_tests.py -- the Gaussian-component width tests of App. A.4.

  python width_tests.py grid      # width law over the 770-configuration grid -> width_grid.csv/.json
  python width_tests.py inject    # width-recovery injections                  -> width_injections.csv/.json

Grid (data). Every combination of
    lower edge  : 400.2, 405, 410, ..., 450 MHz            (11 values)
    upper edge  : 720, 725, ..., 760, 765.6 MHz            (10 values)
    sub-bands   : 6, 7, ..., 12                            ( 7 values)
gives 11 x 10 x 7 = 770 configurations; the fiducial (400.2, 765.6, 9) is one
of them. For each, the sub-bands are rebuilt with equal numbers of unmasked
channels, dedrifted about their flux-weighted nu_eff, fitted with the same
bin-integrated EMG (width free), and every sub-band passing the usual cuts
enters the width law sigma_int = sigma_ref (nu/nu_ref)^beta. As in the main
analysis the width law is an unweighted least-squares fit in log space with
uncertainties from the residual scatter; the weighted alternative is stored
too. These configurations reuse the same data, so their spread measures
sensitivity to the analysis choices, not an independent uncertainty.

Width-recovery injections (synthetic). Synthetic bursts on the observed grid
(Catalog 2 spectral envelope, channel mask, sampling, per-channel off-pulse
noise, same dedrift and sub-band construction) with
    tau(nu)   = tau_ref (nu/nu_ref)^-4,      tau_ref = measured width-free value (7.98 ms)
    sigma(nu) = sigma_0 (nu/nu_ref)^beta_inj, sigma_0 = fixed-width value (1.12 ms),
                beta_inj = 0 (constant) or -1,  nu_ref = 582.9 MHz (band midpoint)
for lower limits 400.2, 420, 440 MHz (upper 765.6) and 6, 9, 12 sub-bands,
100 noise realisations per width law and configuration. sigma_0, tau_ref and
nu_ref are read from results.json, and written to width_injections.json.

Sources: injected/plotted spectral envelope S(nu) is the running-power-law
model of the Second CHIME/FRB Catalog (CHIME/FRB Collaboration 2026), in the convention of fitburst (Fonseca et al. 2024, ApJS; arXiv:2311.05829).
"""
import itertools
import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

import scatlaw as L
import final_analysis as F

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

GRID_LO = [400.2] + list(np.arange(405.0, 450.1, 5.0))          # 11
GRID_HI = list(np.arange(720.0, 760.1, 5.0)) + [765.6]          # 10
GRID_NSUB = list(range(6, 13))                                   # 7
INJ_LO = [400.2, 420.0, 440.0]
INJ_NSUB = [6, 9, 12]
INJ_BETAS = [0.0, -1.0]
INJ_TRIALS = 100
_G = {}


def _init():
    _G["burst"] = L.Burst(F.H5)
    _G["dtdnu"] = F.drift_slope()[0]


def width_fit(nu, sig, sige, nu_ref):
    """Unweighted (used) and weighted width laws; None if < 3 points."""
    if len(nu) < 3:
        return None, None
    return L.powerlaw_ols(nu, sig, nu_ref), L.powerlaw_wls(nu, sig, sige, nu_ref)


# ---------------------------------------------------------------------------
# 770-configuration grid on the data
# ---------------------------------------------------------------------------
def grid_job(cfg):
    lo, hi, nsub = cfg
    b = _G["burst"]
    sbs = L.build_subbands(b, lo, hi, nsub)
    profs = L.subband_profiles(b.data, b, sbs, _G["dtdnu"])
    fits = F.fit_subbands(profs, b)
    ok = np.array([r["ok"] for r in fits])
    nu = np.array([s["nu_flux"] for s in sbs])[ok]
    sig = np.array([r["sigma"] for r in fits])[ok]
    sige = np.array([r["perr_scaled"][2] for r in fits])[ok]
    nu_ref = 0.5 * (lo + hi)
    u, w = width_fit(nu, sig, sige, nu_ref)
    return dict(lo=lo, hi=hi, nsub=nsub, n_ok=int(ok.sum()), lowest_nu_eff=float(nu.min()) if len(nu) else np.nan,
                n_below_440=int(np.sum(nu < 440.0)),
                beta=u["index"] if u else np.nan, beta_err=u["index_err"] if u else np.nan,
                beta_w=w["index"] if w else np.nan, beta_w_err=w["index_err"] if w else np.nan,
                chi2_red_w=w["chi2_red"] if w else np.nan)


def run_grid():
    cfgs = list(itertools.product(GRID_LO, GRID_HI, GRID_NSUB))
    assert len(cfgs) == 770, len(cfgs)
    with Pool(2, initializer=_init) as p:
        rows = p.map(grid_job, cfgs, chunksize=5)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "width_grid.csv"), index=False)
    good = df[np.isfinite(df.beta)]
    q16, q50, q84 = np.percentile(good.beta, [16, 50, 84])
    below = good[good.n_below_440 > 0]
    above = good[good.n_below_440 == 0]
    summ = dict(
        lower_edges_MHz=[float(v) for v in GRID_LO], upper_edges_MHz=[float(v) for v in GRID_HI],
        n_subbands=GRID_NSUB, n_configurations=len(df), n_with_width_law=int(len(good)),
        width_fit="unweighted least squares in log space, errors from the residual scatter",
        beta_median=float(q50), beta_p16=float(q16), beta_p84=float(q84),
        beta_err_median=float(np.median(good.beta_err)),
        beta_median_with_data_below_440=float(np.median(below.beta)) if len(below) else None,
        n_with_data_below_440=int(len(below)),
        beta_median_without_data_below_440=float(np.median(above.beta)) if len(above) else None,
        n_without_data_below_440=int(len(above)),
        weighted_alternative=dict(beta_median=float(np.median(good.beta_w)),
                                  beta_p16=float(np.percentile(good.beta_w, 16)),
                                  beta_p84=float(np.percentile(good.beta_w, 84)),
                                  beta_err_median=float(np.median(good.beta_w_err)),
                                  chi2_red_median=float(np.median(good.chi2_red_w))),
        fiducial=df[(df.lo == 400.2) & (df.hi == 765.6) & (df.nsub == 9)].iloc[0].to_dict())
    json.dump(summ, open(os.path.join(HERE, "width_grid.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


# ---------------------------------------------------------------------------
# Width-recovery injections
# ---------------------------------------------------------------------------
def _setup_inject():
    _init()
    b = _G["burst"]
    core = json.load(open(os.path.join(HERE, "results.json")))
    A_total = F.fullband(b)["emg"]["A"]
    amp = b.S(b.freqs) * b.good
    amp *= A_total / amp.sum()
    _G.update(core=core, amp=amp, A_total=float(A_total), mu=core["fullband"]["mu"],
              tau_ref=core["free"]["ref"], nu_ref=core["nu_ref"], sigma0=core["sigma_fixed"],
              noise_rms=b.data[:, : b.on_region[0] - 5].std(axis=1))


def clean_subband_profiles(sbs, beta_inj):
    """Noise-free dedrifted sub-band profiles: every unmasked channel gets its
    own EMG with sigma_i = sigma_0 (nu_i/nu_ref)^beta_inj and tau_i =
    tau_ref (nu_i/nu_ref)^-4, onset shifted by the same dedrift as the data."""
    b, g = _G["burst"], _G
    t = np.arange(b.ntime) * b.dt
    tg = (t[:, None] + L._OFFS[None, :] * b.dt).ravel()
    prof = np.zeros((len(sbs), b.ntime))
    for j, sb in enumerate(sbs):
        nu = sb["nu"]
        tau = g["tau_ref"] * (nu / g["nu_ref"]) ** -4.0
        sig = g["sigma0"] * (nu / g["nu_ref"]) ** beta_inj
        mu_i = g["mu"] - g["dtdnu"] * (nu - sb["nu_flux"])
        a_ch = g["amp"][sb["chans"]]
        for i in range(0, len(nu), 64):
            sl = slice(i, i + 64)
            pdf = np.array([L.emg_pdf(tg, m, s, ta) for m, s, ta in zip(mu_i[sl], sig[sl], tau[sl])])
            prof[j] += (a_ch[sl, None] * pdf.reshape(len(pdf), b.ntime, L.OVERSAMP).mean(2)).sum(0)
    return prof


def inject_job(args):
    lo, nsub, beta_inj = args
    if "core" not in _G:
        _setup_inject()
    b, g = _G["burst"], _G
    sbs = L.build_subbands(b, lo, F.BAND[1], nsub)
    clean = clean_subband_profiles(sbs, beta_inj)
    sb_noise = np.array([np.sqrt(np.sum(g["noise_rms"][sb["chans"]] ** 2)) for sb in sbs])
    nu_eff = np.array([sb["nu_flux"] for sb in sbs])
    # injected width "at" each sub-band: flux-weighted mean of the channel widths
    sig_true = np.array([np.sum(b.S(sb["nu"]) * g["sigma0"] * (sb["nu"] / g["nu_ref"]) ** beta_inj)
                         / np.sum(b.S(sb["nu"])) for sb in sbs])
    nu_ref = 0.5 * (lo + F.BAND[1])
    rows = []
    for k in range(INJ_TRIALS):
        rng = np.random.default_rng(int(1e4 * lo) + 100 * nsub + int(10 * abs(beta_inj)) * 7 + 1000003 * k)
        profs = clean + rng.normal(size=clean.shape) * sb_noise[:, None]
        fits = F.fit_subbands(profs, b)
        ok = np.array([r["ok"] for r in fits])
        sig = np.array([r["sigma"] for r in fits])
        sige = np.array([r["perr_scaled"][2] for r in fits])
        u, w = width_fit(nu_eff[ok], sig[ok], sige[ok], nu_ref)
        rows.append(dict(lo=lo, nsub=nsub, beta_inj=beta_inj, trial=k, n_ok=int(ok.sum()),
                         beta=u["index"] if u else np.nan, beta_w=w["index"] if w else np.nan,
                         lowest_sigma=float(sig[0]) if ok[0] else np.nan, lowest_sigma_true=float(sig_true[0]),
                         lowest_nu_eff=float(nu_eff[0])))
    return rows


def run_inject():
    _setup_inject()
    cfgs = list(itertools.product(INJ_LO, INJ_NSUB, INJ_BETAS))
    with Pool(2) as p:
        rows = sum(p.map(inject_job, cfgs, chunksize=1), [])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "width_injections.csv"), index=False)
    per = []
    for (lo, nsub, bi), d in df.groupby(["lo", "nsub", "beta_inj"]):
        per.append(dict(lo=lo, nsub=nsub, beta_inj=bi, beta_mean=float(d.beta.mean()), beta_std=float(d.beta.std()),
                        beta_w_mean=float(d.beta_w.mean()),
                        lowest_sigma_bias_ms=float(d.lowest_sigma.mean() - d.lowest_sigma_true.iloc[0]),
                        lowest_sigma_true_ms=float(d.lowest_sigma_true.iloc[0]), min_n_ok=int(d.n_ok.min())))
    per = pd.DataFrame(per)
    g = _G
    summ = dict(
        scattering_law="tau(nu) = tau_ref (nu/nu_ref)^-4", tau_ref_ms=g["tau_ref"], nu_ref_MHz=g["nu_ref"],
        width_law="sigma(nu) = sigma_0 (nu/nu_ref)^beta_inj", sigma_0_ms=g["sigma0"], width_reference_MHz=g["nu_ref"],
        beta_inj=INJ_BETAS, lower_limits_MHz=INJ_LO, upper_limit_MHz=F.BAND[1], n_subbands=INJ_NSUB,
        trials_per_configuration=INJ_TRIALS, onset_mu_ms=g["mu"], dedrift_slope_ms_per_MHz=g["dtdnu"],
        width_fit="unweighted least squares in log space (as for the data)",
        recovered={str(bi): dict(beta_mean_min=float(d.beta_mean.min()), beta_mean_max=float(d.beta_mean.max()),
                                 beta_std_median=float(d.beta_std.median()),
                                 max_abs_lowest_sigma_bias_ms=float(np.abs(d.lowest_sigma_bias_ms).max()))
                   for bi, d in per.groupby("beta_inj")},
        per_configuration=per.to_dict(orient="records"))
    json.dump(summ, open(os.path.join(HERE, "width_injections.json"), "w"), indent=1, default=float)
    print(per.to_string(index=False))
    print(json.dumps(summ["recovered"], indent=1))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "grid"
    {"grid": run_grid, "inject": run_inject}[what]()
