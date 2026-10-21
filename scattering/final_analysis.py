#!/usr/bin/env python3
"""
final_analysis.py -- master script for the FRB 20210415C scattering-law
paper. Regenerates every number in the main text and Table 1, and writes
results.json (compare with results_paper.json).

    python final_analysis.py                # core analysis -> results.json
    python final_analysis.py --figures      # + Figures 1, 2 (and 3 once injections exist)
    ./run_all.sh                            # everything, in order

Configuration:
  * data: Catalog 2 Stokes-I dynamic spectrum, catalogue channel mask,
    DM 526.2996 pc cm^-3, 162 samples of 0.983 ms
  * band 400.2-765.6 MHz, nine contiguous sub-bands with equal numbers of
    unmasked channels; highest-frequency sub-band excluded (eight retained)
  * flux-weighted nu_eff from the Catalog 2 running power law S(nu); the same
    nu_eff centres the dedrift (dt/dnu = -4.62e-4 ms/MHz from the
    per-channel arrival-time fit, corrected to the exact sample spacing)
  * sub-band fits on the exact sample spacing
  * bin-integrated EMG per sub-band; sigma, tau >= dt/2; retain if converged,
    not at the floor, chi2_red,prof <= 5; tau errors scaled by
    sqrt(chi2_red,prof)
  * log-space WLS power law, nu_ref = band midpoint (582.9 MHz, "583"),
    errors inflated by sqrt(chi2_red,PL) when > 1
  * fixed-width analysis at the median width of the eight retained profiles

Methods and sources: BIC (Schwarz 1978, Ann. Stat. 6, 461) in its
Gaussian-likelihood form from the residual sum of squares, with the
Delta BIC > 6 threshold of Kass & Raftery (1995, JASA 90, 773); drift slope
from the per-channel arrival-time method of Chamma et al. (2025, MNRAS), as
implemented in pipe_instr/ (based on the Arrival Times Pipeline in FRBGui,
https://github.com/mef51/frbgui); spectral model as in scatlaw.py.
"""
import argparse
import json
import os
import warnings

import numpy as np

import scatlaw as L

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

# ============================ CONFIG =====================================
H5 = os.path.join(os.path.dirname(HERE), "data", "FRB20210415C_stokesi_dynamic_spectrum.h5")
BAND = (400.2, 765.6)      # MHz: band bottom to where S(nu) = 4.8% of its peak (5% is at 762.4)
N_SUB = 9
N_EXCLUDE_TOP = 1          # highest-frequency sub-band excluded
PIPE_GEOM = os.path.join(HERE, "pipe_instr", "out", "instr_geom.json")
CHI2_CUT = 5.0
NU_SPLIT_BETA = 440.0      # MHz, beta fit "above 440 MHz"
BETA_WEIGHTED = False      # False: unweighted log-space OLS (paper);       True: WLS on the
                           # width errors, inflated by sqrt(chi2_red) like the tau fits
# =========================================================================


def drift_slope():
    """Per-channel arrival-time slope dt/dnu (ms/MHz) and its error.

    Read from the arrival-time pipeline run (pipe_instr/out/instr_geom.json).
    The pipeline fits each channel on np.linspace(0, n*dt, n) inside an
    n-sample sub-window, which stretches arrival-time differences, and hence
    the slope, by n/(n-1); that factor is removed here. If the pipeline output
    is absent, the slope stored in results_paper.json is used."""
    if not os.path.exists(PIPE_GEOM):
        stored = json.load(open(os.path.join(HERE, "results_paper.json")))
        return stored["dtdnu"], stored["dtdnu_err"]
    g = json.load(open(PIPE_GEOM))
    c = L.pipeline_time_correction(g["subfall"]["0"]["nt"])
    return g["dtdnus"][0][0] * c, g["dtdnus"][0][1] * c


def nu_ref_of(band):
    return 0.5 * (band[0] + band[1])


def fit_subbands(profs, burst, sigma_fixed=None, p0s=None):
    t = L.subband_time_axis(burst.ntime, burst.dt)
    out = []
    for j, y in enumerate(profs):
        noise, _ = L.offpulse_rms(y, burst)
        p0 = p0s[j] if p0s is not None else [np.sum(y) * burst.dt, 79.0, 1.0, 10.0, 0.0]
        try:
            r = L.fit_emg(y, t, burst.dt, noise, sigma_fixed=sigma_fixed, p0=p0)
        except Exception as e:                               # noqa: BLE001
            r = dict(sigma=np.nan, tau=np.nan, perr_scaled=np.full(5, np.nan),
                     chi2_red=np.nan, at_floor=False, popt=None, error=str(e))
        r["ok"] = bool(np.isfinite(r["tau"]) and r["tau"] > 0 and not r["at_floor"]
                       and r["chi2_red"] <= CHI2_CUT)
        out.append(r)
    return out


def analyse(data, burst, band=BAND, nsub=N_SUB, n_exclude=N_EXCLUDE_TOP,
            dtdnu=None, nu_key="nu_flux", sigma_fixed=None, keep_mask=None,
            nu_ref=None):
    """Full two-step analysis on one data array. Returns a dict."""
    nu_ref = nu_ref_of(band) if nu_ref is None else nu_ref
    dtdnu = drift_slope()[0] if dtdnu is None else dtdnu
    sbs = L.build_subbands(burst, band[0], band[1], nsub)
    profs = L.subband_profiles(data, burst, sbs, dtdnu)
    nu = np.array([s[nu_key] for s in sbs])
    free = fit_subbands(profs, burst)
    ok = np.array([r["ok"] for r in free])
    keep = np.arange(nsub) < nsub - n_exclude if keep_mask is None else np.asarray(keep_mask)
    keep = keep & ok
    sig = np.array([r["sigma"] for r in free])
    sige = np.array([r["perr_scaled"][2] for r in free])
    tau = np.array([r["tau"] for r in free])
    taue = np.array([r["perr_scaled"][3] for r in free])
    s_fix = float(np.median(sig[keep])) if sigma_fixed is None else float(sigma_fixed)
    fixed = fit_subbands(profs, burst, sigma_fixed=s_fix,
                         p0s=[r["popt"] if r.get("popt") is not None else None for r in free])
    tauf = np.array([r["tau"] for r in fixed])
    tauef = np.array([r["perr_scaled"][3] for r in fixed])
    keep_fx = keep & np.array([r["ok"] for r in fixed])
    pl_free = L.powerlaw_wls(nu[keep], tau[keep], taue[keep], nu_ref)
    pl_fix = L.powerlaw_wls(nu[keep_fx], tauf[keep_fx], tauef[keep_fx], nu_ref)
    return dict(sbs=sbs, profs=profs, nu=nu, free=free, fixed=fixed, keep=keep,
                sigma=sig, sigma_err=sige, tau=tau, tau_err=taue,
                tau_fixed=tauf, tau_fixed_err=tauef, sigma_fixed=s_fix,
                pl_free=pl_free, pl_fixed=pl_fix, nu_ref=nu_ref, band=band, dtdnu=dtdnu)


def fullband(burst):
    y = burst.data[burst.good].sum(axis=0)          # all unmasked channels
    noise, _ = L.offpulse_rms(y, burst)
    e = L.fit_emg(y, burst.t, burst.dt, noise)
    g = L.fit_gauss(y, burst.t, burst.dt, noise)
    n = len(y)
    # BIC from the residual sum of squares, BIC = n ln(RSS/n) + k ln n;
    # the chi2 form with off-pulse noise gives ~2e4 (see derived_quantities.py).
    rss_e = float(np.sum((y - e["model"]) ** 2))
    rss_g = float(np.sum((y - g["model"]) ** 2))
    bic_e = n * np.log(rss_e / n) + 5 * np.log(n)
    bic_g = n * np.log(rss_g / n) + 4 * np.log(n)
    return dict(profile=y, noise=noise, emg=e, gauss=g, dBIC=float(bic_g - bic_e))


def _width_fits(res, weighted):
    """beta fits (all nine; retained with nu_eff > 440 MHz) and the excess of
    the lowest sub-band over the extrapolated > 440-MHz relation."""
    nu, sig, sige, keep = res["nu"], res["sigma"], res["sigma_err"], res["keep"]
    ok = np.isfinite(sig)
    hi = keep & (nu > NU_SPLIT_BETA)
    if weighted:
        all9 = L.powerlaw_wls(nu[ok], sig[ok], sige[ok], res["nu_ref"])
        above = L.powerlaw_wls(nu[hi], sig[hi], sige[hi], res["nu_ref"])
    else:
        all9 = L.powerlaw_ols(nu[ok], sig[ok], res["nu_ref"])
        above = L.powerlaw_ols(nu[hi], sig[hi], res["nu_ref"])
    lo_idx = int(np.argmin(nu))
    x = np.log(nu[lo_idx] / res["nu_ref"])
    pred = above["ref"] * np.exp(above["index"] * x)
    c = above["cov"]
    pred_err = pred * np.sqrt(c[0, 0] + x * x * c[1, 1] + 2 * x * c[0, 1])
    # significance against the combined uncertainty of the measurement and
    # the extrapolated relation
    excess = (sig[lo_idx] - pred) / np.hypot(sige[lo_idx], pred_err)
    return dict(all=all9, above=above, lowest_nu=float(nu[lo_idx]),
                lowest_sigma=float(sig[lo_idx]), lowest_sigma_err=float(sige[lo_idx]),
                lowest_pred=float(pred), lowest_pred_err=float(pred_err),
                lowest_excess_sigma=float(excess))


def width_law(res):
    """Width-frequency relation sigma_int,j = sigma_ref (nu/nu_ref)^beta.

    Weighting: by default the fits are NOT weighted by the
    profile-fit uncertainties on sigma_int,j. They are ordinary least squares
    in log space, with the parameter uncertainties scaled by the residual
    scatter. Reason: the formal width errors do not describe the scatter of
    the widths about a power law -- a fit weighted by them has chi2_red = 4.9
    for all nine sub-bands and 1.5 above 440 MHz -- so the uncertainties are
    taken from the scatter itself rather than from the formal errors. The
    weighted alternative (with the same sqrt(chi2_red) inflation as the tau
    fits) is computed alongside and stored as "beta_alternative"; set
    BETA_WEIGHTED = True to make it the primary result."""
    out = _width_fits(res, BETA_WEIGHTED)
    out["alternative"] = _width_fits(res, not BETA_WEIGHTED)
    return out


def summarise(burst, fb, res, wl):
    keep = res["keep"]
    dtau = np.abs(res["tau"] - res["tau_fixed"]) / (0.5 * (res["tau_err"] + res["tau_fixed_err"]))
    widths = [s["nu"][-1] - s["nu"][0] for s in res["sbs"]]
    widest_ret = max(w for w, k in zip(widths, keep) if k)
    e = fb["emg"]
    out = {
        "fullband": dict(sigma=e["sigma"], sigma_err=e["perr_scaled"][2], tau=e["tau"],
                         tau_err=e["perr_scaled"][3], mu=e["mu"], chi2_red=e["chi2_red"],
                         dof=e["dof"], sqrt_chi2=float(np.sqrt(e["chi2_red"])), dBIC=fb["dBIC"]),
        "band": list(res["band"]), "nu_ref": res["nu_ref"],
        "subbands": [dict(f_lo=s["f_lo"], f_hi=s["f_hi"], n_chan=int(len(s["chans"])),
                          unmasked_frac=s["unmasked_frac"], nu_flux=s["nu_flux"],
                          nu_arith=s["nu_arith"], sigma=float(res["sigma"][j]),
                          sigma_err=float(res["sigma_err"][j]), tau=float(res["tau"][j]),
                          tau_err=float(res["tau_err"][j]), chi2_red=float(res["free"][j]["chi2_red"]),
                          tau_fixed=float(res["tau_fixed"][j]),
                          tau_fixed_err=float(res["tau_fixed_err"][j]),
                          retained=bool(keep[j]), dtau_over_err=float(dtau[j]))
                     for j, s in enumerate(res["sbs"])],
        "free": {k: res["pl_free"][k] for k in ("index", "index_err", "ref", "ref_err", "chi2_red", "dof")},
        "fixed": {k: res["pl_fixed"][k] for k in ("index", "index_err", "ref", "ref_err", "chi2_red", "dof")},
        "sigma_fixed": res["sigma_fixed"],
        "beta_weighting": ("weighted least squares on the sigma_int,j errors, sqrt(chi2_red)-inflated"
                           if BETA_WEIGHTED else "unweighted least squares in log space, errors from the residual scatter"),
        "beta_all": {k: wl["all"][k] for k in ("index", "index_err", "ref")},
        "beta_above440": {k: wl["above"][k] for k in ("index", "index_err", "ref")},
        "beta_alternative": {
            "weighting": ("unweighted least squares in log space" if BETA_WEIGHTED
                          else "weighted least squares on the sigma_int,j errors, sqrt(chi2_red)-inflated"),
            "all": {k: wl["alternative"]["all"][k] for k in ("index", "index_err", "ref") + (("chi2_red",) if not BETA_WEIGHTED else ())},
            "above440": {k: wl["alternative"]["above"][k] for k in ("index", "index_err", "ref") + (("chi2_red",) if not BETA_WEIGHTED else ())},
            "lowest_excess_sigma": wl["alternative"]["lowest_excess_sigma"]},
        "lowest": {k: wl[k] for k in ("lowest_nu", "lowest_sigma", "lowest_sigma_err",
                                       "lowest_pred", "lowest_pred_err", "lowest_excess_sigma")},
        "dtau_median_retained": float(np.median(dtau[keep])),
        "dtdnu": res["dtdnu"], "dtdnu_err": drift_slope()[1],
        "largest_drift_delay_ms": abs(res["dtdnu"]) * widest_ret,
        "largest_drift_delay_frac_fwhm": abs(res["dtdnu"]) * widest_ret / (2 * np.sqrt(2 * np.log(2)) * e["sigma"]),
        "sigma_range_retained": [float(np.min(res["sigma"][keep])), float(np.max(res["sigma"][keep]))],
        "top_unmasked_frac": res["sbs"][-1]["unmasked_frac"],
        # window after mu divided by the fitted law's tau at the band bottom
        "post_mu_window_ms": float(burst.ntime * burst.dt - e["mu"]),
        "tau_at_band_bottom": float(res["pl_free"]["ref"] * (res["band"][0] / res["nu_ref"]) ** res["pl_free"]["index"]),
        "low_tail_efolds_in_window": float((burst.ntime * burst.dt - e["mu"]) /
                                           (res["pl_free"]["ref"] * (res["band"][0] / res["nu_ref"]) ** res["pl_free"]["index"])),
        # App. A.1 "~82 ms after the tail onset, ~2.3 fitted timescales": window remaining after
        # the start of the catalogue pulse-emission region, in units of tau at the band bottom
        "window_after_emission_onset_ms": float((burst.ntime - burst.on_region[0]) * burst.dt),
        "low_tail_efolds_after_emission_onset": float((burst.ntime - burst.on_region[0]) * burst.dt /
                                                      (res["pl_free"]["ref"] * (res["band"][0] / res["nu_ref"]) ** res["pl_free"]["index"])),
    }
    return out


def report(s):
    fb = s["fullband"]
    print(f"Full band : sigma {fb['sigma']:.3f}±{fb['sigma_err']:.3f} ms, tau {fb['tau']:.2f}±{fb['tau_err']:.2f} ms, "
          f"chi2_red {fb['chi2_red']:.2f} ({fb['dof']} dof, sqrt {fb['sqrt_chi2']:.2f}), dBIC {fb['dBIC']:.0f}")
    print(f"Band {s['band'][0]}-{s['band'][1]} MHz, nu_ref {s['nu_ref']:.1f} MHz; top sub-band unmasked {s['top_unmasked_frac']:.1%}")
    print(" sb   range (MHz)      n    unm   nu_flux  nu_ar   sigma_free        tau_free        chi2  tau_fixed      |dtau|/err ret")
    for j, b in enumerate(s["subbands"]):
        print(f" {j+1:2d} {b['f_lo']:6.1f}-{b['f_hi']:6.1f} {b['n_chan']:5d} {b['unmasked_frac']:5.1%} {b['nu_flux']:7.1f} {b['nu_arith']:6.1f} "
              f"{b['sigma']:6.3f}±{b['sigma_err']:.3f}  {b['tau']:7.3f}±{b['tau_err']:.3f}  {b['chi2_red']:4.2f}  "
              f"{b['tau_fixed']:7.3f}±{b['tau_fixed_err']:.3f}  {b['dtau_over_err']:5.2f}  {'Y' if b['retained'] else '-'}")
    f, x = s["free"], s["fixed"]
    print(f"Width free : alpha {f['index']:.3f}±{f['index_err']:.3f}, tau_ref {f['ref']:.3f}±{f['ref_err']:.3f} ms, chi2_red,PL {f['chi2_red']:.3f} ({f['dof']} dof)")
    print(f"Fixed {s['sigma_fixed']:.4f}: alpha {x['index']:.3f}±{x['index_err']:.3f}, tau_ref {x['ref']:.3f}±{x['ref_err']:.3f} ms, chi2_red,PL {x['chi2_red']:.3f}")
    print(f"beta all nine {s['beta_all']['index']:.3f}±{s['beta_all']['index_err']:.3f}; above 440 {s['beta_above440']['index']:.3f}±{s['beta_above440']['index_err']:.3f}")
    ba = s["beta_alternative"]
    print(f"  ({s['beta_weighting']}); alternative, {ba['weighting']}: all nine {ba['all']['index']:.3f}±{ba['all']['index_err']:.3f}, "
          f"above 440 {ba['above440']['index']:.3f}±{ba['above440']['index_err']:.3f}, lowest-band excess {ba['lowest_excess_sigma']:.1f} sigma")
    lw = s["lowest"]
    print(f"lowest sub-band nu {lw['lowest_nu']:.1f}: sigma {lw['lowest_sigma']:.3f}±{lw['lowest_sigma_err']:.3f} vs extrapolated "
          f"{lw['lowest_pred']:.3f} -> {lw['lowest_excess_sigma']:.1f} sigma")
    print(f"median retained |dtau|/err {s['dtau_median_retained']:.2f}; largest drift delay {s['largest_drift_delay_ms']:.4f} ms "
          f"({s['largest_drift_delay_frac_fwhm']:.1%} of FWHM); retained widths {s['sigma_range_retained'][0]:.2f}-{s['sigma_range_retained'][1]:.2f} ms; "
          f"lowest-band tail e-folds in window {s['low_tail_efolds_in_window']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    burst = L.Burst(H5)
    fb = fullband(burst)
    res = analyse(burst.data, burst)
    wl = width_law(res)
    s = summarise(burst, fb, res, wl)
    report(s)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(s, f, indent=1, default=float)
    np.savez(os.path.join(HERE, "core_products.npz"), profiles=res["profs"],
             fullband_profile=fb["profile"], fullband_model=fb["emg"]["model"],
             free_models=np.array([r["model"] for r in res["free"]]),
             fixed_models=np.array([r["model"] for r in res["fixed"]]))
    if args.figures:
        import figures
        figures.make_all(burst, fb, res, wl, s)
        if os.path.exists(os.path.join(HERE, "inject_out", "trials_main.csv")):
            figures.figure3(s["free"])


if __name__ == "__main__":
    main()
