#!/usr/bin/env python3
"""
derived_quantities.py -- check the core numbers against results_paper.json,
then compute the derived quantities quoted in the paper:

  1. full-band profile-fit details (BIC in both forms, n, k) and a
     multi-start check of every profile fit
  2. tau at 600 MHz for the width-free and fixed-width fits, with the
     tau_ref-alpha covariance propagated (the joint-fit value is from
     joint_600.py)
  3. width-tail correlation from the two-step local covariance matrices

    python derived_quantities.py      -> derived_quantities.json

Methods and sources: BIC as in final_analysis.py (Schwarz 1978).
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

import scatlaw as L
import final_analysis as F

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = json.load(open(os.path.join(HERE, "results_paper.json")))
F.drift_slope = lambda: (PAPER["dtdnu"], PAPER["dtdnu_err"])   # slope used in the paper
NU600 = 600.0


def compare(label, new, old, tol=1e-6):
    ok = abs(new - old) <= tol * max(1.0, abs(old))
    print(f"  {label:34s} {new:12.6f}  paper {old:12.6f}  {'OK' if ok else 'DIFFERS'}")
    return ok


def tau_at(pl, nu_ref, nu):
    """tau(nu) and its error from a power-law fit, propagating the full
    (ln tau_ref, alpha) covariance (already sqrt(chi2_red,PL)-inflated)."""
    x = np.log(nu / nu_ref)
    c = pl["cov"]
    lnt = np.log(pl["ref"]) + pl["index"] * x
    var = c[0, 0] + x * x * c[1, 1] + 2 * x * c[0, 1]
    t = float(np.exp(lnt))
    corr = c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])
    var_nocov = c[0, 0] + x * x * c[1, 1]
    return t, t * np.sqrt(var), t * np.sqrt(var_nocov), corr, x


def main():
    b = L.Burst(F.H5)
    print("== reproduction check against results.json ==")
    fb = F.fullband(b)
    e = fb["emg"]
    allok = True
    allok &= compare("full-band sigma", e["sigma"], PAPER["fullband"]["sigma"])
    allok &= compare("full-band tau", e["tau"], PAPER["fullband"]["tau"])
    allok &= compare("full-band chi2_red", e["chi2_red"], PAPER["fullband"]["chi2_red"])
    allok &= compare("dBIC", fb["dBIC"], PAPER["fullband"]["dBIC"])
    res = F.analyse(b.data, b)
    allok &= compare("free alpha", res["pl_free"]["index"], PAPER["free"]["index"])
    allok &= compare("free tau_ref", res["pl_free"]["ref"], PAPER["free"]["ref"])
    allok &= compare("fixed alpha", res["pl_fixed"]["index"], PAPER["fixed"]["index"])
    allok &= compare("fixed tau_ref", res["pl_fixed"]["ref"], PAPER["fixed"]["ref"])
    allok &= compare("sigma_fixed", res["sigma_fixed"], PAPER["sigma_fixed"])
    print("  ALL REPRODUCED" if allok else "  MISMATCH -- see above")

    out = {}
    # ---------------- 1: BIC, n, k, both forms ----------------
    y, n = fb["profile"], len(fb["profile"])
    rss_e = float(np.sum((y - e["model"]) ** 2))
    rss_g = float(np.sum((y - fb["gauss"]["model"]) ** 2))
    chi2_e, chi2_g = e["chi2"], fb["gauss"]["chi2"]
    bic_chi2_e = chi2_e + 5 * np.log(n)
    bic_chi2_g = chi2_g + 4 * np.log(n)
    out["bic"] = dict(n_bins=n, k_emg=5, k_gauss=4, noise_offpulse=fb["noise"],
                      rss_emg=rss_e, rss_gauss=rss_g, rss_form_dBIC=fb["dBIC"],
                      chi2_form_dBIC_raw_noise=float(bic_chi2_g - bic_chi2_e),
                      chi2_form_dBIC_rescaled_noise=float((chi2_g - chi2_e) / e["chi2_red"] - np.log(n)),
                      chi2_emg=chi2_e, chi2_gauss=chi2_g, gauss_chi2_red=chi2_g / (n - 4),
                      offpulse_bins=int(b.on_region[0] - 5))
    print("\n== 1: BIC ==")
    print(json.dumps(out["bic"], indent=1))

    # multi-start check: full-band and every sub-band, free and fixed width
    print("\n== 1: multi-start check ==")
    starts_sig = [0.6, 1.0, 2.0, 4.0]
    starts_tau = [1.0, 5.0, 10.0, 20.0, 40.0]
    def multistart(yy, tt, noise, ref, sigma_fixed=None):
        dev = []
        for s0 in (starts_sig if sigma_fixed is None else [sigma_fixed]):
            for t0 in starts_tau:
                for mu0 in (75.0, 79.0, 83.0):
                    p0 = [np.sum(yy) * b.dt, mu0, s0, t0, 0.0]
                    try:
                        r = L.fit_emg(yy, tt, b.dt, noise, sigma_fixed=sigma_fixed, p0=p0)
                    except Exception:
                        continue
                    dev.append((r["chi2"] - ref["chi2"], abs(r["tau"] - ref["tau"]) / ref["perr_scaled"][3],
                                abs(r["sigma"] - ref["sigma"]) / max(ref["perr_scaled"][2], 1e-12)))
        dev = np.array(dev)
        return len(dev), float(dev[:, 0].min()), float(dev[:, 1].max()), float(dev[:, 2].max())
    t = L.subband_time_axis(b.ntime, b.dt)
    ms = {}
    nst, dchi, dtau, dsig = multistart(y, b.t, fb["noise"], e)
    ms["fullband"] = (nst, dchi, dtau, dsig)
    print(f"  full band: {nst} starts, min dchi2 {dchi:+.3g}, max |dtau|/err {dtau:.3g}, max |dsigma|/err {dsig:.3g}")
    for j, yy in enumerate(res["profs"]):
        noise, _ = L.offpulse_rms(yy, b)
        nst, dchi, dtau, dsig = multistart(yy, t, noise, res["free"][j])
        nst2, dchi2_, dtau2, _ = multistart(yy, t, noise, res["fixed"][j], sigma_fixed=res["sigma_fixed"])
        ms[f"sb{j+1}"] = (nst, dchi, dtau, dsig, nst2, dchi2_, dtau2)
        print(f"  sb{j+1}: free {nst} starts, min dchi2 {dchi:+.2g}, max |dtau|/err {dtau:.2g}, max |dsig|/err {dsig:.2g}"
              f" | fixed {nst2} starts, min dchi2 {dchi2_:+.2g}, max |dtau|/err {dtau2:.2g}")
    out["multistart"] = ms

    # ---------------- 2: tau at 600 MHz ----------------
    print("\n== 2: tau at 600 MHz (two-step fits) ==")
    for key in ("pl_free", "pl_fixed"):
        pl = res[key]
        t6, e6, e6_nocov, corr, x = tau_at(pl, res["nu_ref"], NU600)
        out[key + "_600"] = dict(tau=t6, err=e6, err_without_cov=e6_nocov, corr_lnref_alpha=float(corr),
                                 x=float(x), tau_ref=pl["ref"], tau_ref_err=pl["ref_err"],
                                 alpha=pl["index"], alpha_err=pl["index_err"], cov=pl["cov"].tolist())
        print(f"  {key}: tau(600) = {t6:.4f} +/- {e6:.4f} ms  (without covariance: +/- {e6_nocov:.4f});"
              f" corr(ln tau_ref, alpha) = {corr:+.3f}; tau_ref {pl['ref']:.4f}+/-{pl['ref_err']:.4f}")

    # ---------------- 3: two-step local width-tail correlation ----------------
    print("\n== 3: two-step local r(sigma, tau) from the curve_fit covariance ==")
    r_loc = []
    for j, (yy, fr) in enumerate(zip(res["profs"], res["free"])):
        noise, _ = L.offpulse_rms(yy, b)
        f = lambda tt, A, mu, s, ta, C: L.emg_binned(tt, A, mu, s, ta, C, b.dt)
        _, pcov = curve_fit(f, t, yy, p0=fr["popt"], sigma=np.full(len(yy), noise), absolute_sigma=True,
                            maxfev=200000)
        r = pcov[2, 3] / np.sqrt(pcov[2, 2] * pcov[3, 3])
        r_loc.append(float(r))
        print(f"  sb{j+1} nu_eff {res['nu'][j]:.1f}: r = {r:+.3f}{'' if res['keep'][j] else '  (excluded)'}")
    out["r_local"] = r_loc
    out["nu_eff"] = res["nu"].tolist()
    out["keep"] = res["keep"].tolist()
    json.dump(out, open(os.path.join(HERE, "derived_quantities.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
