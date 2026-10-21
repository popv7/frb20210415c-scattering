#!/usr/bin/env python3
"""
appendix_tests.py -- robustness tests of the paper (App. A.2, A.3):

  python appendix_tests.py selection   # all nine, leave-one-out, alternative band
  python appendix_tests.py dm          # DM grid (re-dedisperse, remeasure slope, refit)
  python appendix_tests.py joint       # joint eight-profile fit (MCMC)
  python appendix_tests.py foreground  # Galactic NE2001 / YMW16 comparison

DM grid (App. A.2): offsets 0, +/-0.05, +/-0.10, +/-0.25 pc cm^-3. At
each DM the native-resolution data are re-dedispersed (Fourier phase shift at
the true channel frequencies, about the data's 400.195-MHz dedispersion
reference), the drift slope is remeasured
with the per-channel arrival-time pipeline on a matching .npz (the pipeline's
own targetDM path is NOT used: driftrate.dedisperse assumes uniformly spaced
channels, which the masked .npz is not), and both fits are repeated, the fixed width being
recomputed as the median of the retained widths at that DM.

Methods and sources:
  * DM grid: drift slope from the arrival-time pipeline in pipe_instr/
    (per-channel method of Chamma et al. 2025; based on the Arrival Times
    Pipeline in FRBGui, https://github.com/mef51/frbgui);
  * joint fit: emcee (Foreman-Mackey et al. 2013, PASP 125, 306) with its
    differential-evolution and snooker moves (ter Braak 2006, Stat. Comput.
    16, 239; ter Braak & Vrugt 2008, Stat. Comput. 18, 435) and its
    integrated autocorrelation time estimate; maximum-likelihood start with
    scipy.optimize.least_squares;
  * foreground: NE2001 (Cordes & Lazio 2002, arXiv:astro-ph/0207156) and
    YMW16 (Yao, Manchester & Wang 2017, ApJ 835, 29) evaluated with PyGEDM
    (Price et al. 2021, PASA; arXiv:2106.15816); YMW16 tau from the tau-DM relation
    of Krishnakumar et al. (2015, ApJ 804, 23).
"""
import json
import os
import subprocess
import sys
import warnings

import numpy as np

import scatlaw as L
import final_analysis as F

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, "pipe_instr")
NPZ0 = os.path.join(PIPE, "Data", "FRB20210415C.npz")   # built by pipe_instr/make_input.py
DM_OFFSETS = [-0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25]
JOINT_STEPS, JOINT_BURN = 16000, 2000   # >= 50 tau_int after burn-in (checked in the output)


def redisperse(burst, ddm):
    """Dedisperse to DM + ddm with a Fourier phase shift of each native
    channel (relative to the data's 400.195-MHz dedispersion reference).
    Linear-interpolation or integer shifts were tested and rejected: they
    smooth or quantise the channels and, in the fixed-width fits, move alpha
    by up to 0.08 even at |dDM| = 0.05."""
    return burst.redisperse(ddm)


def measure_slope(burst, data, ddm):
    """Write an .npz of the re-dedispersed unmasked channels and run the
    per-channel arrival-time pipeline on it (no-op dedispersion)."""
    z = dict(np.load(NPZ0, allow_pickle=True))
    z["wfall"] = data[burst.good].astype(np.float32)
    z["DM"] = np.float64(burst.dm + ddm)
    os.makedirs(os.path.join(PIPE, "Data_dm"), exist_ok=True)
    fn = os.path.join(PIPE, "Data_dm", f"FRB20210415C_dm{ddm:+.2f}.npz")
    np.savez(fn, **z)
    od = os.path.join(PIPE, f"out_redisp{ddm:+.2f}")
    os.makedirs(od, exist_ok=True)
    code = f"""
import warnings, numpy as np, matplotlib
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import arrivaltimes_stacked_v56_9 as A
A.logdebug = False
cfg = {{'downfactors': (155, 1), 'model_runs': [{{'model': 'EMG', 'subband_plot': 8, 'constrain_index': False,
       'ignore_subindex': [(1)], 'constrain_sigma': [1.05]}}]}}
for kw in A.expand_model_runs(cfg):
    kw = dict(kw); df = kw.pop('downfactors')
    A.measureburst({fn!r}, downfactors=df, outdir={od + '/'!r}, show=False, show_components=False,
                   save=False, targetDM=float(np.load({fn!r})['DM']), **kw)
"""
    subprocess.run([sys.executable, "-c", code], cwd=PIPE, capture_output=True, timeout=900)
    g = json.load(open(os.path.join(od, "instr_geom.json")))
    c = L.pipeline_time_correction(g["subfall"]["0"]["nt"])
    return g["dtdnus"][0][0] * c, g["dtdnus"][0][1] * c


def dm_grid():
    from multiprocessing import Pool
    burst = L.Burst(F.H5)
    with Pool(2) as p:
        slopes = p.starmap(_slope_job, [(d,) for d in DM_OFFSETS])
    rows = []
    for ddm, (sl, sle) in zip(DM_OFFSETS, slopes):
        data = redisperse(burst, ddm)
        r = F.analyse(data, burst, dtdnu=sl)
        rows.append(dict(ddm=ddm, dm=burst.dm + ddm, dtdnu=sl, dtdnu_err=sle,
                         a_free=r["pl_free"]["index"], t_free=r["pl_free"]["ref"],
                         a_fix=r["pl_fixed"]["index"], t_fix=r["pl_fixed"]["ref"],
                         sigma_fixed=r["sigma_fixed"]))
        print(f"DM {burst.dm + ddm:8.3f} ({ddm:+.2f}): dt/dnu {sl:+.3e}±{sle:.1e}  free a {rows[-1]['a_free']:+.3f} "
              f"tau_ref {rows[-1]['t_free']:.3f} | fixed a {rows[-1]['a_fix']:+.3f} tau_ref {rows[-1]['t_fix']:.3f}")
    base = next(r for r in rows if r["ddm"] == 0)
    da = max(max(abs(r["a_free"] - base["a_free"]), abs(r["a_fix"] - base["a_fix"])) for r in rows)
    dt = max(max(abs(r["t_free"] - base["t_free"]), abs(r["t_fix"] - base["t_fix"])) for r in rows)
    print(f"max |d alpha| = {da:.3f}; max |d tau_ref| = {dt:.3f} ms")
    json.dump(rows, open(os.path.join(HERE, "dm_grid.json"), "w"), indent=1)


def _slope_job(ddm):
    burst = L.Burst(F.H5)
    return measure_slope(burst, redisperse(burst, ddm), ddm)


def selection():
    b = L.Burst(F.H5)
    base = F.analyse(b.data, b)
    nu, tau, te, nr = base["nu"], base["tau"], base["tau_err"], base["nu_ref"]
    a9 = L.powerlaw_wls(nu, tau, te, nr)
    k8 = np.arange(F.N_SUB) < F.N_SUB - 1
    loo8 = [L.powerlaw_wls(np.delete(nu[k8], i), np.delete(tau[k8], i), np.delete(te[k8], i), nr)["index"]
            for i in range(k8.sum())]
    loo9 = [L.powerlaw_wls(np.delete(nu, i), np.delete(tau, i), np.delete(te, i), nr)["index"]
            for i in range(F.N_SUB)]
    alt = F.analyse(b.data, b, band=(428.6, 765.6), nsub=8)
    out = dict(all_nine=[a9["index"], a9["index_err"]], loo_from_eight=[min(loo8), max(loo8)],
               loo_from_nine=[min(loo9), max(loo9)],
               alt_band_free=[alt["pl_free"]["index"], alt["pl_free"]["index_err"]],
               alt_band_fixed=[alt["pl_fixed"]["index"], alt["pl_fixed"]["index_err"]],
               alt_band_sigma_fixed=alt["sigma_fixed"])
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "selection_tests.json"), "w"), indent=1)


if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] in ("selection", "dm")):
    what = sys.argv[1] if len(sys.argv) > 1 else "selection"
    {"selection": selection, "dm": dm_grid}[what]()


# ---------------------------------------------------------------------------
# Joint eight-profile fit (App. A.3)
# ---------------------------------------------------------------------------
from scipy.special import log_ndtr, ndtr          # noqa: E402
from scipy.optimize import least_squares          # noqa: E402


def emg_cdf(x, mu, sigma, tau):
    lam = 1.0 / tau
    u = (x - mu) / sigma
    return ndtr(u) - np.exp(-lam * (x - mu) + 0.5 * (lam * sigma) ** 2 + log_ndtr(u - lam * sigma))


def emg_bin_exact(t, A, mu, sigma, tau, C, dt):
    """Exact bin average of the EMG (difference of its CDF across the bin)."""
    return A * (emg_cdf(t + 0.5 * dt, mu, sigma, tau) - emg_cdf(t - 0.5 * dt, mu, sigma, tau)) / dt + C


def joint(nwalk=96, nstep=JOINT_STEPS, burn=JOINT_BURN, noise_mode="scaled", band=None, nsub=None, axis="subband", tag="",
          shared_mu=False, de_moves=True):
    import emcee
    b = L.Burst(F.H5)
    base = F.analyse(b.data, b, band=band or F.BAND, nsub=nsub or F.N_SUB)
    keep = base["keep"]
    profs = base["profs"][keep]
    nu = base["nu"][keep]
    nr = base["nu_ref"]
    t = L.subband_time_axis(b.ntime, b.dt) if axis == "subband" else b.t
    noise = np.array([L.offpulse_rms(y, b)[0] for y in profs])
    if noise_mode == "scaled":                     # same per-profile chi2 rescaling as the two-step fit
        noise = noise * np.sqrt([r["chi2_red"] for r, k in zip(base["free"], keep) if k])
    x = np.log(nu / nr)
    J = len(nu)
    free = [r for r, k in zip(base["free"], keep) if k]
    # Parameters: ln tau_ref, alpha, [mu if shared], then per sub-band
    # (A, sigma, C) [+ mu if not shared]. The Gaussian-component width,
    # amplitude and baseline vary independently between sub-bands; by
    # default the onset mu does too (--shared-mu shares it).
    p0 = [np.log(base["pl_free"]["ref"]), base["pl_free"]["index"]]
    if shared_mu:
        p0 += [float(np.median([r["mu"] for r in free]))]
    for r in free:
        p0 += [r["A"], r["sigma"], r["C"]] + ([] if shared_mu else [r["mu"]])
    p0 = np.array(p0)
    floor = 0.5 * b.dt
    nper = 3 if shared_mu else 4
    off = 3 if shared_mu else 2
    i_sig = 1                                     # index of sigma within a sub-band block

    def model(p):
        lt, a = p[0], p[1]
        taus = np.exp(lt + a * x)
        P = p[off:].reshape(J, nper)
        mus = np.full(J, p[2]) if shared_mu else P[:, 3]
        return np.array([emg_bin_exact(t, P[j, 0], mus[j], P[j, 1], taus[j], P[j, 2], b.dt)
                         for j in range(J)]), taus

    def resid(p):
        m, _ = model(p)
        return ((profs - m) / noise[:, None]).ravel()

    if shared_mu:
        lo = np.r_[-np.inf, -np.inf, t[0], np.tile([0, floor, -np.inf], J)]
        hi = np.r_[np.inf, np.inf, t[-1], np.tile([np.inf, 30, np.inf], J)]
    else:
        lo = np.r_[-np.inf, -np.inf, np.tile([0, floor, -np.inf, t[0]], J)]
        hi = np.r_[np.inf, np.inf, np.tile([np.inf, 30, np.inf, t[-1]], J)]
    ml = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac", max_nfev=20000)
    pml = ml.x
    print(f"joint ML: alpha {pml[1]:.3f}, tau_ref {np.exp(pml[0]):.3f} ms")

    def logp(p):
        if np.any(p < lo) or np.any(p > hi):
            return -np.inf
        _, taus = model(p)
        if np.any(taus < floor):
            return -np.inf
        r = resid(p)
        return -0.5 * np.dot(r, r)

    ndim = len(pml)
    rng = np.random.default_rng(42)
    scale = np.abs(pml) * 1e-4 + 1e-5
    start = pml + scale * rng.normal(size=(nwalk, ndim))
    # differential-evolution moves mix far better than the default stretch move
    # in this ~34-parameter space (tau_int drops several-fold)
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)] if de_moves else None
    sampler = emcee.EnsembleSampler(nwalk, ndim, logp, moves=moves)
    sampler.random_state = np.random.RandomState(42).get_state()   # reproducible chains
    sampler.run_mcmc(start, nstep, progress=False)
    ch = sampler.get_chain(discard=burn, thin=10, flat=True)
    a_s, t_s = ch[:, 1], np.exp(ch[:, 0])
    qa, qt = np.percentile(a_s, [16, 50, 84]), np.percentile(t_s, [16, 50, 84])
    rs = []
    for j in range(J):
        tau_j = np.exp(ch[:, 0] + ch[:, 1] * x[j])
        rs.append(np.corrcoef(ch[:, off + nper * j + i_sig], tau_j)[0, 1])
    try:
        tau_all = sampler.get_autocorr_time(discard=burn, quiet=True)
        tauint, tau_max_all = float(np.max(tau_all[:2])), float(np.max(tau_all))
    except Exception:                                                       # noqa: BLE001
        tauint = tau_max_all = np.nan
    out = dict(alpha=[qa[1], qa[2] - qa[1], qa[1] - qa[0]], tau_ref=[qt[1], qt[2] - qt[1], qt[1] - qt[0]],
               max_abs_r_sigma_tau=float(np.max(np.abs(rs))), r_sigma_tau=[float(v) for v in rs],
               autocorr_steps=float(tauint), autocorr_steps_all_params=tau_max_all,
               n_tau_after_burn=float((nstep - burn) / tauint),
               n_tau_after_burn_all_params=float((nstep - burn) / tau_max_all),
               n_samples=int(len(a_s)), noise_mode=noise_mode, nstep=nstep, burn=burn,
               moves="DE+snooker" if de_moves else "stretch",
               ess=float(nwalk * (nstep - burn) / tau_max_all),
               shared_mu=shared_mu, band=list(band or F.BAND),
               axis="exact")
    print(f"joint MCMC ({noise_mode} noise, shared_mu={shared_mu}, {tag or 'final'}): alpha {qa[1]:.3f} +{qa[2]-qa[1]:.3f}/-{qa[1]-qa[0]:.3f}, "
          f"tau_ref {qt[1]:.3f} +{qt[2]-qt[1]:.3f}/-{qt[1]-qt[0]:.3f} ms, max|r_sigma_tau| {np.max(np.abs(rs)):.2f}, "
          f"autocorr ~{tauint:.0f} steps")
    json.dump(out, open(os.path.join(HERE, f"joint_fit_{noise_mode}{tag}.json"), "w"), indent=1)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "joint":
    # python appendix_tests.py joint [scaled|raw] [--shared-mu] [--intermediate] [--steps N] [--burn N]
    args = sys.argv[2:]
    mode = args[0] if args and not args[0].startswith("--") else "scaled"
    kw = dict(noise_mode=mode, nstep=JOINT_STEPS, burn=JOINT_BURN)
    tag = ""
    if "--steps" in args:
        kw["nstep"] = int(args[args.index("--steps") + 1])
    if "--burn" in args:
        kw["burn"] = int(args[args.index("--burn") + 1])
    if "--shared-mu" in args:
        kw["shared_mu"] = True
        tag += "_sharedmu"
    if "--stretch" in args:
        kw["de_moves"] = False
        tag += "_stretch"
    if "--intermediate" in args:
        kw.update(band=(428.6, 765.6), nsub=8)
        tag += "_intermediate"
    joint(tag=tag, **kw)


def foreground():
    """Galactic pulse-broadening predictions at nu_ref (Sec. 4).
    tau at 1 GHz along l=122.207, b=25.264 from pygedm (values stored here):
    NE2001 4.178e-7 s and YMW16 2.225e-6 s (from its empirical tau-DM
    relation, Krishnakumar et al. 2015), both scaled as nu^-4, as for the
    scintillation bandwidths in ../scintillation/acf_scintillation.py."""
    core = json.load(open(os.path.join(HERE, "results.json")))
    nu = core["nu_ref"] / 1e3
    ne = 4.178e-7 * nu ** -4.0 * 1e6
    ym = 2.225e-6 * nu ** -4.0 * 1e6
    tau = core["free"]["ref"] * 1e3
    out = dict(nu_ref=core["nu_ref"], ne2001_us=ne, ymw16_us=ym, tau_meas_us=tau,
               excess_ne2001=tau / ne, excess_ymw16=tau / ym)
    print(f"at {core['nu_ref']:.1f} MHz: NE2001 {ne:.1f} us, YMW16 {ym:.1f} us; measured {tau / 1e3:.2f} ms "
          f"-> excess {tau / ne:.2e} (NE2001), {tau / ym:.2e} (YMW16)")
    json.dump(out, open(os.path.join(HERE, "foreground.json"), "w"), indent=1)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "foreground":
    foreground()
