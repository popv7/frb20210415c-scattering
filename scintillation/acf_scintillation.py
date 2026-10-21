#!/usr/bin/env python
"""
acf_scintillation.py -- measure the scintillation bandwidth nu_s from a CHIME
Cat2 Stokes-I waterfall, in each clean contiguous block of channels.

Method follows Jahns-Schindler et al. (2026) sec 2.2:
  - on-pulse spectrum, per-channel baseline subtracted
  - one-sided ACF using the MEAN of the intensity product (not the sum), so the
    modulation index m is recoverable as a fit parameter
  - zero lag excluded (radiometer noise lands there)
  - Lorentzian fit  f(dnu) = m^2 / (1 + (dnu/nu_s)^2)
  - a second broad Lorentzian absorbs the burst's own spectral envelope
  - per-lag uncertainties from propagated channel noise AND from an off-pulse
    control; the larger of the two is used as the fit weight

Outputs a fitted nu_s and m per block, a nu^4 scaling test across blocks, and a
diagnostic panel.

Usage:
    python acf_scintillation.py                 # uses the CONFIG block
    python acf_scintillation.py /path/to.h5

The per-block estimator is analyse_block() and the data preparation is
prepare(), so that inject_scintillation.py runs the identical estimator on
injected spectra.

Methods and sources:
  * ACF estimator and per-lag radiometer uncertainties: Jahns-Schindler et
    al. (2026, arXiv:2608.10452), sec. 2.2;
  * ACF normalised as the mean of lagged products of fractional deviations,
    so that ACF(0) -> m^2, following the convention of Masui et al. (2015,
    Nature 528, 523);
  * expected bandwidths from 2 pi nu_s tau = C1 (C1 = 1; Cordes & Rickett
    1998, ApJ 507, 846), with tau at 1 GHz from NE2001 (Cordes & Lazio 2002)
    and YMW16 (Yao, Manchester & Wang 2017) via PyGEDM (Price et al. 2021).
"""

import argparse
import os
import sys

import h5py
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ CONFIG =====================================
HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(os.path.dirname(HERE), "data", "FRB20210415C_stokesi_dynamic_spectrum.h5")
PLOT = os.path.join(HERE, "acf_diagnostic.png")

FMIN, FMAX = 595.0, 715.0   # search this range for clean blocks
MIN_RUN = 300               # channels; shorter runs can't constrain nu_s
MAX_LAG_MHZ = 2.0           # fit out to this lag
SMOOTH_FRAC = 0.10          # envelope smoothing width, as a fraction of block
# =========================================================================

# Galactic predictions for l=122.207, b=25.264 (pygedm). tau at 1 GHz, seconds.
TAU_1GHZ = {"NE2001": 4.178e-07, "YMW16": 2.225e-06}


def nu_s_pred_khz(model, freq_mhz):
    tau = TAU_1GHZ[model] * (freq_mhz / 1000.0) ** -4
    return 1.0 / (2 * np.pi * tau) / 1e3


def lorentzian(dnu, m2, nus):
    return m2 / (1.0 + (dnu / nus) ** 2)


def two_lorentzian(dnu, m2n, nus, m2b, nub):
    """Narrow (scintillation) plus broad (burst spectral envelope)."""
    return lorentzian(dnu, m2n, nus) + lorentzian(dnu, m2b, nub)


def acf_mean_product(spec, ok, nlag):
    """
    One-sided ACF using the mean of the intensity product, so ACF(0) -> m^2.

    spec : channel intensities (arbitrary units), masked entries ignored
    ok   : bool, which channels are usable
    Returns lags 1..nlag and the ACF, plus the pair count at each lag.
    """
    mu = spec[ok].mean()
    d = np.where(ok, (spec - mu) / mu, 0.0)
    v = ok.astype(float)
    acf = np.zeros(nlag)
    npair = np.zeros(nlag)
    for k in range(1, nlag + 1):
        w = v[:-k] * v[k:]
        n = w.sum()
        npair[k - 1] = n
        acf[k - 1] = (d[:-k] * d[k:] * w).sum() / n if n > 0 else np.nan
    return np.arange(1, nlag + 1), acf, npair


def acf_noise_errors(spec, sigma, ok, nlag):
    """
    Per-lag ACF uncertainty propagated from per-channel noise sigma, following
    the structure of eqs (2)-(4) of Jahns-Schindler et al. (2026): each product
    term contributes its own channel noise times the other channel's signal.
    """
    mu = spec[ok].mean()
    s = np.where(ok, spec / mu, 0.0)
    sg = np.where(ok, sigma / mu, 0.0)
    v = ok.astype(float)
    err = np.zeros(nlag)
    for k in range(1, nlag + 1):
        w = v[:-k] * v[k:]
        n = w.sum()
        if n == 0:
            err[k - 1] = np.nan
            continue
        t = (sg[:-k] * s[k:]) ** 2 + (sg[k:] * s[:-k]) ** 2
        err[k - 1] = np.sqrt((t * w).sum()) / n
    return err


def smooth(y, ok, width):
    """Gaussian smooth of a masked series, for envelope removal."""
    x = np.arange(y.size, dtype=float)
    g = np.exp(-0.5 * ((x - x.mean()) / width) ** 2)
    g /= g.sum()
    num = np.convolve(np.where(ok, y, 0.0), g, mode="same")
    den = np.convolve(ok.astype(float), g, mode="same")
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def find_blocks(ok, freqs, fmin, fmax, min_run):
    win = (freqs >= fmin) & (freqs <= fmax)
    m = ok & win
    padded = np.concatenate(([False], m, [False]))
    e = np.diff(padded.astype(np.int8))
    starts, ends = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    return [(a, b - 1) for a, b in zip(starts, ends) if (b - a) >= min_run]


def fit_block(lag_mhz, acf, err, nus_guess):
    """Try the two-component fit, fall back to single Lorentzian."""
    g = [0.5, nus_guess, 0.5, 20 * nus_guess]
    bounds = ([0, nus_guess / 30, 0, nus_guess], [5, nus_guess * 30, 5, 1e4])
    try:
        p, c = curve_fit(two_lorentzian, lag_mhz, acf, p0=g, sigma=err,
                         absolute_sigma=True, bounds=bounds, maxfev=40000)
        return p, np.sqrt(np.diag(c)), "two"
    except Exception:
        pass
    try:
        p, c = curve_fit(lorentzian, lag_mhz, acf, p0=[0.5, nus_guess],
                         sigma=err, absolute_sigma=True,
                         bounds=([0, nus_guess / 30], [5, nus_guess * 30]),
                         maxfev=40000)
        return np.array([p[0], p[1], 0.0, 1e3]), \
               np.array([np.sqrt(c[0, 0]), np.sqrt(c[1, 1]), 0.0, 0.0]), "one"
    except Exception:
        return None, None, "failed"


def load(path):
    """Read the Cat2 waterfall and the attributes the analysis needs."""
    with h5py.File(path, "r") as f:
        freqs = f["index_map/freqs"][:]
        ok_all = f["good_freq"][:].astype(bool)
        data = f["data"][:].astype(np.float64)
        model = f["model"][:].astype(np.float64)
        attrs = dict(f.attrs)
    return freqs, ok_all, data, model, attrs


def prepare(data, attrs):
    """
    Baseline-subtract each channel and form the on-pulse spectrum, its
    per-channel noise, and an off-pulse control spectrum of equal duration.
    Returns (spec, sigma, spec_ctrl, base, (t0, t1)).
    """
    per = np.ravel(attrs["pulse_emission_region"])
    t0, t1 = int(per[0]), int(per[1])
    off = np.ones(data.shape[1], dtype=bool)
    off[max(0, t0 - 5):min(data.shape[1], t1 + 5)] = False

    base = data[:, off].mean(axis=1, keepdims=True)
    resid = data - base
    mad = np.median(np.abs(resid[:, off]
                           - np.median(resid[:, off], axis=1, keepdims=True)),
                    axis=1)
    sigma = 1.4826 * mad * np.sqrt(t1 - t0)
    spec = resid[:, t0:t1].sum(axis=1)

    # Control: an off-pulse stretch of equal duration, same treatment.
    c0 = 5
    spec_ctrl = resid[:, c0:c0 + (t1 - t0)].sum(axis=1)
    return spec, sigma, spec_ctrl, base, (t0, t1)


def analyse_block(sb, sgb, cb, okb, res, fc, nlag, verbose=False,
                  freqs_block=None):
    """
    Run the full per-block estimator on one contiguous block.

    sb, sgb, cb : on-pulse spectrum, its per-channel noise, off-pulse control
    okb         : bool mask of usable channels in the block
    res         : channel width (MHz); fc: block centre (MHz)
    nlag        : maximum lag in channels before the n//3 cap
    Returns a dict, or None if the fit failed.
    """
    n = okb.sum()
    nl = min(nlag, n // 3)

    lags, acf, npair = acf_mean_product(sb, okb, nl)
    e_noise = acf_noise_errors(sb, sgb, okb, nl)
    _, acf_c, _ = acf_mean_product(np.abs(cb) + np.abs(cb).mean(), okb, nl)
    e_ctrl = np.full(nl, np.nanstd(acf_c[nl // 3:]))
    lag_mhz = lags * res

    guess = nu_s_pred_khz("NE2001", fc) / 1e3

    # Adjacent ACF lags are strongly correlated, so the number of INDEPENDENT
    # samples is the number of scintles across the block, not the number of
    # channels.  Without this floor the errors are wildly optimistic and
    # chi2 explodes.  Two passes: floor from the predicted nu_s, refit with
    # the floor recomputed from the fitted nu_s.
    p = pe = None
    nus_it = guess
    for _ in range(2):
        n_scint = max((n * res) / max(nus_it, res), 2.0)
        e_scint = np.full(nl, 1.0 / np.sqrt(n_scint))
        err = np.fmax(np.fmax(e_noise, e_ctrl), e_scint)
        p, pe, kind = fit_block(lag_mhz, acf, err, guess)
        if p is None:
            break
        nus_it = p[1]

    # Cross-check: remove the envelope by subtracting a smoothed spectrum,
    # then fit a single Lorentzian.  Should agree with the two-component fit.
    # The flattened spectrum is offset by the mean on-pulse spectrum, so its
    # ACF is the fractional modulation of the burst, as in the two-component
    # branch.
    env = smooth(sb, okb, max(5.0, SMOOTH_FRAC * n))
    flat = np.where(okb & np.isfinite(env), sb - env, 0.0)
    offset = sb[okb].mean()
    _, acf_f, _ = acf_mean_product(flat + offset, okb, nl)
    try:
        pf, _ = curve_fit(lorentzian, lag_mhz, acf_f, p0=[0.3, guess],
                          bounds=([0, guess / 30], [5, guess * 30]),
                          maxfev=40000)
        nus_flat = pf[1]
    except Exception:
        nus_flat = np.nan

    if p is None:
        return None

    nus, nus_e = p[1], pe[1]
    m = np.sqrt(max(p[0], 0))
    m_e = 0.5 * pe[0] / max(m, 1e-6)

    # If the "broad" component has collapsed onto the narrow one, the two
    # are degenerate and the covariance is meaningless even though the
    # value may be fine.  Detect it and say so rather than quoting nonsense.
    degenerate = (kind == "two" and (p[3] < 4 * p[1] or nus_e > 0.5 * nus))
    acf_used, p_used = acf, p            # what the reported fit was made on
    if degenerate:
        try:
            pf2, cf2 = curve_fit(lorentzian, lag_mhz, acf_f,
                                 p0=[0.3, guess], sigma=err,
                                 absolute_sigma=True,
                                 bounds=([0, guess / 30], [5, guess * 30]),
                                 maxfev=40000)
            nus, nus_e = pf2[1], np.sqrt(cf2[1, 1])
            m, m_e = np.sqrt(max(pf2[0], 0)), 0.5 * np.sqrt(cf2[0, 0]) / \
                max(np.sqrt(max(pf2[0], 0)), 1e-6)
            kind = "one (envelope-subtracted; two-component degenerate)"
            # Plot the ACF that was actually fit, not the raw one -- these
            # are different data and overlaying one on the other's model
            # makes the figure look broken when it isn't.
            acf_used, p_used = acf_f, np.array([pf2[0], pf2[1], 0.0, 1e3])
        except Exception:
            kind = "two (DEGENERATE - error bar unreliable)"
    resid_fit = acf_used - (two_lorentzian(lag_mhz, *p_used)
                            if kind == "two"
                            else lorentzian(lag_mhz, p_used[0], p_used[1]))
    chi2 = np.nansum((resid_fit / err) ** 2) / max(nl - 4, 1)

    # A detection needs a resolved width AND a significant modulation.
    # Either one alone is meaningless.
    sig_nus = nus / nus_e if nus_e > 0 else 0.0
    sig_m = m / m_e if m_e > 0 else 0.0
    detected = (sig_nus > 3) and (sig_m > 3) and (nus > 1.5 * res)

    if verbose:
        a_f, b_f = freqs_block[0], freqs_block[-1]
        print(f"\n--- {a_f:.2f} - {b_f:.2f} MHz  "
              f"(centre {fc:.1f}, {n} chans, {kind}) ---")
        print(f"  nu_s          = {nus*1e3:7.1f} +/- {nus_e*1e3:.1f} kHz "
              f"( {nus/res:.1f} chan, {sig_nus:.1f} sigma )")
        print(f"  m             = {m:.2f} +/- {m_e:.2f}  ( {sig_m:.1f} sigma )")
        print(f"  reduced chi2  = {chi2:.2f}")
        print(f"  envelope-subtracted cross-check nu_s = {nus_flat*1e3:.1f} kHz")
        print(f"  predicted: NE2001 {nu_s_pred_khz('NE2001', fc):.1f} kHz, "
              f"YMW16 {nu_s_pred_khz('YMW16', fc):.1f} kHz")
        print(f"  --> {'DETECTION' if detected else 'NON-DETECTION'}")
        if not detected:
            if sig_m <= 3:
                print("      no significant modulation: nothing to fit a width to")
            if nus <= 1.5 * res:
                print("      width at or below ~1.5 channels: unresolved")
        if chi2 > 3:
            print("  !! poor fit; inspect the residual panel")
        elif chi2 < 0.5:
            print("  !! chi2 << 1: error floor is over-conservative here")

    return dict(fc=fc, lag=lag_mhz, acf=acf_used, acf_raw=acf, err=err,
                acf_ctrl=acf_c, p=p_used, kind=kind, nus=nus, nus_e=nus_e,
                m=m, m_e=m_e, sig_nus=sig_nus, sig_m=sig_m, chi2=chi2,
                detected=detected, nus_flat=nus_flat)


def lag_structure(results, spec_ctrl, res, kmax=8, period=16):
    """Correlated structure at short lags (1..kmax channels): per block, the
    mean envelope-subtracted on-pulse ACF and the largest |ACF| of the
    off-pulse control (as normalised in analyse_block). Then a test for an
    upchannelisation ripple: each block's envelope-subtracted fractional
    spectrum is folded on the 16-channel period (phase = absolute channel
    index mod 16); a ripple common to the band would give folded patterns that
    correlate between blocks, more than for the off-pulse control."""
    print(f"\n=== short-lag structure ({res * 1e3:.0f}-{kmax * res * 1e3:.0f} kHz) ===")
    folds = {"on": [], "control": []}
    for r in results:
        sb, okb = r["spec"], r["ok"]
        n = okb.sum()
        cb = spec_ctrl[r["a"]:r["b"] + 1]
        env = smooth(sb, okb, max(5.0, SMOOTH_FRAC * n))
        flat = np.where(okb & np.isfinite(env), sb - env, 0.0)
        _, acf_f, _ = acf_mean_product(flat + sb[okb].mean(), okb, kmax)
        _, acf_c, _ = acf_mean_product(np.abs(cb) + np.abs(cb).mean(), okb, kmax)
        print(f"  {r['freqs'][0]:.1f} MHz: on-pulse mean ACF {acf_f.mean():.3f}, "
              f"control max |ACF| {np.abs(acf_c).max():.4f}")
        phase = np.arange(r["a"], r["b"] + 1) % period
        for key, x in (("on", sb), ("control", cb + sb[okb].mean())):
            e = smooth(x, okb, max(5.0, SMOOTH_FRAC * n))
            d = np.where(okb & np.isfinite(e), (x - e) / sb[okb].mean(), np.nan)
            folds[key].append([np.nanmean(d[phase == p]) for p in range(period)])
    for key, f in folds.items():
        f = np.array(f)
        rr = [np.corrcoef(f[i], f[j])[0, 1] for i in range(len(f)) for j in range(i + 1, len(f))]
        print(f"  {period}-channel folded pattern, mean between-block correlation ({key}): {np.mean(rr):+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=FILE)
    ap.add_argument("--fmin", type=float, default=FMIN)
    ap.add_argument("--fmax", type=float, default=FMAX)
    ap.add_argument("--min-run", type=int, default=MIN_RUN)
    ap.add_argument("--max-lag", type=float, default=MAX_LAG_MHZ)
    ap.add_argument("--plot", default=PLOT)
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"ERROR: no file at {args.path}", file=sys.stderr)
        return 1

    freqs, ok_all, data, _, attrs = load(args.path)
    res = attrs["res_freq"]                       # MHz per channel
    spec, sigma, spec_ctrl, _, _ = prepare(data, attrs)

    blocks = find_blocks(ok_all, freqs, args.fmin, args.fmax, args.min_run)
    print(f"file      : {args.path}")
    print(f"event_id  : {attrs.get('event_id')}   {attrs.get('tns_name')}")
    print(f"blocks with >= {args.min_run} clean channels in "
          f"{args.fmin:.0f}-{args.fmax:.0f} MHz: {len(blocks)}")
    if not blocks:
        print("no usable blocks -- lower --min-run or widen the range")
        return 1

    nlag = int(args.max_lag / res)
    results = []

    for (a, b) in blocks:
        fc = 0.5 * (freqs[a] + freqs[b])
        sl = slice(a, b + 1)
        okb = ok_all[sl]
        r = analyse_block(spec[sl], sigma[sl], spec_ctrl[sl], okb, res, fc,
                          nlag, verbose=True, freqs_block=freqs[sl])
        if r is None:
            print(f"\n{freqs[a]:.1f}-{freqs[b]:.1f} MHz: fit failed")
            continue
        r.update(a=a, b=b, spec=spec[sl], ok=okb, freqs=freqs[sl])
        results.append(r)

    lag_structure(results, spec_ctrl, res)

    # ---------------- verdict: detection or upper limit ----------------
    ndet = sum(r["detected"] for r in results)
    print(f"\n=== overall: {ndet} of {len(results)} blocks are detections ===")

    if ndet == 0:
        # Nothing to fit a scaling law to.  Combine the modulation indices into
        # a single limit -- that is the publishable number.
        m = np.array([r["m"] for r in results])
        me = np.array([r["m_e"] for r in results])
        ok_m = np.isfinite(m) & np.isfinite(me) & (me > 0)
        if ok_m.any():
            w = 1.0 / me[ok_m] ** 2
            mbar = (m[ok_m] * w).sum() / w.sum()
            mbe = 1.0 / np.sqrt(w.sum())
            print(f"  combined modulation index m = {mbar:.3f} +/- {mbe:.3f}")
            print(f"  -> 2-sigma upper limit       m < {mbar + 2*mbe:.2f}")
        print(f"  -> upper limit on nu_s: below the {res*1e3:.1f} kHz channel "
              f"width (unresolved)")
        pmin = min(nu_s_pred_khz("NE2001", r["fc"]) for r in results)
        pmax = max(nu_s_pred_khz("NE2001", r["fc"]) for r in results)
        print(f"  NE2001 predicts {pmin:.0f}-{pmax:.0f} kHz over these blocks, "
              f"which would have been resolved -- disfavoured")
        print(f"  YMW16 predicts "
              f"{min(nu_s_pred_khz('YMW16', r['fc']) for r in results):.0f}-"
              f"{max(nu_s_pred_khz('YMW16', r['fc']) for r in results):.0f} kHz, "
              f"sub-channel -- consistent with this non-detection")
        print("  NOTE: unresolved and quenched (source broadened by the far "
              "screen) are indistinguishable here")
        print("  no scaling test attempted: it is meaningless without "
              "significant per-block widths")

    elif ndet == 1:
        r = next(x for x in results if x["detected"])
        print(f"  single detection at {r['fc']:.0f} MHz: "
              f"nu_s = {r['nus']*1e3:.1f} +/- {r['nus_e']*1e3:.1f} kHz, "
              f"m = {r['m']:.2f} +/- {r['m_e']:.2f}")
        print("  no scaling test possible with one point -- the nu^4 check "
              "needs at least two detected blocks")
        print("  treat as provisional: intrinsic spectral structure can mimic "
              "a width in a single block")

    else:
        det = [r for r in results if r["detected"]]
        fc = np.array([r["fc"] for r in det])
        ns = np.array([r["nus"] for r in det])
        ne = np.array([r["nus_e"] for r in det])
        x, y, ey = np.log(fc), np.log(ns), ne / ns
        print("\n=== nu_s scaling (detections only) ===")
        if len(det) == 2:
            dx = x[1] - x[0]
            sl = (y[1] - y[0]) / dx
            sl_e = np.hypot(ey[0], ey[1]) / abs(dx)
        else:
            # cov='unscaled' is essential: np.polyfit otherwise rescales the
            # covariance by chi2/dof, which shrinks the error whenever the
            # points are unconstrained -- exactly when you must not.
            sl_arr, cov = np.polyfit(x, y, 1, w=1.0 / ey, cov="unscaled")
            sl, sl_e = sl_arr[0], np.sqrt(cov[0, 0])
        print(f"  nu_s ~ nu^({sl:.2f} +/- {sl_e:.2f})   "
              f"[expect +4 for scintillation, ~0 for intrinsic structure]")
        print(f"  lever arm: {fc.min():.0f} -> {fc.max():.0f} MHz "
              f"= factor {(fc.max()/fc.min())**4:.2f} in nu_s")
        if sl - 2 * sl_e > 1.0:
            print("  -> inconsistent with flat: supports scintillation")
        else:
            print("  -> not distinguishable from flat; supporting evidence only")

    # ---------------- diagnostic panel ----------------
    nb = len(results)
    fig = plt.figure(figsize=(13, 3.0 * (nb + 1)), constrained_layout=True)
    gs = fig.add_gridspec(nb + 1, 2)

    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(r["freqs"][r["ok"]], r["spec"][r["ok"]], lw=0.4)
        ax.set(xlabel="Frequency (MHz)", ylabel="on-pulse",
               title=f"Spectrum {r['freqs'][0]:.1f}-{r['freqs'][-1]:.1f} MHz")

        ax = fig.add_subplot(gs[i, 1])
        ax.errorbar(r["lag"] * 1e3, r["acf"], yerr=r["err"], fmt=".", ms=2,
                    lw=0.4, alpha=0.6, label="ACF")
        xf = np.linspace(r["lag"][0], r["lag"][-1], 600)
        model = (two_lorentzian(xf, *r["p"]) if r["kind"] == "two"
                 else lorentzian(xf, r["p"][0], r["p"][1]))
        ax.plot(xf * 1e3, model, "r-", lw=1.5, label="fit")
        ax.plot(xf * 1e3, lorentzian(xf, r["p"][0], r["p"][1]), "r--", lw=1,
                label="narrow only")
        ax.plot(r["lag"] * 1e3, r["acf_ctrl"], color="grey", lw=0.5,
                alpha=0.7, label="off-pulse control")
        ax.axvline(r["nus"] * 1e3, color="g", ls=":",
                   label=f"nu_s={r['nus']*1e3:.0f} kHz")
        ax.axvline(nu_s_pred_khz("NE2001", r["fc"]), color="b", ls=":",
                   alpha=0.6, label="NE2001")
        ax.set(xlabel="Lag (kHz)", ylabel="ACF",
               xlim=(0, min(600, r["lag"][-1] * 1e3)),
               title=f"centre {r['fc']:.0f} MHz, chi2r={r['chi2']:.2f}, "
                     f"{'DET' if r['detected'] else 'non-det'}")
        ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[nb, :])
    if results:
        fc = np.array([r["fc"] for r in results])
        det = np.array([r["detected"] for r in results])
        ns = np.array([r["nus"] * 1e3 for r in results])
        ne = np.array([r["nus_e"] * 1e3 for r in results])
        if det.any():
            ax.errorbar(fc[det], ns[det], yerr=ne[det], fmt="o",
                        label="detection")
        if (~det).any():
            # Non-detections are upper limits at the channel width, not
            # measurements -- drawing them as points with error bars would
            # imply a constraint that is not there.
            ax.errorbar(fc[~det], np.full((~det).sum(), res * 1e3),
                        yerr=res * 1e3 * 0.35, uplims=True, fmt="v",
                        color="grey", label="non-detection (upper limit)")
        ff = np.linspace(fc.min() * 0.95, fc.max() * 1.05, 100)
        for model, c in (("NE2001", "b"), ("YMW16", "orange")):
            ax.plot(ff, [nu_s_pred_khz(model, x) for x in ff], c, ls="--",
                    alpha=0.7, label=model)
        if det.any():
            f0, n0 = fc[det][0], ns[det][0]
            ax.plot(ff, n0 * (ff / f0) ** 4, "g:", label="nu^4 through pt 1")
        ax.axhline(res * 1e3, color="k", lw=0.8)
        ax.text(ff[0], res * 1e3 * 1.05, "1 channel", fontsize=8)
        ax.set(xlabel="Frequency (MHz)", ylabel="nu_s (kHz)", yscale="log",
               title="Scintillation bandwidth vs frequency")
        ax.legend(fontsize=8)

    fig.savefig(args.plot, dpi=130)
    print(f"\nplot written to {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
