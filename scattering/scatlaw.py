"""
scatlaw.py -- core library for the FRB 20210415C frequency-resolved
scattering-law analysis: data access, sub-band construction, dedrifting,
bin-integrated EMG and Gaussian profile fits, and power-law fits. Choices not
stated in the paper are marked ASSUMED where they are made.

Methods and sources:
  * data: Stokes-I dynamic spectrum, channel mask, DM and burst parameters from
    the Second CHIME/FRB Catalog (CHIME/FRB Collaboration 2026), doi:10.11570/25.0066;
  * S(nu): the catalogue's running-power-law spectral model, in the
    convention of fitburst (Fonseca et al. 2024, ApJS; arXiv:2311.05829);
  * profile model: exponentially modified Gaussian (Gaussian convolved with a
    one-sided exponential), evaluated in a numerically stable form with the
    scaled complementary error function (scipy.special.erfcx) and averaged
    over each time bin;
  * dedispersion delays use the catalogue's dispersion constant.
"""
import json

import h5py
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc, erfcx

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
NATIVE_RES_MHZ = 0.0244140625


class Burst:
    """Catalog 2 Stokes-I dynamic spectrum, native resolution."""

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            self.freqs = f["index_map/freqs"][:]                 # ascending MHz
            self.good = f["good_freq"][:].astype(bool)           # catalogue mask
            self.data = f["data"][:].astype(np.float64)          # (nchan, ntime)
            self.dt = float(f.attrs["res_time"]) * 1e3           # ms
            self.dm = float(f.attrs["dm_incoherent"])
            self.dm_const = float(f.attrs["dm_const"])           # MHz^2 s / (pc cm^-3)
            self.dedisp_ref = float(f.attrs["ref_freq"])         # MHz
            self.on_region = tuple(int(v) for v in np.ravel(f.attrs["pulse_emission_region"])[:2])
            bp = json.loads(f.attrs["burst_parameters_json"])
        self.spec_index = bp["spectral_index"][0]
        self.spec_running = bp["spectral_running"][0]
        self.spec_ref = bp["ref_freq"][0]
        self.ntime = self.data.shape[1]
        # sample-centre times, sample 0 centred at t = 0 (ms)
        self.t = np.arange(self.ntime) * self.dt

    def S(self, nu):
        """Catalog 2 running-power-law spectral model (fitburst convention)."""
        x = np.log(np.asarray(nu, float) / self.spec_ref)
        return np.exp(self.spec_index * x + self.spec_running * x * x)

    def redisperse(self, ddm):
        """Return a copy of the data dedispersed to DM + ddm (Fourier shift)."""
        if ddm == 0:
            return self.data.copy()
        # extra delay of channel nu relative to the dedispersion reference
        delay_ms = 1e3 * self.dm_const * ddm * (self.freqs ** -2 - self.dedisp_ref ** -2)
        return fourier_shift_rows(self.data, -delay_ms / self.dt)


def fourier_shift_rows(arr, shift_samples):  # used for DM re-dedispersion
    """Shift each row of arr by shift_samples[i] (positive = later) using a
    Fourier phase ramp (exact for band-limited signals, preserves noise)."""
    n = arr.shape[1]
    k = np.fft.rfftfreq(n)
    ph = np.exp(-2j * np.pi * np.outer(np.atleast_1d(shift_samples), k))
    return np.fft.irfft(np.fft.rfft(arr, axis=1) * ph, n=n, axis=1)


# ---------------------------------------------------------------------------
# Sub-band construction (Sec. 2.2)
# ---------------------------------------------------------------------------
def build_subbands(burst, lo, hi, nsub):
    """nsub contiguous sub-bands with equal numbers of unmasked channels in
    [lo, hi] MHz (np.array_split: counts differ by at most one)."""
    sel = burst.good & (burst.freqs >= lo) & (burst.freqs <= hi)
    parts = np.array_split(np.flatnonzero(sel), nsub)
    sbs = []
    for p in parts:
        nu = burst.freqs[p]
        w = burst.S(nu)
        sbs.append(dict(
            chans=p, nu=nu,
            nu_flux=float(np.sum(w * nu) / np.sum(w)),     # eq. effective_frequency
            nu_arith=float(nu.mean()),
            f_lo=float(nu[0]), f_hi=float(nu[-1]),
            unmasked_frac=len(p) / (p[-1] - p[0] + 1),
        ))
    return sbs


def dedrift_rows(rows, nu, nu_c, dtdnu, dt):
    """Shift each channel by dt_i = -dtdnu*(nu_i - nu_c) (eq. dedrift) using
    linear interpolation between samples (np.interp). Fourier and
    cubic-spline shifts give the same alpha to <0.001."""
    if dtdnu == 0:
        return rows
    sh = -dtdnu * (nu - nu_c) / dt                 # samples, positive = later
    k = np.arange(rows.shape[1])
    return np.array([np.interp(k - s, k, r) for r, s in zip(rows, sh)])


def subband_profiles(data, burst, sbs, dtdnu, centre="nu_flux"):
    """Dedrift each unmasked channel about the sub-band centre and sum."""
    return np.array([dedrift_rows(data[sb["chans"]], sb["nu"], sb[centre], dtdnu,
                                  burst.dt).sum(axis=0) for sb in sbs])


def subband_time_axis(n, dt):
    """Time axis for the sub-band EMG fits (sample centres, sample 0 at t=0)."""
    return np.arange(n) * dt


def pipeline_time_correction(n):
    """Factor converting a duration measured on the per-channel arrival-time
    pipeline's linspace(0, n*dt, n) axis (step dt*n/(n-1)) to true time:
    (n - 1) / n."""
    return (n - 1) / n


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def emg_pdf(t, mu, sigma, tau):
    """Unit-area EMG (Gaussian * causal exponential), numerically stable."""
    t = np.asarray(t, float)
    lam = 1.0 / tau
    z = (mu + lam * sigma ** 2 - t) / (np.sqrt(2) * sigma)
    out = np.empty_like(t)
    pos = z >= 0
    out[pos] = 0.5 * lam * np.exp(-0.5 * ((t[pos] - mu) / sigma) ** 2) * erfcx(z[pos])
    a = 0.5 * lam * (2 * mu + lam * sigma ** 2 - 2 * t[~pos])
    out[~pos] = 0.5 * lam * np.exp(a) * erfc(z[~pos])
    return out


OVERSAMP = 32
_OFFS = (np.arange(OVERSAMP) + 0.5) / OVERSAMP - 0.5


def emg_binned(t, A, mu, sigma, tau, C, dt):
    """Bin-integrated EMG + baseline: mean of the model across each bin."""
    tg = (np.asarray(t)[:, None] + _OFFS[None, :] * dt).ravel()
    return A * emg_pdf(tg, mu, sigma, tau).reshape(len(t), OVERSAMP).mean(1) + C


def gauss_binned(t, A, mu, sigma, C, dt):
    tg = (np.asarray(t)[:, None] + _OFFS[None, :] * dt).ravel()
    g = np.exp(-0.5 * ((tg - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    return A * g.reshape(len(t), OVERSAMP).mean(1) + C


# ---------------------------------------------------------------------------
# Profile fitting
# ---------------------------------------------------------------------------
def offpulse_rms(y, burst):
    """Per-bin noise: standard deviation (numpy default, ddof=0) of the
    leading off-pulse samples [0, t0 - 5), where t0 is the start of the
    catalogue pulse_emission_region. The trailing side is not used because
    the low-frequency tail runs to the end of the window. Gives the paper's
    full-band chi2_red,prof = 7.3 (157 dof)."""
    sel = np.arange(len(y)) < burst.on_region[0] - 5
    return float(np.std(y[sel])), sel


def fit_emg(y, t, dt, noise, sigma_fixed=None, p0=None, floor=None):
    """Bin-integrated EMG fit with absolute per-bin noise.

    Returns dict with best-fit values, formal errors, chi2_red and errors
    rescaled by sqrt(chi2_red) (the profile residual-variance rescaling)."""
    floor = 0.5 * dt if floor is None else floor          # sigma, tau >= dt/2
    ipk = int(np.argmax(y))
    if p0 is None:
        base0 = np.median(y[: max(10, ipk // 2)])
        area0 = max(np.sum(y - base0) * dt, 1e-12)
        p0 = [area0, t[ipk] - 1.0, 1.0, 5.0, base0]
    lo = [0.0, t[0], floor, floor, -np.inf]
    hi = [np.inf, t[-1], 30.0, 500.0, np.inf]
    if sigma_fixed is not None:
        lo[2], hi[2] = sigma_fixed * (1 - 1e-9), sigma_fixed * (1 + 1e-9)
        p0 = list(p0)
        p0[2] = sigma_fixed
    p0 = [min(max(v, l + 1e-12 if np.isfinite(l) else v), h - 1e-12 if np.isfinite(h) else v)
          for v, l, h in zip(p0, lo, hi)]
    f = lambda tt, A, mu, s, ta, C: emg_binned(tt, A, mu, s, ta, C, dt)
    popt, pcov = curve_fit(f, t, y, p0=p0, bounds=(lo, hi), sigma=np.full(len(y), noise),
                           absolute_sigma=True, maxfev=200000)
    model = f(t, *popt)
    npar = 4 if sigma_fixed is not None else 5
    chi2 = float(np.sum(((y - model) / noise) ** 2))
    dof = len(y) - npar
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    if sigma_fixed is not None:
        perr[2] = 0.0
    scale = np.sqrt(chi2 / dof)
    return dict(A=popt[0], mu=popt[1], sigma=popt[2], tau=popt[3], C=popt[4],
                popt=popt, perr=perr, chi2=chi2, dof=dof, chi2_red=chi2 / dof,
                perr_scaled=perr * scale, model=model,
                at_floor=bool((popt[3] <= floor * 1.001) or
                              (sigma_fixed is None and popt[2] <= floor * 1.001)))


def fit_gauss(y, t, dt, noise, p0=None):
    ipk = int(np.argmax(y))
    if p0 is None:
        base0 = np.median(y[: max(10, ipk // 2)])
        p0 = [max(np.sum(y - base0) * dt, 1e-12), t[ipk], 3.0, base0]
    f = lambda tt, A, mu, s, C: gauss_binned(tt, A, mu, s, C, dt)
    popt, pcov = curve_fit(f, t, y, p0=p0, sigma=np.full(len(y), noise),
                           absolute_sigma=True, maxfev=200000,
                           bounds=([0, t[0], 0.5 * dt, -np.inf], [np.inf, t[-1], 100, np.inf]))
    model = f(t, *popt)
    chi2 = float(np.sum(((y - model) / noise) ** 2))
    return dict(popt=popt, chi2=chi2, dof=len(y) - 4, model=model)


# ---------------------------------------------------------------------------
# Power laws
# ---------------------------------------------------------------------------
def powerlaw_wls(nu, val, err, nu_ref, inflate=True):
    """Weighted least squares of ln(val) = ln(ref) + idx*ln(nu/nu_ref) with
    weights 1/(err/val)^2 (absolute). If inflate and chi2_red > 1, the
    covariance is multiplied by chi2_red (errors by sqrt(chi2_red))."""
    nu, val, err = map(lambda a: np.asarray(a, float), (nu, val, err))
    x, y, s = np.log(nu / nu_ref), np.log(val), err / val
    w = 1.0 / s ** 2
    X = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = cov @ (X.T @ (w * y))
    resid = y - X @ beta
    chi2 = float(np.sum(w * resid ** 2))
    dof = len(x) - 2
    chi2r = chi2 / dof if dof > 0 else np.nan
    fac = chi2r if (inflate and chi2r > 1) else 1.0
    cov_s = cov * fac
    ref = float(np.exp(beta[0]))
    return dict(index=float(beta[1]), index_err=float(np.sqrt(cov_s[1, 1])),
                ref=ref, ref_err=float(ref * np.sqrt(cov_s[0, 0])),
                chi2=chi2, dof=dof, chi2_red=chi2r, cov=cov_s,
                index_err_formal=float(np.sqrt(cov[1, 1])))


def powerlaw_ols(nu, val, nu_ref):
    """Unweighted least squares of ln(val) on ln(nu/nu_ref), with the
    covariance scaled by the residual variance (np.polyfit(..., cov=True)
    convention). This is how the width-frequency index beta was fitted:
    it reproduces beta = -1.35 +/- 0.22 (all nine) and -0.93 +/- 0.19
    (above 440 MHz); the weighted fit does not."""
    nu, val = np.asarray(nu, float), np.asarray(val, float)
    x, y = np.log(nu / nu_ref), np.log(val)
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(x) - 2
    cov = np.linalg.inv(X.T @ X) * (resid @ resid) / dof
    ref = float(np.exp(beta[0]))
    return dict(index=float(beta[1]), index_err=float(np.sqrt(cov[1, 1])),
                ref=ref, ref_err=float(ref * np.sqrt(cov[0, 0])), dof=dof, cov=cov)
