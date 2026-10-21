#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 23:21:03 2025

@author: victorpop

Based on the Arrival Times Pipeline by Mohammed Chamma
(arrivaltimes.py in https://github.com/mef51/frbgui; originally
ArrivalTimes.ipynb in https://github.com/mef51/subdriftlaw), Copyright (c)
2022 Mohammed Chamma, MIT licence (see LICENSE-FRBGui). The per-channel
arrival-time method is that of Chamma et al. (2025, MNRAS).

Added in this version (by Victor Pop): EMG per-channel and sub-band fits, with arrival times
at the EMG's leading-edge half-maximum (EMG_halfwidth.py); emcee MCMC fits
(Foreman-Mackey et al. 2013); stacked-channel fits. Other sources used
inside: drifting-Gaussian models of Jahns et al. (2023, MNRAS 519, 666) via
driftrate.py; DM_phase (Seymour, Michilli & Pleunis 2019, ascl:1910.004),
called only with recalculateDM=True; the reduced chi-squared of a scipy.odr
fit follows https://stackoverflow.com/a/21406281 (CC BY-SA).

Modified to include MCMC fitting for full-band intensity profile

v56: Added sub-band scattering index (alpha) recovery and fiducial τ(ν) curve.
     When USE_EMG_MIX is True, splits band into N sub-bands, fits EMG in each,
     recovers τ(ν) = τ_ref*(ν/ν_ref)^α power law, uses closest-to-grid FWHM
     classification, and plots the fiducial scattering envelope on the waterfall.
     Works with both stacked and non-stacked pipelines.

"""

import numpy as np
import matplotlib 
matplotlib.use("Agg")  # patched for headless run
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.markers import MarkerStyle
from matplotlib.widgets import Button
import scipy, glob
import warnings
from itertools import zip_longest, cycle
from tqdm import tqdm
import pandas as pd
# from sklearn.mixture import GaussianMixture
import driftrate
from driftrate import scilabel, subburst_suffixes
import DM_phase
from scipy.special import erfc, erfcx
from EMG_halfwidth import emg_left_hwhm
import emcee  # for MCMC fitting
import acf_slope  # 2D-autocorrelation sub-burst slope (ACF mode)
import scint_bw  # diffractive scintillation-bandwidth (Delta_nu_d) measurement



# Based on https://github.com/mef51/subdriftlaw/blob/master/ArrivalTimes.ipynb
plt.rcParams['toolbar'] = 'toolbar2'
"""
When fitting rows, noise is sampled over a fraction of the data, from the start.
i.e. `data_sigma = np.std(data[0:len(data)//NOISE_WIDTH_FACTOR])`
"""
NOISE_WIDTH_FACTOR = 5

# === STACKED-FIT SNR FILTERS ===
# After stacking N channels per stack, the per-channel noise drops by ~sqrt(N)
# but the fitter can still happily converge to a near-zero EMG that "fits"
# pure noise (χ²_red ≈ 1 because residuals match the noise std). To reject
# these, every stacked TOA must clear two amplitude-SNR thresholds:
#
#   STACKED_AMP_SNR_MIN  : amp / amp_err must exceed this. This is the
#                          "Z-score" of the fitted amplitude against the null
#                          hypothesis A=0, computed from the curve_fit
#                          covariance matrix.
#   STACKED_PEAK_SNR_MIN : amp / std(off_pulse) must exceed this. This is an
#                          *external* check that the fitted amplitude rises
#                          above the off-pulse noise floor of the stacked
#                          profile itself, independent of the fit's pcov.
#                          The off-pulse window is the leading
#                          len(profile)//NOISE_WIDTH_FACTOR samples (matches
#                          fitemg's noise estimator).
#
# Set either to None to disable that filter. Both apply identically to the
# Gaussian-stacking path (stacked_gauss_df).
STACKED_AMP_SNR_MIN  = 4.0   # 3σ fit-covariance detection
STACKED_PEAK_SNR_MIN = 5.0   # 4× off-pulse-noise peak amplitude
# === ARRIVAL TIME CONVENTION TOGGLE ===
# When False (default): Arrival times are at the leading edge of the intensity profile
#   - Gaussian: tstart = mu - sqrt(2 ln 2) * sigma  (left half-maximum crossing)
#   - EMG: t_emg_hwhm = left half-width at half-maximum crossing point
# When True: Arrival times are at the peak (maximum intensity) of the fitted profile
#   - Gaussian: tstart = mu (the center/peak of the Gaussian)
#   - EMG: t_emg_peak = numerically determined peak of the EMG function
PEAK_INTENSITY_TOA = False

# === FIT TOGGLES (set exactly one to True) ===
USE_GAUSS_MIX = False   # runs like arrivaltimes_final (Gaussian time-domain fits)
USE_EMG_MIX   = True   # runs like arrivaltimes_scatter (EMG time-domain fits)

# safety: ensure only one is active
if USE_GAUSS_MIX and USE_EMG_MIX:
	raise RuntimeError("Only one of USE_GAUSS_MIX or USE_EMG_MIX may be True.")

if not (USE_GAUSS_MIX or USE_EMG_MIX):
	# default to Gaussian if neither specified
	USE_GAUSS_MIX = False
	
# === SIGNAL STACKING TOGGLE ===
USE_SIGNAL_STACKING =   False   # Enable/disable signal stacking
STACK_SIZE = 5               # Number of frequency channels to stack
stacked_wfall_visualization = False  # NEW: Enable/disable stacked waterfall visualization
# === END STACKING TOGGLE ===    

# === DIAGNOSTICS TOGGLE ===
SHOW_DIAGNOSTICS = False  # Master toggle - set to True to enable diagnostic categories below

# === DETAILED DIAGNOSTIC CONTROLS ===
# Each category can be individually enabled/disabled.
# Only active when SHOW_DIAGNOSTICS = True

DIAG = {
	# --- ERROR BAR ANALYSIS (for debugging large error bars) ---
	'ERROR_BAR_SUMMARY': True,       # Consolidated error bar diagnostic summary at end
	'EMG_COVARIANCE': False,          # EMG mixture fit covariance matrix analysis
	'SPECTRAL_COVARIANCE': False,     # Spectral fit covariance analysis
	'ERROR_BAR_COMPONENT': False,     # Per-component error bar comparison
	'TRACING': False,                 # Tracing horizontal error bar origin
	
	# --- FIT QUALITY DIAGNOSTICS ---
	'INTENSITY_PROFILE': False,       # DIAG 1: Intensity profile fit parameters
	'PER_ROW_FIT': False,             # DIAG 2: Per-row fitting results
	'FILTER_CHAIN': False,            # DIAG 3: Filter chain (SNR, amplitude, uncertainty, time)
	'FINAL_SUMMARY': False,           # DIAG 4: Final summary across components
	'LINE_FIT': False,                # Line fit diagnostic
	'CHI_SQUARED': False,             # Chi-squared diagnostic report
	
	# --- TIME/COORDINATE DIAGNOSTICS ---
	'SUBFALL': False,                 # Subfall window diagnostic
	'SUBPKTIME': False,               # Subpktime calculation diagnostic
	'TIME_FILTER': False,             # Time filter diagnostic (detailed)
	'ARRIVAL_TIME_PLOT': False,       # Arrival time plotting diagnostic
	'SLOPE_PLOT': False,              # Slope line plotting diagnostic
	'BOTTOM_SUBPLOT': False,          # Bottom subplot (slope) diagnostic
	
	# --- STACKING DIAGNOSTICS ---
	'STACKING_EMG': False,            # EMG signal stacking diagnostics
	'STACKING_GAUSS': False,          # Gaussian signal stacking diagnostics
	'STACKING_FILTER': False,         # Stacked TOA filter diagnostics
	
	# --- OTHER ---
	'MODEL_HEADER': False,            # Model comparison diagnostics header
	'SNR': False,                     # SNR calculation diagnostics
	'EMG_FIT_SUMMARY': False,         # EMG fit summary for per-row fits
	'PER_ROW_EMG': False,             # Per-row EMG fit diagnostics (detailed)
	'BURST_DURATION': True,           # Burst duration result (usually want this on)
}

def diag_enabled(key):
	"""Check if a specific diagnostic category is enabled."""
	return SHOW_DIAGNOSTICS and DIAG.get(key, False)

# === END DIAGNOSTICS TOGGLE ===

# === MCMC TOGGLE AND CONFIGURATION ===
USE_MCMC = False              # Enable/disable MCMC fitting for full-band profile (only works with USE_EMG_MIX)
MCMC_NWALKERS = 32           # Number of MCMC walkers
MCMC_NSTEPS = 10000          # Number of MCMC steps
MCMC_BURNIN = 1000           # Burnin steps to discard
MCMC_PRIOR_RANGE = 0.2       # Prior range factor for MCMC
MCMC_OFF_PULSE_BINS = 50     # Manual off-pulse bins for baseline estimation
# === END MCMC CONFIGURATION ===

# Validate MCMC can only be used with EMG
if USE_MCMC and not USE_EMG_MIX:
	raise RuntimeError("USE_MCMC can only be True when USE_EMG_MIX is True.")

# === ADAPTIVE EMG WINDOW TOGGLE ===
# When enabled, the EMG fit window expands asymmetrically at lower frequencies
# to capture frequency-dependent scattering tails (tau_sc scales with frequency)
USE_ADAPTIVE_EMG_WINDOW = False  # Enable/disable adaptive EMG fit windows for stacking
# === END ADAPTIVE EMG WINDOW TOGGLE ===

# === SUBBAND ALPHA (SCATTERING INDEX) FIT TOGGLE ===
# When enabled (and USE_EMG_MIX is True), splits the band into N sub-bands,
# fits EMG in each, and recovers the scattering power-law τ(ν) = τ_ref*(ν/ν_ref)^α.
# Also uses the closest-to-grid FWHM classification from the sub-band-corrected
# (σ_hi, τ_ref) instead of the old fixed τ/σ threshold approach.
USE_SUBBAND_ALPHA_FIT = False    # Module-level default. Sub-band α fitting is now
                                  # OPT-IN per burst via the `subband_plot` kwarg —
                                  # see the validation block in measureburst() that
                                  # rejects subband-only kwargs (constrain_index,
                                  # bw_width_factor, bw_channel_override) on
                                  # components without a truthy subband_plot entry.
                                  # Flip to True to globally enable for every burst
                                  # (the legacy behavior); not recommended.
N_SUBBANDS_ALPHA = 4             # Number of sub-bands to split the waterfall into
# === END SUBBAND ALPHA FIT TOGGLE ===

########## === END TOGGLES ===

# === TELESCOPE NATIVE TIME RESOLUTION (ms) ===
# These represent the minimum resolvable timescale for each instrument.
# If fitted sigma or tau falls below this, the parameter is unphysical.
# Override or add entries as needed for your datasets.
TELESCOPE_NATIVE_RESOLUTION_MS = {
	'chime':      0.983,      # CHIME intensity data
	'arecibo':    0.01024,    # Arecibo PUPPI coherent filterbank
	'gbt':        0.01024,    # GBT GUPPI coherent filterbank
	'fast':       0.04915,    # FAST tracking mode (49.152 us)
	'effelsberg': 0.05469,    # Effelsberg PFFTS (54.6875 us typical)
	'lofar':      1.31,       # LOFAR beamformed (standard pulsar pipeline)
}

# Map dataset/source name patterns to telescope identifiers.
# Keys are lowercase substrings matched against the source/dataset name.
# Add your own mappings here. First match wins.
DATASET_TELESCOPE_MAP = {
	'chime':      'chime',
	'gbt':        'gbt',
	'greenbank':  'gbt',
	'green_bank': 'gbt',
	'arecibo':    'arecibo',
	'puppi':      'arecibo',
	'fast':       'fast',
	'crafts':     'fast',
	'effelsberg': 'effelsberg',
	'eff100':     'effelsberg',
	'lofar':      'lofar',
}

def identify_telescope(dataset_name):
	"""
	Identify the telescope from a dataset/source name string.
	Matches against DATASET_TELESCOPE_MAP (case-insensitive, first match wins).
	
	Returns:
	--------
	telescope : str or None
		Telescope key (e.g. 'chime'), or None if no match found.
	"""
	name_lower = dataset_name.lower()
	for pattern, telescope in DATASET_TELESCOPE_MAP.items():
		if pattern in name_lower:
			return telescope
	return None

def get_native_resolution(telescope):
	"""
	Get the native time resolution (ms) for a telescope.
	
	Returns:
	--------
	float or None
		Native resolution in ms, or None if telescope not in lookup.
	"""
	if telescope is None:
		return None
	return TELESCOPE_NATIVE_RESOLUTION_MS.get(telescope, None)

# === END TELESCOPE RESOLUTION ===


# === BIN-INTEGRATED EMG MODEL (Option A: oversample and average) ===

def bin_integrated_emg(t_centers, A, mu, sigma, lamb, C, dt_ms, oversamp=16):
	"""
	Bin-integrated EMG: average the continuous EMG model over each time bin.
	
	This accounts for the finite time resolution of the instrument. Each
	recorded sample is not the instantaneous EMG value, but the average
	over a bin of width dt_ms. Fitting an instantaneous model to binned
	data biases sigma, tau, and mu when they are comparable to or smaller
	than dt_ms.
	
	Parameters:
	-----------
	t_centers : (N,) array
		Centers of the time bins (ms).
	A, mu, sigma, lamb, C : float
		EMG parameters.
	dt_ms : float
		Bin width / time resolution (ms). E.g. 0.983 for CHIME, or
		res_time_ms after downsampling.
	oversamp : int
		Number of sub-samples per bin (16 is typical; 32-64 for narrow pulses).
	
	Returns:
	--------
	(N,) array
		Bin-averaged EMG model values.
	"""
	t_centers = np.asarray(t_centers, dtype=float)
	offs = (np.arange(oversamp) + 0.5) / oversamp - 0.5   # in [-0.5, +0.5)
	t_grid = t_centers[:, None] + offs[None, :] * dt_ms     # (N, oversamp)
	return emg_model(t_grid, A, mu, sigma, lamb, C).mean(axis=1)

def bin_integrated_emgmix(t_centers, dt_ms, oversamp, *p):
	"""
	Bin-integrated EMG mixture model. Same parameter layout as emgmix_model.
	"""
	t_centers = np.asarray(t_centers, dtype=float)
	n = (len(p) - 1) // 4
	C_val = p[4*n]
	offs = (np.arange(oversamp) + 0.5) / oversamp - 0.5
	t_grid = t_centers[:, None] + offs[None, :] * dt_ms
	y = C_val + np.zeros_like(t_grid)
	for i in range(n):
		A     = p[i]
		mu    = p[n + i]
		sigma = p[2*n + i]
		lamb  = p[3*n + i]
		y += emg_model(t_grid, A, mu, sigma, lamb, 0.0)
	return y.mean(axis=1)

def pick_oversamp(dt_ms, sigma_guess, tau_guess, base=16):
	"""
	Adaptively choose oversampling factor based on how narrow the pulse
	is relative to the bin width.
	"""
	m = np.nanmin([sigma_guess, tau_guess])
	if not np.isfinite(m) or m <= 0:
		return base
	if m < 0.25 * dt_ms:
		return 64
	if m < 0.5 * dt_ms:
		return 32
	return base

# === END BIN-INTEGRATED EMG ===


# === EMG FWHM AND SCATTERING REGIME FUNCTIONS ===

def emg_fwhm(mu, sigma, lamb, A=1.0, C=0.0, n_grid=5000):
	"""
	Numerically compute the true FWHM of an EMG profile.
	
	The EMG FWHM has no closed-form expression. This function evaluates
	the EMG on a fine grid and finds the half-maximum crossings via
	interpolation.
	
	Parameters:
	-----------
	mu, sigma, lamb : float
		EMG parameters (Gaussian center, Gaussian width, decay rate).
	A : float
		Amplitude (default 1.0; only the shape matters for FWHM).
	C : float
		Baseline (subtracted before computing FWHM).
	n_grid : int
		Number of grid points for evaluation.
	
	Returns:
	--------
	float
		FWHM in the same units as mu/sigma, or np.nan on failure.
	"""
	if not (np.isfinite(mu) and np.isfinite(sigma) and np.isfinite(lamb)):
		return np.nan
	if sigma <= 0 or lamb <= 0:
		return np.nan
	
	tau = 1.0 / lamb
	t_min = mu - 5 * sigma
	t_max = mu + 5 * sigma + 8 * tau
	
	t = np.linspace(t_min, t_max, n_grid)
	y = emg_model(t, A, mu, sigma, lamb, C) - C   # remove baseline
	
	idx_peak = np.argmax(y)
	y_peak = y[idx_peak]
	if y_peak <= 0:
		return np.nan
	
	half = y_peak / 2.0
	
	# Left crossing: find last point below half-max to the left of peak
	left_below = np.where(y[:idx_peak] < half)[0]
	if len(left_below) == 0:
		return np.nan
	li = left_below[-1]
	if li + 1 >= idx_peak:
		return np.nan
	# Linear interpolation for left crossing
	t_left = np.interp(half, [y[li], y[li+1]], [t[li], t[li+1]])
	
	# Right crossing: find first point below half-max to the right of peak
	right_below = np.where(y[idx_peak:] < half)[0]
	if len(right_below) == 0:
		return np.nan
	ri = idx_peak + right_below[0]
	if ri < 1:
		return np.nan
	# Linear interpolation for right crossing
	t_right = np.interp(half, [y[ri], y[ri-1]], [t[ri], t[ri-1]])
	
	fwhm = t_right - t_left
	return fwhm if fwhm > 0 else np.nan


def emg_fwhm_err(sigma, lamb, sigma_err, lamb_err, cov_sigma_lamb=0.0,
                 mu=0.0, A=1.0, C=0.0, n_grid=5000, eps_rel=1e-3):
	"""1σ uncertainty on the numerical EMG FWHM via linear propagation.

	The EMG FWHM is computed by ``emg_fwhm`` and has no closed form, so
	the partial derivatives are taken numerically (centred finite
	differences). The propagation formula is

	    σ_FWHM² = (∂FWHM/∂σ)² σ_σ²
	             + (∂FWHM/∂λ)² σ_λ²
	             + 2 (∂FWHM/∂σ)(∂FWHM/∂λ) · Cov(σ, λ)

	The cross term matters because σ and λ are typically anti-correlated
	in EMG fits (σ-τ degeneracy), and omitting it tends to over-state
	the uncertainty in the wings of that correlation.

	Parameters
	----------
	sigma, lamb : float
	    Central values for σ_Gaussian and decay rate λ = 1/τ.
	sigma_err, lamb_err : float
	    1σ uncertainties on σ and λ.
	cov_sigma_lamb : float
	    Off-diagonal element ``pcov[sigma_idx, lambda_idx]`` from the
	    EMG-mixture fit. Pass 0 if not available; the result then
	    reduces to the diagonal-only propagation.
	mu, A, C, n_grid : passed through to ``emg_fwhm``.
	eps_rel : float
	    Relative step size for the finite-difference partials. The step
	    is ``max(eps_rel * |x|, eps_rel)`` so it works near zero too.

	Returns
	-------
	float
	    1σ uncertainty on the FWHM (same units as ``sigma``). NaN if
	    any input is non-finite, σ ≤ 0, λ ≤ 0, or the partials cannot
	    be computed.
	"""
	if not all(np.isfinite([sigma, lamb, sigma_err, lamb_err])):
		return np.nan
	if sigma <= 0 or lamb <= 0:
		return np.nan

	# Centred finite-difference partials. The grid evaluation is cheap
	# (n_grid points, single argmax + two np.interp's) so four calls
	# add negligible cost.
	d_sig = max(eps_rel * abs(sigma), eps_rel)
	d_lam = max(eps_rel * abs(lamb),  eps_rel)
	f_sp = emg_fwhm(mu=mu, sigma=sigma + d_sig, lamb=lamb,         A=A, C=C, n_grid=n_grid)
	f_sm = emg_fwhm(mu=mu, sigma=sigma - d_sig, lamb=lamb,         A=A, C=C, n_grid=n_grid)
	f_lp = emg_fwhm(mu=mu, sigma=sigma,         lamb=lamb + d_lam, A=A, C=C, n_grid=n_grid)
	f_lm = emg_fwhm(mu=mu, sigma=sigma,         lamb=lamb - d_lam, A=A, C=C, n_grid=n_grid)
	if not all(np.isfinite([f_sp, f_sm, f_lp, f_lm])):
		return np.nan
	dF_dsig = (f_sp - f_sm) / (2.0 * d_sig)
	dF_dlam = (f_lp - f_lm) / (2.0 * d_lam)

	var = (dF_dsig**2) * (sigma_err**2) + (dF_dlam**2) * (lamb_err**2)
	if np.isfinite(cov_sigma_lamb):
		var += 2.0 * dF_dsig * dF_dlam * float(cov_sigma_lamb)
	if not np.isfinite(var) or var < 0:
		# Negative variance can arise from a strongly-correlated fit
		# where the linear approximation overshoots; treat as
		# undetermined rather than imaginary.
		return np.nan
	return float(np.sqrt(var))


def compute_onpulse_intervals_and_offpulse_mask(
	tmix_popt, n_bursts, USE_EMG_MIX,
	times_ms, res_time_ms,
	xos=None,
	cuts=None,
	decay_frac_list=(0.01, 0.05, 0.10, 0.20),
	min_offpulse_bins=50,
	leading_pad_sigma=0.5,
):
	"""
	Compute on-pulse intervals and an off-pulse boolean mask. Two
	modes:

	  • `cuts` supplied (non-empty) → HARD ISOLATION MODE. Each
	    component's on-pulse interval IS its region
	    `[cuts[i-1], cuts[i]]` for inner components. For the leading
	    component, the right edge is `cuts[0]` and the left edge is
	    `max(t_min, μ_a − k·σ_a − leading_pad_sigma·σ_a)` (so the
	    pre-burst noise region is correctly off-pulse). Symmetrically
	    for the trailing component on its right side, using the
	    decay-to-`decay_frac × peak` calculation from the EMG/Gauss
	    profile. Adaptive threshold cascade for the leading/trailing
	    decay-driven edges only; the cuts-driven edges are exact.
	  • `cuts` empty or None → LEGACY ADAPTIVE-DECAY MODE. Each
	    component's interval is `[μ − leading_pad_sigma·σ − k·σ,
	    t_decay]` where `t_decay` is the EMG/Gauss decay-fraction
	    crossing, with the adaptive 1% → 5% → 10% → 20% cascade if
	    the strict mask gives <`min_offpulse_bins` off-pulse bins.

	In both modes, on-pulse intervals returned are the SINGLE SOURCE
	OF TRUTH used by both the noise estimation and the green/orange
	visual bar.

	Parameters
	----------
	tmix_popt : array-like
		Mixture fit parameters. Layout depends on USE_EMG_MIX:
		  EMG:   [A_1..A_n, μ_1..μ_n, σ_1..σ_n, λ_1..λ_n, C]
		  Gauss: [A_1..A_n, μ_1..μ_n, σ_1..σ_n, C]
	n_bursts : int
		Number of components in the mixture.
	USE_EMG_MIX : bool
		If True, use EMG decay logic. If False, Gaussian.
	times_ms : 1D array
		Full time axis of the waterfall in ms (UN-shifted by pktime).
	res_time_ms : float
		Time resolution in ms.
	xos : list of float, optional
		Component center positions in ms. Used to identify which
		component is leading (smallest xo) and trailing (largest xo)
		in hard-isolation mode. If None, derives from tmix_popt[μ].
	cuts : list of float, optional
		User-supplied component time boundaries in ms. If non-empty,
		switches to hard-isolation mode.
	decay_frac_list : tuple of float
		Adaptive thresholds to try (strictest first). In hard-isolation
		mode, only used for the leading/trailing component edges that
		aren't bounded by cuts.
	min_offpulse_bins : int
		Minimum number of off-pulse bins to require before accepting
		a threshold.
	leading_pad_sigma : float
		Extra padding on the leading edge of each component, in σ units.

	Returns
	-------
	on_pulse_intervals_ms : list of (float, float)
		One (t_lo, t_hi) per component, in component order. NaN
		intervals filtered out.
	off_pulse_mask : 1D bool array, shape (n_time,)
		True where the time bin is off-pulse.
	decay_frac_used : float
		The decay threshold that ended up being used (or the
		threshold used for the leading/trailing edges in hard
		isolation mode).
	n_offpulse_bins : int
		Count of True entries in off_pulse_mask.
	"""
	n_time = len(times_ms)
	t0 = float(times_ms[0])
	t1 = float(times_ms[-1])

	# Helper: trailing-edge decay time for one EMG component.
	def _emg_t_decay(mu, sg, lam, decay_frac):
		"""Time at which the EMG profile drops below decay_frac × peak
		on its trailing side. Returns mu+8σ as a defensive fallback."""
		if not (np.isfinite(mu) and np.isfinite(sg) and sg > 0):
			return mu + 8.0 * sg if (np.isfinite(mu) and np.isfinite(sg) and sg > 0) else mu
		if lam is None or not np.isfinite(lam) or lam <= 0:
			# Degenerate λ → pure Gaussian behaviour
			return float(mu + sg * np.sqrt(-2.0 * np.log(decay_frac)))
		tau = 1.0 / lam
		# Fine grid extending far enough to catch the tail crossing
		_t_max = mu + max(8.0 * sg, max(12.0 * tau, sg * np.sqrt(-2.0 * np.log(decay_frac)) + 12.0 * tau))
		_grid = np.linspace(mu - 2.0 * sg, _t_max, 8192)
		try:
			# Use a normalised EMG (A=1 doesn't affect peak ratio)
			_prof = emg_model(_grid, 1.0, mu, sg, lam, 0.0)
		except Exception:
			return float(mu + 8.0 * sg + 12.0 * tau)
		if not np.all(np.isfinite(_prof)):
			return float(mu + 8.0 * sg + 12.0 * tau)
		_peak = float(np.max(_prof))
		if _peak <= 0:
			return float(mu + 8.0 * sg + 12.0 * tau)
		_thr = decay_frac * _peak
		_argmax = int(np.argmax(_prof))
		_tail = _prof[_argmax:]
		_tail_t = _grid[_argmax:]
		_below = np.where(_tail <= _thr)[0]
		if len(_below) == 0:
			# Tail didn't cross threshold within the searched grid;
			# return the grid's right edge as the on-pulse boundary.
			return float(_tail_t[-1])
		return float(_tail_t[int(_below[0])])

	# === HARD-ISOLATION MODE (cuts supplied) ===
	# Each component's on-pulse interval is dictated by its region in
	# `cuts`. Inner components' intervals are exactly their region
	# boundaries. Leading/trailing components keep their cuts-bounded
	# edge and compute the unbounded edge from the EMG/Gauss decay so
	# pre-burst and post-burst noise regions stay off-pulse. The
	# adaptive decay cascade applies only to those unbounded edges —
	# if the strictest threshold makes the trailing component's tail
	# extend past `t_max`, the next looser threshold is tried.
	if cuts is not None and len(cuts) > 0:
		if not (n_bursts >= 1 and isinstance(tmix_popt, (list, tuple, np.ndarray))):
			# No mixture fit → can't compute decay edges. Return
			# cuts-only intervals with empty leading/trailing pads.
			cuts_sorted = sorted(float(c) for c in cuts)
			region_edges = [t0] + cuts_sorted + [t1]
			on_intervals = [(region_edges[i], region_edges[i+1]) for i in range(len(region_edges)-1)]
			off_mask = np.zeros(n_time, dtype=bool)
			# Anything strictly outside [region_edges[0], region_edges[-1]] is off-pulse,
			# but since region_edges spans [t0, t1] there is no such bin → all on-pulse.
			# That's fine — caller will warn about empty off-pulse mask.
			return on_intervals, off_mask, decay_frac_list[0], int(np.sum(off_mask))

		# Parse mixture parameters.
		_amps = np.atleast_1d(tmix_popt[0:n_bursts])
		_mus  = np.atleast_1d(tmix_popt[n_bursts:2*n_bursts])
		_sgs  = np.atleast_1d(tmix_popt[2*n_bursts:3*n_bursts])
		if USE_EMG_MIX:
			_lms = np.atleast_1d(tmix_popt[3*n_bursts:4*n_bursts])
		else:
			_lms = [None] * n_bursts

		# Determine component order by μ (region order). The first
		# region (leftmost) holds the smallest-μ component; the last
		# holds the largest. xos isn't strictly needed here; we use
		# _mus from the fit which already reflects current μ positions.
		_mus_list = [(float(_mus[i]), i) for i in range(n_bursts)
		             if np.isfinite(_mus[i]) and np.isfinite(_sgs[i]) and _sgs[i] > 0]
		if not _mus_list:
			# Same fallback as the no-fit case
			cuts_sorted = sorted(float(c) for c in cuts)
			region_edges = [t0] + cuts_sorted + [t1]
			on_intervals = [(region_edges[i], region_edges[i+1]) for i in range(len(region_edges)-1)]
			off_mask = np.zeros(n_time, dtype=bool)
			return on_intervals, off_mask, decay_frac_list[0], int(np.sum(off_mask))
		_mus_sorted_with_idx = sorted(_mus_list, key=lambda p: p[0])
		# Map: region position j → original component index i
		region_to_orig = [p[1] for p in _mus_sorted_with_idx]

		cuts_sorted = sorted(float(c) for c in cuts)
		region_edges = [t0] + cuts_sorted + [t1]
		n_regions = len(region_edges) - 1

		if n_regions != len(region_to_orig):
			# Mismatch: number of regions doesn't match number of valid
			# components. Fall through to legacy adaptive-decay mode.
			pass
		else:
			# Try the decay thresholds in order for the leading and
			# trailing edges only. Inner regions are exactly their cuts.
			def _build_hard_for_threshold(decay_frac):
				on_in_region_order = []
				for j in range(n_regions):
					r_lo = region_edges[j]
					r_hi = region_edges[j + 1]
					orig_i = region_to_orig[j]
					_mu = float(_mus[orig_i])
					_sg = abs(float(_sgs[orig_i]))
					_lm = (float(_lms[orig_i])
					       if (USE_EMG_MIX and _lms[orig_i] is not None
					           and np.isfinite(_lms[orig_i]) and _lms[orig_i] > 0)
					       else None)
					# Leading edge: if this is the first region, compute
					# from decay (we want pre-burst noise to be off-pulse);
					# otherwise the cut is the hard left edge.
					if j == 0:
						_core_lead = 2.0 if USE_EMG_MIX else 3.0
						lead_edge = _mu - (leading_pad_sigma + _core_lead) * _sg
						t_lo = max(r_lo, lead_edge)
					else:
						t_lo = r_lo
					# Trailing edge: if this is the last region, compute
					# from decay so post-burst noise stays off-pulse;
					# otherwise the cut is the hard right edge.
					if j == n_regions - 1:
						if USE_EMG_MIX:
							trail_edge = _emg_t_decay(_mu, _sg, _lm, decay_frac)
						else:
							trail_edge = _mu + _sg * np.sqrt(-2.0 * np.log(decay_frac))
						t_hi = min(r_hi, trail_edge)
					else:
						t_hi = r_hi
					on_in_region_order.append((t_lo, t_hi))

				# Re-order to ORIGINAL component order so downstream code
				# that indexes by original component i reads the right
				# interval.
				on_intervals_ordered = [None] * n_bursts
				for j, orig_i in enumerate(region_to_orig):
					on_intervals_ordered[orig_i] = on_in_region_order[j]
				# Filter out None entries (shouldn't happen here, but guard).
				on_intervals = [iv for iv in on_intervals_ordered if iv is not None]

				# Off-pulse mask: anything outside the union of intervals.
				off_mask = np.ones(n_time, dtype=bool)
				for (t_lo, t_hi) in on_intervals:
					_i_lo = max(0, int(np.floor((t_lo - t0) / res_time_ms)))
					_i_hi = min(n_time - 1, int(np.ceil((t_hi - t0) / res_time_ms)))
					if _i_hi >= _i_lo:
						off_mask[_i_lo:_i_hi + 1] = False
				return on_intervals, off_mask

			# Adaptive cascade applies to leading/trailing edges only.
			last_intervals, last_mask, last_count, last_thr = None, None, 0, decay_frac_list[-1]
			for thr in decay_frac_list:
				intervals, mask = _build_hard_for_threshold(thr)
				n_off = int(np.sum(mask))
				last_intervals, last_mask, last_count, last_thr = intervals, mask, n_off, thr
				if n_off >= min_offpulse_bins:
					return intervals, mask, thr, n_off
			# Loosest threshold's result is what we return if nothing
			# met the minimum.
			return last_intervals, last_mask, last_thr, last_count
	# === END HARD-ISOLATION MODE; fall through to legacy adaptive-decay ===

	# Helper: build intervals and mask for a given decay threshold.
	def _build_for_threshold(decay_frac):
		on_intervals = []
		if not (n_bursts >= 1 and isinstance(tmix_popt, (list, tuple, np.ndarray))):
			# No mixture fit available → empty intervals, all-off-pulse mask.
			return on_intervals, np.ones(n_time, dtype=bool)
		_amps = np.atleast_1d(tmix_popt[0:n_bursts])
		_mus  = np.atleast_1d(tmix_popt[n_bursts:2*n_bursts])
		_sgs  = np.atleast_1d(tmix_popt[2*n_bursts:3*n_bursts])
		if USE_EMG_MIX:
			_lms = np.atleast_1d(tmix_popt[3*n_bursts:4*n_bursts])
		else:
			_lms = [None] * n_bursts
		for _mu, _sg, _lm in zip(_mus, _sgs, _lms):
			if not (np.isfinite(_mu) and np.isfinite(_sg) and _sg > 0):
				continue
			_sg_abs = abs(float(_sg))
			# Leading edge: μ − pad·σ − (2 or 3)σ depending on model.
			# 2σ for EMG (where the core is already narrower than Gauss
			# under the same σ symbol), 3σ for Gauss (where 3σ ≈ 1.1%
			# of peak — close to our default 1% threshold).
			_core_lead = 2.0 if USE_EMG_MIX else 3.0
			t_lo = float(_mu) - (leading_pad_sigma + _core_lead) * _sg_abs
			if USE_EMG_MIX:
				_lam_val = float(_lm) if (_lm is not None and np.isfinite(_lm) and _lm > 0) else None
				t_hi = _emg_t_decay(float(_mu), _sg_abs, _lam_val, decay_frac)
			else:
				# Gaussian trailing edge at decay_frac × peak
				t_hi = float(_mu) + _sg_abs * np.sqrt(-2.0 * np.log(decay_frac))
			on_intervals.append((t_lo, t_hi))

		# Build off-pulse mask from the union of on-pulse intervals.
		off_mask = np.ones(n_time, dtype=bool)
		for (t_lo, t_hi) in on_intervals:
			# Convert ms to bin indices on the times_ms axis.
			_i_lo = int(np.floor((t_lo - t0) / res_time_ms))
			_i_hi = int(np.ceil((t_hi - t0) / res_time_ms))
			_i_lo = max(0, _i_lo)
			_i_hi = min(n_time - 1, _i_hi)
			if _i_hi >= _i_lo:
				off_mask[_i_lo:_i_hi + 1] = False
		return on_intervals, off_mask

	# Adaptive loop: try strictest threshold first.
	last_intervals = []
	last_mask = np.ones(n_time, dtype=bool)
	last_count = 0
	last_thr = decay_frac_list[-1]
	for thr in decay_frac_list:
		intervals, mask = _build_for_threshold(thr)
		n_off = int(np.sum(mask))
		last_intervals = intervals
		last_mask = mask
		last_count = n_off
		last_thr = thr
		if n_off >= min_offpulse_bins:
			# Threshold is OK — keep these and stop.
			return intervals, mask, thr, n_off
	# Even the loosest threshold didn't give enough off-pulse bins.
	# Return the loosest-threshold result anyway and let the caller
	# notice via the low n_off count.
	return last_intervals, last_mask, last_thr, last_count


def classify_scattering_regime(sigma, tau, weak_threshold=0.5, strong_threshold=2.0,
                                sigma_err=None, tau_err=None, cov_sigma_lamb=0.0):
	"""
	Classify the scattering regime based on the ratio tau/sigma.
	
	Also computes all three FWHM approximations and the numerically
	exact EMG FWHM.
	
	Parameters:
	-----------
	sigma : float
		Gaussian core width from EMG fit (ms).
	tau : float
		Scattering timescale = 1/lambda from EMG fit (ms).
	weak_threshold : float
		tau/sigma below this → 'weak' (default 0.5).
	strong_threshold : float
		tau/sigma above this → 'strong' (default 2.0).
	
	Returns:
	--------
	dict with keys:
		'regime'           : str, one of 'weak', 'intermediate', 'strong', or 'unknown'
		'tau_over_sigma'   : float, the ratio tau/sigma
		'fwhm_weak'        : float, FWHM ≈ 2.355 * sigma
		'fwhm_intermediate': float, FWHM ≈ 2.355 * sqrt(sigma^2 + tau^2)
		'fwhm_strong'      : float, FWHM ≈ tau * ln(2)
		'fwhm_emg'         : float, numerically computed true EMG FWHM
		'fwhm_emg_err'     : float, 1σ uncertainty on fwhm_emg (NaN if
		                     no error info passed in)
		'fwhm_reported'    : float, the FWHM corresponding to the assigned regime
	"""
	result = {
		'regime': 'unknown',
		'tau_over_sigma': np.nan,
		'fwhm_weak': np.nan,
		'fwhm_intermediate': np.nan,
		'fwhm_strong': np.nan,
		'fwhm_emg': np.nan,
		'fwhm_emg_err': np.nan,
		'fwhm_reported': np.nan,
	}
	
	if not (np.isfinite(sigma) and np.isfinite(tau) and sigma > 0 and tau > 0):
		return result
	
	ratio = tau / sigma
	result['tau_over_sigma'] = ratio
	
	# All three approximations (always computed)
	result['fwhm_weak']         = 2.355 * sigma
	result['fwhm_intermediate'] = 2.355 * np.sqrt(sigma**2 + tau**2)
	result['fwhm_strong']       = tau * np.log(2)
	
	# Numerical EMG FWHM (and its 1σ uncertainty when σ, λ errors and
	# their covariance are provided). The error propagation goes
	# through λ = 1/τ, so the caller passes the λ-space error directly
	# rather than τ-space; the covariance is in (σ, λ) space too. We
	# convert tau_err → lam_err here so callers can pass τ-space errors
	# transparently. cov_sigma_lamb stays in (σ, λ) space — that's
	# what the EMG fit's pcov natively reports.
	lamb = 1.0 / tau
	result['fwhm_emg'] = emg_fwhm(mu=0.0, sigma=sigma, lamb=lamb)
	if (sigma_err is not None and tau_err is not None
	        and np.isfinite(sigma_err) and np.isfinite(tau_err)
	        and sigma_err > 0 and tau_err >= 0):
		# Propagate τ_err to λ_err: λ = 1/τ → σ_λ = σ_τ / τ²
		lamb_err = tau_err / (tau * tau)
		result['fwhm_emg_err'] = emg_fwhm_err(
			sigma=sigma, lamb=lamb,
			sigma_err=sigma_err, lamb_err=lamb_err,
			cov_sigma_lamb=cov_sigma_lamb,
		)
	
	# Classify regime
	if ratio < weak_threshold:
		result['regime'] = 'weak'
		result['fwhm_reported'] = result['fwhm_weak']
	elif ratio > strong_threshold:
		result['regime'] = 'strong'
		result['fwhm_reported'] = result['fwhm_strong']
	else:
		result['regime'] = 'intermediate'
		result['fwhm_reported'] = result['fwhm_intermediate']
	
	return result

# === END EMG FWHM AND SCATTERING REGIME ===


# ═══════════════════════════════════════════════════════════════════════════════
# SUBBAND SCATTERING INDEX (ALPHA) RECOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def _power_law_log(log_nu, log_tau_ref, alpha):
	"""log(τ) = log(τ_ref) + α * log(ν/ν_ref)  (ν_ref absorbed into centering)."""
	return log_tau_ref + alpha * log_nu


def classify_regime_by_fwhm(sigma, tau):
	"""
	Classify scattering regime by finding which analytical FWHM formula
	is closest to the numerically exact EMG FWHM (closest-to-grid method).

	This replaces the fixed τ/σ threshold approach with a physics-driven
	classification: the regime label tells you which simplified model best
	describes the actual pulse shape.

	Parameters
	----------
	sigma : float   Gaussian core width (ms)
	tau   : float   Scattering timescale 1/λ (ms)

	Returns
	-------
	dict with keys:
		'regime'            : 'weak', 'intermediate', or 'strong'
		'fwhm_grid'         : numerically exact FWHM
		'fwhm_weak'         : 2.355 * σ
		'fwhm_intermediate' : 2.355 * √(σ² + τ²)
		'fwhm_strong'       : τ * ln(2)
		'fwhm_reported'     : the analytical FWHM closest to the grid FWHM
		'residual_weak'     : |fwhm_grid - fwhm_weak|
		'residual_inter'    : |fwhm_grid - fwhm_intermediate|
		'residual_strong'   : |fwhm_grid - fwhm_strong|
		'tau_over_sigma'    : τ/σ
	"""
	result = {
		'regime': 'unknown',
		'fwhm_grid': np.nan,
		'fwhm_weak': np.nan,
		'fwhm_intermediate': np.nan,
		'fwhm_strong': np.nan,
		'fwhm_reported': np.nan,
		'residual_weak': np.nan,
		'residual_inter': np.nan,
		'residual_strong': np.nan,
		'tau_over_sigma': np.nan,
	}

	if not (np.isfinite(sigma) and np.isfinite(tau) and sigma > 0 and tau > 0):
		return result

	lamb = 1.0 / tau

	# Numerical FWHM on fine grid
	fwhm_grid = emg_fwhm(mu=0.0, sigma=sigma, lamb=lamb)
	if not np.isfinite(fwhm_grid):
		return result

	# Three analytical approximations
	fwhm_weak  = 2.355 * sigma
	fwhm_inter = 2.355 * np.sqrt(sigma**2 + tau**2)
	fwhm_strong = tau * np.log(2)

	# Absolute residuals
	res_weak  = abs(fwhm_grid - fwhm_weak)
	res_inter = abs(fwhm_grid - fwhm_inter)
	res_strong = abs(fwhm_grid - fwhm_strong)

	# Pick the closest
	residuals = {'weak': res_weak, 'intermediate': res_inter, 'strong': res_strong}
	best_regime = min(residuals, key=residuals.get)

	fwhm_map = {
		'weak': fwhm_weak,
		'intermediate': fwhm_inter,
		'strong': fwhm_strong,
	}

	result['fwhm_grid'] = fwhm_grid
	result['fwhm_weak'] = fwhm_weak
	result['fwhm_intermediate'] = fwhm_inter
	result['fwhm_strong'] = fwhm_strong
	result['residual_weak'] = res_weak
	result['residual_inter'] = res_inter
	result['residual_strong'] = res_strong
	result['regime'] = best_regime
	result['fwhm_reported'] = fwhm_map[best_regime]
	result['tau_over_sigma'] = tau / sigma

	return result


def _fit_emg_subband(data_1d, duration_ms, sigma_seed=None, tau_seed=None, dt_ms=None,
                     sigma_fixed=None):
	"""
	Fit a single-component EMG to a 1D sub-band time series.

	Lightweight wrapper around emg_model for sub-band fitting.  Uses
	bin-integrated fitting when dt_ms is provided (matching the main
	pipeline's approach for per-row fits).

	When ``sigma_fixed`` is supplied, σ is pinned to that value by
	narrowing its bounds to a ±1e-6·σ_fixed window. The remaining
	parameters (A, μ, λ, C) are optimised normally. Used by the
	`constrain_sigma` diagnostic to test how much of the recovered τ(ν)
	power-law depends on σ being free to trade against τ.

	Returns (popt, pcov, redchisq) or (None, None, nan) on failure.
	popt = [A, mu, sigma, lamb, C]
	"""
	n = len(data_1d)
	t = np.linspace(0, duration_ms, n)
	y = data_1d.copy()

	edge = max(1, n // 10)
	noise_region = np.concatenate([y[:edge], y[-edge:]])
	noise_std = np.std(noise_region)
	if noise_std <= 0:
		noise_std = np.std(y) * 0.1

	y_bg = np.median(noise_region)
	y_peak = np.max(y) - y_bg
	if y_peak <= 0:
		y_peak = np.std(y)

	idx_peak = np.argmax(y)
	mu0 = t[idx_peak]

	var_guess = np.sum((y - y_bg) * (t - mu0)**2) / max(np.sum(y - y_bg), 1e-30)
	sig0 = sigma_seed if (sigma_seed and np.isfinite(sigma_seed) and sigma_seed > 0) \
		else max(np.sqrt(abs(var_guess)) / 3, duration_ms / 50)
	tau0 = tau_seed if (tau_seed and np.isfinite(tau_seed) and tau_seed > 0) else sig0
	lam0 = 1.0 / max(tau0, 1e-6)

	p0 = [y_peak, mu0, sig0, lam0, y_bg]
	# σ and τ floors: half a time-bin.
	# Background: in unresolved-rise sub-bands (rise faster than dt), the
	# optimiser can collapse σ → 0 because the bin-integrated EMG with
	# any σ ≪ dt looks identical (a step function at μ followed by
	# exponential decay), making the χ² landscape flat in that regime.
	# curve_fit then wanders into the flat valley and stops at a value
	# determined by tolerance rather than data — typically ~1/100 dt,
	# which is unphysical to report as an intrinsic Gaussian width.
	# Same pathology occurs in reverse for τ → 0 in sub-bands where
	# the EMG collapses to a Gaussian (no measurable scattering tail).
	#
	# A half-bin floor (dt_ms / 2) acknowledges the resolution limit
	# while still letting the fit indicate "below the resolution"
	# (rather than pinning exactly at one bin). When dt_ms is not
	# supplied — e.g. for legacy callers using the instantaneous EMG —
	# we fall back to the original sig0*0.01 / lam0*1000 behaviour so
	# this change is bounded to the bin-integrated path.
	if dt_ms is not None and dt_ms > 0:
		_half_bin = 0.5 * float(dt_ms)
		_lam_hi   = 1.0 / _half_bin   # τ ≥ half a bin
		_sig_lo   = _half_bin
	else:
		_sig_lo   = sig0 * 0.01
		_lam_hi   = lam0 * 1000
	bounds_lo = [0, 0, _sig_lo, lam0 * 0.001, -abs(y_bg) * 5 - 1]
	bounds_hi = [y_peak * 50, duration_ms, duration_ms / 2, _lam_hi, abs(y_bg) * 5 + 1]

	# Pin σ to a fixed value (constrain_sigma diagnostic) by narrowing
	# its bounds to a ±1e-6·sigma_fixed window. The optimiser cannot
	# move σ outside this band, so the fit reduces to a 4-parameter
	# search over (A, μ, λ, C) with σ held at `sigma_fixed`.
	if sigma_fixed is not None and np.isfinite(sigma_fixed) and sigma_fixed > 0:
		_sig_eps = max(1e-6, 1e-6 * float(sigma_fixed))
		bounds_lo[2] = float(sigma_fixed) - _sig_eps
		bounds_hi[2] = float(sigma_fixed) + _sig_eps
		p0[2] = float(sigma_fixed)

	for i in range(len(p0)):
		p0[i] = max(bounds_lo[i], min(p0[i], bounds_hi[i]))

	data_sigma = np.full(n, noise_std)

	# Choose model: bin-integrated or instantaneous
	if dt_ms is not None and dt_ms > 0:
		tau_guess = 1.0 / max(lam0, 1e-9)
		oversamp = pick_oversamp(dt_ms, sig0, tau_guess)
		fit_model = lambda t, A, mu, sigma, lamb, C: bin_integrated_emg(
			t, A, mu, sigma, lamb, C, dt_ms=dt_ms, oversamp=oversamp
		)
	else:
		fit_model = emg_model

	try:
		popt, pcov = scipy.optimize.curve_fit(
			fit_model, t, y, p0=p0,
			bounds=(bounds_lo, bounds_hi),
			sigma=data_sigma, maxfev=200000
		)
		residuals = (y - fit_model(t, *popt)) / noise_std
		dof = max(n - len(popt), 1)
		redchisq = np.sum(residuals**2) / dof
		return popt, pcov, redchisq
	except Exception:
		return None, None, np.nan


# === SMOOTHED σ(ν) POWER-LAW FIT (ε-fit) ===
# Fits the per-sub-band Gaussian widths σ_j to a monotonically decreasing
# power law σ(ν) = A · (ν/ν_ref)^β with β ≤ 0 enforced, producing a
# "smoothed" sigma envelope across the band. The result complements the
# τ(ν) sub-band fit and is intended for downstream combination with the
# scattering tail.
#
# Fit is performed in log–log space via scipy.optimize.curve_fit with a
# bound on β to enforce monotonic decrease. Returns NaNs if fewer than 3
# good points are available or if the fit fails to converge.
def _fit_sigma_smooth_powerlaw(sub_freqs, sub_sigmas, freq_center):
	"""
	Fit σ(ν) = A · (ν/ν_ref)^β to per-sub-band Gaussian widths.

	Parameters
	----------
	sub_freqs : 1D array of sub-band centre frequencies (MHz).
	sub_sigmas : 1D array of fitted σ per sub-band (ms).
	freq_center : float
		Reference frequency ν_ref (MHz). A is defined as σ(ν_ref).
		Use the same reference as the τ(ν) fit for consistency.

	Returns
	-------
	dict with keys:
		'amp'      : A in ms (σ at ν_ref), NaN on failure
		'amp_err'  : 1σ uncertainty on A (ms)
		'beta'     : exponent (dimensionless, ≤ 0), NaN on failure
		'beta_err' : 1σ uncertainty on β
		'success'  : True if the fit converged with ≥3 finite points
	"""
	out = {'amp': np.nan, 'amp_err': np.nan,
	       'beta': np.nan, 'beta_err': np.nan, 'success': False}

	if (sub_freqs is None or sub_sigmas is None
	        or not np.isfinite(freq_center) or freq_center <= 0):
		return out

	nu  = np.asarray(sub_freqs,  dtype=float)
	sig = np.asarray(sub_sigmas, dtype=float)
	mask = np.isfinite(nu) & np.isfinite(sig) & (nu > 0) & (sig > 0)
	if int(np.sum(mask)) < 3:
		return out

	nu_g  = nu[mask]
	sig_g = sig[mask]

	# Fit in log-log space: log(σ) = log(A) + β · log(ν/ν_ref).
	# This linearises the power law and avoids dynamic-range issues
	# when amplitudes span <1 ms while frequencies are ~10^3 MHz.
	log_sig = np.log(sig_g)
	x_log   = np.log(nu_g / float(freq_center))

	def _model_loglog(x, logA, beta):
		return logA + beta * x

	# β ≤ 0 enforced (strictly monotonic non-increasing).
	# logA is free; initial guess from a quick polyfit.
	try:
		_p0_beta, _p0_logA = np.polyfit(x_log, log_sig, 1)
		# Clamp the initial β into the allowed region so curve_fit
		# starts feasible. If polyfit gave a positive β (uptick in
		# data, e.g. the rightmost CHIME sub-band noise), seed at 0.
		_p0_beta = min(_p0_beta, 0.0)
		p0 = [_p0_logA, _p0_beta]
		bounds = ([-np.inf, -np.inf], [np.inf, 0.0])
		popt, pcov = scipy.optimize.curve_fit(
			_model_loglog, x_log, log_sig, p0=p0, bounds=bounds, maxfev=10000)
		logA_fit, beta_fit = popt
		# Standard errors from covariance (NaN-safe)
		perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.array([np.nan, np.nan])
		logA_err, beta_err = perr[0], perr[1]
		A_fit = float(np.exp(logA_fit))
		# Error propagation: σ_A = A · σ_{logA}
		A_err = float(A_fit * logA_err) if np.isfinite(logA_err) else np.nan
		out.update({
			'amp': A_fit, 'amp_err': A_err,
			'beta': float(beta_fit), 'beta_err': float(beta_err),
			'success': True,
		})
	except Exception as _ex:
		print(f"  σ-smooth ε fit: power-law fit failed "
		      f"({type(_ex).__name__}: {_ex}).")

	return out
# === END SMOOTHED σ(ν) POWER-LAW FIT ===


def fit_subband_alpha(wfall, freqs, duration_ms, res_time_ms, n_sub=4,
                      fb_sigma=None, fb_tau=None, bname='', constrain_alpha=False,
                      bw_width_factor=2.0, channel_override=None,
                      spec_center_outer=None, spec_sigma_outer=None,
                      ignore_subindex=None, sigma_fixed=None):
	"""
	Sub-band scattering index (alpha) recovery.

	Splits the waterfall into n_sub frequency sub-bands, fits an EMG in
	each, then fits a power law τ(ν) = τ_ref * (ν/ν_ref)^α to recover
	(τ_ref, α).  Also returns the best σ from the highest-frequency
	sub-band and the closest-to-grid FWHM regime classification.

	Parameters
	----------
	wfall : ndarray (n_freq, n_time)
		Waterfall data (dedispersed).
	freqs : ndarray (n_freq,)
		Frequency array in MHz.
	duration_ms : float
		Total waterfall duration in ms.
	res_time_ms : float
		Time resolution (bin width) in ms.
	n_sub : int
		Number of sub-bands (default 4).
	fb_sigma : float or None
		Full-band σ from mixture fit, used as seed.
	fb_tau : float or None
		Full-band τ from mixture fit, used as seed.
	bname : str
		Burst name for printing.
	constrain_alpha : bool
		If True, constrain the power-law index α during the τ(ν) fit.
	bw_width_factor : float
		Multiplier on σ_ν from the spectral Gaussian fit that sets the
		half-width of the channel mask used for sub-band splitting. The
		mask keeps channels in [ν₀ − bw_width_factor·σ_ν, ν₀ + bw_width_factor·σ_ν].
		Default 2.0 (i.e. ±2σ_ν). Pass the same value used for the outer
		arrival-time spectral filter to keep the two windows consistent.
		Ignored if channel_override is not None.
	channel_override : tuple of (int, int) or None
		If provided, (i_lo, i_hi) inclusive channel indices that override
		the spectral Gaussian fit and define the sub-band splitting range
		directly. Use when the spectral profile is unreliable (truncated,
		scintillated, noisy). Indices are clamped to the valid range with
		a warning if out of bounds. Default None (use Gaussian fit).
	spec_center_outer : float or None
		Spectral Gaussian mean (MHz) from the caller's outer spectral fit
		(subband_popt[1], a.k.a. pkfreq). When both this and spec_sigma_outer
		are provided AND channel_override is None, these values replace the
		inner-fit's spec_center/spec_sigma so the sub-band splitting window
		matches the outer arrival-time spectral filter exactly. The inner
		curve_fit still runs (its result is returned via spec_popt_inner
		for diagnostic purposes) but does not drive the sub-band window.
		Default None (fall back to computing the inner fit's values as
		authoritative — legacy behaviour).
	spec_sigma_outer : float or None
		Spectral Gaussian σ (MHz) from the caller's outer spectral fit
		(subband_popt[2], a.k.a. bwidth). See spec_center_outer.

	Returns
	-------
	dict with keys:
		'alpha'            : scattering index (power-law exponent)
		'alpha_err'        : uncertainty on alpha
		'tau_ref'          : reference scattering timescale at ν_center (ms)
		'tau_ref_err'      : uncertainty on tau_ref
		'best_sigma'       : σ from highest-frequency sub-band (ms)
		'freq_center'      : reference frequency ν_ref (MHz)
		'n_good_subbands'  : number of successful sub-band fits
		'sub_freqs'        : effective frequency of each sub-band (MHz)
		'sub_taus'         : fitted τ per sub-band (ms)
		'sub_tau_errs'     : uncertainty on τ per sub-band
		'sub_sigmas'       : fitted σ per sub-band (ms)
		'sub_popts'        : EMG fit parameters per sub-band
		'sub_labels'       : frequency range label per sub-band
		'regime_grid'      : closest-to-grid regime classification dict
		'regime_threshold' : threshold-based regime classification dict
		'success'          : bool, True if alpha fit succeeded
	"""
	n_freq, n_time = wfall.shape
	freq_center = (freqs.min() + freqs.max()) / 2.0

	# Default (failed) result
	null_regime_grid = classify_regime_by_fwhm(np.nan, np.nan)
	null_result = {
		'alpha': np.nan, 'alpha_err': np.nan,
		'tau_ref': np.nan, 'tau_ref_err': np.nan,
		'best_sigma': np.nan, 'freq_center': freq_center,
		'n_good_subbands': 0,
		'sub_freqs': np.array([]), 'sub_taus': np.array([]),
		'sub_tau_errs': np.array([]), 'sub_sigmas': np.array([]),
		'sub_sigma_errs': np.array([]),
		'sub_popts': [], 'sub_labels': [],
		'sub_freq_los': np.array([], dtype=float),
		'sub_freq_his': np.array([], dtype=float),
		'all_sub_freq_los': np.array([], dtype=float),
		'all_sub_freq_his': np.array([], dtype=float),
		'regime_grid': null_regime_grid,
		'regime_threshold': {'regime': 'unknown', 'tau_over_sigma': np.nan},
		'success': False,
		'freq_lo': freqs.min(), 'freq_hi': freqs.max(),
		'spec_popt_inner': None,
	}

	# ── Build channel mask: either from explicit indices or from spectral fit ──
	bw_mask = None
	freq_lo = freq_hi = None
	spec_popt_inner = None  # (a, center, sigma) from inner spectral Gaussian fit, if run

	if channel_override is not None:
		# Manual override: use the supplied (i_lo, i_hi) inclusive channel range.
		# Defensive clamp against the local n_freq (which may differ from the
		# outer waterfall if a sub-component slice was passed in).
		try:
			_i_lo_raw, _i_hi_raw = int(channel_override[0]), int(channel_override[1])
			if _i_lo_raw > _i_hi_raw:
				_i_lo_raw, _i_hi_raw = _i_hi_raw, _i_lo_raw
			_i_lo = max(0, _i_lo_raw)
			_i_hi = min(n_freq - 1, _i_hi_raw)
			if _i_lo != _i_lo_raw or _i_hi != _i_hi_raw:
				print(f"  Subband α fit: channel_override=({_i_lo_raw}, {_i_hi_raw}) "
				      f"clamped to ({_i_lo}, {_i_hi}) — wfall has {n_freq} channels")
			if _i_hi - _i_lo + 1 < n_sub * 2:
				print(f"  Subband α fit: channel_override range too narrow "
				      f"({_i_hi - _i_lo + 1} channels for {n_sub} sub-bands), using full band")
				bw_mask = np.ones(n_freq, dtype=bool)
				freq_lo, freq_hi = freqs.min(), freqs.max()
			else:
				bw_mask = np.zeros(n_freq, dtype=bool)
				bw_mask[_i_lo:_i_hi + 1] = True  # +1 for inclusive upper bound
				freq_lo = float(min(freqs[_i_lo], freqs[_i_hi]))
				freq_hi = float(max(freqs[_i_lo], freqs[_i_hi]))
				print(f"  Subband α fit: using channel_override [{_i_lo}, {_i_hi}] inclusive "
				      f"→ [{freq_lo:.1f}, {freq_hi:.1f}] MHz "
				      f"({bw_mask.sum()}/{n_freq} channels)")
		except (TypeError, ValueError, IndexError) as e:
			print(f"  Subband α fit: channel_override={channel_override} could not be parsed "
			      f"({e}), falling back to spectral Gaussian fit")
			bw_mask = None  # fall through to Gaussian fit below

	if bw_mask is None:
		# ── Restrict to ±bw_width_factor·σ of the spectral profile ─────────
		# Compute time-integrated spectrum and fit a Gaussian to determine the
		# burst bandwidth, then only use channels within ±bw_width_factor·σ of
		# the center. Default bw_width_factor=2.0 (i.e. ±2σ).
		#
		# NOTE on two fits: historically this function ran its own "inner"
		# Gaussian fit here and used it for the sub-band splitting window.
		# The caller separately runs an "outer" fit (subband_popt / pkfreq,
		# bwidth) that drives the arrival-time spectral filter and the cyan
		# shaded band in the side panel. For broadband bursts the two fits
		# can disagree — same spectrum, different seeds and weights — which
		# produced inconsistent windows between the subband-α analysis and
		# the TOA selection.
		#
		# Current behaviour: we still compute the inner fit (so that
		# spec_popt_inner is available for diagnostic plotting), but if the
		# caller supplies spec_center_outer / spec_sigma_outer we use those
		# values to build the window — unifying the subband-α window with
		# the outer arrival-time filter window.
		spectrum = wfall.sum(axis=1)
		try:
			spec_nonzero = spectrum[spectrum != 0]
			freq_nonzero = freqs[spectrum != 0]
			if len(freq_nonzero) < 4:
				raise RuntimeError("Too few non-zero channels")
			f0_est = np.sum(freq_nonzero * spec_nonzero) / np.sum(spec_nonzero)
			sig_est = np.sqrt(np.abs(np.sum(spec_nonzero * (freq_nonzero - f0_est)**2) / np.sum(spec_nonzero)))
			_spec_popt, _ = scipy.optimize.curve_fit(
				gauss_model, freq_nonzero, spec_nonzero,
				p0=[spec_nonzero.max(), f0_est, sig_est]
			)
			spec_center_inner = _spec_popt[1]
			spec_sigma_inner = abs(_spec_popt[2])
			spec_popt_inner = (float(_spec_popt[0]), float(spec_center_inner), float(spec_sigma_inner))

			# Decide which (center, sigma) drives the sub-band window.
			# Prefer outer-fit values when the caller provides them and they're
			# finite; otherwise fall back to inner-fit values (legacy path).
			_use_outer = (
				spec_center_outer is not None
				and spec_sigma_outer is not None
				and np.isfinite(spec_center_outer)
				and np.isfinite(spec_sigma_outer)
				and spec_sigma_outer > 0
			)
			if _use_outer:
				spec_center = float(spec_center_outer)
				spec_sigma = float(abs(spec_sigma_outer))
				_fit_src = "outer"
			else:
				spec_center = spec_center_inner
				spec_sigma = spec_sigma_inner
				_fit_src = "inner"

			freq_lo = spec_center - bw_width_factor * spec_sigma
			freq_hi = spec_center + bw_width_factor * spec_sigma
			# Clamp to actual frequency range
			freq_lo = max(freq_lo, freqs.min())
			freq_hi = min(freq_hi, freqs.max())
			bw_mask = (freqs >= freq_lo) & (freqs <= freq_hi)
			if bw_mask.sum() < n_sub * 2:
				print(f"  Subband α fit: ±{bw_width_factor:g}σ range too narrow ({bw_mask.sum()} channels), using full band")
				bw_mask = np.ones(n_freq, dtype=bool)
				freq_lo, freq_hi = freqs.min(), freqs.max()
			else:
				print(f"  Subband α fit: using ±{bw_width_factor:g}σ range [{freq_lo:.1f}, {freq_hi:.1f}] MHz "
					  f"(center={spec_center:.1f}, σ_ν={spec_sigma:.1f}, {bw_mask.sum()}/{n_freq} channels) "
					  f"[source={_fit_src}; inner μ={spec_center_inner:.1f}, σ={spec_sigma_inner:.1f}]")
		except Exception as e:
			print(f"  Subband α fit: spectral Gaussian fit failed ({e}), using full band")
			bw_mask = np.ones(n_freq, dtype=bool)
			freq_lo, freq_hi = freqs.min(), freqs.max()

	# Apply the bandwidth mask
	wfall_sub = wfall[bw_mask, :]
	freqs_sub = freqs[bw_mask]
	n_freq_sub = wfall_sub.shape[0]
	freq_center = (freqs_sub.min() + freqs_sub.max()) / 2.0

	# Update null result with actual freq range
	null_result['freq_lo'] = freq_lo
	null_result['freq_hi'] = freq_hi
	null_result['freq_center'] = freq_center

	chan_per_sub = n_freq_sub // n_sub
	if chan_per_sub < 2:
		_range_label = "channel_override range" if channel_override is not None else f"±{bw_width_factor:g}σ range"
		print(f"  Subband α fit: not enough channels for {n_sub} sub-bands ({n_freq_sub} channels in {_range_label})")
		return null_result

	# ── Sub-band EMG fits ─────────────────────────────────────────────────
	sub_freqs, sub_taus, sub_sigmas = [], [], []
	sub_tau_errs, sub_popts, sub_labels = [], [], []
	sub_sigma_errs = []  # per-sub-band σ uncertainty (sqrt(pcov[2,2]))
	# Exact per-sub-band float frequency bounds (NOT the rounded ones in
	# sub_labels — those are display strings). Reconstructing the sub-band
	# time series from the rounded label drops or adds channels at the edges
	# whenever freqs_sub[i_lo] or freqs_sub[i_hi-1] is non-integer (which is
	# almost always for CHIME). Storing the exact floats lets downstream
	# code (the 'q'-key popup and Panel 1 of the diagnostic plot) sum
	# precisely the channels the fit actually saw.
	sub_freq_los, sub_freq_his = [], []
	# All geometric sub-band slot bounds, regardless of fit success.
	# Used by the 'q'-key popup to show subfall data for sub-bands
	# whose EMG fit was rejected by quality cuts — the user can still
	# inspect the raw data even when there's no fit to overlay.
	all_sub_freq_los, all_sub_freq_his = [], []

	sigma_seed = float(fb_sigma) if (fb_sigma is not None and np.isfinite(fb_sigma)) else None
	tau_seed = float(fb_tau) if (fb_tau is not None and np.isfinite(fb_tau)) else None

	for i in range(n_sub):
		i_lo = i * chan_per_sub
		i_hi = (i + 1) * chan_per_sub if i < n_sub - 1 else n_freq_sub
		sub_ts = wfall_sub[i_lo:i_hi, :].sum(axis=0)
		sub_freq_arr = freqs_sub[i_lo:i_hi]
		nu_eff = np.mean(sub_freq_arr)
		# Exact float bounds of THIS sub-band (used by the 'q'-key popup
		# to reconstruct the same sub_ts the fit consumed).
		sb_f_lo_exact = float(sub_freq_arr.min())
		sb_f_hi_exact = float(sub_freq_arr.max())
		# Always record this geometric slot, even if the fit fails.
		all_sub_freq_los.append(sb_f_lo_exact)
		all_sub_freq_his.append(sb_f_hi_exact)

		popt, pcov, redchisq = _fit_emg_subband(
			sub_ts, duration_ms,
			sigma_seed=sigma_seed, tau_seed=tau_seed,
			dt_ms=res_time_ms
		)

		label = f"{sub_freq_arr.min():.0f}-{sub_freq_arr.max():.0f}"

		if popt is not None:
			sig_i = popt[2]
			tau_i = 1.0 / popt[3]
			lam_err = np.sqrt(pcov[3, 3]) if pcov is not None else np.nan
			tau_err = lam_err / (popt[3]**2) if lam_err > 0 else np.nan
			# σ uncertainty from the SAME EMG covariance matrix. popt[2]
			# is the Gaussian width σ, so pcov[2, 2] is its variance.
			# Mirrors the tau_err extraction one line above; feeds the
			# error bars on the σ-vs-ν panel (ax3).
			sig_err_i = np.sqrt(pcov[2, 2]) if pcov is not None else np.nan

			# Quality cuts: reject fits with pathologically narrow σ,
			# poor reduced chi-squared, or non-finite τ
			if sig_i < 0.001 * duration_ms or redchisq > 5 or not np.isfinite(tau_i):
				continue

			sub_freqs.append(nu_eff)
			sub_taus.append(tau_i)
			sub_sigmas.append(sig_i)
			sub_tau_errs.append(tau_err)
			sub_sigma_errs.append(sig_err_i)
			sub_popts.append(popt)
			sub_labels.append(label)
			sub_freq_los.append(sb_f_lo_exact)
			sub_freq_his.append(sb_f_hi_exact)

	sub_freqs = np.array(sub_freqs)
	sub_taus = np.array(sub_taus)
	sub_sigmas = np.array(sub_sigmas)
	sub_tau_errs = np.array(sub_tau_errs)
	sub_sigma_errs = np.array(sub_sigma_errs)
	sub_freq_los = np.array(sub_freq_los, dtype=float)
	sub_freq_his = np.array(sub_freq_his, dtype=float)
	all_sub_freq_los = np.array(all_sub_freq_los, dtype=float)
	all_sub_freq_his = np.array(all_sub_freq_his, dtype=float)

	# ── Apply user-requested ignore_subindex mask ─────────────────────────
	# User convention: sub-band index 1 = highest frequency, descending
	# (so index N = lowest frequency). Our sub_freqs array is sorted
	# ascending in frequency (sub_freqs[0] = lowest, sub_freqs[-1] =
	# highest), so the conversion is:
	#     array_idx = len(sub_freqs) - user_idx
	# i.e. user index 1 → array index len-1 (highest-freq), and
	#      user index N → array index 0 (lowest-freq).
	#
	# Out-of-range user indices (e.g. ignoring index 8 when only 7
	# sub-bands exist) raise ValueError here, since this is the first
	# point at which we know N_subbands.
	#
	# Two parallel arrays are kept:
	#   * `sub_*` arrays — UNMODIFIED, all sub-bands intact, used for
	#     plotting (so ignored points still show, just greyed out) and
	#     for n_good_subbands accounting.
	#   * `_fit_*` arrays — masked subset of kept sub-bands, used as
	#     input to the α(ν) and ε(σ) fits.
	# `_ignore_mask` (boolean, True for ignored array indices) and
	# `ignore_subindex` (the original user tuple) are stashed in the
	# result dict so the diagnostic plot can render greyed-out points.
	_ignore_mask = np.zeros(len(sub_freqs), dtype=bool)
	_user_ignore_tuple = ignore_subindex if ignore_subindex else None
	if _user_ignore_tuple:
		_N = len(sub_freqs)
		_bad = [_u for _u in _user_ignore_tuple if (_u < 1 or _u > _N)]
		if _bad:
			raise ValueError(
				f"`ignore_subindex` for burst '{bname}' references sub-band "
				f"index(es) {_bad} which do not exist — only {_N} sub-band(s) "
				f"are available (valid range: 1..{_N}, where 1 = highest "
				f"frequency)."
			)
		for _u in _user_ignore_tuple:
			_ignore_mask[_N - _u] = True

	_kept_mask = ~_ignore_mask
	_n_kept = int(np.sum(_kept_mask))
	if _user_ignore_tuple is not None:
		print(f"  Subband α fit: ignoring user-specified sub-band index(es) "
		      f"{list(_user_ignore_tuple)} for '{bname}' "
		      f"(array positions {sorted(np.where(_ignore_mask)[0].tolist())}); "
		      f"{_n_kept}/{len(sub_freqs)} sub-bands retained for fit.")
		if _n_kept < 3:
			print(f"  WARNING: only {_n_kept} sub-band(s) remain after applying "
			      f"`ignore_subindex` — α fit needs ≥3 points. Skipping the τ "
			      f"power-law and ε σ-smooth fits for this component.")

	# Fit-only views (masked). `sub_*` arrays themselves remain
	# untouched so the diagnostic plot and ε-fit plot can still show
	# the ignored points.
	_fit_freqs    = sub_freqs[_kept_mask]
	_fit_taus     = sub_taus[_kept_mask]
	_fit_sigmas   = sub_sigmas[_kept_mask]
	_fit_tau_errs = sub_tau_errs[_kept_mask]

	# ── Power-law fit: τ(ν) = τ_ref * (ν/ν_ref)^α ────────────────────────
	alpha_fit = tau_ref_fit = alpha_err = tau_ref_err = np.nan

	if _n_kept >= 3:
		log_nu = np.log(_fit_freqs / freq_center)
		log_tau = np.log(_fit_taus)
		if np.all(np.isfinite(_fit_tau_errs)) and np.all(_fit_tau_errs > 0):
			log_tau_err = _fit_tau_errs / _fit_taus   # δ(ln τ) ≈ δτ / τ
		else:
			log_tau_err = None

		try:
			_alpha_bounds = ([-np.inf, -4.4], [np.inf, -3.6]) if constrain_alpha else ([-np.inf, -np.inf], [np.inf, np.inf])
			popt_pl, pcov_pl = scipy.optimize.curve_fit(
				_power_law_log, log_nu, log_tau,
				p0=[np.log(np.median(_fit_taus)), -4.0],
				sigma=log_tau_err,
				absolute_sigma=True if log_tau_err is not None else False,
				bounds=_alpha_bounds
			)
			tau_ref_fit = np.exp(popt_pl[0])
			alpha_fit = popt_pl[1]
			alpha_err = np.sqrt(pcov_pl[1, 1])
			tau_ref_err = tau_ref_fit * np.sqrt(pcov_pl[0, 0])
		except Exception as _pl_e:
			# Weighted fit failed. Common causes: bad per-point σ (zeros / near-zeros
			# from under-constrained EMG λ covariance), degenerate log_nu spacing,
			# or bounds interaction with a tiny-weight outlier. Retry unweighted —
			# if the underlying points are reasonable the slope still recovers,
			# just without error-bar information folded in.
			print(f"  Subband α fit: weighted power-law fit failed ({type(_pl_e).__name__}: {_pl_e}); "
			      f"retrying unweighted.")
			try:
				popt_pl, pcov_pl = scipy.optimize.curve_fit(
					_power_law_log, log_nu, log_tau,
					p0=[np.log(np.median(_fit_taus)), -4.0],
					bounds=_alpha_bounds
				)
				tau_ref_fit = np.exp(popt_pl[0])
				alpha_fit = popt_pl[1]
				alpha_err = np.sqrt(pcov_pl[1, 1])
				tau_ref_err = tau_ref_fit * np.sqrt(pcov_pl[0, 0])
				print(f"  Subband α fit: unweighted fallback succeeded "
				      f"(uncertainties are formal, not from per-point σ).")
			except Exception as _pl_e2:
				print(f"  Subband α fit: unweighted power-law fit also failed "
				      f"({type(_pl_e2).__name__}: {_pl_e2}).")

	# ── Best σ from highest-frequency sub-band ─────────────────────────────
	# The highest-freq sub-band has the least scattering contamination in σ.
	# This uses the ORIGINAL (un-masked) sub_sigmas so the σ-vs-ν panel
	# diagnostic anchor is unaffected by `ignore_subindex` per the
	# user's spec (σ panel keeps everything as-is).
	best_sigma = sub_sigmas[-1] if len(sub_sigmas) > 0 else (fb_sigma if fb_sigma is not None else np.nan)

	# ── Smoothed σ(ν) power-law fit (ε-fit) ────────────────────────────────
	# σ(ν) = A · (ν/freq_center)^β with β ≤ 0 enforced (strictly monotonic
	# non-increasing). A is in ms (σ at freq_center); β is dimensionless.
	# Same reference frequency as the τ(ν) fit for consistency. Used by
	# the diagnostic plot and persisted to the CSV as ε columns.
	#
	# Per user spec (item 4): the ε fit uses ALL sub-bands regardless
	# of `ignore_subindex`. The ignore filter applies ONLY to the τ
	# power-law (α, τ_ref) fit. σ measurements are treated as
	# independent of the user's τ-fit exclusions here.
	_eps_sigma_fit = _fit_sigma_smooth_powerlaw(sub_freqs, sub_sigmas, freq_center)

	# ── Regime classification using sub-band-corrected values ──────────────
	regime_grid = classify_regime_by_fwhm(best_sigma, tau_ref_fit)

	# Also compute threshold-based classification for comparison
	if np.isfinite(tau_ref_fit) and np.isfinite(best_sigma) and best_sigma > 0:
		ratio_sb = tau_ref_fit / best_sigma
		regime_thr = 'weak' if ratio_sb < 0.5 else ('strong' if ratio_sb > 2.0 else 'intermediate')
	else:
		ratio_sb = np.nan
		regime_thr = 'unknown'

	success = np.isfinite(alpha_fit) and np.isfinite(tau_ref_fit)

	# ── Print results ─────────────────────────────────────────────────────
	if success:
		print(f"  Subband α fit ({len(sub_freqs)}/{n_sub} sub-bands): "
			  f"α = {alpha_fit:.2f} ± {alpha_err:.2f}, "
			  f"τ_ref = {tau_ref_fit:.4f} ms @ {freq_center:.1f} MHz, "
			  f"σ_hi = {best_sigma:.4f} ms")
		print(f"    Regime (closest-to-grid): {regime_grid['regime']}  "
			  f"(τ/σ = {regime_grid['tau_over_sigma']:.3f})")
		if _eps_sigma_fit['success']:
			print(f"    σ-smooth ε fit: ε = {_eps_sigma_fit['amp']:.4f} ± "
			      f"{_eps_sigma_fit['amp_err']:.4f} ms @ {freq_center:.1f} MHz, "
			      f"β = {_eps_sigma_fit['beta']:.2f} ± {_eps_sigma_fit['beta_err']:.2f}")
	else:
		fallback = "full-band" if fb_tau is not None else "none"
		if len(sub_freqs) < 3:
			print(f"  Subband α fit: FAILED — only {len(sub_freqs)} good sub-bands "
				  f"(need ≥3). Falling back to {fallback}.")
		else:
			print(f"  Subband α fit: FAILED — power-law fit did not converge "
				  f"despite {len(sub_freqs)} good sub-bands (see exception above). "
				  f"Falling back to {fallback}.")

	# ── σ-constrained second pass (constrain_sigma diagnostic) ────────────
	# Repeats the per-sub-band EMG fit with σ pinned to `sigma_fixed`,
	# refits the τ(ν) power law on the resulting τ values, and computes
	# Δτ / σ_τ per sub-band against the free-σ fit. The free-σ outputs
	# above (`alpha_fit`, `tau_ref_fit`, `sub_taus`, ...) are NOT modified
	# — Option C of the design: free fit is the headline, constrained
	# fit is reported alongside. All diagnostic outputs are NaN / None
	# when `sigma_fixed` is not requested.
	csig_alpha = csig_alpha_err = np.nan
	csig_tau_ref = csig_tau_ref_err = np.nan
	csig_sub_freqs = []
	csig_sub_taus = []
	csig_sub_tau_errs = []
	csig_sub_sigmas = []
	csig_sub_popts = []
	csig_dtau_over_sigma_tau_per_sb = []
	csig_dtau_over_sigma_tau_median = np.nan
	csig_dtau_over_sigma_tau_max = np.nan
	csig_all_sb_failed = False
	_csig_ignore_mask = np.array([], dtype=bool)

	if sigma_fixed is not None and np.isfinite(sigma_fixed) and sigma_fixed > 0:
		print(f"  σ-constrained diagnostic: re-fitting with σ = {sigma_fixed:.4f} ms")
		# Re-walk the same sub-band slicing the free pass used. The
		# wfall_sub / freqs_sub / chan_per_sub / n_freq_sub variables
		# are defined above (inside this function) and bound to the same
		# channel selection the free pass operated on, so the constrained
		# pass and the free pass differ ONLY in σ being pinned.
		for _i_c in range(n_sub):
			_i_lo_c = _i_c * chan_per_sub
			_i_hi_c = (_i_c + 1) * chan_per_sub if _i_c < n_sub - 1 else n_freq_sub
			_sub_ts = wfall_sub[_i_lo_c:_i_hi_c, :].sum(axis=0)
			_sub_freq_arr = freqs_sub[_i_lo_c:_i_hi_c]
			_nu_eff = float(np.mean(_sub_freq_arr))
			_popt, _pcov, _redchisq = _fit_emg_subband(
				_sub_ts, duration_ms,
				sigma_seed=sigma_fixed,
				tau_seed=tau_seed,
				dt_ms=res_time_ms,
				sigma_fixed=float(sigma_fixed),
			)
			if _popt is None:
				continue
			_sig_i = _popt[2]
			_tau_i = 1.0 / _popt[3]
			_lam_err = np.sqrt(_pcov[3, 3]) if _pcov is not None else np.nan
			_tau_err = _lam_err / (_popt[3]**2) if (np.isfinite(_lam_err) and _lam_err > 0) else np.nan
			if _sig_i < 0.001 * duration_ms or _redchisq > 5 or not np.isfinite(_tau_i):
				continue
			csig_sub_freqs.append(_nu_eff)
			csig_sub_taus.append(_tau_i)
			csig_sub_sigmas.append(_sig_i)
			csig_sub_tau_errs.append(_tau_err)
			csig_sub_popts.append(_popt)

		csig_sub_freqs    = np.array(csig_sub_freqs)
		csig_sub_taus     = np.array(csig_sub_taus)
		csig_sub_sigmas   = np.array(csig_sub_sigmas)
		csig_sub_tau_errs = np.array(csig_sub_tau_errs)

		# ── Apply ignore_subindex to the σ-constrained pass ──
		# The σ-free pass applies `ignore_subindex` to `_fit_*` arrays
		# above; the σ-constrained α power-law fit must use the SAME
		# excluded set, otherwise the comparison between free α and
		# σ-constrained α is unfair. The matching is by frequency
		# (within 0.1 MHz, same tolerance as the Δτ/σ_τ pairing
		# below) because the σ-constrained pass walks the same sub-band
		# slicing but only includes sub-bands whose fits converged —
		# its array length and ordering may differ from the free pass.
		_csig_ignore_mask = np.zeros(len(csig_sub_freqs), dtype=bool)
		if _user_ignore_tuple is not None and len(csig_sub_freqs) > 0:
			for _j_c, _nu_c in enumerate(csig_sub_freqs):
				_m = np.where(np.abs(np.asarray(sub_freqs) - _nu_c) < 0.1)[0]
				if len(_m) > 0 and _ignore_mask[int(_m[0])]:
					_csig_ignore_mask[_j_c] = True
		_csig_kept_mask = ~_csig_ignore_mask
		_csig_n_kept    = int(np.sum(_csig_kept_mask))
		if _user_ignore_tuple is not None:
			print(f"  σ-constrained diagnostic: ignore_subindex applied "
			      f"({_csig_n_kept}/{len(csig_sub_freqs)} σ-constrained sub-bands retained for α fit)")

		if len(csig_sub_taus) == 0:
			csig_all_sb_failed = True
			print(f"  σ-constrained diagnostic: ALL sub-bands failed the constrained fit.")
		else:
			# Power-law fit on the constrained τ values, after dropping
			# the ignore_subindex sub-bands (matched to the free pass).
			_csig_fit_freqs    = csig_sub_freqs[_csig_kept_mask]
			_csig_fit_taus     = csig_sub_taus[_csig_kept_mask]
			_csig_fit_tau_errs = csig_sub_tau_errs[_csig_kept_mask]
			if len(_csig_fit_taus) >= 3:
				_log_nu_c = np.log(_csig_fit_freqs / freq_center)
				_log_tau_c = np.log(_csig_fit_taus)
				if (np.all(np.isfinite(_csig_fit_tau_errs))
				    and np.all(_csig_fit_tau_errs > 0)):
					_log_tau_err_c = _csig_fit_tau_errs / _csig_fit_taus
				else:
					_log_tau_err_c = None
				try:
					_alpha_bounds_c = ([-np.inf, -4.4], [np.inf, -3.6]) if constrain_alpha \
					                  else ([-np.inf, -np.inf], [np.inf, np.inf])
					_popt_plc, _pcov_plc = scipy.optimize.curve_fit(
						_power_law_log, _log_nu_c, _log_tau_c,
						p0=[np.log(np.median(_csig_fit_taus)), -4.0],
						sigma=_log_tau_err_c,
						absolute_sigma=True if _log_tau_err_c is not None else False,
						bounds=_alpha_bounds_c,
					)
					csig_tau_ref     = float(np.exp(_popt_plc[0]))
					csig_alpha       = float(_popt_plc[1])
					csig_alpha_err   = float(np.sqrt(_pcov_plc[1, 1]))
					csig_tau_ref_err = float(csig_tau_ref * np.sqrt(_pcov_plc[0, 0]))
				except Exception as _e_c:
					print(f"  σ-constrained diagnostic: power-law fit failed "
					      f"({type(_e_c).__name__}: {_e_c}).")

			# Δτ / σ_τ per sub-band. Match by frequency (within 0.1 MHz)
			# since the free and constrained passes use the same sub-band
			# slicing and should produce identical `nu_eff` values modulo
			# floating-point. A sub-band that succeeded in only one of
			# the two passes is recorded as NaN — we can only compute Δτ
			# where both fits gave a τ.
			#
			# `_dtau_aligned` is index-aligned to `csig_sub_freqs` so the
			# diagnostic plot can map each Δτ to its sub-band frequency
			# and grey out ignored points. The summary median/max
			# statistics use only KEPT (non-ignored) sub-bands so they
			# reflect the same point set as the σ-constrained α fit.
			_dtau_aligned = np.full(len(csig_sub_freqs), np.nan, dtype=float)
			for _k_c, (_nu_c, _tau_c, _tauerr_c) in enumerate(zip(
			    csig_sub_freqs, csig_sub_taus, csig_sub_tau_errs)):
				_match = np.where(np.abs(np.asarray(sub_freqs) - _nu_c) < 0.1)[0]
				if len(_match) == 0:
					continue
				_j = int(_match[0])
				_tau_f      = float(sub_taus[_j])
				_tau_f_err  = float(sub_tau_errs[_j])
				if not (np.isfinite(_tau_f) and np.isfinite(_tauerr_c)
				        and np.isfinite(_tau_f_err)):
					continue
				_mean_sigma_tau = 0.5 * (_tauerr_c + _tau_f_err)
				if _mean_sigma_tau <= 0:
					continue
				_dtau_aligned[_k_c] = abs(_tau_c - _tau_f) / _mean_sigma_tau
			csig_dtau_over_sigma_tau_per_sb = _dtau_aligned
			# Summary stats over KEPT sub-bands only (mirrors the α fit).
			_dtau_kept = _dtau_aligned[_csig_kept_mask] if len(_csig_kept_mask) == len(_dtau_aligned) else _dtau_aligned
			_dtau_kept_finite = _dtau_kept[np.isfinite(_dtau_kept)]
			if len(_dtau_kept_finite) > 0:
				csig_dtau_over_sigma_tau_median = float(np.median(_dtau_kept_finite))
				csig_dtau_over_sigma_tau_max    = float(np.max(_dtau_kept_finite))
				print(f"  σ-constrained diagnostic: Δτ/σ_τ median = {csig_dtau_over_sigma_tau_median:.3f}, "
				      f"max = {csig_dtau_over_sigma_tau_max:.3f} "
				      f"(per-sb kept: {[f'{x:.2f}' for x in _dtau_kept_finite]})")
			if np.isfinite(csig_alpha):
				print(f"    σ-constrained α = {csig_alpha:.3f} ± {csig_alpha_err:.3f}, "
				      f"τ_ref = {csig_tau_ref:.4f} ms (vs free α = {alpha_fit:.3f} ± {alpha_err:.3f})")

	return {
		'alpha': alpha_fit, 'alpha_err': alpha_err,
		'tau_ref': tau_ref_fit, 'tau_ref_err': tau_ref_err,
		'best_sigma': best_sigma, 'freq_center': freq_center,
		# --- ignore_subindex bookkeeping (for diagnostic plot + CSV) ---
		'ignore_subindex_used': _user_ignore_tuple,
		'_ignore_mask':         _ignore_mask,
		# --- Smoothed σ(ν) power-law (ε) fit ---
		'eps_sigma_amp':      _eps_sigma_fit['amp'],
		'eps_sigma_amp_err':  _eps_sigma_fit['amp_err'],
		'eps_sigma_beta':     _eps_sigma_fit['beta'],
		'eps_sigma_beta_err': _eps_sigma_fit['beta_err'],
		'eps_sigma_success':  _eps_sigma_fit['success'],
		# --- σ-constrained diagnostic (Option C: parallel reporting) ---
		# All NaN / empty when `sigma_fixed` was not requested. When set,
		# these mirror the corresponding free-fit keys above so downstream
		# consumers can render both fits side-by-side.
		'sigma_constrained_value_ms':           float(sigma_fixed) if sigma_fixed is not None else np.nan,
		'csig_alpha':                            csig_alpha,
		'csig_alpha_err':                        csig_alpha_err,
		'csig_tau_ref':                          csig_tau_ref,
		'csig_tau_ref_err':                      csig_tau_ref_err,
		'csig_sub_freqs':                        csig_sub_freqs if len(csig_sub_freqs) else np.array([]),
		'csig_sub_taus':                         csig_sub_taus if len(csig_sub_taus) else np.array([]),
		'csig_sub_tau_errs':                     csig_sub_tau_errs if len(csig_sub_tau_errs) else np.array([]),
		'csig_sub_sigmas':                       csig_sub_sigmas if len(csig_sub_sigmas) else np.array([]),
		'csig_sub_popts':                        csig_sub_popts,
		'csig_ignore_mask':                      _csig_ignore_mask if sigma_fixed is not None else np.array([], dtype=bool),
		'csig_dtau_over_sigma_tau_per_sb':       csig_dtau_over_sigma_tau_per_sb,
		'csig_dtau_over_sigma_tau_median':       csig_dtau_over_sigma_tau_median,
		'csig_dtau_over_sigma_tau_max':          csig_dtau_over_sigma_tau_max,
		'csig_all_sb_failed':                    csig_all_sb_failed,
		'n_good_subbands': len(sub_freqs),
		'sub_freqs': sub_freqs, 'sub_taus': sub_taus,
		'sub_tau_errs': sub_tau_errs, 'sub_sigmas': sub_sigmas,
		'sub_sigma_errs': sub_sigma_errs,
		'sub_popts': sub_popts, 'sub_labels': sub_labels,
		'sub_freq_los': sub_freq_los, 'sub_freq_his': sub_freq_his,
		'all_sub_freq_los': all_sub_freq_los, 'all_sub_freq_his': all_sub_freq_his,
		'regime_grid': regime_grid,
		'regime_threshold': {'regime': regime_thr, 'tau_over_sigma': ratio_sb},
		'success': success,
		'freq_lo': freq_lo, 'freq_hi': freq_hi,
		'spec_popt_inner': spec_popt_inner,
	}

# === END SUBBAND ALPHA FIT ===


def make_subband_diagnostic_plots(burst_name, wfall, freqs, duration_ms,
                                  sb_result, constrain_index=False,
                                  fb_tau=None, fb_sigma=None, res_time_ms=None,
                                  save=False, outdir='', show=True,
                                  outfmt='.png', comp_label=None,
                                  model_tag=None, name_suffix=''):
    # name_suffix (v56_10): appended to the saved filename and the suptitle so
    # the two pop-ups produced per component — one for the index-free fit and
    # one for the index-fixed (α pinned to ~-4) fit — do not overwrite each
    # other. Pass e.g. 'idxfree' / 'idxfixed'. Empty string = legacy behaviour.
    """
    Create a single figure with 3 subplots from sub-band EMG fitting results:
      1. Sub-band profiles + EMG fits
      2. Scattering timescale τ(ν) with power-law fit + α=−4 reference
      3. Gaussian width σ across sub-bands

    Parameters
    ----------
    burst_name : str
    wfall : 2D array (freq × time)
    freqs : 1D array (MHz)
    duration_ms : float
    sb_result : dict from fit_subband_alpha
    constrain_index : bool — if True, label fit as constrained
    fb_tau : float — full-band τ (ms), for reference line
    fb_sigma : float — full-band σ (ms), for reference line
    res_time_ms : float or None
        Time-bin width (ms) used when the EMG fits were made. Required to
        evaluate the *bin-integrated* model that the fit actually minimised
        against; if None, falls back to duration_ms / n_time which is the
        same value the caller used (subfall.shape[1] * res_time_ms / n_time).
        The fine-grid instantaneous curve does not depend on this value.
    """
    if not sb_result or sb_result['n_good_subbands'] < 2:
        print(f"  Subband plot: only {sb_result['n_good_subbands'] if sb_result else 0} good sub-bands, skipping plots")
        return None

    sub_freqs    = sb_result['sub_freqs']
    sub_taus     = sb_result['sub_taus']
    sub_tau_errs = sb_result['sub_tau_errs']
    sub_sigmas   = sb_result['sub_sigmas']
    # σ uncertainties per sub-band (sqrt(pcov[2,2]) from each EMG fit).
    # .get() with a NaN-filled fallback keeps older sb_result dicts that
    # predate this key from breaking the plot.
    sub_sigma_errs = sb_result.get('sub_sigma_errs',
                                   np.full(len(sub_sigmas), np.nan))
    sub_labels   = sb_result['sub_labels']
    sub_popts    = sb_result['sub_popts']
    freq_center  = sb_result['freq_center']
    best_sigma   = sb_result['best_sigma']
    alpha_fit    = sb_result['alpha']
    alpha_err    = sb_result['alpha_err']
    tau_ref_fit  = sb_result['tau_ref']
    tau_ref_err  = sb_result.get('tau_ref_err', np.nan)
    # σ-constrained diagnostic outputs (NaN / [] when not requested).
    # Drawn as a second dashed line in dark blue on the τ(ν) panel,
    # with the Δτ/σ_τ summary annotated in the residual sub-panel.
    csig_sigma_value     = sb_result.get('sigma_constrained_value_ms', np.nan)
    csig_alpha           = sb_result.get('csig_alpha', np.nan)
    csig_alpha_err       = sb_result.get('csig_alpha_err', np.nan)
    csig_tau_ref         = sb_result.get('csig_tau_ref', np.nan)
    csig_dtau_median     = sb_result.get('csig_dtau_over_sigma_tau_median', np.nan)
    csig_dtau_max        = sb_result.get('csig_dtau_over_sigma_tau_max', np.nan)
    csig_all_failed      = bool(sb_result.get('csig_all_sb_failed', False))
    csig_requested       = np.isfinite(csig_sigma_value) and csig_sigma_value > 0

    n_time  = wfall.shape[1]
    n_freq  = wfall.shape[0]
    n_sub   = len(sub_labels)  # total sub-bands attempted (good ones)
    t_axis  = np.linspace(0, duration_ms, n_time)
    chan_per_sub = n_freq // N_SUBBANDS_ALPHA

    # Resolve bin width (for bin-integrated model evaluation). The fits in
    # _fit_emg_subband used dt_ms = res_time_ms passed from fit_subband_alpha.
    # Falling back to duration_ms / n_time recovers the same value the
    # caller passes when duration_ms was set as n_time * res_time_ms.
    _dt_ms = float(res_time_ms) if (res_time_ms is not None and res_time_ms > 0) \
             else (duration_ms / n_time if n_time > 0 else None)
    # Fine grid for smooth instantaneous EMG curve (avoids the jagged
    # line-segment look when σ ≲ dt, e.g. CHIME bursts at narrow σ).
    _fine_factor = 20
    t_axis_fine = np.linspace(0, duration_ms, _fine_factor * n_time)

    # ── Single figure with 3 logical subplots ──
    # ax2 is split into a main panel + a small residuals strip
    # underneath it (showing (data - fit) / σ for each kept sub-band).
    # We use a 1×3 outer GridSpec where the middle cell is further
    # subdivided into a 2-row sub-grid (3:1 height ratio). ax1 and ax3
    # span the full outer-cell height so their geometry matches the
    # original side-by-side layout.
    # ── Layout: 1×3 by default, 2×3 when constrain_sigma was requested ──
    # The bottom row mirrors the top row but uses the σ-constrained
    # per-sub-band fits, so the user can compare side-by-side how
    # pinning σ shifts the recovered τ(ν) and α. Panel 6 (bottom-right)
    # replaces the redundant "σ across sub-bands" panel (which would
    # be a flat line at σ_fixed) with a Δτ/σ_τ panel — i.e. how far
    # each constrained τ shifted from the free τ in units of the
    # mean of the two τ uncertainties.
    if csig_requested:
        fig = plt.figure(figsize=(20, 12))
        _outer_gs = fig.add_gridspec(2, 3, hspace=0.30)
        ax1 = fig.add_subplot(_outer_gs[0, 0])
        _ax2_inner = _outer_gs[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax2 = fig.add_subplot(_ax2_inner[0])
        ax2_res = fig.add_subplot(_ax2_inner[1], sharex=ax2)
        ax3 = fig.add_subplot(_outer_gs[0, 2])
        # Bottom-row axes (drawn after the top row's code below)
        ax1_b = fig.add_subplot(_outer_gs[1, 0])
        _ax2b_inner = _outer_gs[1, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax2_b = fig.add_subplot(_ax2b_inner[0])
        ax2_b_res = fig.add_subplot(_ax2b_inner[1], sharex=ax2_b)
        ax3_b = fig.add_subplot(_outer_gs[1, 2])
    else:
        fig = plt.figure(figsize=(20, 6))
        _outer_gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(_outer_gs[0, 0])
        _ax2_inner = _outer_gs[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax2 = fig.add_subplot(_ax2_inner[0])
        ax2_res = fig.add_subplot(_ax2_inner[1], sharex=ax2)
        ax3 = fig.add_subplot(_outer_gs[0, 2])
        ax1_b = ax2_b = ax2_b_res = ax3_b = None     # not drawn
    fig.suptitle(f'{burst_name} — Sub-band scattering diagnostics', fontsize=14, fontweight='bold')

    # ── Panel 1: Sub-band profiles + EMG fits ──
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(sub_popts)))
    # Prefer exact float bounds (parallel to sub_labels). Falling back to
    # parsing the rounded labels works but drops/adds boundary channels and
    # produces a sub_ts that is multiplicatively offset from what the fit
    # actually saw — visible as the smooth fit floating above the data line
    # in Panel 1 for non-integer-MHz channelisations like CHIME.
    _sb_freq_los = sb_result.get('sub_freq_los', None)
    _sb_freq_his = sb_result.get('sub_freq_his', None)
    _have_exact_bounds = (_sb_freq_los is not None and _sb_freq_his is not None
                          and len(_sb_freq_los) == len(sub_labels)
                          and len(_sb_freq_his) == len(sub_labels))
    for j, (popt_j, lbl_j) in enumerate(zip(sub_popts, sub_labels)):
        # Reconstruct sub-band time series from channel range
        if _have_exact_bounds:
            freq_lo = float(_sb_freq_los[j])
            freq_hi = float(_sb_freq_his[j])
        else:
            freq_lo, freq_hi = [float(x) for x in lbl_j.split('-')]
        mask = (freqs >= freq_lo) & (freqs <= freq_hi)
        sub_ts = wfall[mask, :].sum(axis=0)

        # Common normalisation derived from the data so curve, bin-integrated
        # values, and data line all share the same vertical scale.
        _ts_min = sub_ts.min()
        _ts_range = (sub_ts.max() - _ts_min) + 1e-30

        sub_norm = (sub_ts - _ts_min) / _ts_range
        ax1.plot(t_axis, sub_norm + j * 0.15, color=colors[j], alpha=0.5, lw=0.8)

        # (a) Smooth instantaneous EMG on the fine grid — clean visual curve.
        fit_y_fine = emg_model(t_axis_fine, *popt_j)
        fit_norm_fine = (fit_y_fine - _ts_min) / _ts_range
        tau_j = 1.0 / popt_j[3] if popt_j[3] > 0 else np.nan
        ax1.plot(t_axis_fine, fit_norm_fine + j * 0.15, color=colors[j], lw=2,
                 label=f'{lbl_j} MHz: τ={tau_j:.2f} ms')

        # (b) Bin-integrated EMG at the actual data bin centres — this is
        # the model the fit minimised against. Drawn as small dots so any
        # visible gap from the smooth curve at the peak is the genuine
        # bin-integration effect (matters when σ ≲ dt).
        if _dt_ms is not None:
            _sigma_p = popt_j[2]; _lam_p = popt_j[3]
            _tau_p = (1.0 / _lam_p) if (np.isfinite(_lam_p) and _lam_p > 0) else np.nan
            _ovs = pick_oversamp(_dt_ms, _sigma_p, _tau_p)
            fit_y_bins = bin_integrated_emg(t_axis, *popt_j, dt_ms=_dt_ms, oversamp=_ovs)
            fit_norm_bins = (fit_y_bins - _ts_min) / _ts_range
            ax1.plot(t_axis, fit_norm_bins + j * 0.15, '.', color=colors[j],
                     ms=3, alpha=0.8)

    # Panel 1 legend on upper-left: with many sub-bands ("τ = X.XX ms"
    # one row per band) the legend box gets wide; placing it on the
    # right pushed it up against panel 2's y-axis area in tight_layout.
    # Upper-left keeps it inside panel 1's own free space.
    ax1.legend(fontsize=7, loc='upper left')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Normalized intensity (offset)')
    ax1.set_title('Sub-band profiles + EMG fits')

    # ── Panel 2: τ(ν) power-law ──
    nu_fine = np.linspace(freqs.min(), freqs.max(), 200)

    # Reference α = −4.0 line, anchored at the median measured τ
    tau_anchor = np.median(sub_taus)
    tau_m4_fine = tau_anchor * (nu_fine / freq_center) ** (-4.0)
    ax2.plot(nu_fine, tau_m4_fine, 'k-', lw=2.5, label='Reference: α=−4.0', zorder=1)

    # Ignored sub-bands (per `ignore_subindex`) are drawn in grey
    # alongside the kept (blue) points so the user can see which
    # points the α fit included. Kept-mask defaults to "all kept"
    # if no ignore was applied, in which case the second errorbar
    # call is a no-op.
    _ignore_mask = sb_result.get('_ignore_mask', None)
    if _ignore_mask is None or not isinstance(_ignore_mask, np.ndarray) \
            or _ignore_mask.shape != sub_freqs.shape:
        _ignore_mask_local = np.zeros(len(sub_freqs), dtype=bool)
    else:
        _ignore_mask_local = _ignore_mask
    _kept_mask_local = ~_ignore_mask_local
    _n_kept_local    = int(np.sum(_kept_mask_local))
    _n_ignored_local = int(np.sum(_ignore_mask_local))

    # Decide a "sensible" τ-axis scale from the well-behaved data
    # *before* plotting the kept points. The auto-range otherwise
    # blows up to fit any pathological upper error bar (a failed
    # sub-band fit can produce τ ≈ 0 with σ_τ ≈ 10³ ms, which
    # squashes the actual fit region into a sliver). We anchor the
    # scale on the τ values themselves, the reference α=−4 curve,
    # and the full-band τ — none of which include runaway σ_τ.
    _yref_pool = [sub_taus[_kept_mask_local]] if _n_kept_local > 0 else []
    _yref_pool.append(np.atleast_1d(tau_m4_fine))
    if fb_tau is not None and np.isfinite(fb_tau) and fb_tau > 0:
        _yref_pool.append(np.array([fb_tau]))
    _yref_concat = np.concatenate([np.asarray(_a).ravel() for _a in _yref_pool])
    _yref_pos = _yref_concat[(_yref_concat > 0) & np.isfinite(_yref_concat)]
    if _yref_pos.size > 0:
        _y_top_ref = float(np.max(_yref_pos))
        _y_bot_ref = float(np.min(_yref_pos))
    else:
        _y_top_ref = 1.0
        _y_bot_ref = 1e-2

    # "Offscreen" criterion: a point's upper bar (τ + σ_τ) reaches
    # more than OVERFLOW_FACTOR × the reference top. This catches
    # degenerate fits (τ ≈ 0 with σ_τ orders of magnitude larger
    # than any real τ in the burst) without flagging legitimately
    # large but finite uncertainties. Same logic mirrored on the
    # lower bar for symmetry.
    _OVERFLOW_FACTOR = 10.0
    _y_top_limit = _y_top_ref * 3.0     # actual axis top (padding on ref)
    _y_bot_limit = max(_y_bot_ref / 3.0, 1e-3)

    # Second criterion: floor-pinned τ. The EMG sub-band fitter
    # imposes τ ≥ dt_ms/2 to stop the optimiser from collapsing
    # into the flat-χ² valley at τ → 0 in unresolved sub-bands
    # (see _fit_emg_subband bounds). When the fit comes back at
    # or just above that floor, the reported τ is the bound, not
    # a measurement — by definition the data couldn't resolve a
    # scattering tail at that frequency. We treat those as failed
    # fits for display purposes too, even though their σ_τ is now
    # finite and small enough to escape the offscreen criterion
    # above. Tolerance is +25% of the floor (τ ≤ 0.625 × dt_ms),
    # which catches points sitting on the bound while leaving any
    # genuinely resolved short τ alone.
    _FLOOR_TOL_FACTOR = 1.25
    if _dt_ms is not None and _dt_ms > 0:
        _tau_floor      = 0.5 * float(_dt_ms)
        _tau_floor_thr  = _FLOOR_TOL_FACTOR * _tau_floor
    else:
        _tau_floor_thr  = 0.0     # no dt_ms → can't apply this criterion

    _err_kept_full = sub_tau_errs[_kept_mask_local] if _n_kept_local > 0 else np.array([])
    _tau_kept_full = sub_taus[_kept_mask_local]    if _n_kept_local > 0 else np.array([])
    _upper_bar = _tau_kept_full + np.where(np.isfinite(_err_kept_full),
                                            _err_kept_full, 0.0)
    _overflow_kept = (_upper_bar > _OVERFLOW_FACTOR * _y_top_ref) \
                     | (~np.isfinite(_err_kept_full)) \
                     | (_err_kept_full > _OVERFLOW_FACTOR * _y_top_ref) \
                     | (_tau_kept_full <= _tau_floor_thr)
    _good_kept     = ~_overflow_kept
    _n_good_kept     = int(np.sum(_good_kept))
    _n_overflow_kept = int(np.sum(_overflow_kept))

    # Index back to the original sub_* arrays — useful below when
    # we want to de-emphasise the *same* points in the residual strip.
    _kept_idx_all      = np.where(_kept_mask_local)[0]
    _overflow_idx_orig = _kept_idx_all[_overflow_kept] if _n_kept_local > 0 else np.array([], dtype=int)
    _overflow_mask_full = np.zeros(len(sub_freqs), dtype=bool)
    if _overflow_idx_orig.size > 0:
        _overflow_mask_full[_overflow_idx_orig] = True

    if _n_good_kept > 0:
        _freqs_good = sub_freqs[_kept_mask_local][_good_kept]
        _taus_good  = _tau_kept_full[_good_kept]
        _errs_good  = (_err_kept_full[_good_kept]
                       if np.all(np.isfinite(_err_kept_full)) else None)
        ax2.errorbar(_freqs_good, _taus_good,
                     yerr=_errs_good,
                     fmt='o', ms=8, color='C0', capsize=4, zorder=3,
                     label=f'Sub-band fits (N={_n_kept_local})')
    if _n_overflow_kept > 0:
        # Flagged points — either:
        #   • upper error bar runs offscreen (degenerate σ_τ), or
        #   • τ is pinned at/near the dt/2 floor (unresolved sub-band)
        # Both modes mean the reported (τ, σ_τ) pair isn't a fully
        # trustworthy measurement: the τ is a bound, and the σ_τ
        # reflects fit-covariance behaviour at that bound rather
        # than a clean parameter uncertainty.
        #
        # Per spec: in this panel we still SHOW the error bars
        # (they convey the spread of what the fit returned even
        # if interpretation is qualified), and overlay an angled
        # strikethrough on the marker to flag the point visually.
        # The residual strip below drops the bars entirely.
        _freqs_of = sub_freqs[_kept_mask_local][_overflow_kept]
        _taus_of  = _tau_kept_full[_overflow_kept]
        _errs_of  = (_err_kept_full[_overflow_kept]
                     if np.all(np.isfinite(_err_kept_full)) else None)
        ax2.errorbar(_freqs_of, _taus_of,
                     yerr=_errs_of,
                     fmt='o', ms=8, color='C0', ecolor='C0',
                     capsize=4, alpha=0.55, zorder=2.5)
        # Strikethrough overlay: a wide horizontal-dash marker
        # rotated 45° so it reads as a diagonal slash through the
        # circle. Drawn on top of the marker layer with full alpha
        # so the flag is legible regardless of the underlying
        # marker's translucency.
        _slash = MarkerStyle('_').rotated(deg=45)
        ax2.plot(_freqs_of, _taus_of,
                 marker=_slash, ms=16, mew=2.4,
                 color='C0', alpha=0.95,
                 linestyle='None', zorder=3.6)
    if _n_ignored_local > 0:
        # Greyed-out ignored points (both marker and errorbar). ecolor
        # set explicitly so the cap and bar lines also use grey rather
        # than inheriting the marker colour.
        ax2.errorbar(sub_freqs[_ignore_mask_local], sub_taus[_ignore_mask_local],
                     yerr=(sub_tau_errs[_ignore_mask_local]
                           if np.all(np.isfinite(sub_tau_errs)) else None),
                     fmt='o', ms=8, color='#B0B0B0', ecolor='#B0B0B0',
                     capsize=4, alpha=0.6, zorder=2,
                     label=f'Ignored (N={_n_ignored_local})')

    if np.isfinite(alpha_fit) and np.isfinite(tau_ref_fit):
        tau_fit_fine = tau_ref_fit * (nu_fine / freq_center) ** alpha_fit
        lbl = f'Fit: α={alpha_fit:.2f}'
        if np.isfinite(alpha_err):
            lbl += f'±{alpha_err:.2f}'
        if constrain_index:
            lbl += ' (constrained)'

        # ±1σ confidence band around the fit line.
        # Computed via Monte Carlo from the (α, τ_ref) marginal
        # uncertainties — sampling independently is a simplification
        # (the joint posterior has correlation between α and τ_ref),
        # but for ±1σ display purposes this gives a sensible band
        # width without needing the full covariance. The band gets
        # filled BEFORE the fit line so the dashed line draws on top.
        if (np.isfinite(alpha_err) and np.isfinite(tau_ref_err)
                and alpha_err > 0 and tau_ref_err > 0):
            _N_MC = 2000
            _rng = np.random.default_rng(0)  # deterministic for reproducibility
            _alpha_samp   = _rng.normal(alpha_fit, alpha_err, _N_MC)
            _tau_ref_samp = _rng.normal(tau_ref_fit, tau_ref_err, _N_MC)
            # Vectorise: shape (_N_MC, len(nu_fine))
            _fit_samples = (_tau_ref_samp[:, None]
                            * (nu_fine[None, :] / freq_center) ** _alpha_samp[:, None])
            # Only keep positive fits for percentile (negatives from
            # extreme tail draws would corrupt the percentile values
            # in log space). MC samples should be overwhelmingly positive
            # for reasonable σs anyway.
            _fit_samples = np.where(_fit_samples > 0, _fit_samples, np.nan)
            _band_lo = np.nanpercentile(_fit_samples, 16, axis=0)
            _band_hi = np.nanpercentile(_fit_samples, 84, axis=0)
            ax2.fill_between(nu_fine, _band_lo, _band_hi, color='red',
                              alpha=0.18, zorder=1.5, label='Fit ±1σ band')

        ax2.plot(nu_fine, tau_fit_fine, 'r--', lw=2, alpha=0.8, label=lbl, zorder=2)
        ax2.plot(freq_center, tau_ref_fit, 'r*', ms=12, zorder=4)

        # σ-constrained second fit (Option C: parallel reporting). Drawn
        # as a dark-blue dashed line atop the free-fit ±1σ band so the
        # comparison is immediately visible. Only renders when the user
        # requested `constrain_sigma` AND the constrained α fit
        # converged. If the constrained per-sub-band fits all failed,
        # we draw nothing here and let the residual sub-panel display
        # the "all sub-bands failed σ-constrained fit" notice below.
        #
        # Suppressed when the 2×3 layout is active (`ax2_b is not None`):
        # the bottom-middle panel shows the σ-fixed fit on its own
        # axes, and duplicating it here would clutter the top panel
        # with information that already has a dedicated home.
        if (csig_requested and np.isfinite(csig_alpha) and np.isfinite(csig_tau_ref)
                and ax2_b is None):
            _csig_tau_fine = csig_tau_ref * (nu_fine / freq_center) ** csig_alpha
            _csig_lbl = f'Fit (σ-fixed @ {csig_sigma_value:.2f} ms): α={csig_alpha:.2f}'
            if np.isfinite(csig_alpha_err):
                _csig_lbl += f'±{csig_alpha_err:.2f}'
            ax2.plot(nu_fine, _csig_tau_fine,
                     ls='--', lw=2, alpha=0.9,
                     color='#1f3a93',  # dark blue
                     label=_csig_lbl, zorder=3)
            ax2.plot(freq_center, csig_tau_ref,
                     marker='*', ms=12, color='#1f3a93', zorder=4)

        # χ²_red of the fit, computed only on the KEPT sub-bands
        # (the ones that fed the α fit — ignored points are excluded
        # by ignore_subindex). Annotated in the top-right of ax2.
        if _n_kept_local >= 2:
            _nu_kept   = sub_freqs[_kept_mask_local]
            _tau_kept  = sub_taus[_kept_mask_local]
            _err_kept  = sub_tau_errs[_kept_mask_local]
            _tau_pred_at_data = tau_ref_fit * (_nu_kept / freq_center) ** alpha_fit
            _valid_err = np.isfinite(_err_kept) & (_err_kept > 0)
            if np.sum(_valid_err) >= 2:
                _chi2 = np.sum(((_tau_kept[_valid_err] - _tau_pred_at_data[_valid_err])
                                / _err_kept[_valid_err]) ** 2)
                # dof = N - 2 regardless of whether α was constrained.
                # Even with α held fixed by a prior, τ_ref is still
                # being fit — reporting dof relative to both nominal
                # parameters (α, τ_ref) gives a more conservative
                # χ²_red that doesn't take credit for the constraint.
                # This matches the convention preferred for the
                # methodology chapter.
                _dof = int(np.sum(_valid_err)) - 2
                if _dof > 0:
                    _chi2_red = _chi2 / _dof
                    ax2.text(
                        0.97, 0.97,
                        f'$\\chi^2_{{\\rm red}}$ = {_chi2_red:.2f}  (dof={_dof})',
                        transform=ax2.transAxes, ha='right', va='top',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='#888', alpha=0.85),
                        zorder=10,
                    )
                # Residuals subplot — (data - fit) / σ for each kept
                # point. ±1σ band in light red as reference. Points
                # outside ±1σ are honestly visible as outliers.
                #
                # Points whose σ_τ is so large that their error bar
                # blew past the τ-panel axis ("overflow") get rendered
                # translucent here too. With σ_τ huge, the normalised
                # residual is mechanically driven toward zero (any real
                # deviation is divided away) and the ±1σ bar would
                # imply a meaningful constraint that doesn't exist —
                # showing them at full opacity is misleading.
                _nu_v   = _nu_kept[_valid_err]
                _tau_v  = _tau_kept[_valid_err]
                _err_v  = _err_kept[_valid_err]
                _pred_v = _tau_pred_at_data[_valid_err]
                _resid_norm = (_tau_v - _pred_v) / _err_v
                # Map validity back to original sub_* indices so we
                # can pick out which residual entries correspond to
                # overflow points in the τ-panel split above.
                _kept_idx_for_valid = _kept_idx_all[_valid_err]
                _resid_overflow = _overflow_mask_full[_kept_idx_for_valid]
                _resid_good     = ~_resid_overflow

                ax2_res.axhline(0, color='black', lw=0.8, zorder=1)
                ax2_res.axhspan(-1, 1, color='red', alpha=0.10, zorder=0)
                if np.any(_resid_good):
                    ax2_res.errorbar(_nu_v[_resid_good], _resid_norm[_resid_good],
                                      yerr=1.0,
                                      fmt='o', ms=6, color='C0', capsize=2, zorder=2)
                if np.any(_resid_overflow):
                    # Same flagged set as the τ panel above. We
                    # drop the ±1σ bar (the underlying σ_τ is not a
                    # real uncertainty — see comment in the τ panel)
                    # and overlay the same angled strikethrough so
                    # the reader can match flagged points across the
                    # two panels by both frequency and visual style.
                    _resid_of_x = _nu_v[_resid_overflow]
                    _resid_of_y = _resid_norm[_resid_overflow]
                    ax2_res.plot(_resid_of_x, _resid_of_y,
                                 marker='o', ms=6, color='C0', alpha=0.45,
                                 linestyle='None', zorder=1.8)
                    _slash_res = MarkerStyle('_').rotated(deg=45)
                    ax2_res.plot(_resid_of_x, _resid_of_y,
                                 marker=_slash_res, ms=12, mew=2.0,
                                 color='C0', alpha=0.95,
                                 linestyle='None', zorder=1.85)
                # Adaptive symmetric y-limits — always fit the worst
                # outlier with comfortable padding so all kept points
                # are visible regardless of how large their residuals
                # get. No hard upper cap: if χ²_red is large because
                # of severe outliers, the residuals panel will scale
                # to show that honestly. Minimum of ±2.5σ to keep the
                # ±1σ reference band visible even for very tight fits.
                _ymax_data = float(np.max(np.abs(_resid_norm))) if len(_resid_norm) > 0 else 1.0
                _ymax_resid = max(_ymax_data * 1.2, 2.5)
                ax2_res.set_ylim(-_ymax_resid, _ymax_resid)

    if fb_tau is not None and np.isfinite(fb_tau) and fb_tau > 0:
        ax2.axhline(fb_tau, color='gray', ls=':', alpha=0.7,
                     label=f'Full-band τ={fb_tau:.2f} ms')

    # ax2 is paired with ax2_res underneath (shared x-axis). The x-label
    # belongs to the bottom of the pair (ax2_res), and ax2's own x-tick
    # labels are hidden to avoid duplication between the two stacked
    # panels.
    ax2.set_xlabel('')
    ax2.set_ylabel('τ (ms)')
    ax2.set_title('Scattering timescale τ(ν)')
    ax2.legend(fontsize=7)
    ax2.set_yscale('log')
    # x-axis kept linear so frequency ticks read as plain MHz values
    # (matching the left/right columns) rather than 6×10³ scientific
    # notation. The power-law τ(ν) fit is still computed in log-log
    # space; only the x rendering is linear.
    ax2.set_xscale('linear')
    _mhz_fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
    _mhz_fmt.set_scientific(False)
    ax2.xaxis.set_major_formatter(_mhz_fmt)
    # Explicit y-limits anchored on the well-behaved data scale
    # (computed above as _y_top_limit / _y_bot_limit). Without this,
    # matplotlib autoscales to include the offscreen error bars of
    # any failed sub-band fits even when they're rendered translucent,
    # which crushes the visible fit region into a sliver. Skipped
    # gracefully if the reference scale wasn't usable (e.g. no kept
    # points and no fb_tau).
    if np.isfinite(_y_top_limit) and np.isfinite(_y_bot_limit) \
            and _y_top_limit > _y_bot_limit > 0:
        ax2.set_ylim(_y_bot_limit, _y_top_limit)
    # Log-axis tick labels: x has been bumped up slightly from 6 to 7
    # (still small enough to avoid overlap on narrow-band sub-band
    # ranges), y is shrunk to 7 from default so it doesn't crowd the
    # panel-1 legend on the left.
    ax2.tick_params(axis='x', which='both', labelsize=7, labelbottom=False)
    ax2.tick_params(axis='y', which='both', labelsize=7)

    # Residuals strip below ax2. Shares x with ax2 (set up when the
    # subplot was created) so it stays aligned during zoom/pan. y-label
    # is the normalised-residual shorthand; x-label moves here from
    # ax2 since this is the bottom of the pair.
    ax2_res.set_xscale('linear')
    _mhz_fmt_res = matplotlib.ticker.ScalarFormatter(useOffset=False)
    _mhz_fmt_res.set_scientific(False)
    ax2_res.xaxis.set_major_formatter(_mhz_fmt_res)
    ax2_res.set_xlabel('Frequency (MHz)')
    ax2_res.set_ylabel('(data − fit) / σ', fontsize=8)
    ax2_res.tick_params(axis='x', which='both', labelsize=7)
    ax2_res.tick_params(axis='y', which='both', labelsize=7)

    # σ-constrained diagnostic summary. When the diagnostic ran, show
    # Δτ/σ_τ median + max inside the residual strip; when ALL sub-bands
    # failed the constrained fit, show a single red warning instead.
    # When the user did not request the diagnostic, nothing is drawn
    # here (preserving the legacy-identical figure layout).
    #
    # Suppressed when the 2×3 layout is active (`ax2_b is not None`):
    # the bottom row's dedicated Δτ panel (panel 6) is the primary
    # home for this information; duplicating it here would clutter
    # the top-row residual strip.
    if csig_requested and ax2_b is None:
        if csig_all_failed:
            ax2_res.text(
                0.02, 0.95, "all sub-bands failed σ-constrained fit",
                transform=ax2_res.transAxes,
                ha='left', va='top', fontsize=7, color='red',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='red', alpha=0.9),
            )
        elif np.isfinite(csig_dtau_median) and np.isfinite(csig_dtau_max):
            ax2_res.text(
                0.02, 0.95,
                (f"Δτ/σ_τ: median={csig_dtau_median:.2f}  "
                 f"max={csig_dtau_max:.2f}   "
                 f"(σ$_t$ fixed = {csig_sigma_value:.3f} ms)"),
                transform=ax2_res.transAxes,
                ha='left', va='top', fontsize=7, color='#1f3a93',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='#1f3a93', alpha=0.9),
            )

    # ── Panel 3: σ across sub-bands ──
    # Per user spec (item 5): this panel is UNCHANGED by
    # `ignore_subindex`. All sub-bands are plotted as a single orange
    # line+markers regardless of the τ-fit ignore mask. The ε-fit also
    # uses all sub-bands (see fit_subband_alpha).
    # σ per sub-band with error bars from each EMG fit's covariance
    # (sub_sigma_errs = sqrt(pcov[2,2])). Bars are only drawn where the
    # uncertainty is finite and positive; sub-bands with a degenerate or
    # failed covariance fall back to no bar (the marker still plots).
    _sig_yerr = np.asarray(sub_sigma_errs, dtype=float)
    _sig_yerr_plot = np.where(np.isfinite(_sig_yerr) & (_sig_yerr > 0),
                              _sig_yerr, 0.0)
    ax3.errorbar(sub_freqs, sub_sigmas, yerr=_sig_yerr_plot,
                 fmt='o-', color='C1', ms=8, capsize=3, elinewidth=1.2,
                 label='Fitted σ per sub-band')
    if np.isfinite(best_sigma):
        ax3.axhline(best_sigma, color='C2', ls='--',
                     label=f'Best σ (hi-ν) = {best_sigma:.3f} ms')
    if fb_sigma is not None and np.isfinite(fb_sigma):
        ax3.axhline(fb_sigma, color='gray', ls=':',
                     label=f'Full-band σ = {fb_sigma:.3f} ms')

    # Smoothed σ(ν) power-law fit (ε): σ(ν) = A · (ν/ν_ref)^β, β ≤ 0.
    # Plotted as a light-but-visible grey curve so it reads as a guide
    # rather than competing with the orange data line.
    _eps_amp      = sb_result.get('eps_sigma_amp',      np.nan)
    _eps_amp_err  = sb_result.get('eps_sigma_amp_err',  np.nan)
    _eps_beta     = sb_result.get('eps_sigma_beta',     np.nan)
    _eps_beta_err = sb_result.get('eps_sigma_beta_err', np.nan)
    if (sb_result.get('eps_sigma_success', False)
            and np.isfinite(_eps_amp) and np.isfinite(_eps_beta)
            and len(sub_freqs) >= 2):
        _nu_eps_curve = np.linspace(float(np.min(sub_freqs)),
                                    float(np.max(sub_freqs)), 200)
        _sig_eps_curve = _eps_amp * (_nu_eps_curve / float(freq_center)) ** _eps_beta
        _eps_label = (r'$\sigma(\nu)=A\,(\nu/\nu_{\mathrm{ref}})^{\beta}$  '
                      f'[A={_eps_amp:.3f} ms, β={_eps_beta:.2f}±{_eps_beta_err:.2f}]')
        ax3.plot(_nu_eps_curve, _sig_eps_curve,
                 color='#9A9A9A', ls='-', lw=1.6, alpha=0.85,
                 label=_eps_label, zorder=1)

    ax3.set_xlabel('Frequency (MHz)')
    ax3.set_ylabel('σ (ms)')
    ax3.set_title('Gaussian width σ across sub-bands')
    ax3.legend(fontsize=7)

    # ════════════════════════════════════════════════════════════════
    # Bottom row (drawn only when constrain_sigma was requested).
    # Mirrors panels 1 and 2 but uses the σ-constrained per-sub-band
    # fits; panel 6 plots Δτ/σ_τ per sub-band instead of the
    # (degenerate) σ-fixed line. When all σ-constrained fits failed,
    # each bottom panel shows a single red annotation in place of
    # plotted content.
    # ════════════════════════════════════════════════════════════════
    if csig_requested and ax1_b is not None:
        csig_sub_freqs    = np.asarray(sb_result.get('csig_sub_freqs', []))
        csig_sub_taus     = np.asarray(sb_result.get('csig_sub_taus',  []))
        csig_sub_tau_errs = np.asarray(sb_result.get('csig_sub_tau_errs', []))
        csig_sub_sigmas   = np.asarray(sb_result.get('csig_sub_sigmas', []))
        csig_sub_popts    = sb_result.get('csig_sub_popts', []) or []
        csig_ignore_mask  = np.asarray(sb_result.get('csig_ignore_mask',
                                                      np.zeros(len(csig_sub_freqs), dtype=bool)))
        csig_dtau_per_sb  = np.asarray(sb_result.get('csig_dtau_over_sigma_tau_per_sb', []),
                                        dtype=float)
        csig_kept_mask    = (~csig_ignore_mask if len(csig_ignore_mask) == len(csig_sub_freqs)
                              else np.ones(len(csig_sub_freqs), dtype=bool))

        # ── Bottom Panel 1: sub-band profiles + σ-constrained EMG fits ──
        if csig_all_failed or len(csig_sub_popts) == 0:
            ax1_b.text(
                0.5, 0.5,
                "All sub-bands failed σ-constrained fit",
                transform=ax1_b.transAxes,
                ha='center', va='center', fontsize=12, color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='red', alpha=0.9),
            )
            ax1_b.set_xticks([]); ax1_b.set_yticks([])
        else:
            # Build a frequency→colour lookup that matches the TOP-row
            # panel-1 colour for each sub-band, so the same sub-band
            # is drawn in the same colour in both panels (visual
            # continuity for the comparison).
            _colors_top = plt.cm.coolwarm(np.linspace(0, 1, len(sub_popts)))
            def _color_for_freq(_nu):
                # Match by nearest sub_freqs entry (within 0.1 MHz).
                _m = np.where(np.abs(np.asarray(sub_freqs) - _nu) < 0.1)[0]
                if len(_m) > 0:
                    return _colors_top[int(_m[0])]
                # Fallback: position-based in the csig array.
                _idx = int(np.argmin(np.abs(np.asarray(sub_freqs) - _nu)))
                return _colors_top[_idx % len(_colors_top)]
            for j, popt_j in enumerate(csig_sub_popts):
                _nu_j = float(csig_sub_freqs[j])
                # Find the matching sub-band's exact freq bounds (parallel
                # to the top-row lookup) to reconstruct the same sub_ts.
                _match = np.where(np.abs(np.asarray(sub_freqs) - _nu_j) < 0.1)[0]
                if len(_match) > 0 and _have_exact_bounds:
                    freq_lo = float(_sb_freq_los[int(_match[0])])
                    freq_hi = float(_sb_freq_his[int(_match[0])])
                elif len(_match) > 0:
                    freq_lo, freq_hi = [float(x) for x in sub_labels[int(_match[0])].split('-')]
                else:
                    continue
                mask = (freqs >= freq_lo) & (freqs <= freq_hi)
                sub_ts = wfall[mask, :].sum(axis=0)
                _ts_min = sub_ts.min()
                _ts_range = (sub_ts.max() - _ts_min) + 1e-30
                sub_norm = (sub_ts - _ts_min) / _ts_range
                _c = _color_for_freq(_nu_j)
                # Greyed out for ignored sub-bands.
                _is_ignored = bool(csig_ignore_mask[j]) if j < len(csig_ignore_mask) else False
                _data_alpha = 0.25 if _is_ignored else 0.5
                _fit_alpha  = 0.4  if _is_ignored else 1.0
                ax1_b.plot(t_axis, sub_norm + j * 0.15, color=_c, alpha=_data_alpha, lw=0.8)

                # Smooth EMG curve
                fit_y_fine = emg_model(t_axis_fine, *popt_j)
                fit_norm_fine = (fit_y_fine - _ts_min) / _ts_range
                tau_j = 1.0 / popt_j[3] if popt_j[3] > 0 else np.nan
                _lbl = f'{freq_lo:.0f}-{freq_hi:.0f} MHz: τ={tau_j:.2f} ms'
                if _is_ignored:
                    _lbl += '  [ignored]'
                ax1_b.plot(t_axis_fine, fit_norm_fine + j * 0.15, color=_c,
                           lw=2, alpha=_fit_alpha, label=_lbl)
            ax1_b.set_xlabel('Time (ms)')
            ax1_b.set_ylabel('Normalized intensity (offset)')
            ax1_b.set_title(f'Sub-band profiles + EMG fits  (σ fixed = {csig_sigma_value:.3f} ms)',
                             fontsize=11, color='#1f3a93')
            ax1_b.legend(fontsize=7, loc='upper left')

        # ── Bottom Panel 2: σ-constrained τ(ν) with residuals ──
        if csig_all_failed or len(csig_sub_taus) == 0:
            ax2_b.text(
                0.5, 0.5,
                "All sub-bands failed σ-constrained fit",
                transform=ax2_b.transAxes,
                ha='center', va='center', fontsize=12, color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='red', alpha=0.9),
            )
            ax2_b.set_xticks([]); ax2_b.set_yticks([])
            ax2_b_res.set_xticks([]); ax2_b_res.set_yticks([])
        else:
            # Reference α=−4 anchored at the same anchor the top row uses
            _tau_anchor_b = np.median(csig_sub_taus[csig_kept_mask]) if csig_kept_mask.sum() > 0 \
                            else np.median(csig_sub_taus)
            _tau_m4_fine_b = _tau_anchor_b * (nu_fine / freq_center) ** (-4.0)
            ax2_b.plot(nu_fine, _tau_m4_fine_b, 'k-', lw=2.5,
                        label='Reference: α=−4.0', zorder=1)

            # Constrained-fit power-law line (if α fit converged)
            if np.isfinite(csig_alpha) and np.isfinite(csig_tau_ref):
                _csig_tau_fine = csig_tau_ref * (nu_fine / freq_center) ** csig_alpha
                _lbl_c = f'Fit (σ fixed): α={csig_alpha:+.2f}'
                if np.isfinite(csig_alpha_err):
                    _lbl_c += f'±{csig_alpha_err:.2f}'
                ax2_b.plot(nu_fine, _csig_tau_fine, '--', color='#1f3a93',
                            lw=2.5, label=_lbl_c, zorder=3)
                ax2_b.plot(freq_center, csig_tau_ref,
                            marker='*', ms=12, color='#1f3a93', zorder=4)

            # Kept (blue) and ignored (grey) constrained-τ points
            if np.any(csig_kept_mask):
                ax2_b.errorbar(csig_sub_freqs[csig_kept_mask], csig_sub_taus[csig_kept_mask],
                                yerr=csig_sub_tau_errs[csig_kept_mask],
                                fmt='s', color='#1f3a93', ms=8, capsize=3,
                                label=f'σ-constrained sub-band fits (N={int(csig_kept_mask.sum())})',
                                zorder=4)
            if np.any(csig_ignore_mask):
                ax2_b.errorbar(csig_sub_freqs[csig_ignore_mask], csig_sub_taus[csig_ignore_mask],
                                yerr=csig_sub_tau_errs[csig_ignore_mask],
                                fmt='s', color='lightgray', ms=8, capsize=3,
                                label=f'Ignored (N={int(csig_ignore_mask.sum())})',
                                zorder=3, mec='gray')
            if fb_tau is not None and np.isfinite(fb_tau) and fb_tau > 0:
                ax2_b.axhline(fb_tau, color='gray', ls=':',
                               label=f'Full-band τ={fb_tau:.2f} ms')

            ax2_b.set_xscale('linear')
            _mhz_fmt_b = matplotlib.ticker.ScalarFormatter(useOffset=False)
            _mhz_fmt_b.set_scientific(False)
            ax2_b.xaxis.set_major_formatter(_mhz_fmt_b)
            ax2_b.set_yscale('log')
            ax2_b.set_ylabel(r'$\tau$ (ms)')
            ax2_b.set_title(f'Scattering timescale τ(ν) — σ fixed = {csig_sigma_value:.3f} ms',
                             fontsize=11, color='#1f3a93')
            ax2_b.legend(fontsize=8, loc='lower left')
            ax2_b.grid(alpha=0.25, which='both')
            # Match the top-row x and y limits so the τ(ν) panels line
            # up visually for direct comparison. ax2's xlim is set by
            # nu_fine range above (extends slightly past the data on
            # both sides); we adopt the same range here.
            ax2_b.set_xlim(ax2.get_xlim())
            ax2_b.set_ylim(ax2.get_ylim())

            # Residual strip: (data − fit) / σ_τ on the σ-constrained α fit
            if np.isfinite(csig_alpha) and np.isfinite(csig_tau_ref) and np.any(csig_kept_mask):
                _nu_k   = csig_sub_freqs[csig_kept_mask]
                _tau_k  = csig_sub_taus[csig_kept_mask]
                _err_k  = csig_sub_tau_errs[csig_kept_mask]
                _pred_k = csig_tau_ref * (_nu_k / freq_center) ** csig_alpha
                _valid  = np.isfinite(_err_k) & (_err_k > 0)
                if np.any(_valid):
                    _chi2_c = np.sum(((_tau_k[_valid] - _pred_k[_valid]) / _err_k[_valid]) ** 2)
                    _dof_c = int(np.sum(_valid)) - 2
                    if _dof_c > 0:
                        _chi2red_c = _chi2_c / _dof_c
                        ax2_b.text(
                            0.97, 0.97,
                            f'$\\chi^2_{{\\rm red}}$ = {_chi2red_c:.2f}  (dof={_dof_c})',
                            transform=ax2_b.transAxes,
                            ha='right', va='top', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      edgecolor='#1f3a93', alpha=0.85),
                        )
                    ax2_b_res.errorbar(_nu_k[_valid],
                                        (_tau_k[_valid] - _pred_k[_valid]) / _err_k[_valid],
                                        yerr=1.0, fmt='s', color='#1f3a93',
                                        ms=6, capsize=2)
                ax2_b_res.axhline(0, color='black', lw=0.6, alpha=0.5)
                ax2_b_res.set_xscale('linear')
                _mhz_fmt_b_res = matplotlib.ticker.ScalarFormatter(useOffset=False)
                _mhz_fmt_b_res.set_scientific(False)
                ax2_b_res.xaxis.set_major_formatter(_mhz_fmt_b_res)
                ax2_b_res.set_xlim(ax2_b.get_xlim())
                ax2_b_res.set_xlabel('Frequency (MHz)')
                ax2_b_res.set_ylabel('(data−fit)/σ', fontsize=8)
                ax2_b_res.grid(alpha=0.25, which='both')
                ax2_b_res.tick_params(axis='y', which='both', labelsize=7)

        # ── Bottom Panel 3: Δτ/σ_τ per sub-band ──
        # Replaces the redundant "σ per sub-band" panel (which would
        # be a flat line at σ_fixed). Shows how much each constrained
        # τ shifted from the free τ, in units of the mean of the two
        # τ uncertainties — a direct sub-band-by-sub-band view of the
        # σ–τ degeneracy magnitude.
        if csig_all_failed or len(csig_dtau_per_sb) == 0:
            ax3_b.text(
                0.5, 0.5,
                "All sub-bands failed σ-constrained fit",
                transform=ax3_b.transAxes,
                ha='center', va='center', fontsize=12, color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='red', alpha=0.9),
            )
            ax3_b.set_xticks([]); ax3_b.set_yticks([])
        else:
            # Plot kept points solid blue, ignored points grey.
            _finite = np.isfinite(csig_dtau_per_sb)
            if csig_kept_mask.shape == csig_dtau_per_sb.shape:
                _kept_finite    = csig_kept_mask & _finite
                _ignored_finite = csig_ignore_mask & _finite
            else:
                _kept_finite    = _finite
                _ignored_finite = np.zeros_like(_finite)
            if np.any(_kept_finite):
                ax3_b.plot(csig_sub_freqs[_kept_finite], csig_dtau_per_sb[_kept_finite],
                            'o-', color='#1f3a93', ms=10, lw=1.5,
                            label='Kept (used in α fit)')
            if np.any(_ignored_finite):
                ax3_b.plot(csig_sub_freqs[_ignored_finite], csig_dtau_per_sb[_ignored_finite],
                            'o', color='lightgray', ms=10, mec='gray',
                            label='Ignored')
            # Reference lines: 1σ, 3σ
            ax3_b.axhline(1.0, color='gray', ls=':', alpha=0.6,
                           label='1σ (free ≈ constrained)')
            ax3_b.axhline(3.0, color='red', ls=':', alpha=0.6,
                           label='3σ (significant shift)')
            # Summary band: median and max over KEPT sub-bands
            if np.isfinite(csig_dtau_median):
                ax3_b.axhline(csig_dtau_median, color='#27ae60', ls='--', alpha=0.8,
                               label=f'Median (kept) = {csig_dtau_median:.2f}')
            ax3_b.set_xlabel('Frequency (MHz)')
            ax3_b.set_ylabel(r'|$\Delta\tau$| / $\sigma_\tau$')
            ax3_b.set_title('σ-τ degeneracy magnitude per sub-band',
                             fontsize=11, color='#1f3a93')
            ax3_b.legend(fontsize=7, loc='best')
            ax3_b.grid(alpha=0.25)

    fig.tight_layout()

    # === Save / show handling (added for batch mode) ===
    # If save=True, write the diagnostic figure to disk under outdir using
    # a name that's unique per burst+component+model so multiple sub-band
    # plots from the same burst (e.g. one per component, or EMG vs
    # Gaussian re-runs) don't overwrite each other.
    if save:
        # Build a safe filename suffix
        _suffix_parts = ['subband']
        if comp_label is not None:
            _suffix_parts.append(f'comp{comp_label}')
        if model_tag is not None:
            _suffix_parts.append(str(model_tag))
        if name_suffix:
            _suffix_parts.append(str(name_suffix))
        _suffix = '_'.join(_suffix_parts)
        if outdir == '' or outdir[-1] == '/':
            _outname = f"{outdir}{burst_name}_{_suffix}{outfmt}"
        else:
            _outname = f"{outdir}/{burst_name}_{_suffix}{outfmt}"
        try:
            fig.savefig(_outname, dpi=120, bbox_inches='tight')
            print(f"Info: Saved {_outname}.")
        except Exception as _e:
            print(f"Warning: Could not save sub-band diagnostic plot to {_outname}: {_e}")
    if show:
        fig.show()
    else:
        # In batch / save-only mode close the figure to free memory and
        # avoid filling the matplotlib state machine with thousands of
        # figures across a multi-burst run.
        plt.close(fig)

    return fig


def line_model(nu, dtdnu, t_b):
	return dtdnu * nu + t_b

def gauss_model(x, a, xo, sigma):
	return a*np.exp(-(x-xo)**2/(2*(sigma**2)))
	
def emg_model(t, A, mu, sigma, lamb, C):
	"""
	Exponentially Modified Gaussian (EMG) model — numerically stable.

	Uses the identity  exp(a)*erfc(z) = exp(a-z²)*erfcx(z)  where
	a-z² = -(t-mu)²/(2σ²), to avoid the exp() overflow that plagues
	the naive formula when λ or σ are large during curve_fit exploration.

	- For z ≥ 0 (left of the EMG mean): use the erfcx form
	  (exp(a) would overflow, but Gaussian envelope × erfcx stays finite).
	- For z < 0 (far right tail): use the original exp(a)*erfc(z)
	  (erfcx would overflow, but exp(a)~0 keeps the product finite).
	"""
	t   = np.asarray(t, dtype=float)
	z   = (mu + lamb * sigma**2 - t) / (np.sqrt(2) * sigma)
	pdf = np.empty_like(t)

	# --- z ≥ 0 branch: erfcx-safe form ---
	pos = z >= 0
	if pos.any():
		gauss_env = np.exp(-0.5 * ((t[pos] - mu) / sigma)**2)
		pdf[pos]  = (lamb / 2.0) * gauss_env * erfcx(z[pos])

	# --- z < 0 branch: original form (exp(a) is small here) ---
	neg = ~pos
	if neg.any():
		a = (lamb / 2.0) * (2.0 * mu + lamb * sigma**2 - 2.0 * t[neg])
		pdf[neg] = (lamb / 2.0) * np.exp(a) * erfc(z[neg])

	return C + A * pdf

def find_emg_peak(mu, sigma, lamb, t_min=None, t_max=None, n_points=1000):
	"""
	Numerically find the peak (maximum) of an EMG function.
	
	The EMG peak location is shifted from mu due to the exponential convolution.
	This function evaluates the EMG over a fine grid and returns the time of maximum.
	
	Parameters:
	-----------
	mu : float
		Gaussian center parameter of the EMG
	sigma : float
		Gaussian width parameter of the EMG
	lamb : float
		Exponential decay rate (1/tau) of the EMG
	t_min : float, optional
		Minimum time for search range. Defaults to mu - 5*sigma
	t_max : float, optional
		Maximum time for search range. Defaults to mu + 5*sigma + 3/lamb
	n_points : int
		Number of points in the fine grid (default 1000)
		
	Returns:
	--------
	t_peak : float
		Time at which the EMG reaches its maximum value
	"""
	if not (np.isfinite(mu) and np.isfinite(sigma) and np.isfinite(lamb)):
		return np.nan
	if sigma <= 0 or lamb <= 0:
		return np.nan
	
	# Set default search range if not provided
	tau = 1.0 / lamb
	if t_min is None:
		t_min = mu - 5 * sigma
	if t_max is None:
		t_max = mu + 5 * sigma + 3 * tau  # Extend to capture scattering tail
	
	# Create fine grid
	t_grid = np.linspace(t_min, t_max, n_points)
	
	# Evaluate EMG (using A=1, C=0 since we only need the location)
	emg_values = emg_model(t_grid, 1.0, mu, sigma, lamb, 0.0)
	
	# Find the peak
	peak_idx = np.argmax(emg_values)
	t_peak = t_grid[peak_idx]
	
	return float(t_peak)

def emgmix_model(t, *p):
	"""
	EMG mixture parameterized by the Gaussian peak mu (the "true peak", x_o).

	Parameter vector layout (N components):
	  [A1..AN, mu1..muN, sigma1..sigmaN, lambda1..lambdaN, C]

	Returns sum_i EMG_i(t; A_i, mu_i, sigma_i, lambda_i) + C.
	The first 3N entries are [A, mu, sigma], so downstream slicing remains identical in shape.
	"""
	n = (len(p) - 1) // 4
	C = p[4*n]
	y = C
	for i in range(n):
		A     = p[i]
		mu    = p[n + i]
		sigma = p[2*n + i]
		lamb  = p[3*n + i]
		y += emg_model(t, A, mu, sigma, lamb, 0.0)
	return y


# ============================================================================
# MCMC FITTING FUNCTIONS (adapted from Band_EMG_MCMC_2.py, an earlier script by Victor Pop)
# ============================================================================

def _emg_no_baseline(t, A, mu, sigma, lamb):
	"""EMG model without baseline for MCMC fitting (after data is shifted).
	Uses the same numerically stable erfcx form as emg_model."""
	return emg_model(t, A, mu, sigma, lamb, 0.0)


def _log_prior_mcmc(theta, p0, prior_range=0.2):
	"""
	Log prior: uniform within ±prior_range of initial guess p0.
	theta = [A, mu, sigma, lamb]
	"""
	A, mu, sigma, lamb = theta
	A0, mu0, sigma0, lamb0 = p0
	
	# Bounds
	if not (A0*(1-prior_range) < A < A0*(1+prior_range)):
		return -np.inf
	if not (mu0*(1-prior_range) < mu < mu0*(1+prior_range)):
		return -np.inf
	if not (sigma0*(1-prior_range) < sigma < sigma0*(1+prior_range)):
		return -np.inf
	if not (lamb0*(1-prior_range) < lamb < lamb0*(1+prior_range)):
		return -np.inf
	if sigma <= 0 or lamb <= 0:
		return -np.inf
	
	return 0.0


def _log_likelihood_mcmc(theta, t, y, sigma_noise):
	"""
	Log likelihood for MCMC.
	theta = [A, mu, sigma, lamb]
	"""
	A, mu, sigma, lamb = theta
	if sigma <= 0 or lamb <= 0:
		return -np.inf
	
	model = _emg_no_baseline(t, A, mu, sigma, lamb)
	residual = y - model
	return -0.5 * np.sum((residual / sigma_noise)**2)


def _log_probability_mcmc(theta, t, y, sigma_noise, p0, prior_range):
	"""Combined log probability for MCMC."""
	lp = _log_prior_mcmc(theta, p0, prior_range)
	if not np.isfinite(lp):
		return -np.inf
	return lp + _log_likelihood_mcmc(theta, t, y, sigma_noise)


def fit_emg_mcmc(t_axis, data, burst_name="", prior_range=0.2, 
				 nwalkers=32, nsteps=10000, burnin=1000, off_pulse_bins=50):
	"""
	Fit EMG model to data using MCMC.
	
	Parameters:
	-----------
	t_axis : array
		Time axis (ms)
	data : array  
		Intensity data (should be full-band time series)
	burst_name : str
		Name of burst for progress display
	prior_range : float
		Prior range factor for MCMC bounds
	nwalkers : int
		Number of MCMC walkers
	nsteps : int
		Number of MCMC steps
	burnin : int
		Number of burnin steps to discard
	off_pulse_bins : int
		Number of bins to use for off-pulse baseline estimation
		
	Returns:
	--------
	mcmc_results : dict
		Dictionary containing:
		- 'status': 'success' or 'fallback'
		- 'median': [A, mu, sigma, lamb] median values
		- 'err_low': lower uncertainties
		- 'err_high': upper uncertainties
		- 'tau': scattering time (1/lamb)
		- 'tau_err_low': tau lower uncertainty
		- 'tau_err_high': tau upper uncertainty
		- 'baseline': baseline value (C)
		- 'samples': MCMC samples (for diagnostics)
	"""
	
	# Adaptive off-pulse bins based on data size
	n_time = len(data)
	n_freq_implied = 200  # typical for your data
	adaptive_bins = min(
		max(20, n_time // 10),
		max(20, n_freq_implied // 4)
	)
	off_pulse_bins_use = min(off_pulse_bins, adaptive_bins)
	
	try:
		# Step 1: Estimate and subtract baseline
		baseline = np.median(data[:off_pulse_bins_use])
		data_shifted = data - baseline
		
		# Step 2: Initial EMG fit using curve_fit for starting point
		A0 = float(np.max(data_shifted))
		mu0 = float(t_axis[np.argmax(data_shifted)])
		
		# Estimate sigma from data width
		above_half = data_shifted > (A0 / 2)
		if np.any(above_half):
			sigma0 = float(np.std(t_axis[above_half]))
		else:
			sigma0 = float((t_axis[-1] - t_axis[0]) / 10.0)
		sigma0 = max(sigma0, 1e-6)
		
		lamb0 = 1.0 / sigma0
		
		# Noise estimate
		sigma_noise = np.std(data_shifted[:off_pulse_bins_use])
		if sigma_noise <= 0:
			sigma_noise = np.std(data_shifted) * 0.1
		
		# Initial guess
		p0 = [A0, mu0, sigma0, lamb0]
		
		# Try curve_fit first for better initial guess
		try:
			popt_init, _ = scipy.optimize.curve_fit(
				_emg_no_baseline, t_axis, data_shifted,
				p0=p0,
				bounds=(
					[0, -np.inf, 1e-9, 1e-9],
					[np.inf, np.inf, np.inf, np.inf]
				),
				maxfev=50000
			)
			p0 = popt_init  # Use fitted values as starting point
		except:
			pass  # Keep original p0 if fit fails
		
		# Step 3: MCMC sampling
		print(f"  Running MCMC for intensity vs time profile of burst {burst_name}...")
		
		ndim = 4  # A, mu, sigma, lamb
		
		# Initialize walkers around p0 with small scatter
		pos = p0 + 1e-4 * np.abs(p0) * np.random.randn(nwalkers, ndim)
		
		# Ensure positivity for sigma and lamb
		pos[:, 2] = np.abs(pos[:, 2]) + 1e-6  # sigma
		pos[:, 3] = np.abs(pos[:, 3]) + 1e-6  # lamb
		
		# Create sampler
		sampler = emcee.EnsembleSampler(
			nwalkers, ndim, _log_probability_mcmc,
			args=(t_axis, data_shifted, sigma_noise, p0, prior_range)
		)
		
		# Run MCMC with progress bar
		print(f"  >>> DEBUG: Starting MCMC chain for burst: {burst_name}")
		sampler.run_mcmc(pos, nsteps, progress=True)
		
		# Step 4: Extract results
		print(f"  >>> DEBUG: Extracting chain for burst: {burst_name}")
		samples = sampler.get_chain(discard=burnin, flat=True)
		print(f"  >>> DEBUG: Chain shape for burst {burst_name}: {samples.shape}")
		
		# Get median and percentiles
		median_params = np.percentile(samples, 50, axis=0)
		lower_params = np.percentile(samples, 16, axis=0)
		upper_params = np.percentile(samples, 84, axis=0)
		
		err_low = median_params - lower_params
		err_high = upper_params - median_params
		
		# Calculate tau and its uncertainty
		lamb_samples = samples[:, 3]
		tau_samples = 1.0 / lamb_samples
		tau_median = np.percentile(tau_samples, 50)
		tau_16 = np.percentile(tau_samples, 16)
		tau_84 = np.percentile(tau_samples, 84)
		
		mcmc_results = {
			'status': 'success',
			'median': median_params,  # [A, mu, sigma, lamb]
			'err_low': err_low,
			'err_high': err_high,
			'tau': tau_median,
			'tau_err_low': tau_median - tau_16,
			'tau_err_high': tau_84 - tau_median,
			'baseline': baseline,
			'samples': samples
		}
		
		return mcmc_results
		
	except Exception as e:
		print(f"  !!! MCMC FAILED for burst: {burst_name}")
		print(f"  !!! Error: {e}")
		print(f"  Falling back to regular EMG fit...")
		
		# Return failure status
		return {
			'status': 'fallback',
			'median': [np.nan, np.nan, np.nan, np.nan],
			'err_low': [np.nan, np.nan, np.nan, np.nan],
			'err_high': [np.nan, np.nan, np.nan, np.nan],
			'tau': np.nan,
			'tau_err_low': np.nan,
			'tau_err_high': np.nan,
			'baseline': np.nan,
			'samples': None
		}

# ============================================================================
# END MCMC FUNCTIONS
# ============================================================================


def smallestdivisor(n):
	for i in range(2, n):
		if n % i == 0:
			return i

def listnpzs(path):
	""" List all npz files in path """
	files = glob.glob(path+'*.npz')
	[print(f) for f in sorted(files)]
	exit()

# N component model
def gaussmix_model(x, *p):
	n = len(p)//3
	model = 0
	for i in range(0, n): # stops at n-1
		model += gauss_model(x, p[0*n+i], p[1*n+i], p[2*n+i])
	return model
class StackFilterAudit:
	"""Records, per stacked-channel index, the pass/fail verdict and the
	tested value at every stacked-TOA filter stage.

	The stacked-TOA filters successively shrink `stacked_df`, so a stack
	rejected early never reaches later filters. This recorder is fed by
	each filter as it runs (so it reflects exactly what the pipeline
	decided, not a popup-time re-derivation) and is later stashed in the
	click-popup state. `render_lines(stack_idx)` produces the text block
	shown in the popup's right-hand audit panel.

	Per stack we store an ordered list of (filter_label, status, detail)
	where status is 'PASS', 'FAIL', or 'SKIP' (not evaluated — the stack
	was already rejected upstream).
	"""

	def __init__(self):
		# stack_idx -> list of (label, status, detail_str)
		self._rows = {}
		# ordered list of filter labels, in the order they ran
		self._filter_order = []
		# final verdict per stack: (retained_bool, rejected_at_label_or_None)
		self._verdict = {}

	def record(self, filter_label, evaluated_indices, pass_mask, detail_fn):
		"""Log one filter stage.

		filter_label : str    — e.g. 'Filter 1 (amp>0)'
		evaluated_indices     — iterable of stack indices this filter tested
		pass_mask             — dict {idx: bool} pass/fail for each tested idx
		detail_fn             — callable(idx) -> short str (value vs threshold)
		"""
		if filter_label not in self._filter_order:
			self._filter_order.append(filter_label)
		for idx in evaluated_indices:
			i = int(idx)
			passed = bool(pass_mask.get(idx, False))
			try:
				detail = detail_fn(idx)
			except Exception:
				detail = ""
			self._rows.setdefault(i, []).append(
				(filter_label, 'PASS' if passed else 'FAIL', detail))

	def finalize(self, retained_indices, all_indices):
		"""Mark which stacks survived all filters; the rest get a SKIP
		entry for every filter after the one that rejected them."""
		retained = set(int(i) for i in retained_indices)
		for i in (int(x) for x in all_indices):
			rows = self._rows.get(i, [])
			rejected_at = None
			for (label, status, _detail) in rows:
				if status == 'FAIL':
					rejected_at = label
					break
			# Append SKIP rows for filters this stack never reached.
			n_reached = len(rows)
			for label in self._filter_order[n_reached:]:
				rows.append((label, 'SKIP', 'not evaluated — rejected upstream'))
			self._rows[i] = rows
			self._verdict[i] = (i in retained, rejected_at)

	def render_lines(self, stack_idx):
		"""Return a list of text lines for the popup audit panel."""
		i = int(stack_idx)
		rows = self._rows.get(i)
		if not rows:
			return [f"Stack {i}: no filter audit recorded",
			        "(stacking may not have run for this component)"]
		retained, rejected_at = self._verdict.get(i, (False, None))
		lines = [f"Stack {i} — filter audit", ""]
		for (label, status, detail) in rows:
			mark = {'PASS': '[PASS]', 'FAIL': '[FAIL]', 'SKIP': '[ -- ]'}[status]
			lines.append(f"{mark} {label}")
			if detail:
				lines.append(f"        {detail}")
		lines.append("")
		if retained:
			lines.append("VERDICT: RETAINED for slope fit")
		else:
			where = rejected_at if rejected_at else "an unrecorded stage"
			lines.append(f"VERDICT: REJECTED at {where}")
		return lines


def fitgauss(data, duration, redchisq=1, mu0_seed=None, sigma0_seed=None,
             sigma_min=None, sigma_max=None, mu_min=None, mu_max=None):
	"""
	Fits a gaussian (no baseline term) to a 1-D row.

	Mirrors :func:`fitemg`'s seed-and-bounds interface — the EMG-vs-Gaussian
	stacking paths are kept at numerical parity so the same fit_per_region /
	per-stack call patterns work for both. Note that `gauss_model` does not
	carry a baseline C parameter; for stacking the caller is expected to
	have pre-subtracted the global baseline (see the unified-baseline block
	near line ~4803).

	Parameters
	----------
	data : array
		1-D intensity data.
	duration : float
		Total duration in ms.
	redchisq : float
		Reduced chi-squared used to scale the per-bin uncertainty estimate.
	mu0_seed : float, optional
		Initial guess for μ (in local coordinates). When omitted falls back
		to the intensity-weighted centroid of `data`.
	sigma0_seed : float, optional
		Initial guess for σ. When omitted falls back to the second moment
		of `data` around the centroid.
	sigma_min, sigma_max : float, optional
		Lower and upper bounds on σ in ms. Pass `sigma_min=dt_ms` to forbid
		sub-resolution fits (matches the σ bound used in fitemg stacking).
		Defaults preserve the legacy unbounded behaviour.
	mu_min, mu_max : float, optional
		Lower and upper bounds on μ in ms. Used by the stacked-fit callers
		to confine low-SNR per-stack fits to a window around the seed
		(e.g. mu_seed ± 3σ_seed) so they cannot drift onto a noise peak
		elsewhere in the window. Defaults (None) preserve the legacy
		unbounded behaviour.

	Returns
	-------
	popt : list of three floats
		[A, μ, σ].
	pcov : 3×3 array
		Covariance matrix from curve_fit (or filled with NaNs on failure).
	redchisq : float
		Reduced chi-squared of the fit (or 0 / NaN on failure, preserving
		legacy semantics).
	"""
	if len(data) == 0:
		return [0, 0, 0], [0, 0, 0], 0
	x  = np.linspace(0, duration, num=len(data))

	# === LOCAL ESTIMATION (fallback for seeds) ===
	xo_local = sum(x * data) / max(np.sum(data), 1e-12)
	var = abs(np.sum(data * (x - xo_local)**2) / max(np.sum(data), 1e-12))
	sig0_local = np.sqrt(var) if var > 0 else duration / 10.0

	# === USE SEEDS IF VALID, ELSE FALL BACK ===
	if mu0_seed is not None and np.isfinite(mu0_seed) and 0 < mu0_seed < duration:
		mu0 = float(mu0_seed)
	else:
		mu0 = float(xo_local)
	if sigma0_seed is not None and np.isfinite(sigma0_seed) and sigma0_seed > 0:
		sig0 = float(sigma0_seed)
	else:
		sig0 = max(float(sig0_local), 1e-6)

	# === RESOLVE σ BOUNDS (mirror fitemg's defaults) ===
	_sig_lo = max(float(sigma_min), 1e-9) if (sigma_min is not None and np.isfinite(sigma_min) and sigma_min > 0) else 1e-9
	_sig_hi = float(sigma_max) if (sigma_max is not None and np.isfinite(sigma_max) and sigma_max > 0) else np.inf
	# Clamp the seed into the bounded box (curve_fit raises if p0 outside).
	sig0 = float(np.clip(sig0, _sig_lo, _sig_hi))

	# === RESOLVE μ BOUNDS ===
	# Defaults are ±inf (legacy unbounded). The stacked-fit callers pass
	# a finite window (mu_seed ± 3σ_seed) so a low-SNR per-stack fit
	# cannot wander onto a noise peak far from the true burst location.
	_mu_lo = float(mu_min) if (mu_min is not None and np.isfinite(mu_min)) else -np.inf
	_mu_hi = float(mu_max) if (mu_max is not None and np.isfinite(mu_max)) else np.inf
	if _mu_lo >= _mu_hi:   # degenerate window -> fall back to unbounded
		_mu_lo, _mu_hi = -np.inf, np.inf
	mu0 = float(np.clip(mu0, _mu_lo, _mu_hi))

	try:
		data_sigma = [np.sqrt(redchisq) * np.std(data[0:len(data)//NOISE_WIDTH_FACTOR])] * len(data)
		popt, pcov = scipy.optimize.curve_fit(
			gauss_model,
			x,
			data,
			p0=[float(np.max(data)), mu0, sig0],
			sigma=data_sigma,
			bounds=(
				[-np.inf, _mu_lo, _sig_lo],
				[ np.inf, _mu_hi, _sig_hi]
			),
			maxfev=200000,
		)
		residuals = (data - gauss_model(x, *popt)) / data_sigma
		chisq = np.sum(residuals ** 2)
		redchisq = chisq / max(len(data) - len(popt), 1)
	except Exception:
		# Matches fitemg's broad-Exception fallback. curve_fit can raise
		# RuntimeError (optimizer didn't converge), ValueError (e.g. p0
		# outside bounds, or full_output+bounds combo on older scipy), or
		# LinAlgError (singular Jacobian). Any of these means the fit
		# failed and downstream code should see the same shape as a
		# successful return.
		popt = [0, 0, 0]
		pcov = np.full((3, 3), np.nan)
		redchisq = np.nan
	return popt, pcov, redchisq

def fitemgmix(data, duration, xos, sigmas=None, lambdas=None, fix_xos=False, tol=0.01, dt_ms=None):

	n = len(xos)
	if n == 0:
		return [0]*(3), np.zeros((3,3)), np.nan

	# Normalize like fitgaussmix does (max=1) so downstream scaling lines still work.
	# OLD: scale = np.max(data) if np.max(data) != 0 else 1.0
	scale = np.nanmax(data) if np.isfinite(np.nanmax(data)) and np.nanmax(data) != 0 else 1.0
	y = (data / scale) if scale != 1.0 else data

	t = np.linspace(0, duration, num=len(y))

	# Guesses
	# NaN-robust reductions: a NaN anywhere in `y` (e.g. an all-NaN time
	# column from nanmean over a fully-masked sub-band) would otherwise
	# poison var_guess/A0/C0, making every seed NaN and the fit infeasible.
	# OLD: if sigmas is None:
	# The caller passes sigmas=[] (an EMPTY list, not None) when no per-component
	# seed is specified — see the "[DEBUG tmix-input] ... sigmas=[]" print. `[] is
	# None` is False, so the old guard skipped seed generation and left sigmas
	# empty. p0 was then [*A0, *xos, *[], *[], C0] -> length 2n+1 while `bounds`
	# is always built for n components (length 4n+1), so curve_fit raised on the
	# length mismatch and the bare except returned silent all-zeros (the original
	# sigma=nan / tau=unresolved symptom). Regenerate whenever sigmas is missing
	# OR has the wrong length so p0 always matches `bounds`.
	if sigmas is None or len(sigmas) != n:
		# OLD: var_guess = abs(np.sum(y*(t - np.mean(xos))**2) / max(np.sum(y), 1e-12))
		var_guess = abs(np.nansum(y*(t - np.mean(xos))**2) / max(np.nansum(y), 1e-12))
		sig0 = np.sqrt(var_guess) / 4 if (np.isfinite(var_guess) and var_guess > 0) else duration/10.0
		sigmas = [sig0]*n
	# OLD: if lambdas is None:
	if lambdas is None or len(lambdas) != n:
		lambdas = [1.0/max(s, 1e-3) for s in sigmas]

	# OLD: A0 = [np.max(y)]*n
	_A0 = np.nanmax(y)
	if not np.isfinite(_A0):
		_A0 = 1.0
	A0 = [_A0]*n

	off_n = max(len(y)//NOISE_WIDTH_FACTOR, 1)
	# OLD: C0 = float(np.median(y[:off_n]))
	C0 = float(np.nanmedian(y[:off_n]))
	if not np.isfinite(C0):
		C0 = 0.0

	# Parameter layout: [A1..AN, mu1..muN, sigma1..sigmaN, lambda1..lambdaN, C]
	p0 = [*A0, *xos, *sigmas, *lambdas, C0]

	# Bounds (keep sigmas, lambdas positive; clamp mu if requested)
	lo_As   = [-np.inf]*n
	hi_As   = [ np.inf]*n
	if fix_xos:
		lo_mus = [xi - tol for xi in xos]
		hi_mus = [xi + tol for xi in xos]
	else:
		lo_mus = [-np.inf]*n
		hi_mus = [ np.inf]*n
	lo_sig  = [1e-9]*n
	hi_sig  = [ np.inf]*n
	lo_lam  = [1e-9]*n
	# === LAMBDA UPPER BOUND ===
	# Without an upper bound, curve_fit can drive λ arbitrarily large on
	# low-S/N bursts, producing tau = 1/λ smaller than the time resolution
	# (an unresolved-scattering delta function). The fit becomes degenerate
	# (parameters absorb each other) and the covariance matrix explodes,
	# yielding tau_err values like ±1e10 ms (B17 case: λ=770, τ=0.0013 ms,
	# τ_err=1.1e7 ms). Capping λ at 1/dt_ms forces τ ≥ dt_ms — the smallest
	# physically resolvable scattering timescale — and prevents this
	# degenerate corner. When dt_ms is unavailable (legacy callers passing
	# dt_ms=None), the bound is left at +inf to preserve old behaviour.
	if dt_ms is not None and dt_ms > 0:
		hi_lam = [1.0 / float(dt_ms)] * n
	else:
		hi_lam = [np.inf] * n
	# === END LAMBDA UPPER BOUND ===
	lo_C, hi_C = -np.inf, np.inf

	bounds = (
		[*lo_As, *lo_mus, *lo_sig, *lo_lam, lo_C],
		[*hi_As, *hi_mus, *hi_sig, *hi_lam, hi_C]
	)

	# === CLAMP SEEDS STRICTLY INSIDE BOUNDS =========================
	# ROOT CAUSE FIX (silent all-zeros / "sigma_t = (-nan)x10^-inf,
	# tau = unresolved, chi2 = nan"). The lambda seed is 1/sig0, and the
	# UPPER bound on lambda is 1/dt_ms (added with the bin-integrated EMG).
	# When the variance-based sig0 = sqrt(var)/4 falls BELOW dt_ms — which
	# band-limiting (bw_manual) makes likely, because a cleaner profile is
	# dominated by the narrow burst core so `var` collapses — the seed
	# lambda0 = 1/sig0 EXCEEDS 1/dt_ms. curve_fit then raises
	# "Initial guess is outside of provided bounds" and the bare `except`
	# below returned an all-zeros popt, so the legend showed sigma=0
	# (rendered by scilabel as (-nan)x10^-inf) and tau unresolved.
	# `fitemg` already clamps its seeds this way (see ~line 3433); doing the
	# same here closes the gap. We clamp to bound - eps (a hair inside) so a
	# seed sitting exactly on a finite edge can't trip the strict check, and
	# so the optimizer has room to move off the boundary on the first step.
	_lb = np.asarray(bounds[0], dtype=float)
	_ub = np.asarray(bounds[1], dtype=float)
	_p0 = np.asarray(p0, dtype=float)
	_both_finite = np.isfinite(_lb) & np.isfinite(_ub)
	_eps = np.zeros_like(_p0)
	_eps[_both_finite] = 1e-6 * np.maximum(_ub[_both_finite] - _lb[_both_finite], 1e-12)
	# OLD: p0 = [*A0, *xos, *sigmas, *lambdas, C0]   # (unclamped — could be infeasible)
	p0 = list(np.clip(_p0, _lb + _eps, _ub - _eps))
	# === END CLAMP SEEDS ============================================

	# === DROP NON-FINITE SAMPLES ====================================
	# A NaN/inf anywhere in `y` (e.g. an all-NaN time column produced by
	# nanmean over a fully-masked sub-band) makes curve_fit raise
	# "array must not contain infs or NaNs" -> same silent all-zeros.
	# Fit on the finite (t, y) pairs only; the model is point-wise so
	# subsetting the abscissa is safe.
	_finite = np.isfinite(t) & np.isfinite(y)
	if not _finite.all():
		print(f"  Note: fitemgmix dropping {int((~_finite).sum())} non-finite tseries sample(s) before fit.")
	t_fit = t[_finite]
	y_fit = y[_finite]
	# === END DROP NON-FINITE ========================================

	# === ROBUST data_sigma ==========================================
	# OLD: data_sigma = [np.std(y[:off_n])] * len(y)
	# If the leading off-pulse slab is flat (e.g. a masked region zeroed to
	# a constant), np.std -> 0, curve_fit divides residuals by 0 -> inf ->
	# raises. Fall back to the global std, then to 1.0, so weighting is
	# always finite and positive.
	_ds = float(np.nanstd(y_fit[:off_n]))
	if not np.isfinite(_ds) or _ds <= 0:
		_ds = float(np.nanstd(y_fit))
	if not np.isfinite(_ds) or _ds <= 0:
		_ds = 1.0
	data_sigma = [_ds] * len(y_fit)
	# === END ROBUST data_sigma ======================================

	# === SELECT MODEL: bin-integrated or instantaneous ===
	if dt_ms is not None and dt_ms > 0:
		# Adaptive oversampling from initial guesses
		tau_guesses = [1.0/max(l, 1e-9) for l in lambdas]
		min_sigma = min(sigmas) if sigmas else 1.0
		min_tau = min(tau_guesses) if tau_guesses else 1.0
		oversamp = pick_oversamp(dt_ms, min_sigma, min_tau)
		fit_model = lambda t_c, *p: bin_integrated_emgmix(t_c, dt_ms, oversamp, *p)
		eval_model = fit_model
	else:
		fit_model = emgmix_model
		eval_model = emgmix_model

	try:
		popt, pcov = scipy.optimize.curve_fit(
			fit_model, t_fit, y_fit, p0=p0, bounds=bounds, sigma=data_sigma, maxfev=200000
		)
		residuals = (y_fit - eval_model(t_fit, *popt)) / data_sigma
		chisq = np.sum(residuals**2)
		redchisq = chisq / max(len(y_fit) - len(popt), 1)
	except Exception as _emg_exc:
		# Surface WHY the fit failed instead of returning silent zeros. The
		# all-zeros popt still flows downstream (legend reads sigma=0/tau
		# unresolved) but now the terminal tells you the real reason.
		print(f"  WARNING: fitemgmix curve_fit FAILED -> {type(_emg_exc).__name__}: {_emg_exc}")
		print(f"           seeds(clamped): A={p0[0:n]}, mu={p0[n:2*n]}, "
		      f"sigma={p0[2*n:3*n]}, lambda={p0[3*n:4*n]}, C={p0[4*n]}")
		print(f"           lambda upper bound = {hi_lam} (=1/dt_ms; dt_ms={dt_ms})")
		popt = np.array([0]*(4*n + 1), dtype=float)
		pcov = np.zeros((4*n + 1), dtype=float)
		pcov = np.diag(pcov)
		redchisq = np.nan

	return popt, pcov, redchisq



	
def fitemg(data, duration, redchisq=1, mu0_seed=None, sigma0_seed=None, lambda0_seed=None, dt_ms=None,
           sigma_min=None, sigma_max=None, tau_min=None, tau_max=None, mu_min=None, mu_max=None):
	"""
	Fit EMG (with baseline C) to a 1-D row.
	
	Parameters:
	-----------
	data : array
		1-D intensity data
	duration : float
		Total duration in ms
	redchisq : float
		Reduced chi-squared for uncertainty scaling
	mu0_seed : float, optional
		Initial guess for mu from intensity profile fit (in local coordinates)
	sigma0_seed : float, optional
		Initial guess for sigma from intensity profile fit
	lambda0_seed : float, optional
		Initial guess for lambda from intensity profile fit
	dt_ms : float, optional
		Bin width / time resolution (ms) for bin-integrated fitting.
		If None, fits the instantaneous EMG model (original behaviour).
		If provided, uses oversample-and-average to account for finite
		time resolution, preventing bias when sigma or tau ~ dt_ms.
	sigma_min, sigma_max : float, optional
		Lower and upper bounds on σ in ms. Default None = no bounds beyond
		the legacy 1e-9 floor. Pass dt_ms to forbid sub-resolution σ.
	tau_min, tau_max : float, optional
		Lower and upper bounds on τ in ms (translated internally to
		bounds on λ = 1/τ). Default None = no bounds. Pass tau_min=dt_ms
		to forbid delta-function-like fits.
		
	Returns: popt=[A, mu, sigma, lamb, C], pcov (5x5), redchisq
	"""
	if len(data) == 0:
		return [np.nan]*5, np.full((5,5), np.nan), np.nan

	x  = np.linspace(0, duration, num=len(data))
	
	# === LOCAL ESTIMATION (used as fallback or when seeds not provided) ===
	xo = sum(x * data) / max(np.sum(data), 1e-12)
	var = abs(np.sum(data * (x - xo)**2) / max(np.sum(data), 1e-12))
	sig0_local = np.sqrt(var) if var > 0 else duration / 10.0

	off_n = max(len(data) // NOISE_WIDTH_FACTOR, 1)
	C0    = float(np.median(data[:off_n]))
	A0    = float(np.max(data) - C0)
	
	# === USE SEEDS IF PROVIDED AND VALID, OTHERWISE FALL BACK TO LOCAL ===
	# mu0
	if mu0_seed is not None and np.isfinite(mu0_seed) and 0 < mu0_seed < duration:
		mu0 = float(mu0_seed)
	else:
		mu0 = float(xo)
	
	# sigma0
	if sigma0_seed is not None and np.isfinite(sigma0_seed) and sigma0_seed > 0:
		sig0 = float(sigma0_seed)
	else:
		sig0 = max(sig0_local, 1e-6)
	
	# lambda0
	if lambda0_seed is not None and np.isfinite(lambda0_seed) and lambda0_seed > 0:
		lam0 = float(lambda0_seed)
	else:
		lam0 = 1.0 / max(sig0, 1e-6)

	data_sigma = [np.sqrt(redchisq) * np.std(data[:off_n])] * len(data)

	# === SELECT MODEL: bin-integrated or instantaneous ===
	if dt_ms is not None and dt_ms > 0:
		tau0 = 1.0 / max(lam0, 1e-9)
		oversamp = pick_oversamp(dt_ms, sig0, tau0)
		fit_model = lambda t, A, mu, sigma, lamb, C: bin_integrated_emg(
			t, A, mu, sigma, lamb, C, dt_ms=dt_ms, oversamp=oversamp
		)
		eval_model = fit_model  # for residual evaluation
	else:
		fit_model = emg_model
		eval_model = emg_model

	# === RESOLVE OPTIONAL BOUNDS ===
	# Defaults preserve legacy behaviour (1e-9 floors, no upper bound).
	_sig_lo = max(float(sigma_min), 1e-9) if (sigma_min is not None and np.isfinite(sigma_min) and sigma_min > 0) else 1e-9
	_sig_hi = float(sigma_max) if (sigma_max is not None and np.isfinite(sigma_max) and sigma_max > 0) else np.inf
	# τ bounds translate to λ bounds: λ ∈ [1/τ_max, 1/τ_min]
	_lam_lo = (1.0 / float(tau_max)) if (tau_max is not None and np.isfinite(tau_max) and tau_max > 0) else 1e-9
	_lam_hi = (1.0 / float(tau_min)) if (tau_min is not None and np.isfinite(tau_min) and tau_min > 0) else np.inf
	# μ bounds: default ±inf (legacy). The stacked-fit caller passes a
	# finite window (mu_seed ± 3σ_seed) so a low-SNR per-stack fit cannot
	# drift its mu onto a noise peak elsewhere in the window.
	_mu_lo = float(mu_min) if (mu_min is not None and np.isfinite(mu_min)) else -np.inf
	_mu_hi = float(mu_max) if (mu_max is not None and np.isfinite(mu_max)) else np.inf
	if _mu_lo >= _mu_hi:   # degenerate window -> fall back to unbounded
		_mu_lo, _mu_hi = -np.inf, np.inf
	# Clamp initial guesses into the bounded box; curve_fit will fail
	# loudly if p0 is outside bounds.
	sig0 = float(np.clip(sig0, _sig_lo, _sig_hi))
	lam0 = float(np.clip(lam0, _lam_lo, _lam_hi))
	mu0 = float(np.clip(mu0, _mu_lo, _mu_hi))

	try:
		popt, pcov = scipy.optimize.curve_fit(
			fit_model, x, data,
			p0=[A0, mu0, sig0, lam0, C0],
			sigma=data_sigma,
			bounds=(
				[-np.inf, _mu_lo, _sig_lo, _lam_lo, -np.inf],
				[ np.inf, _mu_hi, _sig_hi, _lam_hi,  np.inf]
			),
			maxfev=200000
		)
		residuals = (data - eval_model(x, *popt)) / data_sigma
		chisq     = np.sum(residuals**2)
		redchisq  = chisq / max(len(data) - len(popt), 1)
	except Exception:
		popt    = [np.nan]*5
		pcov    = np.full((5,5), np.nan)
		redchisq = np.nan

	return popt, pcov, redchisq



def fitgaussmix(data, duration, xos, sigmas=None, fix_xos=False, tol=0.01):
	n = len(xos) # Number of components
	if np.max(data) != 0:
		data = data / np.max(data) # normalize
	x = np.linspace(0, duration, num=len(data))
	# OLD: if sigmas is None:
	# Same empty-list trap as fitemgmix: the caller passes sigmas=[] (not None),
	# so the old guard fell through to the else branch and built guess as
	# [*[max]*n, *xos, *[]] (length 2n) while `bounds` under fix_xos is length 3n.
	# curve_fit then raised a ValueError that the RuntimeError-only except did not
	# catch, and the `finally: return popt` ran with popt unbound -> the
	# "cannot access local variable 'popt'" crash. Regenerate on missing OR wrong
	# length so guess always matches bounds.
	if sigmas is None or len(sigmas) != n:
		sigmas = np.sqrt(abs(sum(data*(x-np.mean(xos))**2)/sum(data)))/4
		guess = [*[np.max(data)]*n, *xos, *[sigmas]*n]
	else:
		guess = [*[np.max(data)]*n, *xos, *sigmas]

	bounds = (-np.inf, np.inf)
	if fix_xos:
		bounds = (  # fix xos
			[*[-np.inf]*n, *[xoi - tol for xoi in xos], *[-np.inf]*n],
			[*[np.inf]*n, *[xoi + tol for xoi in xos], *[np.inf]*n]
		)

	try:
		data_sigma = [np.std(data[0:len(data)//NOISE_WIDTH_FACTOR])]*len(data)
		popt, pcov = scipy.optimize.curve_fit(
			gaussmix_model,
			x,
			data,
			p0=guess,
			bounds=bounds,
			sigma=data_sigma,
		)
		residuals = (data - np.max(data)*gaussmix_model(x, *popt))/data_sigma
		chisq = np.sum((residuals) ** 2)
		redchisq = chisq / (len(data)- len(popt))
	# OLD: except RuntimeError as e:
	# Catch ANY fit failure, not just RuntimeError. With `finally: return popt`
	# below, a failure type not caught here (e.g. ValueError from a p0/bounds
	# length mismatch) leaves popt unbound, and the finally's return then raises a
	# confusing "cannot access local variable 'popt'" that masks the real cause.
	# Degrade to all-zeros like the EMG path and print why.
	except Exception as e:
		print(f"  WARNING: fitgaussmix curve_fit FAILED -> {type(e).__name__}: {e}")
		popt = [0]*3*n
# 		pcov = [0]*3*n 
		pcov = np.zeros((3*n, 3*n))  # Correct: 2D covariance matrix
		redchisq = 0
	finally:
		return popt, pcov, redchisq


def fit_per_region_emgmix(data, duration, xos, cuts, sigmas=None,
                          lambdas=None, fix_xos=False, tol=0.01, dt_ms=None):
	"""
	Per-region independent EMG fits — replacement for the joint
	`fitemgmix` when `cuts` is supplied and the user wants HARD
	component isolation.

	Each region [cuts[i-1], cuts[i]] contains exactly one component
	whose center `xos[i]` falls inside the region. We fit a SINGLE
	EMG to that region's slice of `data` with TIGHT physical bounds:

	  * σ_i ∈ [dt_ms, region_width_i / 2]
	  * λ_i ≥ 1/region_width_i   (i.e. τ_i ≤ region_width_i)
	  * μ_i ∈ [xos[i] − tol, xos[i] + tol]
	  * A_i, C_i unbounded

	These bounds enforce that each component physically fits inside
	its region and the scattering tail can't outlast it. Without these
	bounds the local fit can land on a "wide & low" EMG that produces
	a long tail extending into the next region — even though the fit
	itself only sees its slice, the bad σ/τ values then break the
	display when components are summed.

	Results are assembled into the same parameter vector layout as
	`fitemgmix`:

	    [A_1..A_n, μ_1..μ_n, σ_1..σ_n, λ_1..λ_n, C]

	so all downstream code (`tmix_popt` consumers, the visual bar,
	the slope panel, etc.) reads the result without change. The
	covariance matrix is block-diagonal: each component's 4×4
	(A, μ, σ, λ) covariance is placed in its own block, off-diagonal
	cross-component entries are zero (correct: independent fits are
	statistically independent). The C parameter shares no covariance
	with the per-component parameters since it's a single global
	value (median of the off-region bins).

	The noise estimate used for `curve_fit`'s `sigma` argument comes
	from the pre-burst region of the FULL time series (bins before
	the leftmost component's region), giving every region a robust
	noise level even when the region itself has no clean off-pulse
	bins.

	Returns (popt, pcov, redchisq) matching `fitemgmix`'s signature.
	popt's A and C values are in NORMALIZED units (data / max(data)),
	matching `fitemgmix`'s convention — the caller rescales for
	display via `tmix_popt_plot[A_i] *= max(tseries)`.
	"""
	n = len(xos)
	if n == 0:
		return [0]*(4*n + 1), np.zeros((4*n + 1, 4*n + 1)), np.nan

	# Normalize like fitemgmix does (max=1) so downstream rescaling lines work.
	scale = np.max(data) if np.max(data) != 0 else 1.0
	y_full = (data / scale) if scale != 1.0 else data
	n_time = len(y_full)

	# Build region boundaries in ms: [t_min, cuts[0], cuts[1], ..., cuts[-1], t_max]
	t_axis = np.linspace(0, duration, num=n_time)
	bin_ms = duration / max(n_time, 1)
	t_min, t_max = 0.0, float(duration)
	region_edges_ms = [t_min] + [float(c) for c in cuts] + [t_max]
	if len(region_edges_ms) - 1 != n:
		raise ValueError(
			f"fit_per_region_emgmix: number of regions = "
			f"{len(region_edges_ms) - 1} (from {len(cuts)} cuts) "
			f"does not match number of components = {n}. Each region "
			f"must contain exactly one component."
		)

	# Sort xos so component i is in region i (region order = increasing ms).
	xos_sorted_idx = sorted(range(n), key=lambda i: xos[i])
	xos_in_region_order = [xos[i] for i in xos_sorted_idx]

	# Verify each xos falls inside its expected region.
	for i in range(n):
		r_lo, r_hi = region_edges_ms[i], region_edges_ms[i + 1]
		xo_i = xos_in_region_order[i]
		if not (r_lo <= xo_i <= r_hi):
			print(
				f"  Warning: component {i} center xo = {xo_i:.4f} ms "
				f"falls outside its region [{r_lo:.4f}, {r_hi:.4f}] ms. "
				f"Cuts may be mismatched to xos."
			)

	# Global baseline C: the minimum of the pre-burst and post-burst
	# medians. Reasoning:
	#  - The naïve mask (t < first_xo) | (t > last_xo) is correct when
	#    components are well-separated from their off-pulse regions
	#    on both sides, but for *scattered* bursts the post-burst
	#    region contains a long scattering tail that elevates the
	#    median far above the true noise floor.
	#  - Taking the lower of the two medians sides with whichever
	#    side is cleaner. Usually that's pre-burst (scattering only
	#    extends rightward), but on bursts with a setup glitch or
	#    RFI spike before the burst, the post-burst slab will be
	#    the better estimator.
	# Falls back to the leading NOISE_WIDTH_FACTOR slab if neither
	# side has enough bins.
	_xo_first = xos_in_region_order[0]
	_xo_last  = xos_in_region_order[-1]
	_pre_mask  = t_axis < _xo_first
	_post_mask = t_axis > _xo_last
	_cand_medians = []
	if _pre_mask.sum() >= 5:
		_cand_medians.append(float(np.median(y_full[_pre_mask])))
	if _post_mask.sum() >= 5:
		_cand_medians.append(float(np.median(y_full[_post_mask])))
	if _cand_medians:
		C_global = min(_cand_medians)
	else:
		off_n = max(n_time // NOISE_WIDTH_FACTOR, 1)
		C_global = float(np.median(y_full[:off_n]))

	# GLOBAL noise estimate: σ_noise from the pre-burst region (clean
	# off-pulse bins before the leftmost component's region). Used as
	# `sigma` for every per-region curve_fit so noise is consistent
	# across regions, even when an individual region has no clean
	# off-pulse bins of its own (e.g., the inner regions for tightly-
	# packed multi-component bursts).
	_prepulse_mask = t_axis < region_edges_ms[0] + 0.5 * (xos_in_region_order[0] - region_edges_ms[0])
	# Prefer bins strictly before the leading region; if too few,
	# fall back to leading NOISE_WIDTH_FACTOR slab.
	if _prepulse_mask.sum() >= 5:
		noise_std_global = float(np.std(y_full[_prepulse_mask]))
	else:
		off_n = max(n_time // NOISE_WIDTH_FACTOR, 1)
		noise_std_global = float(np.std(y_full[:off_n]))
	if not (noise_std_global > 0):
		noise_std_global = 1e-6

	# Per-region fits in region order.
	#
	# C is FIXED to C_global for every per-region fit (we subtract
	# C_global from the region's data and fit a baseline-free EMG).
	# This guarantees that the (A, σ, λ) values reported by the fit
	# are consistent with the baseline used at display time. Otherwise
	# curve_fit can pick a different per-region C that minimises that
	# region's chi-square (e.g. fitting through a noisy trough or part
	# of the burst's own rising edge), and when we then display with
	# C_global the resulting EMG floats above or below the data —
	# even though the (σ, λ) shape values are individually fine, the
	# absolute level is inconsistent. With this fix all per-region
	# fits and the displayed stitched curve share one baseline.
	per_comp_popts = []   # each entry: (A, mu_global, sigma, lambda) — NORMALIZED units
	per_comp_pcovs = []   # each entry: 4×4 covariance
	chisq_total = 0.0
	dof_total = 0
	for i in range(n):
		r_lo_ms, r_hi_ms = region_edges_ms[i], region_edges_ms[i + 1]
		region_width = r_hi_ms - r_lo_ms
		# Bin indices for this region's slice of the data
		i_lo = max(0, int(np.floor(r_lo_ms / max(bin_ms, 1e-12))))
		i_hi = min(n_time, int(np.ceil(r_hi_ms / max(bin_ms, 1e-12))))
		# Subtract the global baseline so the fit sees a zero-baseline
		# EMG and we can drop C from the parameter vector entirely.
		region_data = y_full[i_lo:i_hi] - C_global
		t_region_local = t_axis[i_lo:i_hi] - r_lo_ms

		if len(region_data) < 5:
			print(f"  Warning: region {i} has {len(region_data)} bins, too few for EMG fit")
			per_comp_popts.append((np.nan, np.nan, np.nan, np.nan))
			per_comp_pcovs.append(np.full((4, 4), np.nan))
			continue

		# Seeds — small starting σ and τ relative to region width.
		mu_local_seed = float(xos_in_region_order[i] - r_lo_ms)
		sig0 = min(0.08, region_width / 6.0)
		# τ seed bounded by region size
		tau0 = min(0.05, region_width / 6.0)
		lam0 = 1.0 / max(tau0, max(bin_ms, 1e-6))
		# A seed: peak above the now-zero baseline (region_data is
		# already C_global-subtracted).
		A0 = max(float(np.max(region_data)), 1e-4)
		p0 = [A0, mu_local_seed, sig0, lam0]

		# Bounds (now 4-parameter; no C entry):
		#   A unbounded
		#   μ ∈ [mu_seed − tol, mu_seed + tol]
		#   σ ∈ [dt_ms_floor, region_width / 2]
		#   λ ∈ [1/region_width, 1/dt_ms_floor]   (i.e. τ ∈ [dt_ms, region_width])
		#
		# Both ends of the λ range are physically motivated:
		#   * λ ≥ 1/region_width  → τ ≤ region_width  (tail can't outlast the
		#                                              region; bound from below)
		#   * λ ≤ 1/dt_ms         → τ ≥ dt_ms         (tail can't be sub-bin;
		#                                              the fit collapses to a
		#                                              Gaussian-shaped EMG
		#                                              otherwise, which then
		#                                              produces a nonsensical
		#                                              lambda_seed for the
		#                                              downstream stacking
		#                                              pipeline. See e.g.
		#                                              B29 component a: without
		#                                              this bound, λ landed at
		#                                              ~10⁷, τ ~30 ns, and
		#                                              every per-stack fit
		#                                              inherited a degenerate
		#                                              seed.)
		_dt_floor = max(dt_ms if (dt_ms is not None and dt_ms > 0) else bin_ms, bin_ms * 0.5)
		_lam_hi   = 1.0 / _dt_floor
		lo = [
			-np.inf,
			mu_local_seed - tol,
			_dt_floor,
			1.0 / region_width,
		]
		hi = [
			np.inf,
			mu_local_seed + tol,
			region_width / 2.0,
			_lam_hi,
		]
		# Clamp the seed too — curve_fit raises if p0 is outside the box.
		lam0 = float(np.clip(lam0, 1.0 / region_width, _lam_hi))

		# Select EMG model with C hard-wired to 0 (baseline already
		# subtracted from region_data). curve_fit sees a 4-parameter
		# function f(t, A, μ, σ, λ).
		if dt_ms is not None and dt_ms > 0:
			oversamp = pick_oversamp(dt_ms, sig0, tau0)
			_dt_local = dt_ms
			_ovs_local = oversamp
			fit_model = lambda t, A, mu, sigma, lamb: bin_integrated_emg(
				t, A, mu, sigma, lamb, 0.0, dt_ms=_dt_local, oversamp=_ovs_local
			)
		else:
			fit_model = lambda t, A, mu, sigma, lamb: emg_model(
				t, A, mu, sigma, lamb, 0.0
			)

		# Use the global noise estimate (in normalized units, since y_full is normalized).
		sigma_arr = np.full_like(region_data, noise_std_global)

		try:
			popt_i, pcov_i = scipy.optimize.curve_fit(
				fit_model, t_region_local, region_data,
				p0=p0, bounds=(lo, hi),
				sigma=sigma_arr, absolute_sigma=True,
				maxfev=200000,
			)
			# Local μ → global μ
			A_i = float(popt_i[0])
			mu_i_global = float(popt_i[1]) + r_lo_ms
			sg_i = float(popt_i[2])
			lm_i = float(popt_i[3])
			# pcov is already 4×4 since we dropped C from the fit.
			pcov_i_arr = np.asarray(pcov_i)
			if pcov_i_arr.shape == (4, 4):
				cov_4x4 = pcov_i_arr
			else:
				cov_4x4 = np.full((4, 4), np.nan)
			per_comp_popts.append((A_i, mu_i_global, sg_i, lm_i))
			per_comp_pcovs.append(cov_4x4)

			# Per-region chi-square contribution. dof = N − 4 (no
			# longer N − 5; C isn't being fit).
			resid = (region_data - fit_model(t_region_local, *popt_i)) / sigma_arr
			chisq_i = float(np.sum(resid**2))
			chisq_total += chisq_i
			dof_total += max(len(region_data) - 4, 1)
		except Exception as _e:
			print(f"  Warning: per-region EMG fit for component {i} (region "
			      f"[{r_lo_ms:.4f}, {r_hi_ms:.4f}] ms) failed: {_e}")
			per_comp_popts.append((np.nan, np.nan, np.nan, np.nan))
			per_comp_pcovs.append(np.full((4, 4), np.nan))

	# Re-order to ORIGINAL xos order so downstream indexing matches.
	popts_in_original_order = [None] * n
	pcovs_in_original_order = [None] * n
	for j, orig_idx in enumerate(xos_sorted_idx):
		popts_in_original_order[orig_idx] = per_comp_popts[j]
		pcovs_in_original_order[orig_idx] = per_comp_pcovs[j]

	# Assemble parameter vector in fitemgmix's layout.
	As = np.array([p[0] for p in popts_in_original_order], dtype=float)
	mus = np.array([p[1] for p in popts_in_original_order], dtype=float)
	sgs = np.array([p[2] for p in popts_in_original_order], dtype=float)
	lms = np.array([p[3] for p in popts_in_original_order], dtype=float)
	popt = np.concatenate([As, mus, sgs, lms, [C_global]])

	# Block-diagonal covariance with each component's 4×4 in its own block.
	dim = 4 * n + 1
	pcov = np.zeros((dim, dim), dtype=float)
	for i in range(n):
		cov_i = np.asarray(pcovs_in_original_order[i])
		if cov_i.shape != (4, 4) or not np.all(np.isfinite(cov_i)):
			continue
		idx = [i, n + i, 2*n + i, 3*n + i]
		for a in range(4):
			for b in range(4):
				pcov[idx[a], idx[b]] = cov_i[a, b]

	redchisq = (chisq_total / max(dof_total, 1)) if dof_total > 0 else np.nan
	return popt, pcov, redchisq


def fit_per_region_gaussmix(data, duration, xos, cuts, sigmas=None,
                            fix_xos=False, tol=0.01):
	"""
	Per-region independent Gaussian fits — the Gaussian-mode
	counterpart to `fit_per_region_emgmix`. Same design: one fit per
	region using `fitgauss`, results assembled into the fitgaussmix
	layout [A_1..A_n, μ_1..μ_n, σ_1..σ_n, C].

	Returns (popt, pcov, redchisq).
	"""
	n = len(xos)
	if n == 0:
		return [0]*(3*n + 1), np.zeros((3*n + 1, 3*n + 1)), np.nan

	n_time = len(data)
	t_axis = np.linspace(0, duration, num=n_time)
	t_min, t_max = 0.0, float(duration)
	region_edges_ms = [t_min] + [float(c) for c in cuts] + [t_max]
	if len(region_edges_ms) - 1 != n:
		raise ValueError(
			f"fit_per_region_gaussmix: number of regions = "
			f"{len(region_edges_ms) - 1} (from {len(cuts)} cuts) "
			f"does not match number of components = {n}."
		)

	# Normalize like fitgaussmix does (max=1) so that the returned A
	# and C are in normalized units and the downstream display
	# rescaling (tmix_popt_plot[A_i] *= max(tseries)) recovers
	# original units consistently. Without this, A came back in
	# original units and got multiplied by max(tseries) again at
	# display time, producing off-the-chart amplitudes.
	scale = np.max(data) if np.max(data) != 0 else 1.0
	y_full = (data / scale) if scale != 1.0 else np.asarray(data, dtype=float)

	# Global baseline C: minimum of pre-burst and post-burst medians.
	# See the matching comment in fit_per_region_emgmix for rationale —
	# scattering tails contaminate the post-burst median for scattered
	# bursts, so taking the lower of the two sides defaults to whichever
	# region is cleaner.
	xos_sorted_idx = sorted(range(n), key=lambda i: xos[i])
	_xo_first = xos[xos_sorted_idx[0]]
	_xo_last  = xos[xos_sorted_idx[-1]]
	_pre_mask  = t_axis < _xo_first
	_post_mask = t_axis > _xo_last
	_cand_medians = []
	if _pre_mask.sum() >= 5:
		_cand_medians.append(float(np.median(y_full[_pre_mask])))
	if _post_mask.sum() >= 5:
		_cand_medians.append(float(np.median(y_full[_post_mask])))
	if _cand_medians:
		C_global = min(_cand_medians)
	else:
		off_n = max(n_time // NOISE_WIDTH_FACTOR, 1)
		C_global = float(np.median(y_full[:off_n]))

	# C is FIXED to C_global for every per-region fit. `gauss_model`
	# is already baseline-free (no C parameter), so the only thing
	# needed to make the displayed baseline match the fitted shape
	# is to subtract C_global from each region's data BEFORE the
	# fit. Without this, fitgauss's amplitude A absorbs the
	# region's apparent baseline (e.g. ~2.3 here for 11A), and when
	# we then display gauss(t) + C_global, the curve floats above
	# the data wherever the region's apparent baseline exceeded
	# C_global. See the matching comment in fit_per_region_emgmix.
	per_comp_popts = []
	per_comp_pcovs = []
	chisq_total = 0.0
	dof_total = 0
	for j, orig_i in enumerate(xos_sorted_idx):
		r_lo_ms, r_hi_ms = region_edges_ms[j], region_edges_ms[j + 1]
		bin_ms = duration / max(n_time, 1)
		i_lo = max(0, int(np.floor(r_lo_ms / max(bin_ms, 1e-12))))
		i_hi = min(n_time, int(np.ceil(r_hi_ms / max(bin_ms, 1e-12))))
		region_data = y_full[i_lo:i_hi] - C_global
		region_duration_ms = max((i_hi - i_lo), 1) * bin_ms
		region_left_edge_ms = i_lo * bin_ms
		try:
			popt_i, pcov_i, redchisq_i = fitgauss(
				region_data, region_duration_ms, redchisq=1.0
			)
			A_i  = float(popt_i[0])
			mu_i_local = float(popt_i[1])
			sg_i = float(popt_i[2])
			mu_i_global = mu_i_local + region_left_edge_ms
			pcov_i_arr = np.asarray(pcov_i)
			cov_3x3 = pcov_i_arr if pcov_i_arr.shape == (3, 3) else np.full((3, 3), np.nan)
			per_comp_popts.append((A_i, mu_i_global, sg_i))
			per_comp_pcovs.append(cov_3x3)
			if np.isfinite(redchisq_i) and len(region_data) > 3:
				chisq_i = float(redchisq_i) * max(len(region_data) - 4, 1)
				chisq_total += chisq_i
				dof_total += max(len(region_data) - 4, 1)
		except Exception as _e:
			print(f"  Warning: per-region Gaussian fit for component {orig_i} failed: {_e}")
			per_comp_popts.append((np.nan, np.nan, np.nan))
			per_comp_pcovs.append(np.full((3, 3), np.nan))

	popts_in_original_order = [None] * n
	pcovs_in_original_order = [None] * n
	for j, orig_idx in enumerate(xos_sorted_idx):
		popts_in_original_order[orig_idx] = per_comp_popts[j]
		pcovs_in_original_order[orig_idx] = per_comp_pcovs[j]

	As = np.array([p[0] for p in popts_in_original_order], dtype=float)
	mus = np.array([p[1] for p in popts_in_original_order], dtype=float)
	sgs = np.array([p[2] for p in popts_in_original_order], dtype=float)
	popt = np.concatenate([As, mus, sgs, [C_global]])

	dim = 3 * n + 1
	pcov = np.zeros((dim, dim), dtype=float)
	for i in range(n):
		cov_i = np.asarray(pcovs_in_original_order[i])
		if cov_i.shape != (3, 3) or not np.all(np.isfinite(cov_i)):
			continue
		idx = [i, n + i, 2*n + i]
		for a in range(3):
			for b in range(3):
				pcov[idx[a], idx[b]] = cov_i[a, b]

	redchisq = (chisq_total / max(dof_total, 1)) if dof_total > 0 else np.nan
	return popt, pcov, redchisq


def fitrows(wfall, dt, freqs, plot=False, bname=''):
	if type(plot) == list:
		plot = [int(p) for p in plot]
	fitdata = np.zeros((wfall.shape[0], 11))
	for i, row in enumerate(wfall):
		popt, pcov, redchisq = fitgauss(row, wfall.shape[1]*dt)
		perr = np.sqrt(np.diag(pcov))
		# print(f"{freqs[i]:.2f}: {popt = } {perr = } {redchisq = } {np.max(row) = }")

		if logdebug and not np.isnan(redchisq) and redchisq > 0:
			# sanity check: is the reduced chisq=1 after rescaling the data uncertainties?
			_, _, post_redchisq = fitgauss(row, wfall.shape[1]*dt, redchisq)
			printd(f"{freqs[i]:.2f}: {redchisq = :.2f}, post fit reduced χ² = {post_redchisq:.2f}")

		if len(perr.shape) == 2: perr = np.diag(perr) # handles when pcov is nans

		sigma = abs(popt[2])
		# Arrival time convention depends on PEAK_INTENSITY_TOA toggle
		if PEAK_INTENSITY_TOA:
			# Peak mode: arrival time at Gaussian center (peak of intensity)
			tstart = popt[1]
			tstart_err = perr[1]  # Just the uncertainty on mu
		else:
			# Leading edge mode: arrival time at left half-maximum crossing
			tstart = (popt[1]-np.sqrt(2*np.log(2))*sigma)
			tstart_err = np.sqrt(perr[1]**2 + 2*np.log(2)*perr[2]**2)
		# print(f"{freqs[i] = }, {tstart = }, {tstart_err = }")
		# print(f"{row_sigma = :.4f}, {redchisq = :.3f}, {chisq = :.3f}")

		tend   = (popt[1]+np.sqrt(2)*sigma)
		fitdata[i,:] = [
			freqs[i],
			tstart,
			tend,
			popt[0],
			popt[1],
			tstart_err,
			sigma,
			redchisq,
			*perr # (3 values)
		]
		# print(fitdata.shape)
		# exit()

		# Diagnostic plot for row fit
		if (plot and type(plot) == bool) or ((type(plot) == list) and int(freqs[i]) in plot):
			xs = np.linspace(0, wfall.shape[1]*dt, num=len(row))
			data_sigma = [np.std(row[0:len(row)//NOISE_WIDTH_FACTOR])]*len(row)
			residuals = (row - gaussmix_model(xs, *popt))/data_sigma
			fig, axs = plt.subplots(2,1, figsize=(10,7), height_ratios=[3,1])
			axs[0].plot(xs, row)
			axs[0].plot(
				xs,
				gauss_model(xs, *popt),
				label='$a, t_0, ÃÆ’$ = '+', '.join([f"{float(p):.3f}" for p in popt])
			)
			axs[0].axvline(
				x=tstart,
				c='k',
				linestyle='-.',
				label=f'$t_\\text{{start}}$ = {scilabel(tstart, tstart_err)}'
			)
			axs[0].axvline(
					x=xs[len(row)//NOISE_WIDTH_FACTOR],
					label=f'noise sample boundary ($\\sigma_{{\\text{{data}}}} = $ {data_sigma[0]:.2f})',
					c='r', linestyle='--'
				)
			axs[0].axvline(x=min(xs), alpha=0, label=f"Ãâ€¡$^2_{{\\text{{red}}}}=${redchisq:.3f}")
			axs[0].set_xlim(min(xs), max(xs))
			axs[0].set_title(f"{bname} freq: {freqs[i]:.2f} MHz")
			axs[0].legend(frameon=False)

			kstest = scipy.stats.kstest(residuals, scipy.stats.norm.cdf)
			if not np.isnan(tstart_err) and not np.isnan(residuals.min()):
				_, bins, _ = axs[1].hist(
					np.random.standard_normal(len(residuals)),
					bins='auto',
					label='Sampled $N(0,1)$',
					histtype='step',
					density=True,
					linewidth=2,
					range=(residuals.min(), residuals.max())
				)
				nx = np.linspace(*axs[1].get_xlim())
				axs[1].plot(
					nx,
					gauss_model(nx, 1/np.sqrt(2*np.pi), 0, 1),
					'k-',
					label='$N(0,1)$'
				)
				axs[1].hist(
					residuals,
					bins=bins,
					label='residuals',
					density=True
					# histtype='step'
				)
				axs[1].plot(0, 'w.', alpha=0,
					label=f"K-S test $p = {kstest.pvalue:.2f}$\nstatistic $= {kstest.statistic:.2f}$"
				)
				axs[1].legend(frameon=False, fontsize='small')
			axs[1].set_title("Distribution of normalized residuals")
			plt.tight_layout()
			plt.show(block=True)
			plt.close()

	return pd.DataFrame(data=fitdata, columns=[
		'freqs',
		'tstart',
		'tend',
		'amp',
		'xo',
		'tstart_err',
		'sigma',
		'red_chisq',
		'amp_err',
		'xo_err',
		'sigma_err'
	])

def fitrowsemg(wfall, dt, freqs, plot=False, bname='',
               mu0_seed=None, sigma0_seed=None, lambda0_seed=None):
	"""
	Row-by-row EMG fit mirroring `fitrows`, with tail-aware windowing and extra outputs.

	Choices implemented:
	  - Center (x_o) = mu (Gaussian-core peak / "true peak")
	  - Width for windowing = sigma_emg_std = sqrt(sigma^2 + 1/lambda^2)
	  - Keep `sigma` column as the EMG's *Gaussian-core* sigma (backwards compatible)
	  - Extra columns: lambda, tau, lambda_err, tau_err, sigma_emg_std, sigma_emg_std_err
	  - Window edges: tstart/tend built from x_o and sigma_emg_std with k = sqrt(2)

	Parameters (new):
	  mu0_seed, sigma0_seed, lambda0_seed : float or None
	    Initial-guess seeds forwarded to fitemg() for every row.
	    Passing sigma0_seed and lambda0_seed from the intensity-profile
	    fit prevents the sigma-tau degeneracy from killing per-row fits.
	"""
	# normalize plot argument
	if type(plot) == list:
		plot = [int(p) for p in plot]

	n_rows = wfall.shape[0]
	# original 11 cols + 9 new = 19
	fitdata = np.full((n_rows, 19), np.nan, dtype=float)

	total_duration = wfall.shape[1] * dt
	k_window = np.sqrt(2)

	for i, row in enumerate(wfall):
		# --- EMG fit (expects you already defined fitemg and emg) ---
		popt, pcov, redchisq = fitemg(row, total_duration, dt_ms=dt,
		                              mu0_seed=mu0_seed,
		                              sigma0_seed=sigma0_seed,
		                              lambda0_seed=lambda0_seed)
		# popt = [A, mu, sigma, lamb, C]
		# pcov is 5x5; may be nan on failure
  
		
######## Unregularized
		# guard: extract 1-D perr; if pcov bad, fill NaNs
		if isinstance(pcov, np.ndarray) and pcov.ndim == 2:
			perr = np.sqrt(np.diag(pcov))
		else:
			perr = np.full(5, np.nan, dtype=float)

###### REGULARIZATION
####################################################################
		# # === Solution 3: Regularize ill-conditioned covariance ===
		# if isinstance(pcov, np.ndarray) and pcov.ndim == 2:
		#     try:
		#         cond_num = np.linalg.cond(pcov)
		#         if cond_num > 1e8:
		#             reg_factor = 1e-8 * np.trace(pcov) / pcov.shape[0]
		#             pcov = pcov + reg_factor * np.eye(pcov.shape[0])
		#             # print(f"  Row {i}: Regularized pcov (cond was {cond_num:.2e})")
		#     except np.linalg.LinAlgError:
		#         cond_num = np.inf
####################################################################

		# try a diagnostic / reweighting pass like fitgauss
		if 'logdebug' in globals() and logdebug and np.isfinite(redchisq) and redchisq > 0:
			try:
				_, _, post_redchisq = fitemg(row, total_duration, redchisq, dt_ms=dt)
				if 'printd' in globals():
					printd(f"{freqs[i]:.2f}: redchisq={redchisq:.2f}, post-fit χ²_red={post_redchisq:.2f}")
			except Exception:
				pass

		# guard: perr from diag; handle pathological shapes
		try:
			perr = np.sqrt(np.diag(pcov))
		except Exception:
			perr = np.array([np.nan]*5, dtype=float)

		# unpack parameters
		A, mu, sig_core, lam, C = (np.nan,)*5
		if popt is not None and len(popt) == 5:
			A, mu, sig_core, lam, C = popt

		# === ADD ADDITION 1 HERE ===
		# === DIAGNOSTIC: Check fit quality (suppressed when stacking is enabled) ===
		if i == 0 and not USE_SIGNAL_STACKING:  # Print header once, only if not stacking
			print("\n=== Per-Row EMG Fit Diagnostics ===")
			print(f"{'Row':<4} {'Freq':>8} {'redchi':>7} {'A':>8} {'mu':>8} {'sigma':>8} {'lambda':>8} "
				  f"{'pcov_OK':<8} {'cond_num':>10} {'t_err_OK':<9}")
		
		pcov_ok = "OK" if (isinstance(pcov, np.ndarray) and pcov.ndim == 2 and np.all(np.isfinite(pcov))) else "BAD"
		
		pcov_ok = "OK" if (isinstance(pcov, np.ndarray) and pcov.ndim == 2 and np.all(np.isfinite(pcov))) else "BAD"

		if isinstance(pcov, np.ndarray) and pcov.ndim == 2:
			try:
				cond_num = np.linalg.cond(pcov)
			except np.linalg.LinAlgError:
				# SVD didn't converge - matrix is extremely ill-conditioned
				cond_num = np.inf
				pcov_ok = "SVD_FAIL"
		else:
			cond_num = np.nan
		
		# This will be set later, but we'll check at the end
		t_err_status = "PENDING"
		
		# Per-row diagnostics (suppressed when stacking is enabled)
		if not USE_SIGNAL_STACKING:
			if np.isnan(redchisq) or redchisq > 50:
				print(f"{i:<4d} {freqs[i]:>8.2f} {redchisq:>7.2f} {'POOR FIT':>50}")
			elif not np.isfinite([A, mu, sig_core, lam]).all():
				print(f"{i:<4d} {freqs[i]:>8.2f} {'NaN PARAMS':>60}")
			elif sig_core < 0.001 * total_duration:
				print(f"{i:<4d} {freqs[i]:>8.2f} {redchisq:>7.2f} SIGMA TOO SMALL: {sig_core:.6f}")
			elif lam > 10000 or lam < 0.01:
				print(f"{i:<4d} {freqs[i]:>8.2f} {redchisq:>7.2f} LAMBDA EXTREME: {lam:.2f}")
			elif pcov_ok == "BAD":
				print(f"{i:<4d} {freqs[i]:>8.2f} {redchisq:>7.2f} COVARIANCE BAD")
			elif cond_num > 1e10:
				print(f"{i:<4d} {freqs[i]:>8.2f} {redchisq:>7.2f} ILL-CONDITIONED: cond={cond_num:.2e}")


		# pull variances/covariances (with guards)
		def _safe_cov(ii, jj):
			try:
				return float(pcov[ii, jj])
			except Exception:
				return np.nan

		mu_var   = _safe_cov(1,1)
		sig_var  = _safe_cov(2,2)
		lam_var  = _safe_cov(3,3)
		mu_lam_c = _safe_cov(1,3)
		sig_lam_c= _safe_cov(2,3)

		# --- Derived scattering quantities ---
		# lambda_err, tau (=1/lam), tau_err
		lambda_err = np.sqrt(lam_var) if np.isfinite(lam_var) and lam_var >= 0 else np.nan
		tau        = (1.0/lam) if (lam is not None and np.isfinite(lam) and lam > 0) else np.nan
		tau_err    = (lambda_err / (lam**2)) if (np.isfinite(lambda_err) and np.isfinite(lam) and lam > 0) else np.nan
		
		# --- EMG arrival time: depends on PEAK_INTENSITY_TOA toggle ---
		# When PEAK_INTENSITY_TOA=False (default): use left HWHM crossing (leading edge)
		# When PEAK_INTENSITY_TOA=True: use peak of EMG function
		t_emg_hwhm = np.nan
		try:
			if np.isfinite(mu) and np.isfinite(sig_core) and np.isfinite(tau) and (sig_core > 0) and (tau > 0):
				if PEAK_INTENSITY_TOA:
					# Peak mode: find the maximum of the EMG function numerically
					t_emg_hwhm = find_emg_peak(mu=float(mu), sigma=float(sig_core), lamb=float(lam))
				else:
					# Leading edge mode: use left HWHM crossing
					stats = emg_left_hwhm(mu=float(mu), sigma=float(sig_core), tau=float(tau))
					# stats includes: x_mode, f_mode, x_left, HWHM_left; we use the LEFT crossing time as arrival time
					t_emg_hwhm = float(stats["x_left"])
		except Exception:
			# leave as NaN on failure; downstream masking already handles low-S/N rows
			pass
		
		
		##USE GAUSSIAN FALLBACK ROWS  
##################################################################        
		# # === CRITICAL: Fallback for t_emg_hwhm itself when HWHM calculation fails ===
		# if not np.isfinite(t_emg_hwhm):
		#     # When HWHM can't be computed, fall back to using the Gaussian-style tstart
		#     # This keeps the row usable even if EMG-specific HWHM fails
		#     if np.isfinite(mu) and np.isfinite(sig_core):
		#         # Use Gaussian-style leading edge: mu - sqrt(2) * sigma
		#         t_emg_hwhm = mu - np.sqrt(2) * abs(sig_core)
		#     elif np.isfinite(mu):
		#         # Last resort: use mu itself (center of Gaussian core)
		#         t_emg_hwhm = mu
		#     else:
		#         # Absolute fallback: use time window center
		#         t_emg_hwhm = total_duration / 2.0
				
#################################################################
				
		# --- Arrival-time uncertainty via numerical propagation ---
		t_emg_hwhm_err = np.nan
		gradient_failure_reason = None
		
		# For PEAK_INTENSITY_TOA mode, we need to propagate uncertainty differently
		# For leading edge mode, we use the original gradient-based propagation
		try:
			if PEAK_INTENSITY_TOA:
				# Peak mode: propagate uncertainty using numerical gradients for the peak location
				if (np.isfinite(t_emg_hwhm) and 
					np.isfinite([mu, sig_core, lam]).all() and 
					isinstance(pcov, np.ndarray) and pcov.shape == (5,5) and 
					np.all(np.isfinite(pcov))):
					
					# Step sizes
					dmu  = max(1e-6, 0.001 * abs(mu)) if mu != 0 else 1e-6
					dsig = max(1e-6, 0.001 * abs(sig_core)) if sig_core != 0 else 1e-6
					dlam = max(1e-6, 0.001 * abs(lam)) if lam != 0 else 1e-6
					
					# Central differences for peak location wrt mu, sigma, lambda
					def xpeak(mu_, sig_, lam_):
						return find_emg_peak(mu=float(mu_), sigma=float(sig_), lamb=float(lam_))
					
					x_center = xpeak(mu, sig_core, lam)
					if np.isfinite(x_center):
						g_mu = (xpeak(mu+dmu, sig_core, lam) - xpeak(mu-dmu, sig_core, lam)) / (2*dmu)
						g_sig = (xpeak(mu, sig_core+dsig, lam) - xpeak(mu, sig_core-dsig, lam)) / (2*dsig)
						g_lam = (xpeak(mu, sig_core, lam+dlam) - xpeak(mu, sig_core, lam-dlam)) / (2*dlam)
						
						if np.isfinite([g_mu, g_sig, g_lam]).all():
							# Gradient vector J wrt parameters in pcov order [A, mu, sigma, lambda, C]
							J = np.array([0.0, g_mu, g_sig, g_lam, 0.0], dtype=float)
							# Variance = J^T * Cov * J
							var_x = float(J @ pcov @ J)
							if np.isfinite(var_x) and var_x >= 0:
								t_emg_hwhm_err = np.sqrt(var_x)
			else:
				# Leading edge mode: original gradient-based propagation
				if (np.isfinite(t_emg_hwhm) and 
					np.isfinite([mu, sig_core, lam]).all() and 
					isinstance(pcov, np.ndarray) and pcov.shape == (5,5) and 
					np.all(np.isfinite(pcov))):
				
					# Step sizes: make them adaptive to parameter scales
					dmu  = max(1e-6, 0.001 * abs(mu)) if mu != 0 else 1e-6
					dsig = max(1e-6, 0.001 * abs(sig_core)) if sig_core != 0 else 1e-6
					dtau = max(1e-6, 0.001 / abs(lam)) if lam != 0 else 1e-6
		
					# Central differences for x_left wrt mu, sigma, tau
					def xleft(mu_, sig_, tau_):
						result = emg_left_hwhm(mu=float(mu_), sigma=float(sig_), tau=float(tau_))
						return float(result["x_left"])
		
					# Test the function first
					try:
						x_center = xleft(mu, sig_core, 1.0/lam)
						if not np.isfinite(x_center):
							gradient_failure_reason = "center_hwhm_nonfinite"
							raise ValueError("emg_left_hwhm returned non-finite at center")
					except Exception as e:
						gradient_failure_reason = f"center_hwhm_failed: {str(e)[:30]}"
						raise
		
					# Compute gradients
					try:
						x_mu_plus = xleft(mu+dmu, sig_core, 1.0/lam)
						x_mu_minus = xleft(mu-dmu, sig_core, 1.0/lam)
						g_mu = (x_mu_plus - x_mu_minus) / (2*dmu)
						
						x_sig_plus = xleft(mu, sig_core+dsig, 1.0/lam)
						x_sig_minus = xleft(mu, sig_core-dsig, 1.0/lam)
						g_sig = (x_sig_plus - x_sig_minus) / (2*dsig)
						
						x_tau_plus = xleft(mu, sig_core, 1.0/lam+dtau)
						x_tau_minus = xleft(mu, sig_core, 1.0/lam-dtau)
						g_tau = (x_tau_plus - x_tau_minus) / (2*dtau)
						g_lam = g_tau * (-1.0 / (lam**2))
						
						if not np.isfinite([g_mu, g_sig, g_lam]).all():
							gradient_failure_reason = f"gradient_nonfinite: mu={g_mu:.2e}, sig={g_sig:.2e}, lam={g_lam:.2e}"
							raise ValueError("Gradient contains non-finite values")
							
					except Exception as e:
						gradient_failure_reason = f"gradient_calc_failed: {str(e)[:30]}"
						raise
		
					# Gradient vector J wrt parameters in pcov order [A, mu, sigma, lambda, C]
					J = np.array([0.0, g_mu, g_sig, g_lam, 0.0], dtype=float)
		
					# Variance = J^T * Cov * J
					try:
						var_x = float(J @ pcov @ J)
						if not np.isfinite(var_x):
							gradient_failure_reason = f"variance_nonfinite: {var_x}"
							raise ValueError("Variance is non-finite")
						if var_x < 0:
							gradient_failure_reason = f"variance_negative: {var_x:.2e}"
							raise ValueError("Variance is negative")
						t_emg_hwhm_err = np.sqrt(var_x)
					except Exception as e:
						gradient_failure_reason = f"variance_calc_failed: {str(e)[:30]}"
						raise
						
				else:
					# Identify which condition failed (for leading edge mode)
					if not np.isfinite(t_emg_hwhm):
						gradient_failure_reason = "t_emg_hwhm_nonfinite"
					elif not np.isfinite([mu, sig_core, lam]).all():
						gradient_failure_reason = f"params_nonfinite: mu={mu}, sig={sig_core}, lam={lam}"
					elif not isinstance(pcov, np.ndarray) or pcov.shape != (5,5):
						gradient_failure_reason = f"pcov_wrong_shape: {type(pcov)}, shape={getattr(pcov, 'shape', None)}"
					elif not np.all(np.isfinite(pcov)):
						gradient_failure_reason = "pcov_contains_nonfinite"
					else:
						gradient_failure_reason = "unknown_precondition"
					
		except Exception as e:
			if gradient_failure_reason is None:
				gradient_failure_reason = f"unexpected_exception: {str(e)[:30]}"
				
		# --- Centers and widths ---
		# x_o := mu (true Gaussian peak) -> goes to CSV
		xo_out = mu
		xo_var = mu_var
		xo_err = np.sqrt(xo_var) if np.isfinite(xo_var) and xo_var >= 0 else np.nan

		# Define burst center x_c := mu + 1/lambda (mean) if lambda>0, else fallback to mu
		if (lam is not None and np.isfinite(lam) and lam > 0):
			xc_out = mu + 1.0/lam
			# Var(x_c) ≈ Var(mu) + (1/lam^4) Var(lam) - 2*(1/lam^2) Cov(mu,lam)
			xc_var = (mu_var if np.isfinite(mu_var) else 0.0) \
					 + ((lam_var if np.isfinite(lam_var) else 0.0) / (lam**4)) \
					 - (2.0 * (mu_lam_c if np.isfinite(mu_lam_c) else 0.0) / (lam**2))
		else:
			xc_out = mu
			xc_var = mu_var
		xc_err = np.sqrt(xc_var) if np.isfinite(xc_var) and xc_var >= 0 else np.nan

		# core sigma (as in old pipeline) for red underline at x_o
		sigma_core = abs(sig_core) if np.isfinite(sig_core) else np.nan
		sigma_core_err = np.sqrt(sig_var) if np.isfinite(sig_var) and sig_var >= 0 else np.nan

		# tail-aware std: s = sqrt(sig^2 + 1/lam^2) for purple underline and windowing
		if (lam is not None and np.isfinite(lam) and lam > 0 and np.isfinite(sig_core)):
			sigma_emg_std = float(np.sqrt(sig_core**2 + (1.0/lam**2)))
			# propagate with covariance:
			# ds/dsig = sig/s, ds/dlam = (-1)/(lam^3 * s)
			ds_dsig = sigma_core / sigma_emg_std
			ds_dlam = -1.0 / ( (lam**3) * sigma_emg_std )
			var_s = 0.0
			if np.isfinite(sig_var):  var_s += (ds_dsig**2) * sig_var
			if np.isfinite(lam_var):  var_s += (ds_dlam**2) * lam_var
			if np.isfinite(sig_lam_c): var_s += 2.0 * ds_dsig * ds_dlam * sig_lam_c
			sigma_emg_std_err = np.sqrt(var_s) if np.isfinite(var_s) and var_s >= 0 else np.nan
		else:
			sigma_emg_std = sigma_core
			sigma_emg_std_err = sigma_core_err
				
	
		# === FALLBACK: Use simple approximation when propagation fails ===
		if not np.isfinite(t_emg_hwhm_err):
			# Use a fraction of the pulse width as uncertainty
			if np.isfinite(sigma_emg_std) and sigma_emg_std > 0:
				t_emg_hwhm_err = 0.2 * sigma_emg_std  # 20% of width
				fallback_used = "sigma_emg_std"
			elif np.isfinite(sig_core) and sig_core > 0:
				t_emg_hwhm_err = 0.2 * sig_core  # 20% of core width
				fallback_used = "sig_core"
			elif np.isfinite(total_duration):
				t_emg_hwhm_err = 0.01 * total_duration  # 1% of duration
				fallback_used = "duration"
			else:
				t_emg_hwhm_err = np.nan
				fallback_used = "none"
			
			# Report fallback usage (suppressed when stacking is enabled)
			if not USE_SIGNAL_STACKING and (i % 50 == 0 or True):  # Change True to False to reduce spam
				print(f"  Row {i} (freq {freqs[i]:.2f}): Fallback uncertainty used ({fallback_used})")
				if gradient_failure_reason:
					print(f"    Reason: {gradient_failure_reason}")

		
		
#         # PREVIOUS EMG ERROR BLOCK
#         # --- NEW: numerical uncertainty for EMG arrival time (left HWHM) ---
#         t_emg_hwhm_err = np.nan
#         try:
#             # need covariance on [mu, sigma, lambda]
#             if np.all(np.isfinite([mu, sig_core, lam])) and lam > 0 and isinstance(pcov, np.ndarray) and pcov.shape == (5,5):
#                 # finite-diff steps
#                 rel = 1e-3
#                 dmu   = rel * max(abs(mu), 1e-6)
#                 dsig  = rel * max(abs(sig_core), 1e-6)
#                 dtau  = rel * max(1.0/lam, 1e-6)

#         #         # If the calculation produced NaN or extreme values, use fallback
#         #         if not np.isfinite(t_emg_hwhm_err) or t_emg_hwhm_err > 10.0 * total_duration:
#         #             # Fallback: use a fraction of the pulse width as uncertainty estimate
#         #             t_emg_hwhm_err = 0.1 * sigma_emg_std  # 10% of pulse width
#         # except Exception:
#         #     # Fallback on error
#         #     if np.isfinite(sigma_emg_std):
#         #         t_emg_hwhm_err = 0.1 * sigma_emg_std
#         #     else:
#         #         t_emg_hwhm_err = 0.01  # last-resort fallback (10 µs)



#                 # central diffs for x_left wrt mu, sigma, tau
#                 def xleft(mu_, sig_, tau_):
#                     return float(emg_left_hwhm(mu=float(mu_), sigma=float(sig_), tau=float(tau_))["x_left"])

#                 # ∂x/∂mu
#                 g_mu  = (xleft(mu+dmu, sig_core, 1.0/lam) - xleft(mu-dmu, sig_core, 1.0/lam)) / (2*dmu)
#                 # ∂x/∂sigma
#                 g_sig = (xleft(mu, sig_core+dsig, 1.0/lam) - xleft(mu, sig_core-dsig, 1.0/lam)) / (2*dsig)
#                 # ∂x/∂tau, then convert to ∂x/∂lambda via d(tau)/d(lambda) = -1/lambda^2
#                 g_tau = (xleft(mu, sig_core, 1.0/lam+dtau) - xleft(mu, sig_core, 1.0/lam-dtau)) / (2*dtau)
#                 g_lam = g_tau * (-1.0 / (lam**2))

#                 # gradient vector J wrt parameters in pcov order [A, mu, sigma, lambda, C]
#                 J = np.array([0.0, g_mu, g_sig, g_lam, 0.0], dtype=float)

#                 # variance = J^T * Cov * J
#                 var_x = float(J @ pcov @ J)
#                 t_emg_hwhm_err = np.sqrt(var_x) if (np.isfinite(var_x) and var_x >= 0) else np.nan
#         except Exception:
#             t_emg_hwhm_err = np.nan
# #########################################

		# --- Window edges built from tail-aware width ---
		tstart = xc_out - k_window * sigma_emg_std
		tend   = xc_out + k_window * sigma_emg_std
		# propagate ignoring cross-cov(xo, s) like the original code:
		tstart_err = np.sqrt(
			(xc_err if np.isfinite(xc_err) else 0.0)**2 +
			(k_window**2) * (sigma_emg_std_err if np.isfinite(sigma_emg_std_err) else 0.0)**2
		)

		# basic amplitude errors (A from diag), keep naming consistent with original
		amp_err = perr[0] if (perr.size >= 1 and np.isfinite(perr[0])) else np.nan


		# --- Save row (17 columns total) ---
		fitdata[i, :] = [
			freqs[i],             # 1  freqs
			tstart,               # 2  tstart  (tail-aware window)
			tend,                 # 3  tend    (tail-aware window)
			A,                    # 4  amp     (EMG amplitude scale)
			xo_out,               # 5  xo      (EMG mean)
			tstart_err,           # 6  tstart_err
			sigma_core,           # 7  sigma   (core Gaussian ÃÆ’; back-compat)
			redchisq,             # 8  red_chisq
			amp_err,              # 9  amp_err
			xo_err,               # 10 xo_err  (propagated for mean)
			sigma_core_err,       # 11 sigma_err (core)
			lam,                  # 12 lambda (decay rate)
			tau,                  # 13 tau = 1/lambda
			lambda_err,           # 14 lambda_err
			tau_err,              # 15 tau_err
			sigma_emg_std,        # 16 sigma_emg_std (tail-aware)
			sigma_emg_std_err,    # 17 sigma_emg_std_err
			t_emg_hwhm,           #EMG per-row arrival time (left HWHM crossing)
			t_emg_hwhm_err        # Numerical EMG ToA uncertainty (ms)
		]

		# --- Optional diagnostics/plots (EMG model & residuals) ---
		should_plot = (plot and isinstance(plot, bool)) or (isinstance(plot, list) and int(freqs[i]) in plot)
		if should_plot and np.all(np.isfinite([A, mu, sig_core, lam, C])):
			xs = np.linspace(0, total_duration, num=len(row))
			# noise sigma from early off-pulse slice
			off_n = max(len(row) // NOISE_WIDTH_FACTOR, 1)
			data_sigma = [np.std(row[:off_n])] * len(row)
			residuals = (row - emg_model(xs, *popt)) / data_sigma

			fig, axs = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1])
			axs[0].plot(xs, row, label='data')
			axs[0].plot(xs, emg_model(xs, *popt),
						label=r'$A,\ \mu,\ \sigma,\ \lambda,\ C$ = ' + ', '.join([f"{float(p):.3f}" for p in popt]))
			axs[0].axvline(x=tstart, c='k', linestyle='-.',
						   label=f'$t_\\text{{start}}$ = {scilabel(tstart, tstart_err)}')
			axs[0].axvline(x=xs[off_n],
						   label=rf'noise boundary ($\sigma_{{data}} = $ {data_sigma[0]:.2f})',
						   c='r', linestyle='--')
			axs[0].axvline(x=min(xs), alpha=0, label=rf"$\chi^2_{{red}}$={redchisq:.3f}")
			axs[0].set_xlim(min(xs), max(xs))
			axs[0].set_title(f"{bname} freq: {freqs[i]:.2f} MHz")
			axs[0].legend(frameon=False)

			# residual diagnostics vs N(0,1)
			try:
				kstest = scipy.stats.kstest(residuals, scipy.stats.norm.cdf)
				if not (np.isnan(tstart_err) or np.isnan(np.min(residuals))):
					_, bins, _ = axs[1].hist(
						np.random.standard_normal(len(residuals)),
						bins='auto', label='Sampled $N(0,1)$', histtype='step',
						density=True, linewidth=2, range=(np.min(residuals), np.max(residuals))
					)
					nx = np.linspace(*axs[1].get_xlim())
					# standard normal pdf for reference (reuse existing gauss_model if available)
					if 'gauss_model' in globals():
						axs[1].plot(nx, gauss_model(nx, 1/np.sqrt(2*np.pi), 0, 1), 'k-', label='$N(0,1)$')
					else:
						axs[1].plot(nx, np.exp(-0.5*nx**2)/np.sqrt(2*np.pi), 'k-', label='$N(0,1)$')
					axs[1].hist(residuals, bins=bins, label='residuals', density=True)
					axs[1].plot(0, 'w.', alpha=0,
								label=f"K-S test $p = {kstest.pvalue:.2f}$\nstat={kstest.statistic:.2f}")
					axs[1].legend(frameon=False, fontsize='small')
			except Exception:
				pass

			axs[1].set_title("Distribution of normalized residuals")
			plt.tight_layout()
			plt.show(block=True)
			plt.close()
			
			# === ADD ADDITION 3 HERE (after line 804, before line 805) ===
			# === SUMMARY: EMG fit quality ===
			print(f"\n=== EMG Fit Summary for {bname} ===")
			valid_fits = np.isfinite(fitdata[:, 7])  # red_chisq column
			print(f"Valid fits: {valid_fits.sum()}/{n_rows} ({100*valid_fits.sum()/n_rows:.1f}%)")
			
			# Analyze t_emg_hwhm_err (column 18)
			t_err_finite = np.isfinite(fitdata[:, 18])
			print(f"Finite t_emg_hwhm_err: {t_err_finite.sum()}/{n_rows} ({100*t_err_finite.sum()/n_rows:.1f}%)")
			
			if t_err_finite.any():
				t_errs = fitdata[t_err_finite, 18]
				print(f"  t_emg_hwhm_err range: [{t_errs.min():.6f}, {t_errs.max():.6f}]")
				print(f"  median: {np.median(t_errs):.6f}")
			
			# Analyze covariance issues by checking all error columns
			all_errs = fitdata[:, [5, 8, 9, 10, 13, 14, 16, 18]]  # All error columns
			rows_with_all_finite_errs = np.all(np.isfinite(all_errs), axis=1)
			print(f"Rows with all finite uncertainties: {rows_with_all_finite_errs.sum()}/{n_rows}")
			
			# Check parameter ranges
			if valid_fits.any():
				sigmas = fitdata[valid_fits, 6]
				lambdas = fitdata[valid_fits, 11]
				print(f"\nParameter ranges for valid fits:")
				print(f"  sigma: [{np.nanmin(sigmas):.6f}, {np.nanmax(sigmas):.6f}], median: {np.nanmedian(sigmas):.6f}")
				print(f"  lambda: [{np.nanmin(lambdas):.6f}, {np.nanmax(lambdas):.6f}], median: {np.nanmedian(lambdas):.6f}")
				
				# Flag extreme values
				extreme_sigma = (sigmas < 0.001) | (sigmas > 10)
				extreme_lambda = (lambdas < 0.1) | (lambdas > 1000)
				if extreme_sigma.any():
					print(f"  WARNING: {extreme_sigma.sum()} rows have extreme sigma values")
				if extreme_lambda.any():
					print(f"  WARNING: {extreme_lambda.sum()} rows have extreme lambda values")
						


	# assemble DataFrame with labeled columns
	return pd.DataFrame(
		data=fitdata,
		columns=[
			'freqs',
			'tstart',              # from xo (mean) and sigma_emg_std
			'tend',                # from xo (mean) and sigma_emg_std
			'amp',                 # EMG amplitude A
			'xo',                  # EMG mean (mu + 1/lambda)
			'tstart_err',
			'sigma',               # EMG core Gaussian ÃÆ’ (back-compat)
			'red_chisq',
			'amp_err',
			'xo_err',
			'sigma_err',           # error on core ÃÆ’
			'lambda',              # EMG decay rate
			'tau',                 # 1/lambda
			'lambda_err',
			'tau_err',
			'sigma_emg_std',       # tail-aware width
			'sigma_emg_std_err',
			't_emg_hwhm',           # arrival time used in EMG mode
			't_emg_hwhm_err'      # Numerical EMG ToA uncertainty
		]
	)

def get_stack_channels(stack_idx, stack_size, n_freq):
	"""
	Get the channel indices that belong to a given stack.
	
	Parameters:
	-----------
	stack_idx : int
		Index of the stack (0-indexed)
	stack_size : int
		Number of channels per stack
	n_freq : int
		Total number of frequency channels
		
	Returns:
	--------
	list : List of channel indices in this stack
	"""
	start_idx = stack_idx * stack_size
	end_idx = min((stack_idx + 1) * stack_size, n_freq)
	return list(range(start_idx, end_idx))

def should_ignore_stack(stack_idx, stack_size, n_freq, ignore_channels):
	"""
	Check if a stack should be ignored based on the channels it contains.
	
	Parameters:
	-----------
	stack_idx : int
		Index of the stack (0-indexed)
	stack_size : int
		Number of channels per stack
	n_freq : int
		Total number of frequency channels
	ignore_channels : list
		List of channel indices to ignore
		
	Returns:
	--------
	bool : True if the stack should be ignored (contains any ignored channel)
	"""
	if not ignore_channels:
		return False
	
	# Expand any range objects in ignore_channels
	expanded_ignore = []
	for item in ignore_channels:
		if hasattr(item, '__iter__') and not isinstance(item, (int, np.integer)):
			expanded_ignore.extend(list(item))
		else:
			expanded_ignore.append(item)
	
	stack_channels = get_stack_channels(stack_idx, stack_size, n_freq)
	return any(ch in expanded_ignore for ch in stack_channels)

def stack_channels(wfall, stack_size):
	"""
	Stack waterfall into non-overlapping blocks of channels.
	
	Parameters:
	-----------
	wfall : ndarray (n_freq, n_time)
		Waterfall data
	stack_size : int
		Number of channels to stack together
		
	Returns:
	--------
	stacked_profiles : ndarray (n_stacks, n_time)
		Stacked profiles (sum of channels)
	stack_center_indices : ndarray (n_stacks,)
		Center channel index for each stack
	"""
	n_freq = wfall.shape[0]
	n_stacks = int(np.ceil(n_freq / stack_size))
	n_time = wfall.shape[1]
	
	stacked_profiles = np.zeros((n_stacks, n_time))
	stack_center_indices = np.zeros(n_stacks)
	
	for i in range(n_stacks):
		start_idx = i * stack_size
		end_idx = min((i + 1) * stack_size, n_freq)
		
		# Sum the channels in this block
		stacked_profiles[i, :] = np.sum(wfall[start_idx:end_idx, :], axis=0)
		
		# Center frequency index of this stack
		stack_center_indices[i] = (start_idx + end_idx - 1) / 2.0
		
	return stacked_profiles, stack_center_indices.astype(int)

def visualize_stacked_waterfall(stacked_profiles, stack_center_indices, freqs_full, 
								 res_time_ms, duration, stack_size, burst_name=""):
	"""
	Create a visualization of the stacked waterfall in a separate window.
	
	Parameters:
	-----------
	stacked_profiles : ndarray (n_stacks, n_time)
		Stacked intensity profiles (sum of channels)
	stack_center_indices : ndarray (n_stacks,)
		Center channel indices for each stack
	freqs_full : ndarray
		Full frequency array for all channels
	res_time_ms : float
		Time resolution in ms
	duration : float
		Total duration in ms
	stack_size : int
		Number of channels stacked together
	burst_name : str
		Name of burst for title
	"""
	n_stacks = stacked_profiles.shape[0]
	n_time = stacked_profiles.shape[1]
	
	# Create frequency array for stacked channels
	stacked_freqs = freqs_full[stack_center_indices]
	
	# Create time axis
	time_axis = np.linspace(0, duration, n_time)
	
	# Create new figure window
	fig, ax = plt.subplots(figsize=(12, 8))
	
	# Calculate extent for imshow
	extent = [
		0,                          # time start
		duration,                   # time end
		stacked_freqs.min(),        # freq start
		stacked_freqs.max()         # freq end
	]
	
	# Plot stacked waterfall
	im = ax.imshow(
		stacked_profiles,
		aspect='auto',
		origin='lower',
		interpolation='none',
		cmap='viridis',
		extent=extent,
		vmax=np.quantile(stacked_profiles, 0.999)
	)
	
	# Add colorbar
	cbar = plt.colorbar(im, ax=ax, label='Summed Intensity')
	
	# Labels and title
	ax.set_xlabel('Time (ms)', fontsize=12)
	ax.set_ylabel('Frequency (MHz)', fontsize=12)
	ax.set_title(f'Stacked Waterfall Visualization ({stack_size} channels per stack)\n{burst_name}', 
				 fontsize=14, fontweight='bold')
	
	# Add annotation showing stacking info
	info_text = f"Stacks: {n_stacks}\nChannels/stack: {stack_size}\nTotal time bins: {n_time}"
	ax.text(0.02, 0.98, info_text,
			transform=ax.transAxes,
			verticalalignment='top',
			bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
			fontsize=10)
	
	plt.tight_layout()
	plt.show(block=False)  # Non-blocking so it doesn't pause execution
	
	return fig


# === ADD ADDITION 4 HERE (between lines 868-870) ===
def diagnose_emg_row(row, dt, freq_mhz, total_duration=None, plot=True):
	"""
	Diagnose why an EMG fit might produce NaN uncertainties.
	
	Parameters:
	- row: 1D array of intensities
	- dt: time resolution (ms)
	- freq_mhz: frequency of this row (MHz)
	- total_duration: total duration in ms (if None, computed from row length * dt)
	- plot: whether to show diagnostic plots
	"""
	if total_duration is None:
		total_duration = len(row) * dt
	
	print(f"\n{'='*60}")
	print(f"Diagnosing EMG fit for frequency {freq_mhz:.2f} MHz")
	print(f"{'='*60}")
	
	# Fit the row
	popt, pcov, redchisq = fitemg(row, total_duration)
	A, mu, sig_core, lam, C = popt
	
	print(f"\n1. FIT PARAMETERS:")
	print(f"   A (amplitude): {A:.4f}")
	print(f"   mu (peak time): {mu:.4f} ms")
	print(f"   sigma (core width): {sig_core:.6f} ms")
	print(f"   lambda (decay rate): {lam:.4f} 1/ms")
	print(f"   C (baseline): {C:.4f}")
	print(f"   reduced chi-squared: {redchisq:.4f}")
	
	print(f"\n2. COVARIANCE MATRIX:")
	if isinstance(pcov, np.ndarray) and pcov.ndim == 2:
		print(f"   Shape: {pcov.shape}")
		print(f"   All finite? {np.all(np.isfinite(pcov))}")
		print(f"   Condition number: {np.linalg.cond(pcov):.2e}")
		print(f"   Diagonal (variances): {np.diag(pcov)}")
		print(f"   Any negative diagonal? {np.any(np.diag(pcov) < 0)}")
		
		# Check correlations
		if np.all(np.diag(pcov) > 0):
			corr_matrix = pcov / np.sqrt(np.outer(np.diag(pcov), np.diag(pcov)))
			print(f"\n   Correlation matrix:")
			print(f"   {corr_matrix}")
			max_corr = np.max(np.abs(corr_matrix - np.eye(5)))
			print(f"   Max off-diagonal correlation: {max_corr:.3f}")
	else:
		print(f"   PROBLEM: pcov is not a proper array")
		print(f"   Type: {type(pcov)}, Value: {pcov}")
	
	# Try to compute t_emg_hwhm
	try:
		result = emg_left_hwhm(mu=float(mu), sigma=float(sig_core), tau=float(1.0/lam))
		t_hwhm = result["x_left"]
		print(f"\n3. EMG LEFT HWHM:")
		print(f"   t_emg_hwhm: {t_hwhm:.6f} ms")
		print(f"   Finite? {np.isfinite(t_hwhm)}")
	except Exception as e:
		print(f"\n3. EMG LEFT HWHM CALCULATION FAILED:")
		print(f"   Error: {e}")
		t_hwhm = np.nan
	
	# Try numerical gradient
	print(f"\n4. NUMERICAL GRADIENT TEST:")
	if np.isfinite([mu, sig_core, lam, t_hwhm]).all():
		try:
			dmu  = max(1e-6, 0.001 * abs(mu))
			dsig = max(1e-6, 0.001 * abs(sig_core))
			dtau = max(1e-6, 0.001 / abs(lam))
			
			print(f"   Step sizes: dmu={dmu:.2e}, dsig={dsig:.2e}, dtau={dtau:.2e}")
			
			def xleft(mu_, sig_, tau_):
				return float(emg_left_hwhm(mu=float(mu_), sigma=float(sig_), tau=float(tau_))["x_left"])
			
			x_center = xleft(mu, sig_core, 1.0/lam)
			x_mu_p = xleft(mu+dmu, sig_core, 1.0/lam)
			x_mu_m = xleft(mu-dmu, sig_core, 1.0/lam)
			x_sig_p = xleft(mu, sig_core+dsig, 1.0/lam)
			x_sig_m = xleft(mu, sig_core-dsig, 1.0/lam)
			x_tau_p = xleft(mu, sig_core, 1.0/lam+dtau)
			x_tau_m = xleft(mu, sig_core, 1.0/lam-dtau)
			
			g_mu = (x_mu_p - x_mu_m) / (2*dmu)
			g_sig = (x_sig_p - x_sig_m) / (2*dsig)
			g_tau = (x_tau_p - x_tau_m) / (2*dtau)
			g_lam = g_tau * (-1.0 / (lam**2))
			
			print(f"   ∂x/∂mu: {g_mu:.4e}")
			print(f"   ∂x/∂sigma: {g_sig:.4e}")
			print(f"   ∂x/∂lambda: {g_lam:.4e}")
			print(f"   All finite? {np.isfinite([g_mu, g_sig, g_lam]).all()}")
			
			if isinstance(pcov, np.ndarray) and np.all(np.isfinite(pcov)):
				J = np.array([0.0, g_mu, g_sig, g_lam, 0.0])
				var_x = float(J @ pcov @ J)
				print(f"\n5. UNCERTAINTY PROPAGATION:")
				print(f"   var(x) = J^T @ Cov @ J: {var_x:.4e}")
				print(f"   Finite? {np.isfinite(var_x)}")
				print(f"   Positive? {var_x > 0}")
				if np.isfinite(var_x) and var_x > 0:
					t_err = np.sqrt(var_x)
					print(f"   t_emg_hwhm_err: {t_err:.6f} ms")
					print(f"   Relative uncertainty: {100*t_err/abs(t_hwhm):.1f}%")
				else:
					print(f"   PROBLEM: Cannot compute sqrt of variance")
			
		except Exception as e:
			print(f"   GRADIENT COMPUTATION FAILED: {e}")
	else:
		print(f"   Cannot test: prerequisites not finite")
	
	# Plot if requested
	if plot and np.isfinite([A, mu, sig_core, lam, C]).all():
		t = np.linspace(0, total_duration, len(row))
		model = emg_model(t, A, mu, sig_core, lam, C)
		
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
		
		ax1.plot(t, row, 'o', label='Data', alpha=0.5)
		ax1.plot(t, model, 'r-', label='EMG Fit', linewidth=2)
		if np.isfinite(t_hwhm):
			ax1.axvline(t_hwhm, color='g', linestyle='--', label=f'Left HWHM: {t_hwhm:.3f} ms')
		ax1.axvline(mu, color='purple', linestyle='--', label=f'mu: {mu:.3f} ms', alpha=0.5)
		ax1.set_xlabel('Time (ms)')
		ax1.set_ylabel('Intensity')
		ax1.set_title(f'EMG Fit for {freq_mhz:.2f} MHz (redchi={redchisq:.2f})')
		ax1.legend()
		ax1.grid(True, alpha=0.3)
		
		# Residuals
		residuals = row - model
		ax2.plot(t, residuals, 'o-', alpha=0.5)
		ax2.axhline(0, color='k', linestyle='--')
		ax2.set_xlabel('Time (ms)')
		ax2.set_ylabel('Residuals')
		ax2.set_title('Fit Residuals')
		ax2.grid(True, alpha=0.3)
		
		plt.tight_layout()
		plt.show()
	
	print(f"{'='*60}\n")
	
	return popt, pcov, redchisq




def plotburst(data, band, retfig=False, extent=None):
	fig, axs = plt.subplot_mosaic(
		'''
		T.
		WB
		''',
		figsize=(8, 7),
		width_ratios=[3,1],
		height_ratios=[1,3],
		gridspec_kw={
			'hspace': 0,
			'wspace': 0
		}
	)
	axs['W'].imshow(
		data,
		aspect='auto',
		origin='lower',
		interpolation='none',
		extent=extent,
		norm='linear',
		vmax=np.quantile(data, 0.999),
	)
	if not extent:
		extent = [0, data.shape[1], 0, data.shape[0]]
	axs['T'].plot(np.linspace(*extent[:2], num=data.shape[1]), np.nanmean(data, axis=0))
	axs['B'].stairs(band, np.linspace(*extent[2:], num=len(band)+1), orientation='horizontal')
	if extent:
		axs['W'].set_xlabel('Time (ms)')
		axs['W'].set_ylabel('Frequency (MHz)')

	axs['B'].yaxis.set_tick_params(labelleft=False)

	axs['T'].sharex(axs['W'])
	axs['B'].sharey(axs['W'])
	if retfig:
		return fig, axs
	else:
		plt.show(block=True)
		plt.close()

logdebug = False
def printd(*args):
	if logdebug: print(*args)

results_columns = [
	'name',
	'DM',
	't0 (ms)',
	't0_err',
	'center_f (MHz)',
	'center_f_err',
	'duration (ms)',
	'duration_err',
	'bandwidth (MHz)',
	'bandwidth_err',
	'dtdnu (ms/MHz)',
	'dtdnu_err',
	'redchisq_t',
	'redchisq_nu',
	'redchisq_dtdnu',
	'tb (ms)',
	'tb_err',
	'num_arrtimes',
	'model',
	'signal_stacking',
	'tau_sc (ms)',
	'tau_sc_err',
	# --- Scattering regime characterization (full-band EMG) ---
	'scattering_regime',    # 'weak', 'intermediate', 'strong', or 'unknown'
	'tau_over_sigma',       # ratio tau/sigma from full-band fit
	'fwhm_weak (ms)',       # 2.355 * sigma
	'fwhm_intermediate (ms)', # 2.355 * sqrt(sigma^2 + tau^2)
	'fwhm_strong (ms)',     # tau * ln(2)
	'fwhm_emg (ms)',        # numerically computed true EMG FWHM
	'fwhm_emg_err (ms)',    # 1σ uncertainty on fwhm_emg (analytic
	                        #  linear propagation through (σ, λ) with
	                        #  the σ-λ covariance from the EMG fit)
	'fwhm_reported (ms)',   # FWHM matching the assigned regime
	'below_native_res',     # True if sigma or tau < native telescope resolution
	# --- Sub-band scattering index recovery (EMG only) ---
	# v56_10: ALL FOUR (index × σ) variations are always recorded. To avoid
	# new columns, the α/τ_ref columns are 2-lists over the INDEX axis
	# [idx_free, idx_fixed] (idx_fixed pins α≈-4); the plain columns are the
	# σ-FREE family and the *_sigma_constrained columns are the σ-FIXED family.
	# Per-sub-band point columns (sb_tau_per_sb, sb_nu_ratio_kept) are instead
	# 2-lists over the σ axis [σ-free, σ-fixed]. Parse with a NaN-aware
	# evaluator (entries may be `nan`). See the encoding block near results.append.
	'sb_alpha',             # 2-list [idx_free, idx_fixed] α @ σ-free
	'sb_alpha_err',         # 2-list [idx_free, idx_fixed] α-err @ σ-free
	'sb_tau_ref (ms)',      # 2-list [idx_free, idx_fixed] τ_ref @ σ-free
	'sb_tau_ref_err',       # 2-list [idx_free, idx_fixed] τ_ref-err @ σ-free
	'sb_sigma_hi (ms)',     # σ from highest-frequency sub-band
	'sb_freq_center (MHz)', # reference frequency for τ_ref
	'sb_n_subbands',        # number of good sub-band fits
	'sb_regime_grid',       # closest-to-grid regime from sub-band (σ_hi, τ_ref)
	'sb_fwhm_grid (ms)',    # numerically exact FWHM from sub-band values
	'sb_fwhm_reported (ms)', # analytical FWHM closest to grid FWHM
	'sb_alpha_restricted',   # v56_10: PRIMARY variation tag (e.g. 'idx_free,sig_free'); was 'Yes'/'No'
	'bw_override_used',      # 'Yes' if bw_channel_override was used (manual bandwidth), else 'No'
	'bw_ToA_override_used',  # 'Yes' if bw_ToA_override was used (hard-coded slope-fit channel range), else 'No'
	# Comma-separated list of user-specified 1-indexed sub-band labels
	# excluded from the τ(ν) and ε(σ) fits (1 = highest-frequency
	# sub-band, descending). Empty string when no sub-bands were
	# ignored for this component.
	'sb_ignored_subindex',
	# --- Smoothed σ(ν) power-law fit (ε) ---
	# σ(ν) = A · (ν/sb_freq_center)^β with β ≤ 0 enforced. A is in ms (σ at
	# the same ν_ref used for τ_ref); β is dimensionless. Provides a
	# monotonic smoothed-σ envelope across the band, intended for
	# combination with the τ(ν) scattering envelope downstream.
	'eps_sigma_amp (ms)',    # A: σ at sb_freq_center from the ε power-law fit
	'eps_sigma_amp_err',     # 1σ uncertainty on A (ms)
	'eps_sigma_beta',        # β: power-law exponent (dimensionless, ≤ 0)
	'eps_sigma_beta_err',    # 1σ uncertainty on β
	# --- σ-constrained sub-band diagnostic (Option C; v56_10 `constrain_sigma`) ---
	# Populated only when `constrain_sigma` was set for this component.
	# The free-σ fit's α / τ_ref / per-sub-band τ values continue to live
	# in the `sb_*` columns above (unchanged headline); these columns
	# report the σ-constrained re-fit alongside, plus the per-sub-band
	# Δτ/σ_τ ratios that measure how much σ–τ trading was happening.
	# All NaN / empty when the diagnostic was not requested.
	'sigma_constrained_value (ms)',      # σ value used to pin the diagnostic fit
	'sb_alpha_sigma_constrained',         # α from the σ-constrained τ(ν) fit
	'sb_alpha_err_sigma_constrained',     # 1σ on the constrained α
	'sb_tau_ref_sigma_constrained (ms)',  # τ_ref from the constrained fit
	'dtau_over_sigma_tau_median',         # median |Δτ|/σ_τ across sub-bands
	'dtau_over_sigma_tau_max',            # max |Δτ|/σ_τ across sub-bands
	'dtau_over_sigma_tau_per_sb',         # python-list-ish string, e.g. "[0.50, 0.80, 1.00, 0.20]"
	# --- Per-sub-band τ(ν) raw values (for downstream auto-classifier) ---
	# `sb_tau_per_sb` and `sb_nu_per_sb` are aligned Python-list-ish
	# strings (e.g. "[1.95, 2.39, 2.09, ...]" and "[1006.86, 1022.55, ...]").
	# They list every sub-band that survived the σ-free per-sub-band
	# EMG fit's quality cuts, ordered by ascending frequency. Empty
	# string when no sub-band fits succeeded.
	#
	# `sb_nu_ratio_kept` is max(ν) / min(ν) computed over only the
	# sub-bands KEPT after `ignore_subindex` filtering (i.e. the
	# point set that actually fed the α power-law fit). This is the
	# tight ν-leverage available to the α measurement, distinct from
	# the full-band `bandwidth`/`center_f` ratio which over-estimates
	# leverage when sub-bands at the band edges were ignored.
	# Equals NaN when fewer than 2 sub-bands were kept.
	'sb_tau_per_sb',
	'sb_tau_err_per_sb',    # 2-list [σ-free errs, σ-fixed errs], parallel to sb_tau_per_sb (for weighted fits / error bars)
	'sb_nu_per_sb',
	'sb_nu_ratio_kept',
	# --- Intensity-profile fit diagnostics (for BIC/AIC model selection) ---
	'n_profile_bins',        # number of time samples in the integrated time-series fit
	'n_params_t',            # number of free parameters in the tmix fit (3*n_bursts+1 Gaussian, 4*n_bursts+1 EMG)
	# --- ACF (2D-autocorrelation) sub-burst slope, per component ---
	# Populated only when ACF mode ran for the component; NaN/False
	# otherwise. The slope (`acf_dtdnu`) is the ACF measurement, in the
	# pipeline's dt/dnu (ms/MHz) convention. The `duration (ms)` column is
	# UNCHANGED (still EMG/Gaussian-derived) so the sub-burst slope law
	# plots slope=acf_dtdnu vs duration=EMG width.
	'ACF',                  # bool: True if ACF mode ran for this component
	'acf_dtdnu (ms/MHz)',   # ACF slope, pipeline dt/dnu convention (signed)
	'acf_dtdnu_err',        # 1-sigma error on acf_dtdnu
	'acf_slope (MHz/ms)',   # raw FRBGui slope dnu/dt (provenance/audit)
	'acf_theta',            # resolved major-axis angle (rad)
	'acf_theta_err',        # 1-sigma error on acf_theta
	'acf_sigma_t (ms)',     # ACF burst-frame time sigma (ACF sigma / sqrt2)
	'acf_sigma_nu (MHz)',   # ACF burst-frame freq sigma (ACF sigma / sqrt2)
	'acf_redchisq',         # reduced chi^2 of the 2D Gaussian to the ACF,
	                        #  driftrate/FRBGui-native value (unnormalised
	                        #  ACF, first-50-columns noise). This is the
	                        #  value shown in the bottom-panel title.
	'acf_redchisq_norm',    # alternative reduced chi^2 on the zero-lag-
	                        #  normalised ACF with a corner-based noise
	                        #  estimate. Neither is a calibrated absolute
	                        #  goodness-of-fit -- both are relative only.
	# --- Ridge (centroid) ACF drift: Var_nu/Cov of the same 2D-Gaussian fit,
	#     well-behaved where the major-axis acf_dtdnu above blows up. ---
	'acf_ridge_dtdnu (ms/MHz)',   # ridge drift, pipeline dt/dnu convention (signed)
	'acf_ridge_drift (MHz/ms)',   # ridge drift dnu/dt (= central value of the fit)
	'acf_ridge_dtdnu_err',        # 1-sigma error on acf_ridge_dtdnu (ms/MHz), from the noise bootstrap
	'acf_ridge_drift_err',        # 1-sigma error on the drift (MHz/ms), from the noise bootstrap
	# --- Effective ACF dt/dnu: ridge where the major axis is blown up, major
	#     otherwise. The single trustworthy ACF slope for downstream use. ---
	'acf_effective_dtdnu (ms/MHz)',  # chosen ACF dt/dnu (ridge or major)
	'acf_effective_dtdnu_err',       # 1-sigma error on acf_effective_dtdnu
	'acf_effective_source',          # which readout was chosen: 'ridge' | 'major' | 'none'
	# --- Diffractive scintillation bandwidth (Delta_nu_d), per component ---
	# Populated only when `scint` mode ran for the component; NaN/False
	# otherwise. Measured from the Lorentzian fit to the frequency ACF of the
	# per-component spectrum (scint_bw.py). `scint_tau_recip` is the reciprocity
	# inversion 2*pi*tau*Delta_nu_d = C1 (C1 = scint_C1, 1.16 Kolmogorov default).
	# `scint_resolved` is False when Delta_nu_d falls below ~2 channel widths,
	# in which case the value is an upper limit, not a measurement.
	'scint',                  # bool: True if scint mode ran for this component
	'scint_bw (MHz)',         # Delta_nu_d (decorrelation bandwidth, HWHM)
	'scint_bw_err',           # 1-sigma error (fit + finite-scintle, quadrature)
	'scint_tau_recip (ms)',   # tau from reciprocity inversion
	'scint_tau_recip_err',    # 1-sigma error on scint_tau_recip
	'scint_redchisq',         # reduced chi^2 of the Lorentzian ACF fit
	'scint_mod_index',        # modulation index m = sqrt(fit amplitude)
	'scint_n_scintles',       # B / Delta_nu_d (number of scintles across band)
	'scint_resolved',         # bool: Delta_nu_d >= ~2 channels (else upper limit)
]
# === FIT ALIASES (resolved after function definitions) ===
if USE_EMG_MIX:
	SINGLE_FIT_FUNC = fitemg          # time-integrated 1D fit
	MIX_FIT_FUNC    = fitemgmix       # multi-component time fit
	ROWS_FIT_FUNC   = fitrowsemg      # per-row arrival-time fit
	MIX_MODEL_FUNC  = emgmix_model    # plotting model for time mixture
else:
	SINGLE_FIT_FUNC = fitgauss
	MIX_FIT_FUNC    = fitgaussmix
	ROWS_FIT_FUNC   = fitrows
	MIX_MODEL_FUNC  = gaussmix_model
# === END ALIASES ===


# === MULTI-MODEL RUN EXPANSION ===
# Lets a single per-burst config dict request several `measureburst()`
# runs with different models (e.g. EMG and Gaussian), so that downstream
# model-selection (BIC/AIC/χ²_red comparison) has both rows available
# in the same results CSV.
#
# Schema (calling-script side):
#   files = {
#       '9-04.npz': {
#           'xos':  [27.9, 64.11],
#           'cuts': [47.62],
#           # ... all settings shared across runs ...
#           'model_runs': [
#               {'model': 'EMG',      'constrain_index': [True, True], 'subband_plot': [7,5]},
#               {'model': 'Gaussian'},
#           ],
#       },
#       '9-03.npz': {
#           'xos':  [...],
#           'model': 'EMG',          # no model_runs → single run, unchanged
#       },
#   }
#
# Recommended use in the calling script:
#   for filename, cfg in files.items():
#       for kwargs in arrivaltimes.expand_model_runs(cfg):
#           burst_results = arrivaltimes.measureburst(filename, ..., **kwargs)
#           ...
#           plt.show()    # blocks until plots are closed → sequential UI
#
# Rules:
#   * If `cfg` has no 'model_runs' key, returns [cfg] unchanged
#     (full backwards-compat — bursts not using the feature see no diff).
#   * If 'model_runs' is present, each entry is merged with the outer
#     config: outer keys provide defaults, inner keys override.
#   * Duplicate keys (present in both outer cfg and inner entry) → the
#     inner entry wins; a UserWarning is emitted so the user knows the
#     outer setting was shadowed.
#   * The 'model_runs' key itself is stripped from the merged dicts
#     (it's a meta-key, not a measureburst() kwarg).
#   * Each inner entry MUST contain a 'model' key (so the CSV's 'model'
#     column actually distinguishes the rows). Raises ValueError if not.
def expand_model_runs(cfg):
	"""
	Expand a per-burst config dict into one or more `measureburst()`
	kwargs dicts.

	Parameters
	----------
	cfg : dict
		Per-burst configuration as written in the calling script's
		`files = {...}` dict. May contain a 'model_runs' key (list of
		per-run override dicts) to request multiple model variants.

	Returns
	-------
	list of dict
		One kwargs dict per model run. Length 1 if 'model_runs' is
		absent. Each returned dict is a fresh copy — caller can mutate
		it freely without affecting the original `cfg`.

	Raises
	------
	ValueError
		If 'model_runs' is present but not a non-empty list/tuple, or
		if any inner entry is not a dict, or if any inner entry is
		missing the required 'model' key.
	"""
	# `warnings` is imported at module top alongside scipy/glob.
	# Don't re-import per user preferences.

	if not isinstance(cfg, dict):
		raise ValueError(
			f"expand_model_runs: expected a dict, got {type(cfg).__name__}: {cfg!r}"
		)

	if 'model_runs' not in cfg:
		# Backwards-compat fast path: single run, no expansion.
		return [dict(cfg)]

	runs = cfg['model_runs']
	if not isinstance(runs, (list, tuple)) or len(runs) == 0:
		raise ValueError(
			f"`model_runs` must be a non-empty list/tuple of dicts; got "
			f"{type(runs).__name__}: {runs!r}"
		)

	# Shared outer config = cfg minus the meta-key 'model_runs'. Copy
	# so the original is never mutated.
	_outer = {k: v for k, v in cfg.items() if k != 'model_runs'}

	expanded = []
	for _i, _entry in enumerate(runs):
		if not isinstance(_entry, dict):
			raise ValueError(
				f"`model_runs[{_i}]` must be a dict; got "
				f"{type(_entry).__name__}: {_entry!r}"
			)
		if 'model' not in _entry:
			raise ValueError(
				f"`model_runs[{_i}]` is missing the required 'model' key. "
				f"Each entry must specify 'model' (e.g. 'EMG' or 'Gaussian') "
				f"so the CSV's 'model' column distinguishes the runs."
			)
		# Warn on duplicate keys (outer key shadowed by inner). Use a
		# UserWarning so it's filterable but visible by default in
		# Jupyter/terminal output.
		_dupes = sorted(set(_outer.keys()) & set(_entry.keys()))
		if _dupes:
			warnings.warn(
				f"expand_model_runs: in run {_i} (model={_entry.get('model')!r}), "
				f"inner override(s) for key(s) {_dupes} shadow the outer config "
				f"value(s). The inner value(s) will be used.",
				UserWarning,
				stacklevel=2,
			)
		# Merge: outer first, inner overrides. dict-update semantics.
		_merged = dict(_outer)
		_merged.update(_entry)
		expanded.append(_merged)

	return expanded
# === END MULTI-MODEL RUN EXPANSION ===


def measureburst(
	filename,
	xos=[],
	cuts=[],
	sigmas=None,
	fix_xos=False,
	tolms=0.01,
	band_int_width_fac=1,
	targetDM=None,
	recalculateDM=False,
	correctTimes=False,
	downfactors=(1,1),
	trim_downfactors=None, # Trim frequency channels from the high-freq end ("top"/"Top"/"TOP") or low-freq end ("bottom"/"Bottom"/"BOTTOM") of the raw .npz waterfall so that wfall.shape[0] becomes divisible by downfactors[0]. The trim is the MINIMUM needed (= wfall.shape[0] % downfactors[0]). Trimmed channels are excluded from the dynamic spectrum plot and from ALL downstream fitting. Mask indices in `masks=[...]` remain on the original raw .npz grid (re-indexed internally to the trimmed grid). bw_channel_override raw indices likewise stay on the original raw grid. None (default) = no trim; the existing divisibility check raises if the dimension is not already divisible.
	subtractbg=False,
	bw_filter='data_cutoff',
	bw_width_factor=3, # burst based filter factors are likely to help a lot with complex and blended components, and avoids having to do submasks
	bw_channel_override=None, # Manually specify the burst bandwidth, bypassing the spectral Gaussian fit. Accepts either {'raw': (i_lo, i_hi)} (indices on the raw pre-downsample grid, internally mapped through downfactors[0]), {'downsampled': (i_lo, i_hi)} / {'ds': (...)} (indices already on the downsampled grid, used directly), or a bare tuple (i_lo, i_hi) for back-compat (treated as raw, with a prompt to switch to the explicit form). Use when the spectral profile is unreliable (truncated, noisy, etc.).
	bw_ToA_override=None,    # Manually restrict which channels' ToAs are eligible for the slope fit. Same input schema as bw_channel_override: {'raw': (i_lo, i_hi)} / {'ds': (i_lo, i_hi)} / bare tuple. Per-component support via list. Functions like bw_filter='model_width' but with a hard-coded channel range instead of the spectral Gaussian fit. Acts ONLY on the slope-fit ToA list (subdf / fit_df) — bw_channel_override still independently controls the sub-band α fit's channel range. ToAs outside the range are rendered as ignored (grey hollow circles), same visual treatment as the `ignore` filter.
	bw_manual=None,          # [NEW] Restrict the frequency band (MHz) over which the FULL-BAND intensity profile (tseries) is integrated before the duration/tau fit. None (default) = full band (existing behaviour). A bare (freq_lo, freq_hi) tuple in MHz pins the band for the whole burst. A per-component list e.g. [(600,720), None, (540,600)] is accepted for forward-compat but, because the time-domain fit is a SINGLE global mixture over one full-band tseries, the entries are collapsed to their UNION (min lo, max hi); None entries are skipped. Affects ONLY the tseries integration (duration & tau_sc) - the spectrum panel, slope-fit ToAs, and sub-band alpha fit are unaffected (use bw_filter / bw_ToA_override / bw_channel_override for those).
	subband_diagnostic=None, # Diagnostic mode: list of 1-indexed component indices to process. Per-channel ToA fits (the slow step, 60-90s per component) and sub-band α fits are SKIPPED for components NOT in this list. CSV writing is force-disabled. A red banner is drawn on the pop-up so you don't forget you're in diagnostic mode. Use during iterative tuning of component-specific settings (subband_plot, constrain_index, ignore_subindex, bw_channel_override) — once happy with each component's config, remove this kwarg and run the full pipeline. e.g. subband_diagnostic=[1] processes only component a. None (default) = process all components as usual.
	subbandslice=None,
	snr_cutoff=3,
	t_filter_factor=2,
	crop=None,
	# postcrop=None, # unimplemented
	masks=[],
	ignore=[],
	# ── Per-component arrival-time ignore (added v56_10) ─────────────
	# When `ignore` is a flat list of channel indices applied to every
	# component's slope fit, `ignore_per_comp` lets you exclude
	# channels FROM A SPECIFIC COMPONENT'S slope fit only. Useful when
	# one component has scintillation outliers or sub-burst structure
	# that contaminates only its arrival-time fit, while the other
	# component's ToAs at those frequencies are clean.
	#
	# Format: list of length `len(xos)`, one entry per component.
	# Each entry may be:
	#   - None or () or []        → no extra ignores for this component
	#   - a list/tuple/range      → channel indices to ignore for THIS
	#                                component's slope fit only
	#   - a nested iterable       → flattened to channel indices
	#                                (e.g. (range(0,153), 255) becomes
	#                                channels 0..152 plus 255)
	# Indices refer to the post-downsampling channel grid (same
	# convention as the burst-wide `ignore`).
	#
	# The effective ignore set for component i is the UNION of
	# `ignore` (burst-wide) and `ignore_per_comp[i]` (per-component).
	#
	# Example: 2-component burst, ignore channels 240-255 for the
	# slope fit of component b only:
	#   ignore=[],
	#   ignore_per_comp=[(), range(240, 256)],
	ignore_per_comp=None,
	submasks=None,
	bandmask_thres=None,
	measure_drift=True,
	show=True,
	figsize=(10, 8), # (10, 9) for paper, (10,8) for screen
	show_components=False,
	show_arrtime=False,
	cmap_norm='linear',
	cmap='viridis',
	save=True,
	outdir='',
	outfmt='.png',
	return_arrivaltimes=False,
	return_fig=False,
	loadonly=False,
	save_solutions=False,
	load_solutions=None,
	hide_legend=False,
	legendloc=1,
	label_components=False,
	pslopeidx=-1,
	tpoint='tstart',
	model='Gaussian',
	stacking=False,
	stack_size=5,
	diagnostics=False,
	subband_plot=False,
	constrain_index=False,
	ignore_subindex=None,
	constrain_sigma=None,
	ACF=False, # Per-component 2D-autocorrelation slope mode. Accepts the same shapes as subband_plot: False (off, default), True (all components), or a list. A list may be per-component bools (length 1 or n components) OR a list of 1-indexed component indices to enable. When ACF is on for a component, the per-channel ToA fit AND stacking are REPLACED by a 2D-autocorrelation slope measurement (driftrate.processBurst). The EMG/Gaussian intensity-profile fit is unchanged, so `duration (ms)` and `tau_sc` still come from that fit. The ACF slope is written to new `acf_*` CSV columns (the existing `dtdnu` column is also set from the ACF result for ACF components). On fit failure the acf_* columns are NaN and a warning is printed.
	ACF_flag_axis_ratio=2.0,      # flag if resolution-normalised ACF Delta_t/Delta_nu exceeds this (major-axis blow-up symptom; ridge is robust to it)
	ACF_flag_slope_msMHz=4.0,     # plain-ACF: also flag if |dt/dnu| (ms/MHz) exceeds this (blow-up symptom)
	ACF_ridge=True,               # compute the ridge (centroid) ACF drift = Var_nu/Cov for every ACF component. Reads the same 2D-Gaussian fit, not the major-axis tan(theta), so narrow-band/scattered bursts do not blow up. Populates the acf_ridge_* CSV columns; leaves the legacy acf_dtdnu (major-axis) untouched.
	ACF_ridge_nboot=24,           # noise-bootstrap realisations for the ridge-drift error (the random half of acf_ridge_drift_err). 0 disables the bootstrap (error = scattering systematic only). Only runs for ACF components, so the cost is confined to those.
	ACF_ridge_blowup_msMHz=4.0,   # acf_effective_dtdnu: substitute the ridge for the major-axis slope when |major dt/dnu| (ms/MHz) exceeds this (blow-up symptom)
	ACF_ridge_ratio=3.0,          # acf_effective_dtdnu: ...or when the major-axis slope exceeds the ridge by this factor (self-calibrating). Ridge used only if it is well-defined (finite, error < magnitude).
	scint=False,                  # Per-component diffractive scintillation-bandwidth (Delta_nu_d) mode. Same shapes as ACF: False (off, default), True (all components), a per-component bool list (length 1 or n components), or a list of 1-indexed component indices. Independent of ACF mode; measured from the Lorentzian fit to the frequency ACF of the per-component spectrum (scint_bw.measure_scint_bw). Writes the scint_* CSV columns; leaves all existing columns untouched. NaN/False when off or the fit fails.
	scint_smooth_mhz=None,        # envelope smoothing scale (MHz) for scint mode; None -> 0.4 * band (unbiased when the burst fills a good fraction of the band). Set to ~(envelope width)/2 for genuinely narrow-band bursts so the Lorentzian wings are not absorbed (biases Delta_nu_d low).
	scint_C1=1.16,                # reciprocity constant in 2*pi*tau*Delta_nu_d = C1 (1.16 uniform Kolmogorov; 1.0 reproduces the simple-screen convention used by e.g. Marcote 2020).
):
	""" Measure spectro-temporal properties of a burst, and output a figure

	Compute the inverse sub-burst slope (dt/dnu) using the per-row arrival time method.

	Compute the duration and bandwidth by finding a 1-dimensional gaussian model
	to the integrated time series and spectrum, respectively.
	The duration and bandwidth are the 1 sigma widths of either fit.

	Compute the center frequency as the center of the 1d spectrum model.

	If multiple components are present, split them up and measure individually. The number
	of components to fit for is equal to ``len(xos)``.

	Args:
		filename (str): filename to .npz of a *dedispersed* burst waterfall. File is expected to follow frbgui's :ref:`burstformat`.
		xos (List[float] or 2-tuple of List[float], optional): List of times in ms of sub-burst centers.
			Can be approximate. If a 2-tuple, the second list is used as the location(s) to cut the waterfall.
			Using the ``cuts`` option instead is equivalent to using the 2 tuple option.
		cuts (List[float], optional): List of times in ms to cut the waterfall. Useful for blended bursts.
			User must make sure their cuts make sense (i.e. in between burst centers).
			Typically, if one cut is needed, then all bursts should be cut as well even if well separated.
		sigmas (List[float], optional): initial guesses for the width sigma when finding the 1-dimensional gaussian model
			to the time series. Must be the same length as ``xos``
		fix_xos (bool, optional): Default False. Whether or not to fix the passed ``xos`` when fitting the 1d model.
			Useful when bursts are blended and one can visually distinguish where a burst should be from the waterfall
			even if it appears completely absorbed in the integrated time series.
		tolms (float, optional): Tolerance in milliseconds to use when ``fix_xos`` is True. Default is 0.01 ms.
		band_int_width_fac (int, optional): When integrating the waterfall to determine the spectrum,
			the factor of ÃÆ’_t,1D to use that determines the on-pulse region. Default 1. Maximum 4.
		targetDM (float, optional): the DM (pc/cm^3) to perform the measurement at.
			Default is to perform the measurement at the DM of the npz file.
		recalculateDM (bool, optional): Recalculate the burst DM using ``DM_phase`` before measurement.
		correctTimes (bool, optional): Shift xos and cuts to account for dispersive shift
			when applying a targetDM other than the burst DM. Note that this shift will occur even when ``fix_xos`` is True.
		downfactors (tuple[int], optional): 2-tuple of factors to downsample by in frequency and time (respectively)
		subtractbg (bool, tuple[bool], optional): Perform a second background subtraction on subbursts.
			By default will do a background subtraction using 10% of channels on the whole waterfall.
			Pass ``(False, False)`` to skip both rounds of background subtraction.
		bw_filter (str, optional): The type of spectral/bandwidth filter to apply on arrival times. Default is ``'data_cutoff'``. Options are

			1. ``'data_cutoff'``: filter out arrival times in channels where the 1ÃÆ’ on-pulse mean amplitude
			is < 3 (see ``snr_cutoff``) times the noise standard deviation
			2. ``'model_cutoff'``: filter out arrival times in channels where the 1d spectral model amplitude
			is < 3 (see ``snr_cutoff``) times the noise standard deviation
			3. ``'model_width'``: filter out arrival times that lie beyond a multiple of the 1d spectral model width (ÃÆ’). See ``bw_width_factor``.
		bw_width_factor (int, optional): When using ``bw_filter=model_width``, 3ÃÆ’ of the burst bandwidth is applied as a spectral filter.
			For bursts with lots of frequency structure this may be inadequate,
			and this parameter can be used to override the filter width. It's recommended to try downsampling first. Note that a
			high ``bw_width_factor`` such as 10-15 likely indicates the bandwidth measurement is being understimated.
		subbandslice (slice, optional): Slice to use when sampling the subband noise standard deviation.
			Overrides ``NOISE_WIDTH_FACTOR``. Use python's slice function. E.g. ``slice(0,25)``
		snr_cutoff (int, optional): The S/N cutoff to use when ``bw_filter='data_cutoff'`` or ``bw_filter='model_cutoff'``.
			By default equals 3.
		t_filter_factor (int, optional): By default 2ÃÆ’ of the burst duration is applied as a temporal filter.
		outdir (str, optional): string of output folder for figures. Defaults to ''.
		crop (tuple[int], optional): pair of indices to crop the waterfall in time
		masks (List[int], optional): frequency indices to mask. Masks are applied before downsampling
		ignore (List[int], optional): frequency channel indices whose arrival times should be excluded from the slope
			calculation. Unlike ``masks``, ignored channels still contribute to the intensity profile fit and
			arrival times are still calculated - they are simply excluded from the dt/dnu slope fit.
			For stacked arrival times, if ANY channel in a stack is in the ignore list, that stack's arrival time
			is excluded. For example, if stack_size=5 and stack i contains channels [10,11,12,13,14], then
			``ignore=[12]`` will exclude that entire stack from the slope calculation.
			This is useful for removing unphysical arrival times from specific channels without affecting
			the overall intensity profile fit.
		submasks (tuple[List[int]], optional): tuple of length `xos` of lists of indices to mask on a subcomponent's waterfall.
			Note that contrary to ``masks``, these are applied after downsampling.
			Indices are scaled from the original size to the downsampled size and so can cover more than one channel.
			The length of ``submasks`` must match the length of ``xos``.
			Example: To specify a mask on the 4th component of a waterfall with 4 components, pass
			``submask=([],[],[],[22])``.
			This is also a good way to filter out misbehaving components in an otherwise well-measured waterfall
			and is useful for complicated bursts.
		bandmask_thres (float or list[float], optional): The intensity threshold in the integrated spectrum above which
			points should be ignored when performing burst fits. This is primarily useful for masking scintillation peaks in order to
			obtain more accurate burst bandwidth measurements. The appropriate level can be inferred from the output plot, or, if there are
			multiple components, using the plot output when ``show_components=True``, since the spectrum normally shown is integrated
			over all components. When a single value is specified, it will be applied to all components if there are more than one. To specify
			a different threshold for each component, pass a list of values, using ``None`` when no spectral masking is to be applied.
			For example: a 3-component waterfall may have ``bandmask_thres=[None, 0.1, 0.12]``. Note that this will only affect the bandwidth
			measurement and not interact with the arrival times in each channel.
		measure_drift (bool, optional): When True (default), and if ``len(xos) > 1`` (i.e. there are multiple burst components), will measure
			the drift rate using the times and center frequencies of the bursts to fit a line. Will also plot a corresponding
			line showing the drift rate measurement. Set to False to disable this behaviour.
		show (bool, optional): if True show interactive figure window for each file
		figsize (tuple, optional): figsize passed to matplotlib used for main output figure.
		show_components (bool, optional): if True show figure window for each sub-burst
		show_arrtime (bool or List[int], optional): if True, show a diagnostic plot with statistical
			and fit information for each frequency channel. If instead a list of frequency channels are provided,
			then only those frequency channels will be shown.
			Frequency channels listed need only be to the nearest integer.
			See also ``return_arrivaltimes`` option.
		cmap_norm (str, optional): The colormap normalization ``norm`` parameter passed to matplotlib's imshow command
			when plotting the waterfall. Default is 'linear', other options are 'log', 'symlog', 'logit',
			or matplotlib's Normalize class.
		cmap (str, optional): matplotlib colormap to use for waterfall
		return_arrivaltimes (bool, optional): If True, will return a dataframe of the arrival times per channel
		return_fig (bool, optional): if True, return the matplotlib figure. The figure will not be closed.
		save (bool, optional): if True save a figure displaying the measurements.
		loadonly (bool, optional): if True will perform loading steps such as masking, dedispersing,
			and downsampling, then return a tuple of wfall, freqs, times_ms, t_popt, DM, etc.
		outfmt (str, optional): string of file format to save figure as. Default is '.png'. Include the '.' character.
		save_solutions (bool, optional): setting to True will save a file inside of the folder specified by outdir
			that contains the fit solution data for the 1d time series and the spectrum of each component. Useful for reviewing
			measurements of bursts with many components that take a long time to analyse. Output filename will be of the form
			``f'{bname}.sols.npz'``. Default False
		load_solutions (str, optional): Filename of solutions file generated by ``save_solutions`` option. Default is None
		hide_legend (bool, optional): Hides the legend in the output if True.
		legendloc (int or str, optional): Set the location of the legend. Passed to matplotlib's loc argument when the legend is called.
		label_components (bool, optional): If True, label components filters in the time series plot. Useful for complicated waterfalls.
		pslopeidx (int, optional): Index of component to show in the time vs. frequency (i.e. slope) plot.
			Default is last component (-1).
		tpoint (str, optional): One of 'tstart' (default), 'tend', or 'xo'.
			Use to measure the slope based on the central peak in each channel, or based on the ending time in each channel.
		model (str, optional): Fitting model to use. Either 'Gaussian' (default) or 'EMG' (Exponentially Modified Gaussian).
			EMG model accounts for scattering and provides tau_sc (scattering timescale) measurements.
		stacking (bool or str, optional): Enable signal stacking to improve SNR. Default is False.
			Can also be 'True'/'False' or 'Yes'/'No' as strings.
		stack_size (int, optional): Number of frequency channels to stack together. Default is 5.
			Only used when stacking=True.
		diagnostics (bool, optional): Enable verbose diagnostic output including SNR calculations,
			filter statistics, and chi-squared analysis. Default is False.

	Returns:
		(list): ``results`` list of lists where each list is the result of the measurement.
			This array can be used to make a pandas dataframe in the following way:

			.. code-block:: python

				resultsdf = pd.DataFrame(
					data=results,
					columns=arrivaltimes.results_columns
				).set_index('name')

			where the columns of the dataframe are

			.. code-block:: python

				'name',
				'DM',
				'center_f (MHz)',
				'center_f_err',
				'duration (ms)',
				'duration_err',
				'bandwidth (MHz)',
				'bandwidth_err',
				'dtdnu (ms/MHz)',
				'dtdnu_err',
				'redchisq_t',     # reduced chi-squared of time series fit
				'redchisq_nu',    # reduced chi-squared of component-wise spectrum fit
				'redchisq_dtdnu', # reduced chi-squared of component-wise slope fit
				'tb (ms)', # t_b
				'tb_err'

		arrtimesdf (pd.DataFrame): Only returned when `return_arrivaltimes` is True.
		fig (matplotlib.fig): Matplotlib figure. Only returned when `return_fig` is True.

	"""
	# === SET MODEL AND STACKING FLAGS BASED ON PARAMETERS ===
	global USE_EMG_MIX, USE_GAUSS_MIX, USE_SIGNAL_STACKING, STACK_SIZE, SHOW_DIAGNOSTICS
	global SINGLE_FIT_FUNC, MIX_FIT_FUNC, ROWS_FIT_FUNC, MIX_MODEL_FUNC
	
	# Set diagnostics flag
	SHOW_DIAGNOSTICS = diagnostics

	# === SUBBAND DIAGNOSTIC MODE SETUP ===
	# If `subband_diagnostic` is provided, skip the expensive per-channel
	# ToA fits and sub-band α fits for components NOT in the list. The
	# mixture fit still runs over all components (it's a single global
	# fit, ~0.5s, and component i's σ depends on the joint optimisation
	# with all other components). Auto-disable CSV writing so iterative
	# tuning doesn't pollute the final spreadsheet. Print a clear banner.
	_diag_mode_active = False
	_diag_comp_set = None  # 0-indexed set of component indices to process
	if subband_diagnostic is not None:
		if not isinstance(subband_diagnostic, (list, tuple)) or len(subband_diagnostic) == 0:
			raise ValueError(
				f"`subband_diagnostic` must be a non-empty list/tuple of "
				f"1-indexed component indices, got {subband_diagnostic!r}"
			)
		try:
			# User supplies 1-indexed; convert to 0-indexed set
			_diag_comp_set = set(int(i) - 1 for i in subband_diagnostic)
		except (TypeError, ValueError) as _e:
			raise ValueError(
				f"`subband_diagnostic` entries must be integers (1-indexed "
				f"component indices), got {subband_diagnostic!r}: {_e}"
			)
		if any(i < 0 for i in _diag_comp_set):
			raise ValueError(
				f"`subband_diagnostic` indices are 1-indexed and must be ≥ 1, "
				f"got {subband_diagnostic!r}"
			)
		_diag_mode_active = True
		# Force CSV writing off — diagnostic mode is for iterative tuning
		save = False
		# Big visible banner
		_banner = ("╔" + "═"*78 + "╗\n"
		           "║" + " SUBBAND DIAGNOSTIC MODE ACTIVE ".center(78) + "║\n"
		           "║" + f" Processing only component(s) {sorted(int(i)+1 for i in _diag_comp_set)} (1-indexed) ".center(78) + "║\n"
		           "║" + " Per-channel ToA fits SKIPPED for other components ".center(78) + "║\n"
		           "║" + " CSV writing DISABLED ".center(78) + "║\n"
		           "╚" + "═"*78 + "╝")
		print(_banner, flush=True)
	# === END SUBBAND DIAGNOSTIC MODE SETUP ===

	# Set model flags
	if model.upper() == 'EMG':
		USE_EMG_MIX = True
		USE_GAUSS_MIX = False
		SINGLE_FIT_FUNC = fitemg
		MIX_FIT_FUNC = fitemgmix
		ROWS_FIT_FUNC = fitrowsemg
		MIX_MODEL_FUNC = emgmix_model
	else:  # Default to Gaussian
		USE_EMG_MIX = False
		USE_GAUSS_MIX = True
		SINGLE_FIT_FUNC = fitgauss
		MIX_FIT_FUNC = fitgaussmix
		ROWS_FIT_FUNC = fitrows
		MIX_MODEL_FUNC = gaussmix_model
	
	# Set stacking flag.
	# `stacking` and `stack_size` may now be per-component lists/tuples. We
	# normalize them later once n_bursts is known. For backward compatibility,
	# the module-level USE_SIGNAL_STACKING / STACK_SIZE retain "any-component"
	# semantics: USE_SIGNAL_STACKING is True if ANY component is stacked, and
	# STACK_SIZE is the first stack size (used only by code paths that don't
	# yet know which component is active).
	def _coerce_stacking_value(v):
		if isinstance(v, str):
			return v.lower() in ['true', 'yes', '1']
		return bool(v)

	if isinstance(stacking, (list, tuple)):
		_stacking_raw = [_coerce_stacking_value(v) for v in stacking]
		USE_SIGNAL_STACKING = any(_stacking_raw)
	else:
		USE_SIGNAL_STACKING = _coerce_stacking_value(stacking)
		_stacking_raw = None  # will broadcast after n_bursts is known

	# Set stack size
	if isinstance(stack_size, (list, tuple)):
		# OLD: int(None) crashes when a component opts out of stacking with
		# stack_size=None (paired with stacking=None/False for that component).
		# _stack_size_raw = [int(s) for s in stack_size]
		# NEW: None -> default 5. Inert: stack_size is ignored for any component
		# whose stacking flag is False/None, so the placeholder is never used.
		_stack_size_raw = [int(s) if s is not None else 5 for s in stack_size]
		STACK_SIZE = _stack_size_raw[0] if len(_stack_size_raw) > 0 else 5
	else:
		STACK_SIZE = int(stack_size)
		_stack_size_raw = None  # will broadcast after n_bursts is known
	
	# Parse subband_plot: True → 4 subbands, int → that many subbands
	if isinstance(subband_plot, (int, float)) and not isinstance(subband_plot, bool):
		_n_subbands = int(subband_plot)
		subband_plot = True
	else:
		_n_subbands = N_SUBBANDS_ALPHA  # default (4)
	
	# Determine arrival time convention label
	toa_convention = "Peak" if PEAK_INTENSITY_TOA else "Leading Edge"
	_stacking_display = _stacking_raw if _stacking_raw is not None else USE_SIGNAL_STACKING
	_stack_size_display = _stack_size_raw if _stack_size_raw is not None else STACK_SIZE
	print(f"Info: Model = {'EMG' if USE_EMG_MIX else 'Gaussian'}, Signal Stacking = {_stacking_display}, Stack Size = {_stack_size_display}, ToA Convention = {toa_convention}")
	# === END MODEL AND STACKING FLAGS ===
	
	# === COMPREHENSIVE DIAGNOSTIC HEADER ===
	if diag_enabled('MODEL_HEADER'):
		print("\n" + "="*80)
		print(f"MODEL COMPARISON DIAGNOSTICS: {'EMG' if USE_EMG_MIX else 'GAUSSIAN'} MODE")
		print("="*80)
	# === END HEADER ===

	if type(xos) == tuple:
		if len(xos) != 2:
			raise "Error: xos must be a list or tuple of two lists"
		cuts = xos[1]
		xos = xos[0]
	if type(xos) != list:
		raise "Error: xos must be a list"

	xos = sorted(xos)
	cuts = sorted(cuts)

	presubtractbg = True
	if type(subtractbg) == tuple:
		presubtractbg = subtractbg[0]
		subtractbg = subtractbg[1]

	if not (0 < band_int_width_fac <= 4):
		band_int_width_fac = 1
		print(f"Warn: {band_int_width_fac =} must be between 0 and 4. Setting to default of 1")

	results = []
	bname = filename.split('/')[-1].split('.npz')[0]
	_npz = np.load(filename, allow_pickle=True)
	# Convert NpzFile -> mutable dict so trim_downfactors can modify wfall/dfs/bandwidth in lockstep.
	data = {k: _npz[k] for k in _npz.files}
	_npz.close()
	wfall = np.copy(data['wfall'])
	print(f"Info: Measuring burst {bname} with {len(xos) if len(xos) != 0 else 1} components...")

	# === TRIM FREQUENCY CHANNELS TO MAKE WFALL DIVISIBLE BY downfactors[0] ===
	# Applied AFTER load, BEFORE dedispersion / masking / cropping / subsampling.
	# Trim is the minimum needed (= wfall.shape[0] % downfactors[0]).
	# wfall row 0 = low freq (bottom of plot, origin='lower'); last row = high freq (top of plot).
	# dfs is sorted ascending (low -> high) and aligned 1:1 with wfall rows.
	# Mask indices in `masks=[...]` stay on the ORIGINAL raw .npz grid; we re-index them onto
	# the trimmed grid immediately below so wfall[mask] = 0 hits the correct physical channels.
	_top_trim = 0   # channels removed from high-freq end (last rows of wfall, last entries of dfs)
	_bot_trim = 0   # channels removed from low-freq end (first rows of wfall, first entries of dfs)
	if trim_downfactors is not None:
		_tdf = str(trim_downfactors).strip().lower()
		if _tdf not in ('top', 'bottom'):
			raise ValueError(
				f"!!! BURST '{bname}': trim_downfactors must be 'Top', 'Bottom', or None "
				f"(case-insensitive). Got {trim_downfactors!r}."
			)
		_remainder = wfall.shape[0] % int(downfactors[0])
		if _remainder == 0:
			print(
				f">>> DEBUG: trim_downfactors={trim_downfactors!r} requested but wfall.shape[0]="
				f"{wfall.shape[0]} is already divisible by downfactors[0]={downfactors[0]}. "
				f"No trim applied."
			)
		else:
			if _tdf == 'top':
				_top_trim = _remainder
				_old_shape = wfall.shape
				wfall = wfall[:-_top_trim, :]
				data['wfall'] = data['wfall'][:-_top_trim, :]
				data['dfs']   = data['dfs'][:-_top_trim]
				# Bandwidth = (max_freq - min_freq) + one channel width (the .npz convention has
				# bin centers, so bandwidth covers half a channel beyond each extreme).
				# Easiest: recompute from the trimmed dfs assuming uniform channel spacing.
				_chw = (data['dfs'][-1] - data['dfs'][0]) / max(len(data['dfs']) - 1, 1)
				data['bandwidth'] = float((data['dfs'][-1] - data['dfs'][0]) + _chw)
				print(
					f">>> DEBUG: trim_downfactors='Top' -> trimmed {_top_trim} channel(s) from "
					f"high-freq end (last rows). wfall.shape {_old_shape} -> {wfall.shape}. "
					f"freq range now {data['dfs'][0]:.4f} -- {data['dfs'][-1]:.4f} MHz, "
					f"bandwidth -> {data['bandwidth']:.4f} MHz."
				)
			else:  # 'bottom'
				_bot_trim = _remainder
				_old_shape = wfall.shape
				wfall = wfall[_bot_trim:, :]
				data['wfall'] = data['wfall'][_bot_trim:, :]
				data['dfs']   = data['dfs'][_bot_trim:]
				_chw = (data['dfs'][-1] - data['dfs'][0]) / max(len(data['dfs']) - 1, 1)
				data['bandwidth'] = float((data['dfs'][-1] - data['dfs'][0]) + _chw)
				print(
					f">>> DEBUG: trim_downfactors='Bottom' -> trimmed {_bot_trim} channel(s) from "
					f"low-freq end (first rows). wfall.shape {_old_shape} -> {wfall.shape}. "
					f"freq range now {data['dfs'][0]:.4f} -- {data['dfs'][-1]:.4f} MHz, "
					f"bandwidth -> {data['bandwidth']:.4f} MHz."
				)

	assert len(data['dfs']) == wfall.shape[0], "dfs must stay aligned 1:1 with wfall rows"
	if targetDM:
		print(f"Info: Dedispersing from {data['DM']} to {targetDM} pc/cm3")
		ddm = targetDM - data['DM']
		wfall = driftrate.dedisperse(
			wfall,
			ddm,
			min(data['dfs']),
			data['bandwidth']/wfall.shape[0],
			1000*data['duration']/wfall.shape[1],
			freqs=np.asarray(data['dfs']),  # FIX 2026-10: masked channels are not contiguous
		)
	else:
		targetDM = data['DM']

	## Vary DM by small amounts and calculate optimal using DM_phase
	if recalculateDM:
		step = 0.1
		ddms = np.arange(-40, 40, step=step)
		step = 0.01 # for very short bursts
		ddms = np.arange(-10, 10, step=step)
		print(
			f"Info: Searching for optimal DM in steps of "
			f"{step} pc/cm3 and range of {targetDM} +/- {ddms[-1]:.0f} pc/cm3..."
		)
		ddmopt, ddmopt_std = DM_phase.get_dm(
			wfall,
			ddms,
			data['duration'] / data['wfall'].shape[1], # s
			np.asarray(data['dfs']),  # FIX 2026-10: true channel frequencies (was a uniform linspace)
			manual_bandwidth=False,
			fname=f'measurements/dmplots/{bname}',
			no_plots=False,
			blackonwhite=False,
			fformat='.pdf',
		)
		print(f"Info: Found {ddmopt = :.3f} +/- {ddmopt_std:.3f}")
		print(f"Info: {targetDM + ddmopt = :.3f} +/- {ddmopt_std:.3f}")
		targetDM = targetDM + ddmopt
		print(f"Info: Dedispersing from {data['DM']} to {targetDM} pc/cm3")
		ddm = targetDM - data['DM']
		wfall = driftrate.dedisperse(
			wfall,
			ddm,
			min(data['dfs']),
			data['bandwidth']/wfall.shape[0],
			1000*data['duration']/wfall.shape[1],
			freqs=np.asarray(data['dfs']),  # FIX 2026-10: masked channels are not contiguous
		)
		plt.close()

	###

	print(f">>> DEBUG: Applying masks for burst '{bname}', wfall.shape={wfall.shape}, masks={masks}")
	if ignore and len(ignore) > 0:
		print(f">>> DEBUG: Ignore filter enabled for burst '{bname}': ignore={ignore} (will be applied to slope calculation only)")

	# Mask indices are on the ORIGINAL raw .npz channel grid. If trim_downfactors removed
	# channels, remap raw -> trimmed:
	#   'Top' trim: removed last _top_trim rows. Raw indices in [0, n_raw - _top_trim) map
	#       unchanged. Raw indices in [n_raw - _top_trim, n_raw) point to trimmed channels
	#       and are dropped with a warning.
	#   'Bottom' trim: removed first _bot_trim rows. Raw indices < _bot_trim point to
	#       trimmed channels and are dropped. All others get _bot_trim subtracted.
	_n_raw = wfall.shape[0] + _top_trim + _bot_trim   # original raw channel count
	def _remap_one_index(_raw_idx):
		"""Return remapped trimmed-grid index, or None if it falls in the trimmed region."""
		_i = int(_raw_idx)
		if _i < _bot_trim:
			return None
		if _i >= (_n_raw - _top_trim):
			return None
		return _i - _bot_trim

	for mask in masks:
		# Handle both single indices and iterables (range, list, np.ndarray) of indices.
		if hasattr(mask, '__iter__') and not isinstance(mask, (int, np.integer)):
			_raw_indices = list(mask)
		else:
			_raw_indices = [mask]

		if _top_trim > 0 or _bot_trim > 0:
			_remapped = []
			_dropped  = []
			for _r in _raw_indices:
				_m = _remap_one_index(_r)
				if _m is None:
					_dropped.append(int(_r))
				else:
					_remapped.append(_m)
			if _dropped:
				print(
					f">>> DEBUG: trim_downfactors dropped {len(_dropped)} mask index(es) "
					f"that fell in the trimmed region for burst '{bname}': {_dropped[:10]}"
					f"{'...' if len(_dropped) > 10 else ''}"
				)
			if not _remapped:
				continue  # nothing left to apply for this mask
			_apply = _remapped
		else:
			_apply = _raw_indices

		max_idx = max(_apply)
		if max_idx >= wfall.shape[0]:
			raise IndexError(
				f"!!! BURST '{bname}': mask index {max_idx} (post-trim-remap) is out of bounds "
				f"for wfall with {wfall.shape[0]} frequency channels"
			)
		wfall[_apply] = 0
	if presubtractbg:
		wfall = driftrate.subtractbg(wfall, 0, int(wfall.shape[1]*0.1))
	if type(crop) == tuple or type(crop) == list and len(crop) == 2:
		wfall = wfall[..., crop[0]:crop[1]]
		print(f"Info: {bname}: cropped to {wfall.shape = }")
	print(f">>> DEBUG: About to subsample burst '{bname}' with shape {wfall.shape}, downfactors={downfactors}, target shape=({wfall.shape[0]//downfactors[0]}, {wfall.shape[1]//downfactors[1]})")
	if wfall.size == 0 or wfall.shape[1]//downfactors[1] == 0:
		raise ValueError(f"!!! BURST '{bname}' has invalid shape {wfall.shape} after crop={crop}. Cannot subsample to 0 time bins with downfactors={downfactors}")
	# Check divisibility to avoid reshape errors
	

	if wfall.shape[0] % downfactors[0] != 0:
		n = wfall.shape[0]; print(f"    Factors of {n}: {sorted([i for i in range(1, n+1) if n % i == 0])}")
		raise ValueError(...)
	if wfall.shape[0] % downfactors[0] != 0:
		raise ValueError(f"!!! BURST '{bname}': frequency dimension {wfall.shape[0]} is not divisible by downfactor {downfactors[0]}. Remainder={wfall.shape[0] % downfactors[0]}")

	if wfall.shape[1] % downfactors[1] != 0:
		n = wfall.shape[1]; print(f"    Factors of {n}: {sorted([i for i in range(1, n+1) if n % i == 0])}")
		raise ValueError(...)
	if wfall.shape[1] % downfactors[1] != 0:
		raise ValueError(f"!!! BURST '{bname}': time dimension {wfall.shape[1]} is not divisible by downfactor {downfactors[1]}. Remainder={wfall.shape[1] % downfactors[1]}")
	wfall = driftrate.subsample(
		wfall,
		wfall.shape[0]//downfactors[0],
		wfall.shape[1]//downfactors[1]
	)

	# determine resolutions accounting for downsampling and cropping
	freqs_bin0  = min(data['dfs'])
	res_freq    = downfactors[0] * data['bandwidth'] / data['wfall'].shape[0] # MHz
	res_time_ms = downfactors[1] * 1000*data['duration'] / data['wfall'].shape[1] # ms
	duration    = wfall.shape[1] * res_time_ms # duration of potentially cropped waterfall
	print(
		f"Info: {res_freq = :.3f} MHz {res_time_ms = :.5f} ms\n"
		f"Info: {min(data['dfs']) = :.2f} -- {max(data['dfs']) = :.2f} MHz"
	)

	if targetDM and correctTimes:
		ddm = targetDM - data['DM']

		a_dm = 4.14937759336e6
		center_i, errorsumi = driftrate.findCenter(wfall)
		center_f = center_i*res_freq + freqs_bin0
		high_ref_freq = max(data['dfs'])

		deltat = - a_dm * (center_f**-2 - high_ref_freq**-2) * ddm
		xos = [x+deltat for x in xos]
		cuts = [c+deltat for c in cuts]
		print(f'Info: shifting xos and cuts by {deltat} ms for {targetDM = } pc/cm3')

	freqs = np.linspace(freqs_bin0, max(data['dfs']), num=wfall.shape[0]) # channel width/2 is already added
	times_ms = np.linspace(0, duration, num=wfall.shape[1]) # array of timestamps
	# === bw_manual: manual full-band integration band (Option A) =====
	# Restrict the channels feeding the full-band intensity profile
	# (`tseries`) before the duration/tau fit. Resolves to a single
	# (freq_lo, freq_hi) MHz band for the whole burst; per-component
	# lists are unioned, since the time-domain fit is one global
	# mixture over one tseries. Only `tseries` is affected here.
	_bw_manual_band = None
	if bw_manual is not None:
		_bm_isnum = lambda x: isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)
		if isinstance(bw_manual, (tuple, list)) and len(bw_manual) == 2 and _bm_isnum(bw_manual[0]) and _bm_isnum(bw_manual[1]):
			_bw_manual_band = (float(min(bw_manual)), float(max(bw_manual)))
		elif isinstance(bw_manual, list):
			_bm_tups = []
			for _bm_e in bw_manual:
				if _bm_e is None:
					continue
				if isinstance(_bm_e, (tuple, list)) and len(_bm_e) == 2 and _bm_isnum(_bm_e[0]) and _bm_isnum(_bm_e[1]):
					_bm_tups.append((float(_bm_e[0]), float(_bm_e[1])))
				else:
					raise ValueError(f"bw_manual list entries must be a (freq_lo, freq_hi) MHz tuple or None; got {_bm_e!r}.")
			if _bm_tups:
				_bm_lo_u = min(t[0] for t in _bm_tups)
				_bm_hi_u = max(t[1] for t in _bm_tups)
				_bw_manual_band = (min(_bm_lo_u, _bm_hi_u), max(_bm_lo_u, _bm_hi_u))
				if len(_bm_tups) > 1:
					print(f"  Note: bw_manual has {len(_bm_tups)} per-component bands; under the single global mixture fit they are unioned to [{_bw_manual_band[0]:.1f}, {_bw_manual_band[1]:.1f}] MHz for the tseries integration.")
		else:
			raise ValueError(f"bw_manual must be None, a (freq_lo, freq_hi) MHz tuple, or a list of such tuples/None; got {type(bw_manual).__name__}.")

	if _bw_manual_band is None:
		# OLD: tseries = np.nanmean(wfall, axis=0)
		tseries = np.nanmean(wfall, axis=0)  # full band (default / bw_manual=None)
	else:
		_bm_lo, _bm_hi = _bw_manual_band
		_bm_mask = (freqs >= _bm_lo) & (freqs <= _bm_hi)
		_bm_n = int(np.count_nonzero(_bm_mask))
		if _bm_n == 0:
			raise ValueError(f"bw_manual band [{_bm_lo:.1f}, {_bm_hi:.1f}] MHz selects 0 channels. Available band is [{float(np.nanmin(freqs)):.1f}, {float(np.nanmax(freqs)):.1f}] MHz.")
		tseries = np.nanmean(wfall[_bm_mask, :], axis=0)
		print(f"Info: bw_manual -> tseries integrated over [{_bm_lo:.1f}, {_bm_hi:.1f}] MHz ({_bm_n} of {wfall.shape[0]} channels); spectrum/ToA/sub-band paths unaffected.")
	# === END bw_manual ================================================


	# `bw_channel_override` resolution moved to the
	# PER-COMPONENT SUB-BAND-FIT NORMALIZATION block below, alongside
	# the other sub-band-fit params, so n_bursts is in scope.

	tpoint = 'tstart' # 'tend', 'xo'
	pktime = np.nanargmax(tseries)*res_time_ms

	# === UNIFIED BASELINE SUBTRACTION ============================
	# Compute one global baseline value `tseries_C_global` from the
	# OFF-PULSE region of the time series and subtract it from
	# `tseries` BEFORE any fitting runs. This guarantees consistency
	# across all four fit paths (single Gaussian, single EMG,
	# multi-component Gaussian, multi-component EMG): every fit sees
	# a baseline-zero time series, and the displayed model and data
	# share the same zero level.
	#
	# Estimator: the LOWER of (pre-burst median, post-burst median).
	# For scattered bursts the post-burst region contains the
	# scattering tail and over-estimates the noise floor; the
	# pre-burst region typically gives a cleaner estimate. Taking
	# min() defaults to whichever side is cleaner without having to
	# hard-code an assumption about scattering direction.
	#
	# On-pulse window: bounded by [min(xos), max(xos)] when xos is
	# given. Otherwise we bracket around `pktime` using a window of
	# 20% of the total duration (a generous default — wider than
	# almost any burst, so the off-pulse mask stays clean even
	# when the burst is broad).
	if xos and len(xos) >= 1:
		_pulse_lo = float(min(xos))
		_pulse_hi = float(max(xos))
	else:
		_pulse_margin = 0.10 * duration  # 10% of duration on each side
		_pulse_lo = float(pktime) - _pulse_margin
		_pulse_hi = float(pktime) + _pulse_margin
	_pre_mask  = times_ms < _pulse_lo
	_post_mask = times_ms > _pulse_hi
	_cand_C = []
	if _pre_mask.sum() >= 5:
		_cand_C.append(float(np.nanmedian(tseries[_pre_mask])))
	if _post_mask.sum() >= 5:
		_cand_C.append(float(np.nanmedian(tseries[_post_mask])))
	if _cand_C:
		tseries_C_global = min(_cand_C)
	else:
		# Last-resort fallback: leading NOISE_WIDTH_FACTOR slab.
		_off_n = max(len(tseries) // NOISE_WIDTH_FACTOR, 1)
		tseries_C_global = float(np.nanmedian(tseries[:_off_n]))

	# Subtract so all downstream fits see zero-baseline data. The
	# subtraction also propagates naturally into the plot, since
	# `tseries` is what gets plotted at line ~9384.
	tseries = tseries - tseries_C_global
	print(f"Info: subtracted global baseline {tseries_C_global:.6g} "
	      f"from tseries (min of pre/post-burst median)")
	# === END UNIFIED BASELINE SUBTRACTION ========================

# 	t_popt, _, _ = fitgauss(tseries, duration) # whether one or many components, for ref in plot

	# 1D time-integrated fit (Gaussian vs EMG based on toggles)
	if USE_EMG_MIX:
		_t1d_popt, _t1d_pcov, _t1d_redchisq = SINGLE_FIT_FUNC(tseries, duration, dt_ms=res_time_ms)
	else:
		_t1d_popt, _t1d_pcov, _t1d_redchisq = SINGLE_FIT_FUNC(tseries, duration)

	# === MCMC FITTING FOR FULL-BAND PROFILE ===
	_mcmc_results = None
	if USE_MCMC and USE_EMG_MIX:
		_mcmc_results = fit_emg_mcmc(
			times_ms, tseries, burst_name=bname,
			prior_range=MCMC_PRIOR_RANGE,
			nwalkers=MCMC_NWALKERS,
			nsteps=MCMC_NSTEPS,
			burnin=MCMC_BURNIN,
			off_pulse_bins=MCMC_OFF_PULSE_BINS
		)
	# === END MCMC FITTING ===

	# For EMG, optionally seed xos from EMG mean (mu + 1/lambda) if caller didn't provide xos
	if USE_EMG_MIX and len(xos) == 0:
		try:
			A, mu, sig_core, lam, C = _t1d_popt
			xo_out = (mu + 1.0/lam) if (np.isfinite(lam) and lam > 0) else mu
			xos = [xo_out]
		except Exception:
			pass

	# Seed component centers (xos) if user didn't provide any.
	if len(xos) == 0:
		if USE_EMG_MIX:
			# Use mu (true peak) from the EMG 1D fit; fall back to pktime on failure.
			try:
				_, mu_seed, _, _, _ = _t1d_popt
				xos = [float(mu_seed)]
			except Exception:
				xos = [float(pktime)]
		else:
			# Gaussian branch: use xo from the 1D Gaussian fit; fall back to pktime on failure.
			try:
				_, xo_seed, _ = _t1d_popt
				xos = [float(xo_seed)]
			except Exception:
				xos = [float(pktime)]
	
	# Don't prefill tmix_sigmas here; it is set from the mixture fit later
	tmix_redchisq = _t1d_redchisq
	# n_bursts will be derived from xos just below; no need to set it here
	
	
	# Don't prefill tmix_sigmas here; it is set from the mixture fit later
	tmix_redchisq = _t1d_redchisq
	# n_bursts will be derived from xos just below; no need to set it here
	
	
	print(f"Info: {bname}: {data['wfall'].shape = }, {wfall.shape = }.")
	print(f"Info: Using {bw_filter = } and {snr_cutoff = }")

	if loadonly:
		return (
			wfall,
			freqs,
			times_ms,
			res_freq,
			res_time_ms,
			targetDM,
# 			t_popt
			_t1d_popt
		)

	if len(xos) == 0:
		xos.append(pktime)

	## Assuming 1 burst:
	# window = t_filter_factor*abs(t_popt[2]) # 2*stddev of a guassian fit to the integrated time series

	##### multi component model: use multiple 1d gaussians to make multiple windows in time,
	# then use the time windows to make frequency windows
	n_bursts = len(xos)

	# === PER-COMPONENT STACKING NORMALIZATION ===
	# Build _stacking_per_comp (list of bool, length n_bursts) and
	# _stack_size_per_comp (list of int, length n_bursts).
	# Scalars broadcast; lists must match n_bursts in length.
	if _stacking_raw is None:
		_stacking_per_comp = [USE_SIGNAL_STACKING] * n_bursts
	else:
		if len(_stacking_raw) == 1:
			_stacking_per_comp = [_stacking_raw[0]] * n_bursts
		elif len(_stacking_raw) == n_bursts:
			_stacking_per_comp = list(_stacking_raw)
		else:
			raise ValueError(
				f"`stacking` was given as a list of length {len(_stacking_raw)} "
				f"but there are {n_bursts} components (xos). "
				f"Pass either a single value or a list of length {n_bursts}."
			)

	if _stack_size_raw is None:
		_stack_size_per_comp = [STACK_SIZE] * n_bursts
	else:
		if len(_stack_size_raw) == 1:
			_stack_size_per_comp = [_stack_size_raw[0]] * n_bursts
		elif len(_stack_size_raw) == n_bursts:
			_stack_size_per_comp = list(_stack_size_raw)
		else:
			raise ValueError(
				f"`stack_size` was given as a list of length {len(_stack_size_raw)} "
				f"but there are {n_bursts} components (xos). "
				f"Pass either a single value or a list of length {n_bursts}."
			)

	# Keep legacy globals consistent with the per-component picture
	USE_SIGNAL_STACKING = any(_stacking_per_comp)
	# STACK_SIZE retains its first-component value as a fallback for any
	# code paths that still reference the global; per-component code should
	# always use _stack_size_per_comp[comp_idx] instead.
	if any(_stacking_per_comp):
		# Pick the first stacked component's size as the legacy default
		for _i, _s in enumerate(_stacking_per_comp):
			if _s:
				STACK_SIZE = _stack_size_per_comp[_i]
				break

	if n_bursts > 1 and (
		any(s != _stacking_per_comp[0] for s in _stacking_per_comp)
		or any(sz != _stack_size_per_comp[0] for sz in _stack_size_per_comp)
	):
		print(f"  Per-component stacking config:")
		for _i, (_s, _sz) in enumerate(zip(_stacking_per_comp, _stack_size_per_comp)):
			_sfx = subburst_suffixes[_i] if _i < len(subburst_suffixes) else str(_i)
			print(f"    Component {_sfx}: stacking={_s}, stack_size={_sz}")
	# === END PER-COMPONENT STACKING NORMALIZATION ===

	# === PER-COMPONENT SUB-BAND-FIT NORMALIZATION ===
	# Allows the caller to pass scalar values (back-compat) or
	# per-component lists for `subband_plot`, `constrain_index`, and
	# `bw_width_factor`. The original scalar values are captured first
	# (so list inputs override the "we already parsed subband_plot"
	# logic from line ~3183), then broadcast/validated against
	# n_bursts. The legacy module-scope scalars are kept consistent
	# with the per-component picture using "any/first" semantics.
	def _broadcast_per_comp(name, raw_value, default_scalar, n, coerce=None):
		"""Return a length-n list. raw_value may be a scalar or list/tuple.

		If a list, must have length 1 (broadcasts) or n (used as-is);
		otherwise raises ValueError. `coerce` is an optional callable
		applied to each element.
		"""
		if isinstance(raw_value, (list, tuple)):
			_items = list(raw_value)
			if coerce is not None:
				_items = [coerce(x) for x in _items]
			if len(_items) == 1:
				return _items * n
			if len(_items) == n:
				return _items
			raise ValueError(
				f"`{name}` was given as a list of length {len(_items)} "
				f"but there are {n} components (xos). Pass either a single "
				f"value or a list of length {n}."
			)
		# Scalar broadcast
		_v = default_scalar if raw_value is None else raw_value
		if coerce is not None:
			_v = coerce(_v)
		return [_v] * n

	# `subband_plot` accepts: False, True, int (treated as n_subbands), or
	# list. The scalar parser above already converted int → (True,
	# _n_subbands); a list overrides that. We track two parallel lists:
	# `_subband_plot_per_comp` (bool: should we draw/fit subbands for this
	# component) and `_n_subbands_per_comp` (int: how many bands).
	if isinstance(subband_plot, (list, tuple)):
		# Each list entry can itself be False, True, or int (count).
		_sbp_raw_list = list(subband_plot)
		_subband_plot_per_comp = []
		_n_subbands_per_comp = []
		for _entry in _sbp_raw_list:
			if isinstance(_entry, bool):
				_subband_plot_per_comp.append(_entry)
				_n_subbands_per_comp.append(N_SUBBANDS_ALPHA)
			elif isinstance(_entry, (int, float)) and not isinstance(_entry, bool):
				_subband_plot_per_comp.append(True)
				_n_subbands_per_comp.append(int(_entry))
			else:
				_subband_plot_per_comp.append(bool(_entry))
				_n_subbands_per_comp.append(N_SUBBANDS_ALPHA)
		# Length-validate / broadcast
		if len(_subband_plot_per_comp) == 1:
			_subband_plot_per_comp = _subband_plot_per_comp * n_bursts
			_n_subbands_per_comp = _n_subbands_per_comp * n_bursts
		elif len(_subband_plot_per_comp) != n_bursts:
			raise ValueError(
				f"`subband_plot` was given as a list of length {len(_subband_plot_per_comp)} "
				f"but there are {n_bursts} components (xos)."
			)
	else:
		# Scalar — broadcast already-parsed bool + count
		_subband_plot_per_comp = [bool(subband_plot)] * n_bursts
		_n_subbands_per_comp = [int(_n_subbands)] * n_bursts

	_constrain_index_per_comp = _broadcast_per_comp(
		'constrain_index', constrain_index, False, n_bursts,
		coerce=bool,
	)
	_bw_width_factor_per_comp = _broadcast_per_comp(
		'bw_width_factor', bw_width_factor, 2.0, n_bursts,
		coerce=float,
	)

	# === PER-COMPONENT ACF NORMALIZATION ===
	# `ACF` accepts: False, True, or a list. A list may be per-component
	# bools (length 1 or n_bursts) OR a list of 1-indexed component indices
	# to enable (like subband_diagnostic). Result: `_acf_per_comp`, a
	# length-n_bursts list of bools.
	if isinstance(ACF, (list, tuple)):
		_acf_list = list(ACF)
		if len(_acf_list) > 0 and all(isinstance(x, bool) for x in _acf_list):
			# per-component bool list
			if len(_acf_list) == 1:
				_acf_per_comp = _acf_list * n_bursts
			elif len(_acf_list) == n_bursts:
				_acf_per_comp = list(_acf_list)
			else:
				raise ValueError(
					f"`ACF` bool list has length {len(_acf_list)} but there "
					f"are {n_bursts} components (xos). Pass length 1 or "
					f"{n_bursts}.")
		else:
			# list of 1-indexed component indices to enable
			try:
				_acf_idx_set = set(int(i) - 1 for i in _acf_list)
			except (TypeError, ValueError) as _e:
				raise ValueError(
					f"`ACF` list must be all-bool (per-component) or all-int "
					f"(1-indexed component indices); got {ACF!r}: {_e}")
			_acf_per_comp = [i in _acf_idx_set for i in range(n_bursts)]
	else:
		_acf_per_comp = [bool(ACF)] * n_bursts

	_acf_any = any(_acf_per_comp)
	if _acf_any:
		print(f"Info: ACF mode enabled for component(s): "
		      f"{[subburst_suffixes[i] for i in range(n_bursts) if (i < len(subburst_suffixes) and _acf_per_comp[i])]}")
	# === END PER-COMPONENT ACF NORMALIZATION ===

	# === PER-COMPONENT SCINT NORMALIZATION ===
	# `scint` accepts the same shapes as `ACF` (False / True / per-component
	# bool list / 1-indexed index list). Result: `_scint_per_comp`, a
	# length-n_bursts list of bools. Independent of `_acf_per_comp`.
	if isinstance(scint, (list, tuple)):
		_scint_list = list(scint)
		if len(_scint_list) > 0 and all(isinstance(x, bool) for x in _scint_list):
			if len(_scint_list) == 1:
				_scint_per_comp = _scint_list * n_bursts
			elif len(_scint_list) == n_bursts:
				_scint_per_comp = list(_scint_list)
			else:
				raise ValueError(
					f"`scint` bool list has length {len(_scint_list)} but there "
					f"are {n_bursts} components (xos). Pass length 1 or {n_bursts}.")
		else:
			try:
				_scint_idx_set = set(int(i) - 1 for i in _scint_list)
			except (TypeError, ValueError) as _e:
				raise ValueError(
					f"`scint` list must be all-bool (per-component) or all-int "
					f"(1-indexed component indices); got {scint!r}: {_e}")
			_scint_per_comp = [i in _scint_idx_set for i in range(n_bursts)]
	else:
		_scint_per_comp = [bool(scint)] * n_bursts

	_scint_any = any(_scint_per_comp)
	if _scint_any:
		print(f"Info: scint (Delta_nu_d) mode enabled for component(s): "
		      f"{[subburst_suffixes[i] for i in range(n_bursts) if (i < len(subburst_suffixes) and _scint_per_comp[i])]}")
	# === END PER-COMPONENT SCINT NORMALIZATION ===

	# === PER-COMPONENT constrain_sigma RESOLUTION ===
	# `constrain_sigma` is the σ-constrained-diagnostic kwarg added in
	# v56_10. It accepts a tuple of length n_bursts (= n components) where
	# each entry is either None (skip diagnostic for that component) or a
	# positive float (σ value in ms to pin the per-sub-band EMG fits at).
	#
	# Behaviour:
	#   - None or unset: identical to the legacy pipeline (no extra fits,
	#     no extra CSV columns populated, identical figures).
	#   - Tuple with one or more numeric entries: for each numeric entry,
	#     re-run the per-sub-band EMG fit with σ pinned, then re-fit the
	#     τ(ν) power law and compute Δτ/σ_τ per sub-band against the
	#     free-σ fit. Reports both the free α and the σ-constrained α
	#     (Option C of the design).
	#
	# Validation rules (raise ValueError on violation):
	#   1. Must be None, a list, or a tuple. Scalars not accepted.
	#   2. Tuple length must equal n_bursts (n components).
	#   3. Each entry must be None or a positive finite float.
	#      Zero, negative, NaN, inf, or anything below res_time_ms/2
	#      is rejected (σ below the time resolution is unphysical and
	#      the fit's σ lower bound would clamp anyway).
	#   4. Must not be set on the Gaussian model run. This is enforced
	#      structurally: constrain_sigma lives inside the EMG model_run
	#      dict (Schema B from the design discussion), so on the Gaussian
	#      run constrain_sigma defaults to None and falls through cleanly.
	#      If a user puts it inside a Gaussian model_run dict by mistake,
	#      the validator catches it here because USE_EMG_MIX is False.
	#   5. For any component i with a numeric constrain_sigma[i], that
	#      component's subband_plot[i] must also be truthy. The σ-
	#      constrained diagnostic re-uses the same sub-band fits that
	#      `subband_plot` triggers; requesting the diagnostic on a
	#      component with sub-band fitting disabled is a config error.
	if constrain_sigma is not None:
		if USE_GAUSS_MIX:
			raise ValueError(
				"`constrain_sigma` is an EMG-only diagnostic (the σ–τ degeneracy "
				"it probes only exists in EMG fits, since Gaussian fits have no "
				"τ parameter). Found it set on a Gaussian model_run. Either move "
				"it inside the EMG model_run dict, or remove it from this run."
			)
		if not isinstance(constrain_sigma, (list, tuple)):
			raise ValueError(
				f"`constrain_sigma` must be None, a list, or a tuple; got "
				f"{type(constrain_sigma).__name__}: {constrain_sigma!r}. "
				f"To skip the diagnostic for a component use None for that entry, "
				f"e.g. constrain_sigma=(None, 9.8, None) for a 3-component burst."
			)
		if len(constrain_sigma) != n_bursts:
			raise ValueError(
				f"`constrain_sigma` length ({len(constrain_sigma)}) must equal "
				f"the number of components ({n_bursts}). Got {constrain_sigma!r}. "
				f"Use None to skip the diagnostic for individual components."
			)
		_dt_floor_for_sigma = 0.5 * float(res_time_ms)
		_constrain_sigma_per_comp = []
		_violations = []
		for _i_cs, _entry_cs in enumerate(constrain_sigma):
			if _entry_cs is None:
				_constrain_sigma_per_comp.append(None)
				continue
			try:
				_val_cs = float(_entry_cs)
			except (TypeError, ValueError):
				_violations.append(
					f"  • component {_i_cs} ({subburst_suffixes[_i_cs] if _i_cs < len(subburst_suffixes) else _i_cs}): "
					f"got {_entry_cs!r} ({type(_entry_cs).__name__}); expected None or a positive float."
				)
				_constrain_sigma_per_comp.append(None)
				continue
			if not np.isfinite(_val_cs):
				_violations.append(
					f"  • component {_i_cs}: got non-finite value {_val_cs} "
					f"(NaN or inf); expected None or a positive finite float."
				)
				_constrain_sigma_per_comp.append(None)
				continue
			if _val_cs <= 0:
				_violations.append(
					f"  • component {_i_cs}: got σ = {_val_cs} ms; "
					f"σ ≤ 0 is unphysical. Use None to skip the diagnostic for this component."
				)
				_constrain_sigma_per_comp.append(None)
				continue
			if _val_cs < _dt_floor_for_sigma:
				_violations.append(
					f"  • component {_i_cs}: got σ = {_val_cs:.4f} ms which is below "
					f"the time-resolution floor (res_time_ms/2 = {_dt_floor_for_sigma:.4f} ms). "
					f"σ below the time resolution cannot be meaningfully measured; "
					f"the per-sub-band fit's σ lower bound would clamp at this floor."
				)
				_constrain_sigma_per_comp.append(None)
				continue
			# Check subband_plot gating: requesting the diagnostic on a
			# component with sub-band fitting disabled is a config error
			# because the diagnostic relies on the per-sub-band fits.
			_sbplot_i = (bool(_subband_plot_per_comp[_i_cs])
			             if _i_cs < len(_subband_plot_per_comp) else False)
			if not _sbplot_i:
				_violations.append(
					f"  • component {_i_cs}: constrain_sigma = {_val_cs:.4f} ms requested, "
					f"but subband_plot[{_i_cs}] is False/0 (sub-band fitting disabled "
					f"for this component). The σ-constrained diagnostic re-uses "
					f"the per-sub-band fits that subband_plot triggers; without "
					f"sub-band fitting there is no per-sub-band τ to compare against. "
					f"Either enable subband_plot for this component or set "
					f"constrain_sigma[{_i_cs}] = None."
				)
				_constrain_sigma_per_comp.append(None)
				continue
			_constrain_sigma_per_comp.append(_val_cs)
		if _violations:
			raise ValueError(
				"`constrain_sigma` validation failed:\n"
				+ "\n".join(_violations)
				+ "\nReminder: constrain_sigma=(None, 9.8, None) means: skip for "
				  "components a and c, pin σ to 9.8 ms for component b. Use None "
				  "(not 0 or False) to skip individual components."
			)
		# All-None tuple = silently equivalent to constrain_sigma=None.
		if all(v is None for v in _constrain_sigma_per_comp):
			_constrain_sigma_per_comp = [None] * n_bursts
		print(f"Info: constrain_sigma = {_constrain_sigma_per_comp} "
		      f"(σ-constrained diagnostic will run on numeric entries)")
	else:
		_constrain_sigma_per_comp = [None] * n_bursts

	# === PER-COMPONENT ignore_subindex RESOLUTION ===
	# `ignore_subindex` lets the user exclude specific sub-band τ
	# measurements from the τ(ν) power-law fit (and the σ(ν) ε-fit),
	# per-component. The accepted value is a tuple/list of 1-indexed
	# sub-band labels using the convention that index 1 is the
	# HIGHEST-frequency sub-band (i.e., labels descend with frequency).
	# None or an empty tuple/list means "ignore nothing".
	#
	# Format: scalar (applies to all components) OR a list of length
	# n_bursts where each entry is itself a tuple/list/None.
	#
	#   ignore_subindex = None                 → no ignore on any
	#   ignore_subindex = (6,7)                → ignore 6,7 on every
	#   ignore_subindex = [(6,7), None]        → comp a ignores 6,7,
	#                                            comp b ignores nothing
	#
	# Validation (per-component, with per-component out-of-range
	# checking deferred to fit_subband_alpha where N_subbands is known):
	#   * 0 or negative index → raise ValueError here
	#   * non-int → raise ValueError here
	#   * empty tuple → coerced to None
	def _coerce_ignore_subindex(raw):
		"""Coerce a single per-component value to a tuple of ints or None."""
		if raw is None:
			return None
		if not isinstance(raw, (list, tuple)):
			raise ValueError(
				f"`ignore_subindex` per-component entry must be None or "
				f"a tuple/list of 1-indexed ints; got {type(raw).__name__}: {raw!r}"
			)
		_items = list(raw)
		if len(_items) == 0:
			# Empty tuple is equivalent to None
			return None
		_out = []
		for _x in _items:
			if isinstance(_x, bool) or not isinstance(_x, int):
				# bool is technically a subclass of int; reject explicitly
				raise ValueError(
					f"`ignore_subindex` index must be an int; got "
					f"{type(_x).__name__}: {_x!r}"
				)
			if _x <= 0:
				raise ValueError(
					f"`ignore_subindex` indices are 1-indexed (1 = highest-freq "
					f"sub-band); got {_x} which is 0 or negative."
				)
			_out.append(int(_x))
		# Deduplicate while preserving order
		_seen = set(); _dedup = []
		for _x in _out:
			if _x not in _seen:
				_seen.add(_x); _dedup.append(_x)
		return tuple(_dedup)

	if isinstance(ignore_subindex, (list, tuple)) and (
		# Distinguish "list of per-component entries" from "a single
		# tuple of indices broadcast to all components". A tuple of
		# ints (e.g. (6,7)) is the latter; a list/tuple that contains
		# None/list/tuple as elements is the former.
		any(_e is None or isinstance(_e, (list, tuple))
		    for _e in ignore_subindex)
	):
		# Per-component list
		_raw_list = list(ignore_subindex)
		if len(_raw_list) == 1:
			_ignore_subindex_per_comp = [_coerce_ignore_subindex(_raw_list[0])] * n_bursts
		elif len(_raw_list) == n_bursts:
			_ignore_subindex_per_comp = [_coerce_ignore_subindex(_e) for _e in _raw_list]
		else:
			raise ValueError(
				f"`ignore_subindex` was given as a list of length "
				f"{len(_raw_list)} but there are {n_bursts} components "
				f"(xos). Pass either a single value or a list of length "
				f"{n_bursts}."
			)
	else:
		# Scalar (None, single tuple/list of ints) broadcast to all comps
		_coerced = _coerce_ignore_subindex(ignore_subindex)
		_ignore_subindex_per_comp = [_coerced] * n_bursts
	# === END PER-COMPONENT ignore_subindex RESOLUTION ===

	# === PER-COMPONENT ignore_per_comp RESOLUTION ===
	# `ignore_per_comp` lets the user exclude specific CHANNEL indices
	# from a particular component's slope-fit arrival times only (not
	# burst-wide). This is the per-component analogue of the burst-wide
	# `ignore` kwarg.
	#
	# Format: None (default — no per-component ignoring) OR a list of
	# length n_bursts where each entry is:
	#   - None / () / []                  → no extra ignores for this comp
	#   - list/tuple/range of ints        → channel indices to ignore
	#   - tuple of nested iterables       → flattened to channel indices
	#                                        (e.g. (range(0,153), 255)
	#                                        becomes channels 0..152 + 255)
	#
	# Validation: per-component entry must be None or an iterable of
	# ints / iterables-of-ints. Negative indices are rejected here.
	# Out-of-range checking happens at the use site once n_freq is known.
	def _coerce_ignore_per_comp_entry(raw):
		"""Flatten one per-component entry to a list of int channel
		indices. None or empty → []. Nested ranges/lists are flattened
		one level (matching the existing burst-wide `ignore` semantics)."""
		if raw is None:
			return []
		if isinstance(raw, (int, np.integer)):
			# Single int — accept but warn implicit type
			if isinstance(raw, bool):
				raise ValueError(
					f"`ignore_per_comp` entry must be None or an iterable "
					f"of ints; got bool: {raw!r}"
				)
			return [int(raw)]
		if not hasattr(raw, '__iter__'):
			raise ValueError(
				f"`ignore_per_comp` entry must be None or an iterable; "
				f"got {type(raw).__name__}: {raw!r}"
			)
		flat = []
		for item in raw:
			if isinstance(item, bool):
				raise ValueError(
					f"`ignore_per_comp` channel indices must be ints; "
					f"got bool: {item!r}"
				)
			if isinstance(item, (int, np.integer)):
				if int(item) < 0:
					raise ValueError(
						f"`ignore_per_comp` channel indices must be "
						f"non-negative; got {item!r}"
					)
				flat.append(int(item))
			elif hasattr(item, '__iter__'):
				# Nested iterable: flatten one level
				for sub in item:
					if not isinstance(sub, (int, np.integer)) or isinstance(sub, bool):
						raise ValueError(
							f"`ignore_per_comp` nested-iterable entries must "
							f"yield ints; got {type(sub).__name__}: {sub!r}"
						)
					if int(sub) < 0:
						raise ValueError(
							f"`ignore_per_comp` channel indices must be "
							f"non-negative; got {sub!r}"
						)
					flat.append(int(sub))
			else:
				raise ValueError(
					f"`ignore_per_comp` entry must be int or iterable; "
					f"got {type(item).__name__}: {item!r}"
				)
		# Deduplicate while preserving order
		_seen = set(); _dedup = []
		for _x in flat:
			if _x not in _seen:
				_seen.add(_x); _dedup.append(_x)
		return _dedup

	if ignore_per_comp is None:
		_ignore_per_comp_resolved = [[] for _ in range(n_bursts)]
	elif not isinstance(ignore_per_comp, (list, tuple)):
		raise ValueError(
			f"`ignore_per_comp` must be None or a list of length n_bursts="
			f"{n_bursts}; got {type(ignore_per_comp).__name__}: "
			f"{ignore_per_comp!r}"
		)
	elif len(ignore_per_comp) != n_bursts:
		raise ValueError(
			f"`ignore_per_comp` was given as a list of length "
			f"{len(ignore_per_comp)} but there are {n_bursts} components "
			f"(xos). Pass a list of length {n_bursts} (use None or () "
			f"for components that need no per-component ignoring)."
		)
	else:
		_ignore_per_comp_resolved = [
			_coerce_ignore_per_comp_entry(_e) for _e in ignore_per_comp
		]
		_nonempty = [i for i, e in enumerate(_ignore_per_comp_resolved) if e]
		if _nonempty:
			print(f"Info: ignore_per_comp = {ignore_per_comp} "
			      f"(per-component arrival-time ignore applies to "
			      f"components {_nonempty} only)")
	# === END PER-COMPONENT ignore_per_comp RESOLUTION ===

	# === PER-COMPONENT bw_channel_override RESOLUTION ===
	# Manually specify the burst bandwidth instead of relying on the spectral
	# Gaussian fit, on a per-component basis. Use case: bursts where the
	# spectrum is truncated, scintillated, or noisy enough that a Gaussian
	# fit to the time-integrated spectrum gives a misleading bandwidth.
	#
	# Accepted scalar override forms (each a "single override"):
	#   {'raw':         (i_lo, i_hi)}  — explicit raw (pre-downsample) channel
	#                                    indices. Internally divided by
	#                                    downfactors[0] to map to the
	#                                    downsampled grid.
	#   {'downsampled': (i_lo, i_hi)}  — explicit downsampled channel indices,
	#   {'ds':          (i_lo, i_hi)}    used directly with no mapping.
	#   (i_lo, i_hi)                   — legacy bare-tuple form. Treated as
	#                                    RAW indices, with a prompt to switch
	#                                    to the explicit dict form.
	#   None                           — no override (use spectral Gaussian fit).
	#
	# Per-component input: a Python `list` (not tuple) where each element is
	# one of the above scalar forms — e.g.
	#   [None, {'ds': (87, 264)}]
	#   [{'raw': (2009, 3724)}, {'raw': (2400, 4100)}]
	# Length must be 1 (broadcasts) or n_bursts. List vs tuple is the
	# disambiguator: a bare 2-tuple stays a scalar override (legacy form);
	# wrap it in `[...]` to opt in to the per-component form.
	#
	# When active for a component:
	#   - That component's arrival-time spectral filter (pkfreq ±
	#     bw_width_factor·bwidth) is bypassed.
	#   - fit_subband_alpha receives that component's override as
	#     channel_override and uses freqs[i_lo:i_hi+1] as its sub-band
	#     splitting range.
	def _resolve_one_override(raw_value, _freqs, _downfactors, _top_trim=0, _bot_trim=0):
		"""Resolve a single override input to a (i_lo, i_hi) tuple or None.

		Mirrors the legacy scalar resolver. Logging is per-component;
		caller passes the component label if needed.

		`_top_trim` / `_bot_trim` account for trim_downfactors having removed
		channels from the high-freq / low-freq end of the raw .npz waterfall.
		Raw indices in `raw_value` refer to the ORIGINAL raw .npz grid; they
		are remapped onto the trimmed-then-downsampled grid here. Downsampled
		indices already refer to the trimmed-then-downsampled grid (which is
		the only "downsampled" grid the pipeline ever produces post-trim) and
		need no remapping.
		"""
		if raw_value is None:
			return None
		try:
			_n_freq_total = len(_freqs)
			_df0 = int(_downfactors[0]) if _downfactors is not None else 1
			# _n_freq_raw_trimmed is the raw-grid size AFTER trim (= downsampled N * df0).
			_n_freq_raw_trimmed = _n_freq_total * _df0
			# _n_freq_raw_original is the raw-grid size BEFORE trim (what the user's
			# raw indices refer to).
			_n_freq_raw_original = _n_freq_raw_trimmed + _top_trim + _bot_trim

			_grid = None
			_indices_in = None
			if isinstance(raw_value, dict):
				_keys = set(raw_value.keys())
				_valid_keys = {'raw', 'downsampled', 'ds'}
				_matched = _keys & _valid_keys
				_unknown = _keys - _valid_keys
				if _unknown:
					raise ValueError(
						f"bw_channel_override dict has unknown key(s) {sorted(_unknown)}; "
						f"expected one of {sorted(_valid_keys)}"
					)
				if len(_matched) != 1:
					raise ValueError(
						f"bw_channel_override dict must have exactly one of "
						f"{sorted(_valid_keys)} as its key (got {sorted(_keys)})"
					)
				_key = _matched.pop()
				_grid = 'raw' if _key == 'raw' else 'downsampled'
				_pair = raw_value[_key]
				_indices_in = (int(_pair[0]), int(_pair[1]))
			else:
				_grid = 'raw'
				_indices_in = (int(raw_value[0]), int(raw_value[1]))
				print(f"  Note: bw_channel_override={raw_value} "
				      f"(bare tuple) is being treated as RAW channel indices. "
				      f"Prefer the explicit form "
				      f"{{'raw': {raw_value}}} or "
				      f"{{'downsampled': (...)}} to make the grid unambiguous.")

			_i_lo_in, _i_hi_in = _indices_in
			if _i_lo_in > _i_hi_in:
				_i_lo_in, _i_hi_in = _i_hi_in, _i_lo_in

			if _grid == 'raw':
				# raw indices refer to the ORIGINAL raw .npz grid. Remap onto the
				# trimmed raw grid by subtracting _bot_trim, and clamp/drop indices
				# that fall in the trimmed region.
				_i_lo_orig, _i_hi_orig = _i_lo_in, _i_hi_in
				# Detect "fully outside the kept range" before any clamping.
				if _i_hi_in < _bot_trim or _i_lo_in >= (_n_freq_raw_original - _top_trim):
					print(f"  Warning: bw_channel_override raw=({_i_lo_orig}, {_i_hi_orig}) "
					      f"lies entirely in the trim_downfactors-trimmed region "
					      f"(kept raw range [{_bot_trim}, {_n_freq_raw_original - _top_trim - 1}]) "
					      f"— ignoring override")
					return None
				# Clamp into the kept raw range, then shift to trimmed raw indices.
				_i_lo_kept = max(_bot_trim, _i_lo_in)
				_i_hi_kept = min(_n_freq_raw_original - _top_trim - 1, _i_hi_in)
				if (_top_trim > 0 or _bot_trim > 0) and (
					_i_lo_kept != _i_lo_in or _i_hi_kept != _i_hi_in
				):
					print(f"  Info: bw_channel_override raw=({_i_lo_orig}, {_i_hi_orig}) "
					      f"clamped to ({_i_lo_kept}, {_i_hi_kept}) — kept raw range after "
					      f"trim_downfactors is [{_bot_trim}, {_n_freq_raw_original - _top_trim - 1}]")
				_i_lo_raw_trimmed = _i_lo_kept - _bot_trim
				_i_hi_raw_trimmed = _i_hi_kept - _bot_trim
				# Now apply the existing clamp against the trimmed raw size (defensive).
				_i_lo_raw_clamped = max(0, _i_lo_raw_trimmed)
				_i_hi_raw_clamped = min(_n_freq_raw_trimmed - 1, _i_hi_raw_trimmed)
				_i_lo = _i_lo_raw_clamped // _df0
				_i_hi = (_i_hi_raw_clamped + _df0 - 1) // _df0
				_i_lo = max(0, min(_n_freq_total - 1, _i_lo))
				_i_hi = max(0, min(_n_freq_total - 1, _i_hi))
				if _df0 != 1:
					print(f"  Info: bw_channel_override raw-trimmed ({_i_lo_raw_clamped}, {_i_hi_raw_clamped}) "
					      f"mapped to downsampled ({_i_lo}, {_i_hi}) with downfactors[0]={_df0} "
					      f"(raw N after trim={_n_freq_raw_trimmed}, downsampled N={_n_freq_total})")
			else:
				_i_lo = max(0, _i_lo_in)
				_i_hi = min(_n_freq_total - 1, _i_hi_in)
				if _i_lo != _i_lo_in or _i_hi != _i_hi_in:
					print(f"  Warning: bw_channel_override downsampled=({_i_lo_in}, {_i_hi_in}) "
					      f"clamped to ({_i_lo}, {_i_hi}) — downsampled waterfall has "
					      f"{_n_freq_total} channels")
				print(f"  Info: bw_channel_override using downsampled indices ({_i_lo}, {_i_hi}) "
				      f"directly (no mapping; downsampled N={_n_freq_total}, downfactors[0]={_df0})")

			if _i_hi - _i_lo < 1:
				print(f"  Warning: bw_channel_override={raw_value} "
				      f"resolves to <2 channels in downsampled grid "
				      f"(downfactors[0]={_df0}) — ignoring override")
				return None
			_freq_lo = _freqs[_i_lo]
			_freq_hi = _freqs[_i_hi]
			print(f"  Manual bandwidth override: downsampled channels [{_i_lo}, {_i_hi}] inclusive "
			      f"→ [{min(_freq_lo, _freq_hi):.2f}, "
			      f"{max(_freq_lo, _freq_hi):.2f}] MHz "
			      f"({_i_hi - _i_lo + 1} channels). "
			      f"Spectral arrival-time filter will be bypassed.")
			return (_i_lo, _i_hi)
		except (TypeError, ValueError, IndexError, KeyError) as _e:
			print(f"  Warning: bw_channel_override={raw_value} could not be parsed "
			      f"({type(_e).__name__}: {_e}) — ignoring override")
			return None

	# Decide: per-component (list) or scalar (None, dict, tuple)?
	# A Python list opts in to per-component; tuple stays scalar so the
	# legacy bare-tuple form keeps working unambiguously.
	if isinstance(bw_channel_override, list):
		_bw_co_raw_list = list(bw_channel_override)
		if len(_bw_co_raw_list) == 1:
			_bw_co_raw_list = _bw_co_raw_list * n_bursts
		elif len(_bw_co_raw_list) != n_bursts:
			raise ValueError(
				f"`bw_channel_override` was given as a list of length "
				f"{len(_bw_co_raw_list)} but there are {n_bursts} components "
				f"(xos). Pass either a single override (scalar) or a list of "
				f"length {n_bursts}."
			)
		_bw_channel_override_per_comp = []
		for _i, _raw in enumerate(_bw_co_raw_list):
			_sfx = subburst_suffixes[_i] if _i < len(subburst_suffixes) else str(_i)
			if _raw is not None:
				print(f"  Resolving bw_channel_override for component {_sfx}:")
			_bw_channel_override_per_comp.append(
				_resolve_one_override(_raw, freqs, downfactors, _top_trim, _bot_trim)
			)
	else:
		# Scalar (None, dict, or bare tuple) — resolve once, broadcast.
		_resolved = _resolve_one_override(bw_channel_override, freqs, downfactors, _top_trim, _bot_trim)
		_bw_channel_override_per_comp = [_resolved] * n_bursts

	# Legacy scalar alias: first component's resolved override.
	# Downstream code that hasn't been updated to per-component indexing
	# still sees a sensible value. Code that HAS been updated reads
	# `_bw_channel_override_per_comp[comp_idx_loop]` directly.
	_bw_channel_override = (
		_bw_channel_override_per_comp[0]
		if n_bursts > 0 else None
	)
	# === END PER-COMPONENT bw_channel_override RESOLUTION ===

	# === SUBBAND-KWARG / SUBBAND-PLOT CONSISTENCY VALIDATION ==========
	# The sub-band α fit is opt-in per burst via `subband_plot`. The
	# other sub-band-only kwargs (`constrain_index`, `bw_width_factor`,
	# `bw_channel_override`) are NO-OPS for components whose
	# `subband_plot` entry is False/0 — they only steer the sub-band
	# fit. Silently ignoring them lets typos and stale config slip
	# through (e.g. `constrain_index=[True, False]` paired with
	# `subband_plot=False` on the burst). Raise an error per
	# offending (kwarg, component) so the user can see exactly what's
	# inconsistent.
	#
	# Defaults treated as "user didn't set it":
	#   constrain_index   → False
	#   bw_width_factor   → 3.0 (function signature default; see line ~4363)
	#   bw_channel_override → None
	# Anything else on a component whose `subband_plot` is False/0 is
	# flagged. Note: a single bw_width_factor=2.0 explicitly typed by
	# the user also flags as "set" — fine, since typing it without
	# subband_plot is meaningless either way.
	_SUBBAND_FLAG_DEFAULTS = {
		'constrain_index'    : False,
		'bw_width_factor'    : 3.0,
		'bw_channel_override': None,
	}
	_subband_per_comp_views = {
		'constrain_index'    : _constrain_index_per_comp,
		'bw_width_factor'    : _bw_width_factor_per_comp,
		'bw_channel_override': _bw_channel_override_per_comp,
	}
	_offending = []
	for _kw_name, _vals in _subband_per_comp_views.items():
		_default = _SUBBAND_FLAG_DEFAULTS[_kw_name]
		for _i in range(n_bursts):
			_v = _vals[_i] if _i < len(_vals) else _default
			_sb_on_for_this_comp = bool(_subband_plot_per_comp[_i]) if _i < len(_subband_plot_per_comp) else False
			# Compare. For floats we tolerate exact equality with the
			# default (the user typed nothing — broadcast list of 3.0).
			if isinstance(_default, float):
				_is_set = (not isinstance(_v, float)) or (_v != _default)
			else:
				_is_set = (_v != _default)
			if _is_set and not _sb_on_for_this_comp:
				_comp_letter = subburst_suffixes[_i] if _i < len(subburst_suffixes) else str(_i)
				_offending.append(
					f"  • {_kw_name}[{_i}]={_v!r} on component {_comp_letter}, "
					f"but subband_plot[{_i}]={_subband_plot_per_comp[_i]!r} "
					f"(component {_comp_letter} won't run sub-band fitting, so "
					f"{_kw_name} has no effect)"
				)
	if _offending:
		raise ValueError(
			"Sub-band-only kwargs were set on components that have "
			"`subband_plot` disabled — these settings would have no "
			"effect and almost certainly indicate stale or incorrect "
			"config. Either enable `subband_plot` for the relevant "
			"component(s), or remove the kwargs flagged below.\n\n"
			+ "\n".join(_offending)
			+ "\n\nReminder: `subband_plot` accepts False/True, an int "
			"(treated as n_subbands, enables sub-band fitting), or a "
			"per-component list. A list entry of 0 or False means "
			"sub-band fitting is OFF for that component."
		)
	# === END SUBBAND-KWARG VALIDATION =================================


	# === PER-COMPONENT bw_ToA_override RESOLUTION ===
	# `bw_ToA_override` restricts which channels' ToAs are eligible
	# for the slope fit, hard-coding what `bw_filter='model_width'`
	# would otherwise derive from the spectral Gaussian fit. Same
	# input schema and per-component behavior as `bw_channel_override`
	# — we reuse `_resolve_one_override` which was defined above for
	# the channel-override path. The two overrides are independent:
	# `bw_channel_override` drives the sub-band α fit's channel range,
	# `bw_ToA_override` drives the slope fit's channel range, and they
	# can be set to different values on the same burst.
	if isinstance(bw_ToA_override, list):
		_bw_toa_raw_list = list(bw_ToA_override)
		if len(_bw_toa_raw_list) == 1:
			_bw_toa_raw_list = _bw_toa_raw_list * n_bursts
		elif len(_bw_toa_raw_list) != n_bursts:
			raise ValueError(
				f"`bw_ToA_override` was given as a list of length "
				f"{len(_bw_toa_raw_list)} but there are {n_bursts} components "
				f"(xos). Pass either a single override (scalar) or a list of "
				f"length {n_bursts}."
			)
		_bw_toa_override_per_comp = []
		for _i, _raw in enumerate(_bw_toa_raw_list):
			_sfx = subburst_suffixes[_i] if _i < len(subburst_suffixes) else str(_i)
			if _raw is not None:
				print(f"  Resolving bw_ToA_override for component {_sfx}:")
			_bw_toa_override_per_comp.append(
				_resolve_one_override(_raw, freqs, downfactors, _top_trim, _bot_trim)
			)
	else:
		# Scalar (None, dict, or bare tuple) — resolve once, broadcast.
		_resolved_toa = _resolve_one_override(bw_ToA_override, freqs, downfactors, _top_trim, _bot_trim)
		_bw_toa_override_per_comp = [_resolved_toa] * n_bursts
	# === END PER-COMPONENT bw_ToA_override RESOLUTION ===


	# Legacy module-scope scalars — keep consistent so any unrefactored
	# downstream code path that still reads `subband_plot` /
	# `_n_subbands` / `constrain_index` / `bw_width_factor` directly
	# sees the "any/first" view rather than stale values.
	subband_plot = any(_subband_plot_per_comp)
	# `_n_subbands` was the global default count. Keep it for code paths
	# that pre-date the per-component split; use the first-enabled
	# component's count.
	for _i, _sp in enumerate(_subband_plot_per_comp):
		if _sp:
			_n_subbands = _n_subbands_per_comp[_i]
			break
	# `constrain_index` and `bw_width_factor` stay as the first
	# component's value for legacy reads. Per-component code should
	# index the lists.
	if n_bursts > 0:
		constrain_index = _constrain_index_per_comp[0]
		bw_width_factor = _bw_width_factor_per_comp[0]

	# Diagnostic line when configs differ across components
	if n_bursts > 1 and (
		len(set(_subband_plot_per_comp)) > 1
		or len(set(_n_subbands_per_comp)) > 1
		or len(set(_constrain_index_per_comp)) > 1
		or len(set(_bw_width_factor_per_comp)) > 1
	):
		print(f"  Per-component sub-band config:")
		for _i in range(n_bursts):
			_sfx = subburst_suffixes[_i] if _i < len(subburst_suffixes) else str(_i)
			print(
				f"    Component {_sfx}: subband_plot={_subband_plot_per_comp[_i]}, "
				f"n_subbands={_n_subbands_per_comp[_i]}, "
				f"constrain_index={_constrain_index_per_comp[_i]}, "
				f"bw_width_factor={_bw_width_factor_per_comp[_i]}"
			)
	# === END PER-COMPONENT SUB-BAND-FIT NORMALIZATION ===

	# Initialized to None so the load_solutions branch (which skips the
	# boundary-pile-up detector) leaves it unset → downstream code treats
	# all λ as "no special boundary info" and falls back to using the
	# values in tmix_popt directly.
	_tmix_lams_orig = None

	if load_solutions:
		solsdata = np.load(load_solutions, allow_pickle=True)
		tmix_popt, tmix_perr = solsdata['tmix_popt'], solsdata['tmix_perr']
		tmix_redchisq = solsdata['tmix_redchisq']
		subbandpopts = list(solsdata['subbandpopts'])
		subbandperrs = list(solsdata['subbandperrs'])
		subbandredchisqs = list(solsdata['subbandredchisqs'])
		subbandmodels = []
	else:
# 		tmix_popt, tmix_pcov, tmix_redchisq = fitgaussmix(
		# === HARD COMPONENT ISOLATION VIA `cuts` ===
		# When `cuts` is non-empty, each component lives strictly
		# inside its region [cuts[i-1], cuts[i]] and is fit
		# INDEPENDENTLY. The per-region fits respect the user's
		# stated component boundaries — no σ or τ bleed across
		# cuts. Falls back to the joint MIX_FIT_FUNC when no cuts
		# are supplied (preserving the legacy default behaviour).
		if len(cuts) > 0:
			if USE_EMG_MIX:
				print(f"  Using per-region independent EMG fits "
				      f"(cuts = {[f'{c:.4f}' for c in cuts]} ms — components are hard-isolated)")
				tmix_popt, tmix_pcov, tmix_redchisq = fit_per_region_emgmix(
					tseries,
					duration,
					xos=xos,
					cuts=cuts,
					sigmas=sigmas,
					fix_xos=fix_xos,
					tol=tolms,
					dt_ms=res_time_ms,
				)
			else:
				print(f"  Using per-region independent Gaussian fits "
				      f"(cuts = {[f'{c:.4f}' for c in cuts]} ms — components are hard-isolated)")
				tmix_popt, tmix_pcov, tmix_redchisq = fit_per_region_gaussmix(
					tseries,
					duration,
					xos=xos,
					cuts=cuts,
					sigmas=sigmas,
					fix_xos=fix_xos,
					tol=tolms,
				)
		else:
			
			print(f"[DEBUG tmix-input] tseries len={len(tseries)} peak={np.nanmax(tseries):.4f} "
			      f"argmax_t={np.linspace(0,duration,len(tseries))[np.nanargmax(tseries)]:.2f}ms "
			      f"first10mean={np.nanmean(tseries[:10]):.4f} "
			      f"std_off={np.nanstd(tseries[:len(tseries)//5]):.4f}")
			print(f"[DEBUG tmix-input] xos={xos}, sigmas={sigmas}, cuts={cuts}, fix_xos={fix_xos}")
			print(f"[DEBUG tmix-input] duration={duration:.4f}, res_time_ms={res_time_ms:.6f}")

			tmix_popt, tmix_pcov, tmix_redchisq = MIX_FIT_FUNC(
				tseries,
				duration,
				xos=xos,
				sigmas=sigmas,
				fix_xos=fix_xos,
				tol=tolms,
				**(dict(dt_ms=res_time_ms) if USE_EMG_MIX else {})
			)
		tmix_perr = np.sqrt(np.diag(tmix_pcov))
		subbandpopts, subbandmodels, subbandperrs, subbandredchisqs = [], [], [], []

		# === LAMBDA BOUNDARY-PILE-UP DETECTION ===
		# fitemgmix now bounds λ ≤ 1/dt_ms to prevent unresolved-scattering
		# delta-function collapses. If the optimizer presses against that
		# cap, the data carries no real τ information — writing the bound
		# value to the CSV would be misleading (it says "tau = dt_ms" when
		# really tau is unresolved). Detect pile-up and NaN those entries.
		# Also NaN entries whose σ_λ exceeds the parameter value by a wide
		# margin, which catches lingering degenerate fits even when λ is
		# inside the bounds (covariance still blew up).
		#
		# We stash the *original* λ values into `_tmix_lams_orig` BEFORE
		# NaN-ing, so downstream plotting can still draw the EMG curve at
		# the bounded λ (the optimizer's actual best fit, just with τ
		# undetermined). The CSV gets NaN; the figure gets a real curve
		# and a "τ = unresolved" annotation.
		# `_tmix_lams_orig` is already initialised to None above (before
		# both load_solutions and fit branches), so the load path stays
		# clean. Here, before NaN-ing, we overwrite it with the pre-NaN
		# values so plotting can use them.
		if USE_EMG_MIX and isinstance(tmix_popt, (list, tuple, np.ndarray)) and len(tmix_popt) >= 4*n_bursts + 1:
			_tmix_popt_arr = np.array(tmix_popt, dtype=float)
			_tmix_perr_arr = np.array(tmix_perr, dtype=float)
			# Stash original λ values (pre-NaN) so plotting can draw the
			# bounded-EMG curve faithfully even when τ is undetermined.
			_tmix_lams_orig = _tmix_popt_arr[3*n_bursts:4*n_bursts].copy()
			_lam_max = 1.0 / res_time_ms if res_time_ms > 0 else np.inf
			_lam_slice = slice(3*n_bursts, 4*n_bursts)
			for _ci in range(n_bursts):
				_lam = _tmix_popt_arr[3*n_bursts + _ci]
				_lam_err = _tmix_perr_arr[3*n_bursts + _ci]
				_at_cap = np.isfinite(_lam_max) and _lam >= 0.99 * _lam_max
				_runaway_err = (np.isfinite(_lam) and _lam > 0
				                and np.isfinite(_lam_err)
				                and _lam_err > 10.0 * abs(_lam))
				if _at_cap or _runaway_err:
					_reason = "pinned at λ_max=1/dt_ms (τ unresolved)" if _at_cap \
					          else "σ_λ > 10·λ (degenerate fit)"
					print(f"  WARNING: tmix fit for component "
					      f"{subburst_suffixes[_ci] if _ci < len(subburst_suffixes) else _ci}: "
					      f"τ = 1/λ undetermined ({_reason}); "
					      f"writing NaN to tau_sc and tau_sc_err.")
					_tmix_popt_arr[3*n_bursts + _ci] = np.nan
					_tmix_perr_arr[3*n_bursts + _ci] = np.nan
			tmix_popt = list(_tmix_popt_arr)
			tmix_perr = _tmix_perr_arr
		# === END LAMBDA BOUNDARY-PILE-UP DETECTION ===

		print(f"[DEBUG tmix-output] tmix_popt={tmix_popt}")
		print(f"[DEBUG tmix-output] tmix_perr={np.sqrt(np.diag(tmix_pcov))}")
		print(f"[DEBUG tmix-output] tmix_redchisq={tmix_redchisq}")
		# Per-component breakdown so the σ and λ values are easy to read.
		try:
			_dbg_n = len(xos)
			with np.printoptions(precision=6, suppress=False):
				print(f"[DEBUG tmix-output] parsed:  "
				      f"A={list(np.atleast_1d(tmix_popt)[0:_dbg_n])},  "
				      f"mu={list(np.atleast_1d(tmix_popt)[_dbg_n:2*_dbg_n])},  "
				      f"sigma={list(np.atleast_1d(tmix_popt)[2*_dbg_n:3*_dbg_n])},  "
				      f"lambda={list(np.atleast_1d(tmix_popt)[3*_dbg_n:4*_dbg_n])},  "
				      f"C={tmix_popt[4*_dbg_n] if len(tmix_popt) > 4*_dbg_n else 'N/A'}")
				_dbg_perr = np.sqrt(np.diag(tmix_pcov))
				print(f"[DEBUG tmix-output] parsed-err: "
				      f"sigma_err={list(_dbg_perr[2*_dbg_n:3*_dbg_n])},  "
				      f"lambda_err={list(_dbg_perr[3*_dbg_n:4*_dbg_n])}")
		except Exception as _e:
			print(f"[DEBUG tmix-output] parse failed: {_e}")

	if len(tmix_perr.shape) == 2: tmix_perr = np.diag(tmix_perr) # handles when pcov is nans

	# === DIAGNOSTIC: EMG MIXTURE FIT COVARIANCE ANALYSIS ===
	# (Calculations kept for summary at end, prints commented out)
	
	# Parameter layout for EMG: [A1..AN, mu1..muN, sigma1..sigmaN, lambda1..lambdaN, C]
	if USE_EMG_MIX:
		_diag_param_names = []
		for i in range(n_bursts):
			_diag_param_names.append(f"A_{subburst_suffixes[i]}")
		for i in range(n_bursts):
			_diag_param_names.append(f"mu_{subburst_suffixes[i]}")
		for i in range(n_bursts):
			_diag_param_names.append(f"sigma_{subburst_suffixes[i]}")
		for i in range(n_bursts):
			_diag_param_names.append(f"lambda_{subburst_suffixes[i]}")
		_diag_param_names.append("C")
	else:
		_diag_param_names = []
		for i in range(n_bursts):
			_diag_param_names.append(f"A_{subburst_suffixes[i]}")
		for i in range(n_bursts):
			_diag_param_names.append(f"xo_{subburst_suffixes[i]}")
		for i in range(n_bursts):
			_diag_param_names.append(f"sigma_{subburst_suffixes[i]}")
		_diag_param_names.append("C")
	
	# Compute correlation matrix (store for summary)
	_diag_tmix_std = np.sqrt(np.diag(tmix_pcov))
	with np.errstate(divide='ignore', invalid='ignore'):
		_diag_tmix_corr = tmix_pcov / np.outer(_diag_tmix_std, _diag_tmix_std)
		_diag_tmix_corr = np.nan_to_num(_diag_tmix_corr, nan=0.0, posinf=0.0, neginf=0.0)
	
	# Condition number (store for summary)
	try:
		_diag_cond_num = np.linalg.cond(tmix_pcov)
	except:
		_diag_cond_num = np.nan
	# === END EMG MIXTURE FIT COVARIANCE (calculations only) ===
	tmix_amps       = tmix_popt[:n_bursts]
	tmix_xos        = tmix_popt[n_bursts:n_bursts*2]
	tmix_xos_errs   = tmix_perr[n_bursts:n_bursts*2]
	tmix_sigmas     = tmix_popt[n_bursts*2:n_bursts*3]
	tmix_sigma_errs = tmix_perr[n_bursts*2:n_bursts*3]
	# --- Part 2 additions: define burst-centers x_c and tail-aware widths for EMG ---
	if USE_EMG_MIX:
		tmix_lambdas = tmix_popt[3*n_bursts:4*n_bursts]
		# x_o = tmix_xos (mu). Define x_c = mu + 1/lambda (mean) if lambda>0, else fallback to mu.
		xcs = []
		for mu_i, lam_i in zip((tmix_xos if isinstance(tmix_xos, (list, np.ndarray)) else tmix_xos.tolist()), np.atleast_1d(tmix_lambdas)):
			if np.isfinite(lam_i) and lam_i > 0:
				xcs.append(float(mu_i + 1.0/lam_i))
			else:
				xcs.append(float(mu_i))
		# Tail-aware std: sqrt(sigma_core^2 + 1/lambda^2)
		tmix_sigmas_std = []
		for sig_i, lam_i in zip(np.atleast_1d(tmix_sigmas), np.atleast_1d(tmix_lambdas)):
			if np.isfinite(lam_i) and lam_i > 0 and np.isfinite(sig_i):
				tmix_sigmas_std.append(float(np.sqrt(sig_i**2 + (1.0/lam_i**2))))
			else:
				tmix_sigmas_std.append(float(sig_i if np.isfinite(sig_i) else np.nan))
	else:
		# For Gaussian mode, x_c == x_o and 'tail-aware' == core
		xcs = [float(mu_i) for mu_i in (tmix_xos if isinstance(tmix_xos, (list, np.ndarray)) else tmix_xos.tolist())]
		tmix_sigmas_std = [float(s) for s in (tmix_sigmas if isinstance(tmix_sigmas, (list, np.ndarray)) else np.atleast_1d(tmix_sigmas))]

	printd(f"'sigmas': {[f for f in tmix_sigmas]}")

	# === DIAGNOSTIC 1: INTENSITY PROFILE FIT PARAMETERS ===
	# print(f"\n{'='*60}")
	# print(f"[DIAG 1] INTENSITY PROFILE FIT ({'EMG' if USE_EMG_MIX else 'Gaussian'})")
	# print(f"{'='*60}")
	# print(f"  Number of components: {n_bursts}")
	# print(f"  Reduced chi-squared: {tmix_redchisq:.3f}")
	# print(f"  Component centers (xos): {[f'{x:.4f}' for x in tmix_xos]}")
	# print(f"  Component centers (xcs): {[f'{x:.4f}' for x in xcs]}")
	# print(f"  Sigmas (core): {[f'{s:.4f}' for s in tmix_sigmas]}")
	# print(f"  Sigmas (std/tail-aware): {[f'{s:.4f}' for s in tmix_sigmas_std]}")
	# if USE_EMG_MIX:
	# 	print(f"  Lambdas: {[f'{l:.4f}' for l in tmix_lambdas]}")
	# 	print(f"  Taus (1/lambda): {[f'{1/l:.4f}' if l > 0 else 'inf' for l in tmix_lambdas]}")
	# print(f"  Amplitudes: {[f'{a:.4f}' for a in tmix_amps]}")
	# print(f"  Center errors: {[f'{e:.4f}' for e in tmix_xos_errs]}")
	# print(f"  Sigma errors: {[f'{e:.4f}' for e in tmix_sigma_errs]}")
	# === END DIAGNOSTIC 1 ===

	# === SUBBAND SCATTERING INDEX (ALPHA) FIT — INITIALISATION ===
	# The actual fit_subband_alpha call has been moved inside the per-component
	# results loop (around line ~5491) so that pkfreq / bwidth (from the
	# per-component outer spectral fit) are in scope and can be passed as
	# spec_center_outer / spec_sigma_outer. This unifies the sub-band splitting
	# window with the spectral Gaussian plotted in the side panel, making the
	# waterfall cyan boundary lines, the side-panel cyan fill, the sub-band
	# dividers, and the spectral Gaussian curve all come from the same fit.
	#
	# Each component produces one _comp_sb_result, appended in order into
	# _comp_sb_results. Downstream code that previously used _sb_alpha_result
	# (the old pre-loop full-waterfall result) now uses _comp_sb_results[0]
	# (assigned below, after the loop) as the representative burst-level
	# result. The CSV row for component i uses _comp_sb_results[i] directly.
	_sb_alpha_result = None
	_comp_sb_results = []  # per-component results, filled inside the loop
	_comp_acf_results = []  # per-component ACF results, parallel to _comp_sb_results
	# === END SUBBAND ALPHA FIT — INITIALISATION ===

	xos = tmix_xos if type(tmix_xos) == list else tmix_xos.tolist() # align to fit component centers
	xos_errs = tmix_xos_errs if type(tmix_xos_errs) == list else tmix_xos_errs.tolist()

	# === DIAGNOSTIC: TRACING ARRIVAL TIME UNCERTAINTIES ===
	# (Commented out - moved to summary at end)
	# print(f"\n{'+'*80}")
	# print(f"+ DIAGNOSTIC: TRACING HORIZONTAL ERROR BAR ORIGIN")
	# print(f"{'+'*80}")
	# print(f"  xos_errs (arrival time uncertainties) = tmix_xos_errs")
	# print(f"  tmix_xos_errs = tmix_perr[n_bursts : 2*n_bursts]")
	# print(f"  tmix_perr = sqrt(diag(tmix_pcov))")
	# print(f"  tmix_pcov = covariance matrix from curve_fit of EMG/Gaussian mixture")
	# print(f"\n  Current values:")
	# for i, (xo, xo_err) in enumerate(zip(xos, xos_errs)):
	# 	print(f"    Component {subburst_suffixes[i]}: xo = {xo:.4f} ms, xo_err = {xo_err:.6f} ms")
	# 	# Show what pcov diagonal element this came from
	# 	mu_idx = n_bursts + i  # index in parameter vector
	# 	if mu_idx < len(tmix_perr):
	# 		variance = tmix_pcov[mu_idx, mu_idx]
	# 		print(f"      -> tmix_pcov[{mu_idx},{mu_idx}] = {variance:.6e}")
	# 		print(f"      -> sqrt({variance:.6e}) = {np.sqrt(variance):.6f} ms")
	# print(f"{'+'*80}\n")
	# === END DIAGNOSTIC ===

	if not submasks:
		submasks = ([],)*len(xos)
	else:
		if len(submasks) != len(xos):
			raise ValueError(f"Please ensure the length of xos and submasks match. {len(xos) = } {len(submasks) = }")

	if not bandmask_thres:
		bandmask_thres = (None,)*len(xos)
	elif type(bandmask_thres) != list: # assuming its a number, apply it to all components
		bandmask_thres = [bandmask_thres,]*len(xos)
	elif type(bandmask_thres) == list and (len(bandmask_thres) != len(xos)):
		raise ValueError(f"Please ensure the length of xos and bandmask_thres match. {len(xos) = } {len(bandmask_thres) = }")

	subfalls = []
	subbands = []
	# sample of noise levels matching
	# number of channels used in corresponding subband integration
	# from beginning of waterfall
	noisesmpls = []
	bandpass = np.zeros(wfall.shape[0])
	xcs_chans = np.floor(np.array(xcs)/res_time_ms) if 'xcs' in locals() or 'xcs' in globals() else np.floor(np.array(xos)/res_time_ms)
	xos_chans = np.floor(np.array(xos)/res_time_ms)
	# noise_edges is populated below by the GLOBAL OFF-PULSE MASK block
	# from the contiguous off-pulse runs in `_offpulse_mask_global`.
	# === ADAPTIVE EMG WINDOW: Compute tau_sc extension for stacking ===
	# When stacking is enabled, extend the subfall window to the right by tau_sc
	# to accommodate the frequency-dependent scattering tail at lower frequencies
	tau_sc_extension_chans = 0  # Default: no extension
	tau_sc_ms = 0.0  # tau_sc in milliseconds for later use
	original_subfall_duration_chans = 0  # Track original window size (before extension)
	
	if USE_ADAPTIVE_EMG_WINDOW and USE_EMG_MIX and USE_SIGNAL_STACKING:
		# Get tau_sc from full-band intensity profile EMG fit
		if len(tmix_lambdas) > 0 and np.isfinite(tmix_lambdas[0]) and tmix_lambdas[0] > 0:
			tau_sc_ms = 1.0 / tmix_lambdas[0]  # scattering timescale in ms
			tau_sc_extension_chans = int(np.ceil(tau_sc_ms / res_time_ms))  # convert to channels
			print(f"  Adaptive EMG window: tau_sc = {tau_sc_ms:.4f} ms = {tau_sc_extension_chans} channels extension")
		else:
			print(f"  Warning: Could not compute tau_sc for adaptive window (lambda={tmix_lambdas[0] if len(tmix_lambdas) > 0 else 'N/A'})")
	# === END ADAPTIVE EMG WINDOW SETUP ===

	# === GLOBAL OFF-PULSE MASK + ROBUST PER-CHANNEL NOISE ESTIMATE ===
	# Replaces the legacy per-component "slab at start of waterfall"
	# noise sample (which fired the "Noise sample overlaps with pulse
	# region" warning whenever the earliest component lived close to
	# the start of the cropped waterfall, and silently inflated noise
	# even when it didn't warn). The new approach:
	#
	#   1. Compute on-pulse intervals from the multi-component intensity
	#      fit, using a strict (1%) decay threshold per component.
	#   2. Off-pulse mask = all bins outside any on-pulse interval.
	#      Adaptive fallback to looser thresholds (5%, 10%, 20%) if
	#      strict mask gives <50 off-pulse bins.
	#   3. Per-channel noise = 1.4826 × MAD over the off-pulse bins
	#      (rescaled so the value is comparable to what std would
	#      have produced under pure Gaussian noise — keeps existing
	#      snr_cutoff calibrations valid).
	#   4. Same noise array is used for ALL components (per-channel
	#      noise is a property of the receiver and the data, not of
	#      which component we're currently fitting).
	#
	# The on-pulse intervals computed here are the SINGLE SOURCE OF
	# TRUTH — the green/orange visual bar reads the same intervals
	# from `_on_segments_global_ms`, so the bar is now a faithful
	# diagnostic of what the noise computation actually sees as
	# on-pulse. If components overlap in time, the bar will show it.
	_on_segments_global_ms, _offpulse_mask_global, _offpulse_thr_used, _offpulse_n_bins = (
		compute_onpulse_intervals_and_offpulse_mask(
			tmix_popt=tmix_popt,
			n_bursts=n_bursts,
			USE_EMG_MIX=USE_EMG_MIX,
			times_ms=times_ms,
			res_time_ms=res_time_ms,
			xos=xos,
			cuts=cuts,
		)
	)
	_hard_iso_active = (cuts is not None and len(cuts) > 0)
	if _hard_iso_active:
		print(
			f"  On-pulse intervals from `cuts` (hard component isolation): "
			f"{[f'[{a:.4f},{b:.4f}]' for (a,b) in _on_segments_global_ms]} ms"
		)
		print(
			f"  Off-pulse mask: {_offpulse_n_bins}/{len(times_ms)} bins "
			f"({100.0*_offpulse_n_bins/max(1,len(times_ms)):.1f}%), "
			f"decay threshold for leading/trailing edges = "
			f"{_offpulse_thr_used*100:.1f}% of peak"
		)
	else:
		print(
			f"  Off-pulse mask: {_offpulse_n_bins}/{len(times_ms)} bins "
			f"({100.0*_offpulse_n_bins/max(1,len(times_ms)):.1f}%), "
			f"decay threshold = {_offpulse_thr_used*100:.1f}% of peak"
		)
	if _offpulse_n_bins < 20:
		print(
			f"  Warning: off-pulse mask has only {_offpulse_n_bins} bins — "
			f"components fill nearly the entire cropped waterfall. Noise "
			f"estimate may be unreliable. Consider widening `crop`."
		)
	# Compute per-channel rescaled-MAD noise once for the whole burst.
	# If the off-pulse mask is empty (degenerate case), fall back to
	# using the whole waterfall — noisy but at least the pipeline runs.
	if _offpulse_n_bins >= 1:
		_offpulse_data = wfall[:, _offpulse_mask_global]
	else:
		print("  Warning: off-pulse mask is empty; using whole waterfall for noise estimate.")
		_offpulse_data = wfall
	_offpulse_median = np.median(_offpulse_data, axis=1, keepdims=True)
	_noisesmpl_global = 1.4826 * np.median(
		np.abs(_offpulse_data - _offpulse_median), axis=1
	)
	# Guard against degenerate channels (all off-pulse values identical
	# → MAD=0). Replace with the median noise across non-zero channels
	# so SNR doesn't blow up to infinity downstream.
	_nz_noise = _noisesmpl_global[_noisesmpl_global > 0]
	if len(_nz_noise) > 0:
		_fallback_noise = float(np.median(_nz_noise))
		_noisesmpl_global = np.where(
			_noisesmpl_global > 0, _noisesmpl_global, _fallback_noise
		)
	# Populate `noise_edges` for the existing red-dashed-line plotter
	# on the dynamic spectrum. Each contiguous off-pulse run becomes
	# one (start_chan, end_chan) tuple. Replaces the previous behaviour
	# of one entry per component pointing at the leading-edge slab.
	noise_edges = []
	if _offpulse_mask_global.any():
		_mask_arr = np.asarray(_offpulse_mask_global).astype(np.int8)
		# Edges where the mask transitions on/off
		_diff = np.diff(np.concatenate(([0], _mask_arr, [0])))
		_starts = np.where(_diff == 1)[0]
		_ends = np.where(_diff == -1)[0] - 1
		for _s, _e in zip(_starts, _ends):
			noise_edges.append((int(_s), int(_e)))
	# === END GLOBAL OFF-PULSE MASK + NOISE ESTIMATE ===

	# Per-component record of the subfall's actual left edge (in raw
	# waterfall bin units). Needed downstream by the shift application
	# at the "subfall-local -> raw-waterfall" coordinate step, which
	# previously used the *uncliped* expression xc - 4*sig_emg_std and
	# silently over-shifted whenever the proposed window ran past the
	# left edge of the waterfall (e.g. L741965_b component a, where
	# 4*sig_emg_std = 114.5 ms but xc = 93 ms, so the proposed left
	# edge is at -21 ms in raw coords). The actual subfall always
	# starts at bin `subfall_left_chan`, regardless of whether the
	# left edge was clipped; using this value is the only way the
	# global shift can be self-consistent with where the subfall
	# actually begins.
	_subfall_left_chan_per_comp = {}

	for _prep_comp_idx, (xoi, s) in enumerate(zip((xcs_chans if USE_EMG_MIX else xos_chans), (tmix_sigmas_std if USE_EMG_MIX else tmix_sigmas))):
		s4 = np.floor(4*np.abs(s)/res_time_ms)
		s1 = np.floor(1*np.abs(s)/res_time_ms)
		if s4 == 0 or s1 == 0:
			s4 = 4 # hack
			s1 = 1 # hack
			lbl = subburst_suffixes[np.where(xos_chans == xoi)[0][0]]
			print(
				f"Warning: Component ({lbl}) has width below the time resolution, possibly due to poor 1D fit. Using 1ÃÆ’ width as 1 channel."
			)


		# Track original symmetric window size (8*sigma_std = 2*s4 channels)
		original_subfall_duration_chans = int(2 * s4 + 1)

		if len(cuts) == 0:
			# Compute right edge extension for adaptive EMG stacking
			_use_stacking_prep = bool(_stacking_per_comp[_prep_comp_idx]) if _prep_comp_idx < len(_stacking_per_comp) else False
			right_extension = tau_sc_extension_chans if (USE_ADAPTIVE_EMG_WINDOW and USE_EMG_MIX and _use_stacking_prep) else 0
			
			# account for when the edge is outside of wfall
			if xoi-s4 < 0:
				# Left edge clipped: extend from 0 to xoi+s4+extension (clamped to wfall size)
				right_edge = min(int(xoi + s4 + right_extension) + 1, wfall.shape[1])
				subfall = wfall[..., :right_edge]
				subfall_left_chan = 0
			else:
				# Normal case: symmetric window + right extension
				left_edge = int(xoi - s4)
				right_edge = min(int(xoi + s4 + right_extension) + 1, wfall.shape[1])
				subfall = wfall[..., left_edge:right_edge]  # extended window for scattering tail
				subfall_left_chan = left_edge
		else:
			cutchans = np.floor(np.array(cuts)/res_time_ms).astype(int)
			if xoi < cutchans[0]: # left edge
				subfall = wfall[..., :cutchans[0]]
				subfall_left_edge_ms = 0.0  # DIAGNOSTIC: track left edge in ms
				subfall_left_chan = 0
				if diag_enabled('SUBFALL'):
					print(f"\n=== SUBFALL DIAGNOSTIC (Component at xoi={xoi:.3f} ms) ===")
					print(f"  Region: LEFTMOST (before first cut)")
					print(f"  subfall slice: wfall[..., 0:{cutchans[0]}]")
					print(f"  subfall_left_edge_ms = {subfall_left_edge_ms:.3f} ms")
					print(f"  subfall duration = {subfall.shape[1] * res_time_ms:.3f} ms")
			elif xoi > cutchans[-1]: # right edge
				subfall = wfall[..., cutchans[-1]:]
				subfall_left_edge_ms = cuts[-1]  # DIAGNOSTIC: track left edge in ms
				subfall_left_chan = int(cutchans[-1])
				if diag_enabled('SUBFALL'):
					print(f"\n=== SUBFALL DIAGNOSTIC (Component at xoi={xoi:.3f} ms) ===")
					print(f"  Region: RIGHTMOST (after last cut)")
					print(f"  subfall slice: wfall[..., {cutchans[-1]}:]")
					print(f"  subfall_left_edge_ms = {subfall_left_edge_ms:.3f} ms")
					print(f"  subfall duration = {subfall.shape[1] * res_time_ms:.3f} ms")
			else: # middle
				ci = 0
				while xoi > cutchans[ci]: ci += 1
				prev_ci = ci-1

				ci = -1
				while xoi < cutchans[ci]: ci -= 1
				next_ci = ci+1

				subfall = wfall[..., cutchans[prev_ci]:cutchans[next_ci]]
				subfall_left_edge_ms = cuts[prev_ci]  # DIAGNOSTIC: track left edge in ms
				subfall_left_chan = int(cutchans[prev_ci])
				if diag_enabled('SUBFALL'):
					print(f"\n=== SUBFALL DIAGNOSTIC (Component at xoi={xoi:.3f} ms) ===")
					print(f"  Region: MIDDLE (between cuts[{prev_ci}] and cuts[{next_ci}])")
					print(f"  subfall slice: wfall[..., {cutchans[prev_ci]}:{cutchans[next_ci]}]")
					print(f"  subfall_left_edge_ms = {subfall_left_edge_ms:.3f} ms")
					print(f"  subfall duration = {subfall.shape[1] * res_time_ms:.3f} ms")

		# Persist the actual left edge of this component's subfall so
		# the downstream "subfall-local -> raw-waterfall" shift can
		# use the *real* left edge rather than the uncliped expression
		# xc - 4*sig_emg_std. See note at the `_subfall_left_chan_per_comp`
		# initialisation above for why this matters (L741965_b component
		# a fires the LEFT-CLIPPED branch and was off by ~21 ms).
		_subfall_left_chan_per_comp[_prep_comp_idx] = int(subfall_left_chan)
		globals().setdefault('_INSTR_SUBFALL', {})[_prep_comp_idx] = dict(left=int(subfall_left_chan), nt=int(subfall.shape[1]))  # INSTRUMENTATION

		# Compute component and sample noise
		# Slicing syntax: a[start:stop]  means items start through stop-1
		# Therefore When slicing we add 1 to the end to include the ending channel.
		# Note: noise is now computed ONCE GLOBALLY before this loop via
		# rescaled-MAD on the off-pulse mask (see "GLOBAL OFF-PULSE MASK
		# + ROBUST PER-CHANNEL NOISE ESTIMATE" block above). Every
		# component gets the same `_noisesmpl_global` array because
		# per-channel noise is a property of the receiver, not of which
		# component is being fit. The `bw_width_factor`-driven half-width
		# `sw` is still used to define the per-component spectral
		# integration strip (`subband`), which is what feeds the spectrum
		# the user sees in the right-hand panel.
		ddof = 0 # Bessel's correction = 1
		sw = band_int_width_fac*s1 # integration width
		if xoi-sw < 0: # left edge
			subband = wfall[..., :int(xoi+sw)+1].mean(axis=1)
		elif xoi-sw > wfall.shape[1]:
			subband = wfall.mean(axis=1) # probably bad fit, take it all as a fallback
			if bw_filter != 'model_width':
				print("Warning: subband strip taken from whole waterfall (component sw exceeds wfall). Spectral filter may be overly aggressive.")
		else: # Compute spectrum by summing only band_int_width_fac*sigma from burst peak
			subband = wfall[..., int(xoi-sw):int(xoi+sw)+1].mean(axis=1)
			printd(
				f"{int(xoi-sw) = }, {int(xoi+sw) = }",
				wfall[..., int(xoi-sw):int(xoi+sw)+1].shape,
				int(xoi-sw),
				int(xoi+sw)
			)
		# All components share the global rescaled-MAD noise estimate.
		noisesmpls.append(_noisesmpl_global)

		bandpass += subband
		if len(cuts) == 0 and subtractbg: # need to be careful about bg subtraction when cutting
			subfall = driftrate.subtractbg(subfall, 0, 10) # subtract bg again, left
			subfall = driftrate.subtractbg(subfall, subfall.shape[1]-1-10, None) # right
		subfalls.append(subfall)
		subbands.append(subband)
		
		# === DIAGNOSTIC: Check subband and noise sample values ===
		if SHOW_DIAGNOSTICS:
			_noisesmpl = noisesmpls[-1]  # most recently added
			print(f"\n=== SNR Calculation Diagnostics ===")
			print(f"  xoi (burst center bin): {xoi:.1f}, sw (integration width bins): {sw:.1f}")
			print(f"  wfall shape: {wfall.shape}")
			print(f"  subband shape: {subband.shape}, noisesmpl shape: {_noisesmpl.shape}")
			print(f"  subband: min={np.nanmin(subband):.4f}, max={np.nanmax(subband):.4f}, "
				  f"mean={np.nanmean(subband):.4f}, NaN count={np.sum(np.isnan(subband))}, zero count={np.sum(subband == 0)}")
			print(f"  noisesmpl: min={np.nanmin(_noisesmpl):.4f}, max={np.nanmax(_noisesmpl):.4f}, "
				  f"mean={np.nanmean(_noisesmpl):.4f}, NaN count={np.sum(np.isnan(_noisesmpl))}, zero count={np.sum(_noisesmpl == 0)}")
			_snr_test = subband / _noisesmpl
			print(f"  SNR (subband/noisesmpl): min={np.nanmin(_snr_test):.4f}, max={np.nanmax(_snr_test):.4f}, "
				  f"NaN count={np.sum(np.isnan(_snr_test))}, Inf count={np.sum(np.isinf(_snr_test))}")
			print(f"  Channels with SNR > {snr_cutoff}: {np.sum(_snr_test > snr_cutoff)} / {len(_snr_test)}")
			# Check for problematic channels
			_nan_channels = np.where(np.isnan(_snr_test))[0]
			if len(_nan_channels) > 0 and len(_nan_channels) <= 10:
				print(f"  NaN SNR at channels: {_nan_channels}")
				for ch in _nan_channels[:5]:  # show first 5
					print(f"    Channel {ch}: subband={subband[ch]:.4f}, noisesmpl={_noisesmpl[ch]:.4f}")
			elif len(_nan_channels) > 10:
				print(f"  NaN SNR at {len(_nan_channels)} channels (showing first 5): {_nan_channels[:5]}")
				for ch in _nan_channels[:5]:
					print(f"    Channel {ch}: subband={subband[ch]:.4f}, noisesmpl={_noisesmpl[ch]:.4f}")
		# === END SNR DIAGNOSTIC ===

	#### Measurements
	dtdnus, intercepts, subdfs = [], [], []
	# Per-component stacked-fit accumulators (one entry per component).
	# Each entry is either None (component not stacked / no surviving fits)
	# or a populated DataFrame. Mirrors `subdfs` indexing.
	stacked_dfs_per_comp = [None] * n_bursts
	stacked_gauss_dfs_per_comp = [None] * n_bursts
	# Per-component ignored-TOA accumulators. The loop-local variables
	# `ignored_stacked_df`, `ignored_stacked_gauss_df`, `ignored_subdf`
	# get overwritten each iteration of the per-component loop, which
	# means anything that reads them outside the loop (e.g. the slope
	# panel re-render via the toggle button) only ever sees the last
	# component's values — producing visual artifacts like component a's
	# slope panel showing component b's ignored stacked X markers. The
	# per-comp arrays below mirror `stacked_dfs_per_comp` and are used
	# by the slope panel and waterfall ignored-marker plots.
	ignored_stacked_dfs_per_comp = [None] * n_bursts
	ignored_stacked_gauss_dfs_per_comp = [None] * n_bursts
	ignored_subdfs_per_comp = [None] * n_bursts
	# Per-component dicts for popup/global state (so each component's
	# stacked profiles/fitdata can be looked up later by component index).
	_stacked_state_per_comp = [None] * n_bursts        # EMG
	_stacked_gauss_state_per_comp = [None] * n_bursts  # Gaussian
	# Per-component marker fill colors for ToA overlays. White was
	# previously the leading color, but component a's white-filled
	# markers were invisible against the bright signal in the
	# dynamic spectrum. Cycle now starts with high-contrast magenta
	# and lime; subsequent colors are chosen to remain distinct on
	# both bright-yellow (signal) and dark-blue (noise floor)
	# backgrounds. The markers are also drawn with a thin black
	# outline (see the ax_wfall.scatter call) so they remain visible
	# regardless of the underlying waterfall pixel value.
	colors = cycle([
		'magenta',
		'lime',
		'red',
		'cyan',
		'orange',
		'yellow',
		'white',
		'black',
	])

	# Capture the burst-wide `ignore` ONCE before entering the per-component
	# loop. Inside the loop we shadow the name `ignore` to inject per-comp
	# extras, so we cannot use `ignore` itself as the source of the
	# burst-wide list on subsequent iterations — that would compound.
	_ignore_burst_wide_const = list(ignore) if ignore else []

	for subfall, subband, xosi, xosi_err, sigma, sigma_err, submask, bandmask_thresi, noisesmpl in zip(
		subfalls,
		subbands,
		xos,
		xos_errs,
		tmix_sigmas,
		tmix_sigma_errs,
		submasks,
		bandmask_thres,
		noisesmpls
	):
		rowname = bname if len(xos) == 1 else f'{bname}_{subburst_suffixes[xos.index(xosi)]}'
		# === PER-COMPONENT STACKING LOCALS ===
		# Within this loop iteration, these variables shadow the legacy
		# USE_SIGNAL_STACKING / STACK_SIZE globals so each component can
		# be processed with its own stacking config.
		comp_idx_loop = xos.index(xosi)

		# === ACF MODE (per component) ===
		# Defined HERE, at the top of the per-component loop, so the
		# variables exist for every component regardless of which later
		# code paths run (the dtdnu override, the results-row append,
		# and the sub-band block all reference them). When ACF is enabled
		# for this component, measure the sub-burst slope from the 2D
		# autocorrelation of the per-component sub-fall instead of from
		# per-channel arrival times. `subfall`, `freqs`, `res_time_ms`
		# are all in scope at this point.
		_comp_acf_on = bool(_acf_per_comp[comp_idx_loop]) \
			if comp_idx_loop < len(_acf_per_comp) else False
		_comp_acf_result = None
		# Skip the (potentially slow) ACF compute for components that
		# subband_diagnostic will skip below — they `continue` before
		# any ACF result is used or stored.
		_acf_diag_skip = (_diag_mode_active
		                  and _diag_comp_set is not None
		                  and comp_idx_loop not in _diag_comp_set)

		# === SCINT MODE (per component) ===
		# Defined at the top of the loop (like the ACF vars) so the row-append
		# below sees them for every non-diag-skipped component. Independent of
		# ACF mode: measures Delta_nu_d from the frequency ACF of the
		# per-component spectrum (subfall collapsed over time). `subfall`,
		# `freqs` are in scope here exactly as for the ACF compute.
		_comp_scint_on = bool(_scint_per_comp[comp_idx_loop]) \
			if comp_idx_loop < len(_scint_per_comp) else False
		_comp_scint_result = None
		if _comp_scint_on and not _acf_diag_skip:
			_scint_spec = np.asarray(subfall, dtype=float).sum(axis=1)
			_scint_cw = float(abs(freqs[1] - freqs[0])) if len(freqs) > 1 else np.nan
			# +1 channel so the band spans edge-to-edge rather than centre-to-centre.
			_scint_bw_total = float(abs(freqs[-1] - freqs[0])) + _scint_cw
			_comp_scint_result = scint_bw.measure_scint_bw(
				_scint_spec, _scint_cw, _scint_bw_total,
				smooth_mhz=scint_smooth_mhz, C1=scint_C1,
			)

		if _comp_acf_on and not _acf_diag_skip:
			_acf_lbl = (subburst_suffixes[comp_idx_loop]
			            if comp_idx_loop < len(subburst_suffixes)
			            else str(comp_idx_loop))
			# --- Plain ACF on the raw sub-fall ---
			_comp_acf_result = acf_slope.acf_measure_component(
				subfall, freqs, res_time_ms,
				comp_label=_acf_lbl, verbose=True,
			)
			# Scattering-pinning flag: note when the major-axis ACF is
			# stretched along the time-lag axis (resolution-normalised axis
			# ratio) and/or |dt/dnu| has blown up — the symptom of a
			# scattering-dominated ACF tilt (see B8/B3/B12/B17/B25).
			if _comp_acf_result is not None and _comp_acf_result.get('ok'):
				_ar = _comp_acf_result.get('axis_ratio_resnorm', np.nan)
				_sl = abs(_comp_acf_result.get('dtdnu', np.nan))
				_ratio_hit = bool(np.isfinite(_ar) and _ar > ACF_flag_axis_ratio)
				_slope_hit = bool(np.isfinite(_sl) and _sl > ACF_flag_slope_msMHz)
				if _ratio_hit or _slope_hit:
					_why = []
					if _ratio_hit:
						_why.append(f"ACF Δt/Δν (res-norm) = {_ar:.2f} > {ACF_flag_axis_ratio:g}")
					if _slope_hit:
						_why.append(f"|dt/dν| = {_sl:.2f} ms/MHz > {ACF_flag_slope_msMHz:g} (blow-up regime)")
					print(f"  [ACF] FLAG: component {_acf_lbl}: " + "; ".join(_why) + ".")
					print(f"  [ACF]   -> this slope is likely scattering-pinned; "
					      f"use the ridge value (acf_ridge_*) for this component "
					      f"instead.")
			# --- Ridge (centroid) drift + error --------------------------
			# Reads the same 2D-Gaussian fit as the centroid slope Var_nu/Cov
			# (well-behaved where the major-axis tan(theta) blows up) and
			# attaches a noise-bootstrap error. Legacy acf_dtdnu (major axis)
			# is left untouched; this only adds the acf_ridge_* fields. NOT
			# gated on _comp_acf_result['ok']: acf_ridge_drift measures the
			# ridge on the raw sub-fall and is failure-safe.
			if ACF_ridge and _comp_acf_result is not None:
				_acf_ridge = acf_slope.acf_ridge_drift(
					subfall, freqs, res_time_ms,
					descattered=None,
					comp_label=_acf_lbl, n_boot=int(ACF_ridge_nboot),
					maskcorrpeak=True,
				)
				_comp_acf_result['ridge'] = _acf_ridge
				print(f"  [ACF-ridge] component {_acf_lbl}: drift = "
				      f"{_acf_ridge['slope']:+.3f} ± {_acf_ridge['err_slope']:.3f} MHz/ms "
				      f"(dt/dnu {_acf_ridge['dtdnu']:+.4f} ± {_acf_ridge['err_dtdnu']:.4f} ms/MHz).")
				# Effective ACF dt/dnu: ridge where the major axis is blown up,
				# major otherwise. Made once here so downstream scripts read a
				# single trustworthy column instead of re-deriving the choice.
				_eff_dtdnu, _eff_err, _eff_src = acf_slope.effective_acf_dtdnu(
					_comp_acf_result, blowup_msMHz=ACF_ridge_blowup_msMHz,
					ratio=ACF_ridge_ratio)
				_comp_acf_result['effective_dtdnu'] = _eff_dtdnu
				_comp_acf_result['effective_dtdnu_err'] = _eff_err
				_comp_acf_result['effective_source'] = _eff_src
				if _eff_src == 'ridge':
					print(f"  [ACF-eff]   component {_acf_lbl}: major-axis blow-up -> "
					      f"using ridge dt/dnu = {_eff_dtdnu:+.4f} ms/MHz")
		# === END ACF MODE ===
		# === PER-COMPONENT ignore LOCAL ===
		# Build the effective ignore list for THIS component as the union
		# of the burst-wide `ignore` (applies to every component) and
		# `_ignore_per_comp_resolved[comp_idx_loop]` (applies to this
		# component only). The downstream code below was originally written
		# to use the burst-wide `ignore` directly, so we shadow that name
		# inside the loop iteration — none of the per-component logic needs
		# to be re-written, it picks up the merged list automatically.
		#
		# Important: this shadow lives only for the current loop iteration.
		# The next iteration computes its own merged list, and code outside
		# the loop continues to see the original `ignore` (burst-wide only).
		_ignore_this_comp_extra = list(_ignore_per_comp_resolved[comp_idx_loop])
		if _ignore_this_comp_extra:
			# Merge: burst-wide items first, then per-component extras.
			# Deduplication of channel ints is done by the downstream
			# expansion logic via `isin()`, so duplicates here are harmless.
			ignore = _ignore_burst_wide_const + _ignore_this_comp_extra
			print(f"  Component {comp_idx_loop} ({rowname}): "
			      f"effective ignore = burst-wide {_ignore_burst_wide_const} "
			      f"+ per-component {_ignore_this_comp_extra}")
		else:
			ignore = list(_ignore_burst_wide_const)
		use_stacking_this_comp = bool(_stacking_per_comp[comp_idx_loop])
		stack_size_this_comp = int(_stack_size_per_comp[comp_idx_loop])
		# Per-component bandwidth override: read once at the top of each
		# loop iteration so the three downstream filters and the
		# fit_subband_alpha call all see the same value for this
		# component without re-indexing the per-comp list each time.
		_bw_co_this = (
			_bw_channel_override_per_comp[comp_idx_loop]
			if comp_idx_loop < len(_bw_channel_override_per_comp)
			else None
		)
		# Same pattern for the ToA-restriction override. Acts only on
		# the slope-fit ToA list (subdf / fit_df), not the sub-band
		# α fit (which uses _bw_co_this).
		_bw_toa_co_this = (
			_bw_toa_override_per_comp[comp_idx_loop]
			if comp_idx_loop < len(_bw_toa_override_per_comp)
			else None
		)
		# === END PER-COMPONENT STACKING LOCALS ===

		# === SUBBAND DIAGNOSTIC MODE: SKIP NON-LISTED COMPONENTS ===
		# When `subband_diagnostic` is set, components not in the list
		# get their expensive per-channel ToA fits (60-90s/component)
		# and sub-band α fits skipped. We push placeholder values to
		# the per-component arrays so downstream code that indexes
		# them by component still finds something at that index, but
		# with no actual data — defensive `len(subdf) > 0` checks
		# already handle empty dataframes throughout the pipeline.
		if _diag_mode_active and comp_idx_loop not in _diag_comp_set:
			print(f"\n>>> DIAGNOSTIC MODE: skipping component "
			      f"{subburst_suffixes[comp_idx_loop]} (1-indexed: {comp_idx_loop+1})",
			      flush=True)
			# Stub per-component arrays so downstream `for ci in range(...)`
			# loops still match `len(xos)`.
			subdfs.append(pd.DataFrame())
			dtdnus.append((np.nan, np.nan, np.nan))
			intercepts.append((np.nan, np.nan))
			_comp_sb_results.append(None)
			_comp_acf_results.append(None)
			# Skip the rest of this iteration's work entirely.
			continue
		# === END SUBBAND DIAGNOSTIC SKIP ===

		for m in submask:
			if type(m) == range:
				m = np.array(m)
			subfall[m//downfactors[0]] = 0

		sigma = abs(sigma)

		# === COMPUTE EMG SEEDS FOR PER-ROW FITS ===
		# Seeding sigma and lambda from the intensity-profile fit prevents the
		# sigma-tau degeneracy from killing per-row EMG fits on noisy channels.
		_row_fit_kwargs = {}
		if USE_EMG_MIX:
			comp_idx = xos.index(xosi)
			_lam_seed = float(tmix_lambdas[comp_idx]) if (
				len(tmix_lambdas) > comp_idx and np.isfinite(tmix_lambdas[comp_idx]) and tmix_lambdas[comp_idx] > 0
			) else None
			_sig_seed = float(sigma) if np.isfinite(sigma) and sigma > 0 else None
			_sig_std  = float(tmix_sigmas_std[comp_idx]) if (
				len(tmix_sigmas_std) > comp_idx and np.isfinite(tmix_sigmas_std[comp_idx])
			) else None

			# mu in subfall-local coordinates:
			#   No cuts: subfall centered on xc = mu + tau, window = ±4*sigma_std
			#            => left_edge ≈ xc - 4*sigma_std  => mu_local ≈ 4*sigma_std - tau
			#   Cuts:    left edge determined by cut boundaries
			_mu_seed = None
			if _sig_std is not None and _lam_seed is not None:
				_tau_seed = 1.0 / _lam_seed
				if len(cuts) == 0:
					_mu_seed = 4.0 * _sig_std - _tau_seed
				else:
					# Replicate the subpktime logic: find which cut region we're in
					_ci, _edge = 0, 0.0
					while _ci < len(cuts) and xosi >= cuts[_ci]:
						_edge = cuts[_ci]
						_ci += 1
					# subpktime ≈ xc_local = xosi - _edge (xosi = mu in global coords)
					# mu_local = xosi - _edge  (since subfall starts at _edge)
					_mu_seed = xosi - _edge
				# Clamp to subfall bounds (left-edge clipping or cuts may shift this)
				subfall_dur = subfall.shape[1] * res_time_ms
				if _mu_seed < 0 or _mu_seed > subfall_dur:
					_mu_seed = None   # fall back to moment estimate inside fitemg

			_row_fit_kwargs = dict(mu0_seed=_mu_seed, sigma0_seed=_sig_seed, lambda0_seed=_lam_seed)
		# === END EMG SEEDS ===

# 		# Fit a 1d gaussian in each row of the waterfall
		subdf = ROWS_FIT_FUNC(subfall, res_time_ms, freqs, plot=show_arrtime, bname=rowname, **_row_fit_kwargs) #add/remove emg
		
		# === DIAGNOSTIC 2: PER-ROW FITTING RESULTS ===
		# print(f"\n{'='*60}")
		# print(f"[DIAG 2] PER-ROW FITTING for {rowname} ({'EMG' if USE_EMG_MIX else 'Gaussian'})")
		# print(f"{'='*60}")
		# print(f"  Total rows (channels): {len(subdf)}")
		# print(f"  Subfall shape: {subfall.shape} (freq x time)")
		# print(f"  Subfall duration: {subfall.shape[1] * res_time_ms:.4f} ms")
		# 
		# # Check fit quality indicators
		# if 'red_chisq' in subdf.columns:
		# 	valid_chisq = subdf['red_chisq'].dropna()
		# 	good_chisq = (valid_chisq < 50).sum()
		# 	print(f"  Reduced chi-sq: valid={len(valid_chisq)}, good(<50)={good_chisq}, bad(>=50 or NaN)={len(subdf)-good_chisq}")
		# 	print(f"    Range: [{valid_chisq.min():.2f}, {valid_chisq.max():.2f}], median={valid_chisq.median():.2f}")
		# 
		# # Check amplitude (indicates successful fit)
		# if 'amp' in subdf.columns:
		# 	valid_amp = subdf['amp'].dropna()
		# 	positive_amp = (valid_amp > 0).sum()
		# 	print(f"  Amplitude: valid={len(valid_amp)}, positive={positive_amp}, zero_or_neg={len(valid_amp)-positive_amp}")
		# 
		# # Time column diagnostics (different for EMG vs Gaussian)
		# if USE_EMG_MIX:
		# 	# EMG-specific columns
		# 	if 't_emg_hwhm' in subdf.columns:
		# 		valid_t = subdf['t_emg_hwhm'].dropna()
		# 		nan_t = subdf['t_emg_hwhm'].isna().sum()
		# 		print(f"  t_emg_hwhm (arrival time): valid={len(valid_t)}, NaN={nan_t}")
		# 		if len(valid_t) > 0:
		# 			print(f"    Range: [{valid_t.min():.4f}, {valid_t.max():.4f}] ms")
		# 	if 't_emg_hwhm_err' in subdf.columns:
		# 		valid_err = subdf['t_emg_hwhm_err'].dropna()
		# 		nan_err = subdf['t_emg_hwhm_err'].isna().sum()
		# 		inf_err = np.isinf(subdf['t_emg_hwhm_err']).sum()
		# 		print(f"  t_emg_hwhm_err: valid={len(valid_err)}, NaN={nan_err}, Inf={inf_err}")
		# 		if len(valid_err) > 0:
		# 			print(f"    Range: [{valid_err.min():.6f}, {valid_err.max():.6f}] ms")
		# 	if 'lambda' in subdf.columns:
		# 		valid_lam = subdf['lambda'].dropna()
		# 		valid_lam_pos = (valid_lam > 0).sum()
		# 		print(f"  Lambda: valid={len(valid_lam)}, positive={valid_lam_pos}")
		# 	if 'tau' in subdf.columns:
		# 		valid_tau = subdf['tau'].dropna()
		# 		print(f"  Tau (1/lambda): valid={len(valid_tau)}")
		# 		if len(valid_tau) > 0:
		# 			print(f"    Range: [{valid_tau.min():.4f}, {valid_tau.max():.4f}] ms")
		# else:
		# 	# Gaussian-specific columns
		# 	if 'tstart' in subdf.columns:
		# 		valid_t = subdf['tstart'].dropna()
		# 		nan_t = subdf['tstart'].isna().sum()
		# 		print(f"  tstart (arrival time): valid={len(valid_t)}, NaN={nan_t}")
		# 		if len(valid_t) > 0:
		# 			print(f"    Range: [{valid_t.min():.4f}, {valid_t.max():.4f}] ms")
		# 	if 'tstart_err' in subdf.columns:
		# 		valid_err = subdf['tstart_err'].dropna()
		# 		nan_err = subdf['tstart_err'].isna().sum()
		# 		print(f"  tstart_err: valid={len(valid_err)}, NaN={nan_err}")
		# 		if len(valid_err) > 0:
		# 			print(f"    Range: [{valid_err.min():.6f}, {valid_err.max():.6f}] ms")
		# 
		# # Common sigma column
		# if 'sigma' in subdf.columns:
		# 	valid_sig = subdf['sigma'].dropna()
		# 	print(f"  Sigma (width): valid={len(valid_sig)}")
		# 	if len(valid_sig) > 0:
		# 		print(f"    Range: [{valid_sig.min():.4f}, {valid_sig.max():.4f}] ms")
		# === END DIAGNOSTIC 2 ===
		
		# === SIGNAL STACKING: Fit stacked profiles if enabled ===
		if USE_EMG_MIX and use_stacking_this_comp:
			if diag_enabled('STACKING_EMG'):
				print(f"\n=== Signal Stacking Enabled (stack size: {stack_size_this_comp}) ===")
			
			# Stack the waterfall channels
			stacked_profiles, stack_center_indices = stack_channels(subfall, stack_size_this_comp)
			
			# Create frequency array for stacked channels (using center frequencies)
			stacked_freqs = freqs[stack_center_indices]
				
			# === EXTRACT SEEDS FROM INTENSITY PROFILE FIT ===
			# Use THIS component's σ/τ from the joint intensity-profile fit
			# as seeds for every stack's per-channel EMG fit. Previously
			# this hardcoded [0], so component a's σ/τ were used as seeds
			# for d's stacks too — which for 11A meant seeding d's wide,
			# scattered stacks with a's narrow values, biasing fits and
			# downstream filters. Use the loop's current component index.
			_seed_idx = comp_idx_loop if comp_idx_loop < len(tmix_sigmas) else 0
			sigma_seed = float(tmix_sigmas[_seed_idx]) if len(tmix_sigmas) > _seed_idx else None
			lambda_seed = float(tmix_lambdas[_seed_idx]) if len(tmix_lambdas) > _seed_idx else None

			# Compute mu in subfall-local coordinates.
			# Two paths:
			#  * No cuts: subfall is centered on xc with width ±4·σ_std,
			#    so mu_local = 4·σ_std - τ.
			#  * With cuts: subfall is the slice between cuts (not
			#    centered on xc), so mu_local = xosi - subfall_left_edge - τ
			#    (since EMG mean = mu + tau, and the data peak sits near xosi).
			# Using the symmetric-formula mu_local in cuts mode was
			# putting the seed peak in the wrong place and causing many
			# stacks to fail Filter 1 (amp > 0) — the optimiser couldn't
			# find the actual signal from a far-off seed.
			# Note: `xosi` is the loop variable IN MILLISECONDS; the
			# nearby variable `xoi` is the same component's center in
			# CHANNEL UNITS (see line ~5393), so do NOT use xoi here.
			#
			# WARNING: do NOT use `subfall_left_chan` from the outer
			# subfall-construction loop. That loop is a SEPARATE
			# iteration over xcs_chans (line ~5393) and leaves
			# `subfall_left_chan` holding whatever value it had at the
			# end of its last iteration — typically the rightmost
			# component's left edge. Using that stale value gave every
			# component d's left edge, making mu_seed wrong for a/b/c.
			# Recompute the left edge here from this component's xosi
			# (in ms) and the current `cuts`.
			if (sigma_seed is not None and lambda_seed is not None and
				np.isfinite(sigma_seed) and np.isfinite(lambda_seed) and lambda_seed > 0):
				tau_seed = 1.0 / lambda_seed
				sigma_std_seed = (
					float(tmix_sigmas_std[_seed_idx])
					if len(tmix_sigmas_std) > _seed_idx
					else np.sqrt(sigma_seed**2 + tau_seed**2)
				)
				if len(cuts) > 0:
					# Determine this component's subfall left edge from
					# its xosi (ms) and the cuts boundaries (ms).
					_cuts_sorted = sorted(float(c) for c in cuts)
					if xosi < _cuts_sorted[0]:
						_this_subfall_left_ms = 0.0
					elif xosi > _cuts_sorted[-1]:
						_this_subfall_left_ms = _cuts_sorted[-1]
					else:
						# Middle: find the cut immediately before xosi.
						_this_subfall_left_ms = max(
							c for c in _cuts_sorted if c <= xosi
						)
					# Quantize to bin grid so the resulting mu_local
					# matches the actual subfall's bin-0 time.
					_this_subfall_left_chan = int(
						np.floor(_this_subfall_left_ms / res_time_ms)
					)
					_subfall_left_ms = _this_subfall_left_chan * res_time_ms
					mu_seed = float(xosi - _subfall_left_ms - tau_seed)
				else:
					# No cuts: symmetric window centred on xc.
					mu_seed = 4.0 * sigma_std_seed - tau_seed
				# Clamp to a reasonable range within the subfall.
				_subfall_dur_ms = res_time_ms * subfall.shape[1]
				mu_seed = float(np.clip(mu_seed, res_time_ms, _subfall_dur_ms - res_time_ms))
			else:
				mu_seed = None
				sigma_seed = None
				lambda_seed = None

			print(f"  Intensity profile seeds (component {subburst_suffixes[comp_idx_loop]}): "
			      f"mu={mu_seed}, sigma={sigma_seed}, lambda={lambda_seed}")
			
			# === ADAPTIVE WINDOW PARAMETERS ===
			if USE_ADAPTIVE_EMG_WINDOW:
				# Compute frequency range for linear interpolation of window extension
				nu_max = np.max(stacked_freqs)  # highest frequency (0 extra time)
				nu_min = np.min(stacked_freqs)  # lowest frequency (tau_sc extra time)
				nu_range = nu_max - nu_min if nu_max != nu_min else 1.0  # avoid division by zero
				
				# Base window duration (original symmetric window before extension)
				base_window_bins = original_subfall_duration_chans
				base_window_ms = res_time_ms * base_window_bins
				
				print(f"  Adaptive window: nu_range=[{nu_min:.2f}, {nu_max:.2f}] MHz, base_window={base_window_ms:.4f} ms ({base_window_bins} bins), tau_sc={tau_sc_ms:.4f} ms")
			else:
				# Non-adaptive: use full subfall for all frequencies
				nu_max = np.max(stacked_freqs)
				nu_min = np.min(stacked_freqs)
				nu_range = nu_max - nu_min if nu_max != nu_min else 1.0
				base_window_bins = subfall.shape[1]
				base_window_ms = res_time_ms * base_window_bins
			# === END ADAPTIVE WINDOW PARAMETERS ===

			# Fit EMG to each stacked profile
			stacked_fitdata = np.full((len(stacked_profiles), 20), np.nan, dtype=float)
			
			for i, stacked_row in enumerate(stacked_profiles):
				# Determine actual number of channels in this stack (last stack may have fewer)
				start_idx = i * stack_size_this_comp
				end_idx = min((i + 1) * stack_size_this_comp, len(freqs))
				n_channels_in_stack = end_idx - start_idx
				
				# === ADAPTIVE WINDOW: Compute extra time for this frequency ===
				if USE_ADAPTIVE_EMG_WINDOW:
					# extra_time scales linearly from 0 at nu_max to tau_sc at nu_min
					stack_freq = stacked_freqs[i]
					extra_time_ms = tau_sc_ms * (nu_max - stack_freq) / nu_range
					extra_bins = int(np.round(extra_time_ms / res_time_ms))
					
					# Total window for this stack: base + extra
					adaptive_window_bins = min(base_window_bins + extra_bins, stacked_row.shape[0])
					adaptive_window_ms = res_time_ms * adaptive_window_bins
					
					# Slice the stacked profile to use only the adaptive window
					# (left side starts at 0, right side is adaptive)
					adaptive_row = stacked_row[:adaptive_window_bins]
				else:
					# Non-adaptive: use full stacked row
					extra_time_ms = 0.0
					adaptive_window_ms = res_time_ms * stacked_row.shape[0]
					adaptive_row = stacked_row
				# === END ADAPTIVE WINDOW ===
				
				# Normalize the stacked profile by dividing by number of channels
				# This gives it the same noise characteristics as individual channels
				# while benefiting from sqrt(n) SNR improvement
				normalized_row = adaptive_row / n_channels_in_stack
				
				# μ bound: confine the per-stack EMG fit to a window
				# around the intensity-profile seed (±3σ_seed). Without
				# it, low-SNR stacked rows let curve_fit drift μ onto an
				# unrelated noise peak — producing arrival times tens of
				# ms from the true burst, which then get rejected
				# wholesale by the stacked-TOA time-window filter.
				if (mu_seed is not None and np.isfinite(mu_seed)
						and sigma_seed is not None and np.isfinite(sigma_seed)
						and sigma_seed > 0):
					_mu_halfwin_e = 3.0 * abs(sigma_seed)
					_mu_lo_stk_e = mu_seed - _mu_halfwin_e
					_mu_hi_stk_e = mu_seed + _mu_halfwin_e
				else:
					_mu_lo_stk_e = _mu_hi_stk_e = None

				popt, pcov, redchisq = fitemg(
					normalized_row, 
					adaptive_window_ms,  # Use adaptive duration (or full duration if toggle off)
					mu0_seed=mu_seed,
					sigma0_seed=sigma_seed,
					lambda0_seed=lambda_seed,
					# Bound the per-stack fit to physical scales: σ and τ
					# can't go below the time bin (sub-resolution is
					# unphysical and causes delta-function pathologies),
					# and τ can't exceed the adaptive window (a tail
					# longer than what we're fitting is meaningless).
					# σ_max generous (window width) so wide stacks still
					# fit. These bounds prevent the optimiser from
					# landing on degenerate corners of EMG parameter
					# space, which was the cause of Filter 1A killing
					# 80%+ of stacks for narrow components like 11A
					# component a.
					sigma_min=float(res_time_ms),
					sigma_max=float(adaptive_window_ms),
					tau_min=float(res_time_ms),
					tau_max=float(adaptive_window_ms),
					mu_min=_mu_lo_stk_e,
					mu_max=_mu_hi_stk_e,
				)
				
				# Extract parameters (same structure as fitrowsemg)
				if isinstance(pcov, np.ndarray) and pcov.ndim == 2:
					perr = np.sqrt(np.diag(pcov))
				else:
					perr = np.full(5, np.nan, dtype=float)
				
				A, mu, sig_core, lam, C = popt if len(popt) == 5 else ([np.nan]*5)
				
				# Rescale amplitude and baseline back to summed scale for consistency
				A = A * n_channels_in_stack
				C = C * n_channels_in_stack
				A_err = (perr[0] * n_channels_in_stack) if len(perr) > 0 else np.nan
				
				# Compute derived quantities (matching fitrowsemg)
				lambda_err = perr[3] if len(perr) > 3 else np.nan
				tau = (1.0/lam) if (np.isfinite(lam) and lam > 0) else np.nan
				tau_err = (lambda_err / (lam**2)) if (np.isfinite(lambda_err) and np.isfinite(lam) and lam > 0) else np.nan
				
				# Compute arrival time: depends on PEAK_INTENSITY_TOA toggle
				# When PEAK_INTENSITY_TOA=False (default): use left HWHM crossing (leading edge)
				# When PEAK_INTENSITY_TOA=True: use peak of EMG function
				t_emg_hwhm = np.nan
				try:
					if np.isfinite(mu) and np.isfinite(sig_core) and np.isfinite(tau) and (sig_core > 0) and (tau > 0):
						if PEAK_INTENSITY_TOA:
							# Peak mode: find the maximum of the EMG function numerically
							t_emg_hwhm = find_emg_peak(mu=float(mu), sigma=float(sig_core), lamb=float(lam))
						else:
							# Leading edge mode: use left HWHM crossing
							stats = emg_left_hwhm(mu=float(mu), sigma=float(sig_core), tau=float(tau))
							t_emg_hwhm = float(stats["x_left"])
				except Exception:
					pass
				
				# === OPTION 2: Better uncertainty estimate for stacked arrival times ===
				# Compute arrival time uncertainty (improved version)
				t_emg_hwhm_err = np.nan
				if np.isfinite(sig_core) and sig_core > 0:
					# Use full EMG width instead of just 0.2*sigma
					# For stacked channels, account for reduced statistical uncertainty
					# but also increased systematic uncertainty from stacking
					t_emg_hwhm_err = sig_core  # Full sigma provides more realistic uncertainty
					# Alternative: t_emg_hwhm_err = sig_core / np.sqrt(stack_size_this_comp)
				# === END OPTION 2 ===
				
				# Store fitted parameters (matching fitrowsemg column structure,
				# extended with C at column 19 so the click-popup can reconstruct
				# the EMG curve with its actual baseline instead of assuming C=0).
				stacked_fitdata[i, :] = [
					stacked_freqs[i],      # 0: freqs
					np.nan,                # 1: tstart (not used)
					np.nan,                # 2: tend (not used)
					A,                     # 3: amp (rescaled)
					mu,                    # 4: xo
					np.nan,                # 5: tstart_err (not used)
					sig_core,              # 6: sigma
					redchisq,              # 7: red_chisq
					A_err,                 # 8: amp_err (rescaled)
					perr[1] if len(perr) > 1 else np.nan,  # 9: xo_err
					perr[2] if len(perr) > 2 else np.nan,  # 10: sigma_err
					lam,                   # 11: lambda
					tau,                   # 12: tau
					lambda_err,            # 13: lambda_err
					tau_err,               # 14: tau_err
					np.nan,                # 15: sigma_emg_std (not computed for stacked)
					np.nan,                # 16: sigma_emg_std_err
					t_emg_hwhm,            # 17: t_emg_hwhm (arrival time)
					t_emg_hwhm_err,        # 18: t_emg_hwhm_err
					C,                     # 19: baseline C (rescaled) — NEW
				]
			
				# === STACKED EMG FIT DIAGNOSTICS ===
			if diag_enabled('STACKING_EMG'):
				print(f"\n=== Stacked EMG Fit Diagnostics ({len(stacked_profiles)} stacks of {stack_size_this_comp} channels) ===")
				if USE_ADAPTIVE_EMG_WINDOW:
					print(f"  Adaptive window enabled: base={base_window_ms:.3f}ms, extension range=[0, {tau_sc_ms:.3f}]ms")
					print(f"{'Stack':<6} {'Chans':<12} {'Freq (MHz)':<12} {'Window(ms)':>10} {'redchi':>8} {'A':>10} {'mu':>10} {'sigma':>10} "
						  f"{'lambda':>10} {'t_arr':>10} {'t_err':>10} {'Status':<20}")
				else:
					print(f"  Adaptive window disabled: using fixed window={base_window_ms:.3f}ms for all frequencies")
					print(f"{'Stack':<6} {'Chans':<12} {'Freq (MHz)':<12} {'redchi':>8} {'A':>10} {'mu':>10} {'sigma':>10} "
						  f"{'lambda':>10} {'t_arr':>10} {'t_err':>10} {'Status':<20}")
				print("-" * 150)
				
				for i in range(len(stacked_profiles)):
					start_ch = i * stack_size_this_comp
					end_ch = min((i + 1) * stack_size_this_comp, len(freqs)) - 1
					freq_center = stacked_freqs[i]
					
					# Recompute adaptive window for display
					if USE_ADAPTIVE_EMG_WINDOW:
						extra_time_diag = tau_sc_ms * (nu_max - freq_center) / nu_range
						adaptive_window_diag = base_window_ms + extra_time_diag
					else:
						extra_time_diag = 0.0
						adaptive_window_diag = base_window_ms
					
					# Extract values from stacked_fitdata
					redchi = stacked_fitdata[i, 7]
					A = stacked_fitdata[i, 3]
					mu = stacked_fitdata[i, 4]
					sig = stacked_fitdata[i, 6]
					lam = stacked_fitdata[i, 11]
					t_arr = stacked_fitdata[i, 17]
					t_err = stacked_fitdata[i, 18]
					
					# Determine status
					status = "OK"
					if np.isnan(redchi) or redchi > 50:
						status = "POOR FIT"
					elif not np.isfinite([A, mu, sig, lam]).all():
						status = "NaN PARAMS"
					elif A <= 0:
						status = "AMP <= 0"
					elif sig < 0.001 * adaptive_window_diag:
						status = f"SIGMA SMALL ({sig:.2e})"
					elif lam > 10000 or lam < 0.01:
						status = f"LAMBDA EXTREME ({lam:.2f})"
					elif not np.isfinite(t_arr):
						status = "t_arr NaN"
					elif not np.isfinite(t_err):
						status = "t_err NaN"
					
					if USE_ADAPTIVE_EMG_WINDOW:
						print(f"{i:<6d} {start_ch:>4d}-{end_ch:<4d}   {freq_center:>10.2f}   {adaptive_window_diag:>10.3f} {redchi:>8.2f} {A:>10.2f} {mu:>10.4f} "
							  f"{sig:>10.4f} {lam:>10.2f} {t_arr:>10.4f} {t_err:>10.4f} {status:<20}")
					else:
						print(f"{i:<6d} {start_ch:>4d}-{end_ch:<4d}   {freq_center:>10.2f}   {redchi:>8.2f} {A:>10.2f} {mu:>10.4f} "
							  f"{sig:>10.4f} {lam:>10.2f} {t_arr:>10.4f} {t_err:>10.4f} {status:<20}")
			
				print("-" * 150)
			
			# Create DataFrame for stacked arrival times.
			# Only pass the first 19 columns (which match subdf's
			# schema); column 19 holds the fitted C baseline, kept in
			# the underlying numpy array so the click-popup can read it
			# back when reconstructing the EMG curve. The DataFrame
			# itself doesn't carry C because none of the downstream
			# filters or plotting code asks for it by column name.
			stacked_df = pd.DataFrame(
				data=stacked_fitdata[:, :19],
				columns=subdf.columns  # Use same columns as subdf
			)
			
			# === FILTER 1: Amplitude filter ===
			# Per-stack filter audit — records each stack's verdict at every
			# stage for the click-popup's right-hand panel.
			_emg_audit = StackFilterAudit()
			_emg_audit_all_idx = list(stacked_df.index)
			count_before = len(stacked_df)
			amp_mask = stacked_df['amp'] > 0
			_f1_idx = list(stacked_df.index)
			_f1_pass = {ix: bool(amp_mask.loc[ix]) for ix in _f1_idx}
			_f1_amp = {ix: stacked_df.loc[ix, 'amp'] for ix in _f1_idx}
			_emg_audit.record(
				'Filter 1 (amp > 0)', _f1_idx, _f1_pass,
				lambda ix: f"amp = {_f1_amp[ix]:.4f}  (need > 0)")
			rejected_amp = stacked_df[~amp_mask]
			stacked_df = stacked_df[amp_mask]
			if diag_enabled('STACKING_FILTER'):
				print(f"\nFilter 1 - Amplitude (amp > 0): {count_before} → {len(stacked_df)} stacks ({count_before - len(stacked_df)} rejected)")
				if len(rejected_amp) > 0:
					for idx, row in rejected_amp.iterrows():
						print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, amp={row['amp']:.4f}")
			
			# === FILTER 1A: Fit-covariance amplitude SNR (amp / amp_err) ===
			# Rejects stacks where the fit's own covariance says the amplitude
			# is consistent with zero. amp_err comes from sqrt(diag(pcov)),
			# rescaled by n_channels_in_stack to match amp's scale.
			# A non-finite or zero amp_err is treated as a failed fit.
			if STACKED_AMP_SNR_MIN is not None and 'amp_err' in stacked_df.columns:
				count_before = len(stacked_df)
				_amp_snr = stacked_df['amp'] / stacked_df['amp_err'].replace(0, np.nan)
				amp_snr_mask = np.isfinite(_amp_snr) & (_amp_snr > STACKED_AMP_SNR_MIN)
				_f1a_idx = list(stacked_df.index)
				_f1a_pass = {ix: bool(amp_snr_mask.loc[ix]) for ix in _f1a_idx}
				_f1a_snr = {ix: _amp_snr.loc[ix] for ix in _f1a_idx}
				_emg_audit.record(
					f'Filter 1A (amp/amp_err > {STACKED_AMP_SNR_MIN:g})',
					_f1a_idx, _f1a_pass,
					lambda ix: f"SNR = {_f1a_snr[ix]:.2f}  (need > {STACKED_AMP_SNR_MIN:g})")
				rejected_amp_snr = stacked_df[~amp_snr_mask]
				stacked_df = stacked_df[amp_snr_mask]
				if diag_enabled('STACKING_FILTER'):
					print(f"Filter 1A - Fit-covariance SNR (amp/amp_err > {STACKED_AMP_SNR_MIN}): "
					      f"{count_before} → {len(stacked_df)} stacks "
					      f"({count_before - len(stacked_df)} rejected)")
					if len(rejected_amp_snr) > 0:
						for idx, row in rejected_amp_snr.iterrows():
							_snr_val = (row['amp'] / row['amp_err']) if (np.isfinite(row['amp_err']) and row['amp_err'] != 0) else np.nan
							print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, "
							      f"amp={row['amp']:.4f}, amp_err={row['amp_err']:.4f}, "
							      f"snr={_snr_val:.2f}")
			
			# === FILTER 1B: Peak-amplitude vs off-pulse-noise SNR ===
			# External sanity check: does the stack's actual data peak rise
			# above the noise floor? Compares the DATA peak of the stacked
			# profile (within the on-pulse region) to the noise std computed
			# from a guaranteed-clean pre-burst window in the FULL waterfall
			# (summed across the same channel range as the stack).
			#
			# Why the DATA peak rather than the fitted EMG 'A' parameter?
			# `A` is the EMG scale parameter, not the model's peak height
			# — for typical FRB σ/τ ratios the actual peak is 1.5–3×
			# larger than A. Using A in the SNR ratio systematically
			# underestimates significance and rejects stacks that
			# visibly contain signal. The data peak in the on-pulse
			# window is the model-independent quantity readers expect
			# when they hear "peak-to-noise SNR > 4".
			#
			# Why the pre-burst noise window (not subfall leading bins)?
			# In hard-isolation mode (`cuts` supplied), a component's
			# subfall starts at the previous cut boundary, which can
			# sit only a few bins before the component's peak. The
			# "first N bins" off-pulse window then includes the signal
			# — inflating noise_std and crushing peak_snr to below
			# threshold even for stacks that clearly contain signal.
			# This was the bug hiding 11A component d's stacked TOAs.
			if (STACKED_PEAK_SNR_MIN is not None
					and 'stacked_profiles' in dir()
					and stacked_profiles is not None
					and len(stacked_df) > 0):
				count_before = len(stacked_df)

				# Pre-burst time window in the FULL wfall: from t=0 up to
				# 3σ before the leftmost component's center. Fall back to
				# 5% of duration if seeds are missing.
				_n_time_full = wfall.shape[1]
				_dur_full_ms = _n_time_full * res_time_ms
				_xo_first = float(min(xos)) if (xos is not None and len(xos) > 0) else (_dur_full_ms * 0.4)
				_sig_first = (
					float(tmix_sigmas_std[xos.index(min(xos))])
					if (tmix_sigmas_std is not None
						and len(tmix_sigmas_std) > 0
						and np.isfinite(tmix_sigmas_std[xos.index(min(xos))]))
					else max(0.05, _dur_full_ms * 0.02)
				)
				_preburst_end_ms = max(_xo_first - 3.0 * _sig_first, _dur_full_ms * 0.05)
				_preburst_end_bin = max(5, int(_preburst_end_ms / res_time_ms))
				_preburst_end_bin = min(_preburst_end_bin, _n_time_full - 1)

				peak_snr_vals = {}
				for idx in stacked_df.index:
					try:
						i_int = int(idx)
						# Channel range that produced this stack
						_start_ch = i_int * stack_size_this_comp
						_end_ch = min((i_int + 1) * stack_size_this_comp, wfall.shape[0])
						if _end_ch <= _start_ch:
							raise ValueError("empty stack channel range")
						# Noise: same channels summed over the pre-burst
						# region of the full waterfall.
						_noise_profile = wfall[_start_ch:_end_ch, :_preburst_end_bin].sum(axis=0)
						if _noise_profile.size < 3:
							raise ValueError("preburst window too narrow")
						_noise_std = float(np.std(_noise_profile))
						# Signal: actual peak of the stack's data profile.
						_peak_signal = float(np.max(stacked_profiles[i_int]))
					except (IndexError, ValueError, TypeError, AttributeError):
						# Fallback: subfall leading slab + fitted A (legacy behaviour).
						try:
							_profile = stacked_profiles[int(idx)]
							_off_n = max(1, len(_profile) // NOISE_WIDTH_FACTOR)
							_noise_std = float(np.std(_profile[:_off_n]))
							_peak_signal = float(stacked_df.loc[idx, 'amp'])
						except (IndexError, ValueError, TypeError):
							_noise_std = np.nan
							_peak_signal = np.nan
					if (np.isfinite(_noise_std) and _noise_std > 0
							and np.isfinite(_peak_signal)):
						peak_snr_vals[idx] = _peak_signal / _noise_std
					else:
						peak_snr_vals[idx] = np.nan
				_peak_snr = pd.Series(peak_snr_vals)
				peak_snr_mask = np.isfinite(_peak_snr) & (_peak_snr > STACKED_PEAK_SNR_MIN)
				_f1b_idx = list(stacked_df.index)
				_f1b_pass = {ix: bool(peak_snr_mask.loc[ix]) for ix in _f1b_idx}
				_emg_audit.record(
					f'Filter 1B (peak/noise SNR > {STACKED_PEAK_SNR_MIN:g})',
					_f1b_idx, _f1b_pass,
					lambda ix: f"peak SNR = {peak_snr_vals.get(ix, float('nan')):.2f}  "
					           f"(need > {STACKED_PEAK_SNR_MIN:g})")
				rejected_peak_snr = stacked_df[~peak_snr_mask]
				stacked_df = stacked_df[peak_snr_mask]
				if diag_enabled('STACKING_FILTER'):
					print(f"Filter 1B - Peak-vs-noise SNR (data_peak/std(off_pulse) > {STACKED_PEAK_SNR_MIN}): "
					      f"{count_before} → {len(stacked_df)} stacks "
					      f"({count_before - len(stacked_df)} rejected)")
					print(f"  Pre-burst noise window: bins [0, {_preburst_end_bin}] "
					      f"(0 to {_preburst_end_ms:.3f} ms in full waterfall)")
					if len(rejected_peak_snr) > 0:
						for idx, row in rejected_peak_snr.iterrows():
							_p_snr = peak_snr_vals.get(idx, np.nan)
							print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, "
							      f"amp={row['amp']:.4f}, peak_snr={_p_snr:.2f}")
			
			# === FILTER 2: Uncertainty ratio filter ===
			if 't_emg_hwhm' in stacked_df.columns and 't_emg_hwhm_err' in stacked_df.columns:
				count_before = len(stacked_df)
				tiny = max(1e-6, 0.1 * res_time_ms)
				denom = stacked_df['t_emg_hwhm'].abs().clip(lower=tiny)
				ratio = stacked_df['t_emg_hwhm_err'] / denom
				ratio_mask = np.isfinite(ratio) & (ratio < 10)
				_f2_idx = list(stacked_df.index)
				_f2_pass = {ix: bool(ratio_mask.loc[ix]) for ix in _f2_idx}
				_f2_ratio = {ix: ratio.loc[ix] for ix in _f2_idx}
				_emg_audit.record(
					'Filter 2 (t_err/|t_arr| < 10)', _f2_idx, _f2_pass,
					lambda ix: f"ratio = {_f2_ratio[ix]:.2f}  (need < 10)")
				rejected_ratio = stacked_df[~ratio_mask]
				stacked_df = stacked_df[ratio_mask]
				if diag_enabled('STACKING_FILTER'):
					print(f"Filter 2 - Uncertainty ratio (t_err/|t_arr| < 10): {count_before} → {len(stacked_df)} stacks ({count_before - len(stacked_df)} rejected)")
					if len(rejected_ratio) > 0:
						for idx, row in rejected_ratio.iterrows():
							r = row['t_emg_hwhm_err'] / max(abs(row['t_emg_hwhm']), tiny)
							print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, ratio={r:.2f} (t_arr={row['t_emg_hwhm']:.4f}, t_err={row['t_emg_hwhm_err']:.4f})")
			
			# === FILTER 3: Absolute uncertainty filter ===
			# Use the burst width (sigma) as reference
			max_abs_error = 5 * sigma  # or use a fixed threshold like 0.1 ms
			if 't_emg_hwhm_err' in stacked_df.columns:
				count_before = len(stacked_df)
				abs_err_mask = stacked_df['t_emg_hwhm_err'] < max_abs_error
				_f3_idx = list(stacked_df.index)
				_f3_pass = {ix: bool(abs_err_mask.loc[ix]) for ix in _f3_idx}
				_f3_err = {ix: stacked_df.loc[ix, 't_emg_hwhm_err'] for ix in _f3_idx}
				_emg_audit.record(
					f'Filter 3 (t_err < {max_abs_error:.3g} ms)', _f3_idx, _f3_pass,
					lambda ix: f"t_err = {_f3_err[ix]:.4f} ms  (need < {max_abs_error:.3g})")
				rejected_abs = stacked_df[~abs_err_mask]
				stacked_df = stacked_df[abs_err_mask]
				if diag_enabled('STACKING_FILTER'):
					print(f"Filter 3 - Absolute uncertainty (t_err < {max_abs_error:.4f} ms): {count_before} → {len(stacked_df)} stacks ({count_before - len(stacked_df)} rejected)")
					if len(rejected_abs) > 0:
						for idx, row in rejected_abs.iterrows():
							print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, t_err={row['t_emg_hwhm_err']:.4f} ms")
			
			if diag_enabled('STACKING_FILTER'):
				print(f"\n=== Stacked TOAs after initial filters: {len(stacked_df)} / {len(stacked_profiles)} ===\n")
			
# 			print(f"Stacked fits: {len(stacked_df)} stacked arrival times from {len(stacked_profiles)} stacks")
			

			# === VISUALIZE STACKED WATERFALL ===
			if stacked_wfall_visualization:
				visualize_stacked_waterfall(
					stacked_profiles=stacked_profiles,
					stack_center_indices=stack_center_indices,
					freqs_full=freqs,
					res_time_ms=res_time_ms,
					duration=res_time_ms * subfall.shape[1],
					stack_size=stack_size_this_comp,
					burst_name=bname if 'bname' in locals() else rowname
				)
			# === END STACKED WATERFALL VISUALIZATION ===

			
			# === STORE GLOBALLY FOR CLICK VISUALIZATION ===
			# Create global copies to avoid affecting stacking calculations
			
			# FIX 1
			global _global_stacked_profiles, _global_stacked_df, _global_stack_center_indices, _global_stacked_fitdata, _global_subfall_duration, _global_subfall_shift, _global_wfall
			global _global_adaptive_window_params  # NEW: Store adaptive window parameters
# 			global _global_stacked_profiles, _global_stacked_df, _global_stack_center_indices, _global_stacked_fitdata, _global_subfall_duration  
# 			global _global_stacked_profiles, _global_stacked_df, _global_stack_center_indices, _global_stacked_fitdata
			_global_stacked_profiles = stacked_profiles.copy()
			_global_stacked_df = stacked_df.copy()
			_global_stack_center_indices = stack_center_indices.copy()
			_global_stacked_fitdata = stacked_fitdata.copy()
			
			#FIX 2
			_global_subfall_duration = res_time_ms * subfall.shape[1]  # <-- NEW LINE
# 			_global_subpktime = subpktime  # Store subfall peak time offset
			_global_wfall = wfall.copy()  # Store full waterfall for visualization
			
			# NEW: Store adaptive window parameters for visualization (only if enabled)
			if USE_ADAPTIVE_EMG_WINDOW:
				_global_adaptive_window_params = {
					'enabled': True,
					'tau_sc_ms': tau_sc_ms,
					'nu_max': nu_max,
					'nu_min': nu_min,
					'nu_range': nu_range,
					'base_window_ms': base_window_ms,
					'base_window_bins': base_window_bins,
					'stacked_freqs': stacked_freqs.copy(),
					'res_time_ms': res_time_ms
				}
			else:
				_global_adaptive_window_params = {'enabled': False}
			# === END GLOBAL STORAGE ===
			
			
		else:
			stacked_df = None
			# Clear globals when stacking is disabled
			_global_stacked_profiles = None
			_global_stacked_df = None
			_global_stack_center_indices = None
			_global_stacked_fitdata = None
			_global_wfall = None
			_global_adaptive_window_params = None  # NEW: Clear adaptive window params
			
			#FIX 3
			_global_subfall_duration = None
			_global_subfall_shift = None
			
		# === END SIGNAL STACKING ===
		
		# === GAUSSIAN SIGNAL STACKING ===
		if USE_GAUSS_MIX and use_stacking_this_comp:
			print(f"\n=== Gaussian Signal Stacking Enabled (stack size: {stack_size_this_comp}) ===")

			# Stack the waterfall channels
			stacked_gauss_profiles, stack_gauss_center_indices = stack_channels(subfall, stack_size_this_comp)

			# Create frequency array for stacked channels (using center frequencies)
			stacked_gauss_freqs = freqs[stack_gauss_center_indices]

			# === SEED SETUP (mirrors the EMG stacking block) ==============
			# Use this component's σ from the intensity-profile fit as the
			# seed for every stack. Mirrors `_seed_idx = comp_idx_loop`
			# in the EMG path so component a's σ isn't used as the seed
			# for d's stacks. (Previously this Gaussian path used
			# `xos.index(xosi)`, which only works when xos is the same
			# list throughout the run; index-based lookup is more robust.)
			_seed_idx = comp_idx_loop if comp_idx_loop < len(tmix_sigmas) else 0
			sigma_seed = float(tmix_sigmas[_seed_idx]) if len(tmix_sigmas) > _seed_idx else None
			if sigma_seed is not None and np.isfinite(sigma_seed):
				sigma_seed = abs(sigma_seed)

			# Compute μ_seed in subfall-local coordinates.
			# Two paths (mirroring fitemg's stacking logic):
			#  * No cuts: subfall is centered on xc with width ±4·σ_std,
			#    so mu_local = 4·σ_std. (No τ to subtract — Gaussian.)
			#  * With cuts: subfall is the slice between cuts (not
			#    centered on xc), so mu_local = xosi - subfall_left_edge.
			# Without this branch, hard-isolated multi-component bursts
			# would seed every Gaussian stack with the wrong μ, biasing
			# the fits and tripping downstream filters.
			if sigma_seed is not None and np.isfinite(sigma_seed) and sigma_seed > 0:
				sigma_std_seed = (
					float(tmix_sigmas_std[_seed_idx])
					if len(tmix_sigmas_std) > _seed_idx
					else float(sigma_seed)
				)
				if len(cuts) > 0:
					_cuts_sorted = sorted(float(c) for c in cuts)
					if xosi < _cuts_sorted[0]:
						_this_subfall_left_ms = 0.0
					elif xosi > _cuts_sorted[-1]:
						_this_subfall_left_ms = _cuts_sorted[-1]
					else:
						_this_subfall_left_ms = max(
							c for c in _cuts_sorted if c <= xosi
						)
					_this_subfall_left_chan = int(
						np.floor(_this_subfall_left_ms / res_time_ms)
					)
					_subfall_left_ms = _this_subfall_left_chan * res_time_ms
					mu_seed = float(xosi - _subfall_left_ms)
				else:
					mu_seed = 4.0 * sigma_std_seed
				# Clamp to within the subfall span.
				_subfall_dur_ms = res_time_ms * subfall.shape[1]
				mu_seed = float(np.clip(mu_seed, res_time_ms, _subfall_dur_ms - res_time_ms))
			else:
				mu_seed = None
				sigma_seed = None

			print(f"  Intensity profile seeds (component {subburst_suffixes[comp_idx_loop]}): "
			      f"mu={mu_seed}, sigma={sigma_seed}")

			# Fit Gaussian to each stacked profile
			# Structure: freqs, tstart, tend, amp, xo, tstart_err, sigma, red_chisq, amp_err, xo_err, sigma_err
			stacked_gauss_fitdata = np.full((len(stacked_gauss_profiles), 11), np.nan, dtype=float)

			for i, stacked_row in enumerate(stacked_gauss_profiles):
				duration_stack = res_time_ms * subfall.shape[1]

				# Determine actual number of channels in this stack (last stack may have fewer)
				start_idx = i * stack_size_this_comp
				end_idx = min((i + 1) * stack_size_this_comp, len(freqs))
				n_channels_in_stack = end_idx - start_idx

				# Normalize the stacked profile by dividing by number of channels
				# This gives it the same noise characteristics as individual channels
				# while benefiting from sqrt(n) SNR improvement
				normalized_row = stacked_row / n_channels_in_stack

				# Single Gaussian fit. Mirrors the per-stack EMG call
				# (line ~6206): seeds from the intensity-profile fit,
				# σ bounded to [res_time_ms, duration_stack] so the
				# fit can't return sub-resolution σ (delta-function
				# pathology that breaks downstream filters) or absurdly
				# wide σ (the whole stack window). The Gaussian path
				# has no adaptive window — that's specifically for the
				# EMG scattering tail — so the σ upper bound uses the
				# full stack duration.
				# μ bound: confine the per-stack fit to a window around the
				# intensity-profile seed. Low-SNR stacked rows otherwise let
				# curve_fit drift μ onto an unrelated noise peak (seen as
				# tstart values tens of ms away from the true burst). Width
				# ±3σ_seed is generous enough for a real fit but blocks the
				# runaway. Skipped when there is no usable seed.
				if (mu_seed is not None and np.isfinite(mu_seed)
						and sigma_seed is not None and np.isfinite(sigma_seed)
						and sigma_seed > 0):
					_mu_halfwin = 3.0 * abs(sigma_seed)
					_mu_lo_stk = mu_seed - _mu_halfwin
					_mu_hi_stk = mu_seed + _mu_halfwin
				else:
					_mu_lo_stk = _mu_hi_stk = None

				popt, pcov, redchisq = fitgauss(
					normalized_row,
					duration_stack,
					mu0_seed=mu_seed,
					sigma0_seed=sigma_seed,
					sigma_min=float(res_time_ms),
					sigma_max=float(duration_stack),
					mu_min=_mu_lo_stk,
					mu_max=_mu_hi_stk,
				)
				
				# Extract parameters
				if isinstance(pcov, np.ndarray) and pcov.ndim == 2:
					perr = np.sqrt(np.diag(pcov))
				else:
					perr = np.full(3, np.nan, dtype=float)
				
				if len(popt) >= 3:
					A, mu, sig = popt[0], popt[1], popt[2]
					# Rescale amplitude back to summed scale for consistency
					A = A * n_channels_in_stack
				else:
					A, mu, sig = np.nan, np.nan, np.nan
				
				if len(perr) >= 3:
					A_err, mu_err, sig_err = perr[0], perr[1], perr[2]
					# Rescale amplitude error back to summed scale
					A_err = A_err * n_channels_in_stack
				else:
					A_err, mu_err, sig_err = np.nan, np.nan, np.nan
				
				# Compute arrival time: depends on PEAK_INTENSITY_TOA toggle
				sigma_abs = abs(sig)
				if PEAK_INTENSITY_TOA:
					# Peak mode: arrival time at Gaussian center (peak of intensity)
					tstart = mu
					tstart_err = mu_err  # Just the uncertainty on mu
				else:
					# Leading edge mode: arrival time at left half-maximum crossing
					tstart = mu - np.sqrt(2*np.log(2)) * sigma_abs
					tstart_err = np.sqrt(mu_err**2 + 2*np.log(2)*sig_err**2) if np.isfinite(mu_err) and np.isfinite(sig_err) else np.nan
				tend = mu + np.sqrt(2) * sigma_abs
				
				# Store fitted parameters (matching fitrows column structure)
				stacked_gauss_fitdata[i, :] = [
					stacked_gauss_freqs[i],   # freqs
					tstart,                    # tstart (arrival time)
					tend,                      # tend
					A,                         # amp
					mu,                        # xo
					tstart_err,                # tstart_err
					sigma_abs,                 # sigma
					redchisq,                  # red_chisq
					A_err,                     # amp_err
					mu_err,                    # xo_err
					sig_err                    # sigma_err
				]
			
			# === STACKED GAUSSIAN FIT DIAGNOSTICS ===
			print(f"\n=== Stacked Gaussian Fit Diagnostics ({len(stacked_gauss_profiles)} stacks of {stack_size_this_comp} channels) ===")
			print(f"{'Stack':<6} {'Chans':<12} {'Freq (MHz)':<12} {'redchi':>8} {'A':>10} {'mu':>10} {'sigma':>10} "
				  f"{'tstart':>10} {'tstart_err':>10} {'Status':<20}")
			print("-" * 120)
			
			for i in range(len(stacked_gauss_profiles)):
				start_ch = i * stack_size_this_comp
				end_ch = min((i + 1) * stack_size_this_comp, len(freqs)) - 1
				freq_center = stacked_gauss_freqs[i]
				
				# Extract values from stacked_gauss_fitdata
				redchi = stacked_gauss_fitdata[i, 7]
				A = stacked_gauss_fitdata[i, 3]
				mu = stacked_gauss_fitdata[i, 4]
				sig = stacked_gauss_fitdata[i, 6]
				t_arr = stacked_gauss_fitdata[i, 1]  # tstart
				t_err = stacked_gauss_fitdata[i, 5]  # tstart_err
				
				# Determine status
				status = "OK"
				if np.isnan(redchi) or redchi > 50:
					status = "POOR FIT"
				elif not np.isfinite([A, mu, sig]).all():
					status = "NaN PARAMS"
				elif A <= 0:
					status = "AMP <= 0"
				elif sig < 0.001 * (res_time_ms * subfall.shape[1]):
					status = f"SIGMA SMALL ({sig:.2e})"
				elif not np.isfinite(t_arr):
					status = "tstart NaN"
				elif not np.isfinite(t_err):
					status = "tstart_err NaN"
				
				print(f"{i:<6d} {start_ch:>4d}-{end_ch:<4d}   {freq_center:>10.2f}   {redchi:>8.2f} {A:>10.2f} {mu:>10.4f} "
					  f"{sig:>10.4f} {t_arr:>10.4f} {t_err:>10.4f} {status:<20}")
			
			print("-" * 120)
			
			# Create DataFrame for stacked arrival times (use fitrows column names)
			stacked_gauss_df = pd.DataFrame(
				data=stacked_gauss_fitdata,
				columns=['freqs', 'tstart', 'tend', 'amp', 'xo', 'tstart_err', 'sigma', 'red_chisq', 
						 'amp_err', 'xo_err', 'sigma_err']
			)
			
			# === FILTER 1: Amplitude filter (same as regular Gaussian) ===
			# Per-stack filter audit for the click-popup (Gaussian path).
			_gauss_audit = StackFilterAudit()
			_gauss_audit_all_idx = list(stacked_gauss_df.index)
			count_before = len(stacked_gauss_df)
			amp_mask = stacked_gauss_df['amp'] > 0
			_g1_idx = list(stacked_gauss_df.index)
			_g1_pass = {ix: bool(amp_mask.loc[ix]) for ix in _g1_idx}
			_g1_amp = {ix: stacked_gauss_df.loc[ix, 'amp'] for ix in _g1_idx}
			_gauss_audit.record(
				'Filter 1 (amp > 0)', _g1_idx, _g1_pass,
				lambda ix: f"amp = {_g1_amp[ix]:.4f}  (need > 0)")
			rejected_amp = stacked_gauss_df[~amp_mask]
			stacked_gauss_df = stacked_gauss_df[amp_mask]
			print(f"\nFilter 1 - Amplitude (amp > 0): {count_before} → {len(stacked_gauss_df)} stacks ({count_before - len(stacked_gauss_df)} rejected)")
			if len(rejected_amp) > 0:
				for idx, row in rejected_amp.iterrows():
					print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, amp={row['amp']:.4f}")

			# === FILTER 1A: Fit-covariance amplitude SNR (Gaussian stacked) ===
			# Same logic as the EMG path — see comments there. Applied here on
			# stacked_gauss_df with stacked_gauss_profiles for the noise floor.
			if STACKED_AMP_SNR_MIN is not None and 'amp_err' in stacked_gauss_df.columns:
				count_before = len(stacked_gauss_df)
				_amp_snr_g = stacked_gauss_df['amp'] / stacked_gauss_df['amp_err'].replace(0, np.nan)
				amp_snr_mask_g = np.isfinite(_amp_snr_g) & (_amp_snr_g > STACKED_AMP_SNR_MIN)
				_g1a_idx = list(stacked_gauss_df.index)
				_g1a_pass = {ix: bool(amp_snr_mask_g.loc[ix]) for ix in _g1a_idx}
				_g1a_snr = {ix: _amp_snr_g.loc[ix] for ix in _g1a_idx}
				_gauss_audit.record(
					f'Filter 1A (amp/amp_err > {STACKED_AMP_SNR_MIN:g})',
					_g1a_idx, _g1a_pass,
					lambda ix: f"SNR = {_g1a_snr[ix]:.2f}  (need > {STACKED_AMP_SNR_MIN:g})")
				rejected_amp_snr_g = stacked_gauss_df[~amp_snr_mask_g]
				stacked_gauss_df = stacked_gauss_df[amp_snr_mask_g]
				print(f"Filter 1A - Fit-covariance SNR (amp/amp_err > {STACKED_AMP_SNR_MIN}): "
				      f"{count_before} → {len(stacked_gauss_df)} stacks "
				      f"({count_before - len(stacked_gauss_df)} rejected)")
				if len(rejected_amp_snr_g) > 0:
					for idx, row in rejected_amp_snr_g.iterrows():
						_snr_val = (row['amp'] / row['amp_err']) if (np.isfinite(row['amp_err']) and row['amp_err'] != 0) else np.nan
						print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, "
						      f"amp={row['amp']:.4f}, amp_err={row['amp_err']:.4f}, "
						      f"snr={_snr_val:.2f}")

			# === FILTER 1B: Peak-amplitude vs off-pulse-noise SNR (Gaussian) ===
			# See the EMG Filter 1B above for full rationale: uses the
			# stack's DATA peak (not the fitted A) compared against noise
			# computed from the pre-burst region of the FULL waterfall
			# (summed across the same channels as the stack).
			if (STACKED_PEAK_SNR_MIN is not None
					and 'stacked_gauss_profiles' in dir()
					and stacked_gauss_profiles is not None
					and len(stacked_gauss_df) > 0):
				count_before = len(stacked_gauss_df)

				_n_time_full_g = wfall.shape[1]
				_dur_full_ms_g = _n_time_full_g * res_time_ms
				_xo_first_g = float(min(xos)) if (xos is not None and len(xos) > 0) else (_dur_full_ms_g * 0.4)
				_sig_first_g = (
					float(tmix_sigmas_std[xos.index(min(xos))])
					if (tmix_sigmas_std is not None
						and len(tmix_sigmas_std) > 0
						and np.isfinite(tmix_sigmas_std[xos.index(min(xos))]))
					else max(0.05, _dur_full_ms_g * 0.02)
				)
				_preburst_end_ms_g = max(_xo_first_g - 3.0 * _sig_first_g, _dur_full_ms_g * 0.05)
				_preburst_end_bin_g = max(5, int(_preburst_end_ms_g / res_time_ms))
				_preburst_end_bin_g = min(_preburst_end_bin_g, _n_time_full_g - 1)

				peak_snr_vals_g = {}
				for idx in stacked_gauss_df.index:
					try:
						i_int = int(idx)
						_start_ch = i_int * stack_size_this_comp
						_end_ch = min((i_int + 1) * stack_size_this_comp, wfall.shape[0])
						if _end_ch <= _start_ch:
							raise ValueError("empty stack channel range")
						_noise_profile = wfall[_start_ch:_end_ch, :_preburst_end_bin_g].sum(axis=0)
						if _noise_profile.size < 3:
							raise ValueError("preburst window too narrow")
						_noise_std = float(np.std(_noise_profile))
						_peak_signal = float(np.max(stacked_gauss_profiles[i_int]))
					except (IndexError, ValueError, TypeError, AttributeError):
						try:
							_profile = stacked_gauss_profiles[int(idx)]
							_off_n = max(1, len(_profile) // NOISE_WIDTH_FACTOR)
							_noise_std = float(np.std(_profile[:_off_n]))
							_peak_signal = float(stacked_gauss_df.loc[idx, 'amp'])
						except (IndexError, ValueError, TypeError):
							_noise_std = np.nan
							_peak_signal = np.nan
					if (np.isfinite(_noise_std) and _noise_std > 0
							and np.isfinite(_peak_signal)):
						peak_snr_vals_g[idx] = _peak_signal / _noise_std
					else:
						peak_snr_vals_g[idx] = np.nan
				_peak_snr_g = pd.Series(peak_snr_vals_g)
				peak_snr_mask_g = np.isfinite(_peak_snr_g) & (_peak_snr_g > STACKED_PEAK_SNR_MIN)
				_g1b_idx = list(stacked_gauss_df.index)
				_g1b_pass = {ix: bool(peak_snr_mask_g.loc[ix]) for ix in _g1b_idx}
				_gauss_audit.record(
					f'Filter 1B (peak/noise SNR > {STACKED_PEAK_SNR_MIN:g})',
					_g1b_idx, _g1b_pass,
					lambda ix: f"peak SNR = {peak_snr_vals_g.get(ix, float('nan')):.2f}  "
					           f"(need > {STACKED_PEAK_SNR_MIN:g})")
				rejected_peak_snr_g = stacked_gauss_df[~peak_snr_mask_g]
				stacked_gauss_df = stacked_gauss_df[peak_snr_mask_g]
				print(f"Filter 1B - Peak-vs-noise SNR (data_peak/std(off_pulse) > {STACKED_PEAK_SNR_MIN}): "
				      f"{count_before} → {len(stacked_gauss_df)} stacks "
				      f"({count_before - len(stacked_gauss_df)} rejected)")
				print(f"  Pre-burst noise window: bins [0, {_preburst_end_bin_g}] "
				      f"(0 to {_preburst_end_ms_g:.3f} ms in full waterfall)")
				if len(rejected_peak_snr_g) > 0:
					for idx, row in rejected_peak_snr_g.iterrows():
						_p_snr = peak_snr_vals_g.get(idx, np.nan)
						print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, "
						      f"amp={row['amp']:.4f}, peak_snr={_p_snr:.2f}")

			# === FILTER 2: Uncertainty ratio filter ===
			if 'tstart' in stacked_gauss_df.columns and 'tstart_err' in stacked_gauss_df.columns:
				count_before = len(stacked_gauss_df)
				tiny = max(1e-6, 0.1 * res_time_ms)
				denom = stacked_gauss_df['tstart'].abs().clip(lower=tiny)
				ratio = stacked_gauss_df['tstart_err'] / denom
				ratio_mask = np.isfinite(ratio) & (ratio < 10)
				_g2_idx = list(stacked_gauss_df.index)
				_g2_pass = {ix: bool(ratio_mask.loc[ix]) for ix in _g2_idx}
				_g2_ratio = {ix: ratio.loc[ix] for ix in _g2_idx}
				_gauss_audit.record(
					'Filter 2 (t_err/|t_arr| < 10)', _g2_idx, _g2_pass,
					lambda ix: f"ratio = {_g2_ratio[ix]:.2f}  (need < 10)")
				rejected_ratio = stacked_gauss_df[~ratio_mask]
				stacked_gauss_df = stacked_gauss_df[ratio_mask]
				print(f"Filter 2 - Uncertainty ratio (tstart_err/|tstart| < 10): {count_before} → {len(stacked_gauss_df)} stacks ({count_before - len(stacked_gauss_df)} rejected)")
				if len(rejected_ratio) > 0:
					for idx, row in rejected_ratio.iterrows():
						r = row['tstart_err'] / max(abs(row['tstart']), tiny)
						print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, ratio={r:.2f} (tstart={row['tstart']:.4f}, tstart_err={row['tstart_err']:.4f})")
			
			# === FILTER 3: Absolute uncertainty filter ===
			max_abs_error = 5 * sigma  # same as regular Gaussian (use burst width as reference)
			if 'tstart_err' in stacked_gauss_df.columns:
				count_before = len(stacked_gauss_df)
				abs_err_mask = stacked_gauss_df['tstart_err'] < max_abs_error
				_g3_idx = list(stacked_gauss_df.index)
				_g3_pass = {ix: bool(abs_err_mask.loc[ix]) for ix in _g3_idx}
				_g3_err = {ix: stacked_gauss_df.loc[ix, 'tstart_err'] for ix in _g3_idx}
				_gauss_audit.record(
					f'Filter 3 (t_err < {max_abs_error:.3g} ms)', _g3_idx, _g3_pass,
					lambda ix: f"t_err = {_g3_err[ix]:.4f} ms  (need < {max_abs_error:.3g})")
				rejected_abs = stacked_gauss_df[~abs_err_mask]
				stacked_gauss_df = stacked_gauss_df[abs_err_mask]
				print(f"Filter 3 - Absolute uncertainty (tstart_err < {max_abs_error:.4f} ms): {count_before} → {len(stacked_gauss_df)} stacks ({count_before - len(stacked_gauss_df)} rejected)")
				if len(rejected_abs) > 0:
					for idx, row in rejected_abs.iterrows():
						print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, tstart_err={row['tstart_err']:.4f} ms")
			
			print(f"\n=== Stacked Gaussian TOAs after initial filters: {len(stacked_gauss_df)} / {len(stacked_gauss_profiles)} ===\n")
			
			# === STORE GLOBALLY FOR CLICK VISUALIZATION ===
			global _global_stacked_gauss_profiles, _global_stacked_gauss_df, _global_stack_gauss_center_indices
			global _global_stacked_gauss_fitdata, _global_subfall_gauss_duration, _global_subfall_gauss_shift, _global_wfall_gauss
			_global_stacked_gauss_profiles = stacked_gauss_profiles.copy()
			_global_stacked_gauss_df = stacked_gauss_df.copy()
			_global_stack_gauss_center_indices = stack_gauss_center_indices.copy()
			_global_stacked_gauss_fitdata = stacked_gauss_fitdata.copy()
			_global_subfall_gauss_duration = res_time_ms * subfall.shape[1]
			_global_wfall_gauss = wfall.copy()
			# _global_subfall_gauss_shift will be set later when times are shifted
			# === END GLOBAL STORAGE ===
			
		else:
			stacked_gauss_df = None
			# Clear Gaussian stacking globals when disabled
			_global_stacked_gauss_profiles = None
			_global_stacked_gauss_df = None
			_global_stack_gauss_center_indices = None
			_global_stacked_gauss_fitdata = None
			_global_wfall_gauss = None
			_global_subfall_gauss_duration = None
			_global_subfall_gauss_shift = None
		# === END GAUSSIAN SIGNAL STACKING ===
		
# 		if len(cuts) == 0:
# 			subpktime = 4*sigma # since we made a 4 sigma window
# 		else:
# 			ci, edge = 0, 0
# 			while ci < len(cuts) and xosi >= cuts[ci]:
# 				edge = cuts[ci]
# 				ci += 1
# 			subpktime = xosi - edge

		# Use a gate width consistent with how the sub-fall was carved
		if USE_EMG_MIX:
			# tail-aware width = sqrt(sigma_core^2 + 1/lambda^2) from the mixture fit
			width_for_gate = float(tmix_sigmas_std[xos.index(xosi)])
		else:
			width_for_gate = float(abs(sigma))

		if len(cuts) == 0:
			# subpktime is the burst-peak location in SUBFALL-LOCAL coords —
			# i.e. raw-waterfall coord of the peak minus the raw-waterfall
			# coord of the subfall's left edge. The legacy formula
			# `subpktime = 4 * width_for_gate` assumed the subfall starts
			# at `xc − 4*width` (i.e. exactly the proposed un-clipped
			# left edge). When the proposed window runs past the start
			# of the waterfall and the subfall extraction at line ~5970
			# clips the left edge to bin 0, that assumption breaks and
			# `subpktime` ends up at a position that does not correspond
			# to any feature in the subfall — biasing the slope-fit's
			# `tb` and the downstream slope-line, fiducial-scattering,
			# and width-envelope overlays that all anchor at `xcs[ci]`
			# under the implicit assumption `subpktime + subfall_left = xcs`.
			# For L741965_b component a this manifested as a ~21 ms left
			# shift of every overlay relative to the (correctly placed)
			# TOAs after the v56_10 shift fix.
			#
			# Look up the actual subfall left edge for this component
			# (stashed by the prep loop in `_subfall_left_chan_per_comp`)
			# and compute subpktime directly as `xc − subfall_left`,
			# which is the proper definition and reduces to `4*width`
			# automatically when no clipping occurs.
			try:
				_comp_idx_sp = xos.index(xosi)
				_lefe_chan_sp = _subfall_left_chan_per_comp.get(_comp_idx_sp, None)
			except Exception:
				_lefe_chan_sp = None
			if _lefe_chan_sp is not None:
				_xc_or_xos_sp = xcs[_comp_idx_sp] if USE_EMG_MIX else xosi
				subpktime = float(_xc_or_xos_sp) - float(_lefe_chan_sp) * res_time_ms
			else:
				# Defensive fallback to legacy formula if the lookup
				# fails for any reason (should not happen on the
				# normal code path).
				subpktime = 4 * width_for_gate
			if diag_enabled('SUBPKTIME'):
				print(f"\n=== SUBPKTIME DIAGNOSTIC (no cuts) ===")
				print(f"  subpktime (clip-aware) = {subpktime:.4f} ms")
				print(f"  legacy 4*width_for_gate would have given {4*width_for_gate:.4f} ms")
		else:
			ci, edge = 0, 0
			while ci < len(cuts) and xosi >= cuts[ci]:
				edge = cuts[ci]
				ci += 1
			subpktime = xosi - edge
			if diag_enabled('SUBPKTIME'):
				print(f"\n=== SUBPKTIME DIAGNOSTIC (with cuts) ===")
				print(f"  xosi (component center, GLOBAL) = {xosi:.4f} ms")
				print(f"  edge (left edge of subfall region) = {edge:.4f} ms")
				print(f"  subpktime = xosi - edge = {xosi:.4f} - {edge:.4f} = {subpktime:.4f} ms")
				print(f"  This is where we expect the peak in LOCAL coordinates")
	

		# Fit 1d gauss to burst spectrum
		if load_solutions:
			subband_popt = subbandpopts[0]
			subband_perr = subbandperrs[0]
			subband_redchisq = subbandredchisqs[0]
			# ingest lists
			subbandpopts = subbandpopts[1:]
			subbandperrs = subbandperrs[1:]
			subbandredchisqs = subbandredchisqs[1:]
		else:
			if bandmask_thresi: # remove points above the threshold for fitting
				subband_fit = subband[subband < bandmask_thresi]
				freqs_fit   = freqs[(subband < bandmask_thresi).nonzero()]
			else:
				subband_fit, freqs_fit = subband, freqs

			try:
				if subbandslice:
					subbandsigma = np.std(subband[subbandslice])
				else:
					subbandsigma = np.std(subband[:len(subband)//NOISE_WIDTH_FACTOR]) # not always a great estimate
				# printd(f"{subbandsigma = :.3f}")
				# [print(f, r) for f, r in zip(freqs_fit, subband_fit)]
				freqs_fit = freqs_fit[subband_fit.nonzero()]
				subband_fit = subband_fit[subband_fit.nonzero()]
				fo = sum(freqs_fit*subband_fit)/sum(subband_fit) # this is an estimate of center_f
				subband_popt, subband_pcov = scipy.optimize.curve_fit(
					gauss_model,
					freqs_fit,
					subband_fit,
					p0=[
						1,
						fo,
						np.sqrt(abs(sum(subband_fit*(freqs_fit-fo)**2)/sum(subband_fit))) # sigma
					],
					sigma=[subbandsigma]*len(subband_fit)
				)

				residuals = (subband_fit - gauss_model(freqs_fit, *subband_popt))/subbandsigma
				# [print(f"{f}\t{r}\t{m}") for f, r, m in zip(freqs_fit, residuals, gauss_model(freqs_fit, *subband_popt))]
				chisq = np.sum((residuals) ** 2)
				subband_redchisq = chisq / (len(subband_fit)- len(subband_popt))
				subband_perr = np.sqrt(np.diag(subband_pcov))

				# === DIAGNOSTIC: SPECTRAL FIT COVARIANCE ANALYSIS ===
				# (Commented out - moved to summary at end)
				# print(f"\n{'@'*80}")
				# print(f"@ DIAGNOSTIC: SPECTRAL FIT COVARIANCE ANALYSIS for {rowname}")
				# print(f"{'@'*80}")
				# 
				# spec_param_names = ['A_spec', 'center_f', 'sigma_f']
				# print(f"\n  Spectral Gaussian fit parameters:")
				# print(f"  {'Parameter':<15} {'Value':>12} {'Uncertainty':>12} {'Rel.Err (%)':>12}")
				# print(f"  {'-'*51}")
				# for i, pname in enumerate(spec_param_names):
				# 	val = subband_popt[i]
				# 	err = subband_perr[i]
				# 	rel_err = 100 * err / abs(val) if val != 0 else np.inf
				# 	print(f"  {pname:<15} {val:>12.4f} {err:>12.4f} {rel_err:>12.1f}%")
				# 
				# print(f"\n  Fit quality: reduced chi-squared = {subband_redchisq:.3f}")
				# print(f"  Number of data points: {len(subband_fit)}")
				# print(f"  Noise estimate (subbandsigma): {subbandsigma:.6f}")
				# 
				# # Covariance matrix for spectral fit
				# print(f"\n  Spectral fit covariance matrix:")
				# for i, pi in enumerate(spec_param_names):
				# 	row_str = "    "
				# 	for j, pj in enumerate(spec_param_names):
				# 		row_str += f"{subband_pcov[i,j]:>12.4e} "
				# 	print(row_str)
				# 
				# # Correlation matrix for spectral fit
				# spec_std = np.sqrt(np.diag(subband_pcov))
				# with np.errstate(divide='ignore', invalid='ignore'):
				# 	spec_corr = subband_pcov / np.outer(spec_std, spec_std)
				# 	spec_corr = np.nan_to_num(spec_corr, nan=0.0, posinf=0.0, neginf=0.0)
				# 
				# print(f"\n  Spectral fit correlation matrix:")
				# print(f"  {'':>15} {'A_spec':>12} {'center_f':>12} {'sigma_f':>12}")
				# for i, pname in enumerate(spec_param_names):
				# 	row_str = f"  {pname:<15}"
				# 	for j in range(3):
				# 		row_str += f"{spec_corr[i,j]:>12.3f} "
				# 	print(row_str)
				# 
				# print(f"\n  FOCUS: Center frequency uncertainty:")
				# print(f"    center_f = {subband_popt[1]:.4f} ± {subband_perr[1]:.4f} MHz")
				# print(f"    -> This becomes center_f_err (vertical error bar) = {subband_perr[1]:.4f} MHz")
				# 
				# # Condition number
				# try:
				# 	spec_cond = np.linalg.cond(subband_pcov)
				# 	print(f"\n  Spectral covariance condition number: {spec_cond:.2e}")
				# except:
				# 	pass
				# 
				# print(f"{'@'*80}\n")
				# === END SPECTRAL FIT COVARIANCE ANALYSIS ===

				## Diagnostic plot for spectrum fit:
				if False:
					# print(f"{subbandsigma = } {subband_redchisq = }")
					fig, axs = plt.subplots(2,1, figsize=(10,7), height_ratios=[3,1])
					if bandmask_thresi:
						axs[0].plot(freqs, subband, 'rX', ms=4)
						axs[0].axhline(y=bandmask_thresi, c='r', ls='--')
					axs[0].plot(freqs_fit, subband_fit)
					axs[0].plot(
						freqs,
						gauss_model(freqs, *subband_popt),
						label=(
							f'$\sigma_{{\\nu}} =$ {scilabel(subband_popt[2], subband_perr[2])} MHz'
							f"\n$\\chi^2_{{\\text{{red}}}} =${subband_redchisq:.2f}"
						)
					)
					axs[0].legend(frameon=False)
					_, bins, _ = axs[1].hist(
						np.random.standard_normal(len(residuals)),
						bins='auto',
						label='Sampled $N(0,1)$',
						histtype='step',
						density=True,
						linewidth=2,
						range=(residuals.min(), residuals.max())
					)
					axs[1].hist(
						residuals,
						bins=bins,  # type: ignore[reportUndefinedVariable]
						label='residuals',
						density=True
						# histtype='step'
					)
					plt.show(block=True);plt.close()

			except (RuntimeError,ValueError) as e:
				print(f"Warning: Spectrum fit failed.", e)
				subband_popt, subband_perr = [0, 1, 1], [0, 0, 0]
				subband_redchisq = 0

		bwidth, bwidth_err = abs(subband_popt[2]), subband_perr[2] # sigma of spectrum fit
		pkfreq, pkfreq_err = subband_popt[1], subband_perr[1] # this is fitted center_f and center_f_err

		## Apply time and spectral filters to points
		printd(f"Debug: pre-filters {len(subdf) = }")
		
		# === DIAGNOSTIC 3: FILTER CHAIN ===
		# print(f"\n{'='*60}")
		# print(f"[DIAG 3] FILTER CHAIN for {rowname} ({'EMG' if USE_EMG_MIX else 'Gaussian'})")
		# print(f"{'='*60}")
		# print(f"  Initial row count: {len(subdf)}")
		# print(f"  Spectral fit: pkfreq={pkfreq:.2f}±{pkfreq_err:.2f} MHz, bwidth={bwidth:.2f}±{bwidth_err:.2f} MHz")
		filter_tracking = {'initial': len(subdf)}
		# === END HEADER ===
		
		# === DIAGNOSTIC 1: Initial state ===
		if SHOW_DIAGNOSTICS:
			print(f"\n=== FILTER DIAGNOSTICS for {rowname} ===")
			print(f"Initial rows: {len(subdf)}")
		
		printd(f"Info: Applying '{bw_filter}' spectral filter ")
		if bw_filter not in ['data_cutoff', 'model_cutoff', 'model_width']:
			print(f"Warning: unrecognized {bw_filter = }. Reverting to 'data_cutoff'")
			bw_filter = 'data_cutoff'
			
		# === STORE COUNT BEFORE SPECTRAL FILTER ===
		count_before_spectral = len(subdf)
			
		if bw_filter == 'data_cutoff':
			if logdebug:
				for f,n,s in zip(freqs,noisesmpl,subband):
					printd(f"{f:.3f} MHz: {n = :.8f}\t{s = :.8f}\t{s/n = :.8f}")

			snr_values = subband/noisesmpl
			if SHOW_DIAGNOSTICS:
				print(f"  SNR range: [{snr_values.min():.2f}, {snr_values.max():.2f}], cutoff={snr_cutoff}")

			subdf = subdf[ # freqs is the implied axis
				subband/noisesmpl > snr_cutoff
			]
			# === DIAGNOSTIC 2: After spectral filter ===
			if SHOW_DIAGNOSTICS:
				print(f"After spectral filter ({bw_filter}, SNR>{snr_cutoff}): {len(subdf)} rows (lost {count_before_spectral - len(subdf)})")
			printd(f"Debug: after spectral data filter {len(subdf) = }")
			
		elif bw_filter == 'model_cutoff' and bwidth != 1: # there must be a fit
			model = gauss_model(subdf['freqs'], *subband_popt)
			subdf = subdf[
				model/noisesmpl > snr_cutoff
			]
			# === DIAGNOSTIC 2: After spectral filter ===
			if SHOW_DIAGNOSTICS:
				print(f"After spectral filter ({bw_filter}, SNR>{snr_cutoff}): {len(subdf)} rows (lost {count_before_spectral - len(subdf)})")
			
		elif bw_filter == 'model_width' and bwidth != 1: # there must be a fit
			if _bw_co_this is not None:
				if SHOW_DIAGNOSTICS:
					print(f"After spectral filter ({bw_filter}): SKIPPED — bw_channel_override active")
			elif _bw_toa_co_this is not None:
				# Skip the spectral-σ-multiplier filter when
				# bw_ToA_override is active — the override fires later
				# (after the ignore filter) and applies a hard-coded
				# channel range to subdf. We skip model_width here so
				# the two filters don't double up on the same data
				# with potentially conflicting windows. Per design:
				# bw_ToA_override wins over model_width.
				if SHOW_DIAGNOSTICS:
					print(f"After spectral filter ({bw_filter}): SKIPPED — bw_ToA_override active")
			else:
				subdf = subdf[
					(pkfreq-bw_width_factor*bwidth < subdf['freqs']) &
					(subdf['freqs'] < pkfreq+bw_width_factor*bwidth)
				]
				# === DIAGNOSTIC 2: After spectral filter ===
				if SHOW_DIAGNOSTICS:
					print(f"After spectral filter ({bw_filter}, {bw_width_factor}×σ_ν): {len(subdf)} rows (lost {count_before_spectral - len(subdf)})")

		# === APPLY SPECTRAL FILTER TO STACKED ARRIVAL TIMES ===
		if USE_EMG_MIX and use_stacking_this_comp and 'stacked_df' in locals() and stacked_df is not None and len(stacked_df) > 0:
			count_before_stacked_filter = len(stacked_df)
			# Filter stacked_df to only include frequencies present in the filtered subdf
# 			if len(subdf) > 0:
# 				freq_min, freq_max = subdf['freqs'].min(), subdf['freqs'].max()
			if _bw_co_this is not None:
				if SHOW_DIAGNOSTICS:
					print(f"\nFilter 4 (Stacked) - Spectral filter: SKIPPED — bw_channel_override active "
					      f"({len(stacked_df)} stacks retained)")
			elif bwidth > 0 and np.isfinite(pkfreq):
				freq_min = pkfreq - bw_width_factor * bwidth
				freq_max = pkfreq + bw_width_factor * bwidth
				spectral_mask = (stacked_df['freqs'] >= freq_min) & (stacked_df['freqs'] <= freq_max)
				if '_emg_audit' in dir():
					_f4_idx = list(stacked_df.index)
					_f4_pass = {ix: bool(spectral_mask.loc[ix]) for ix in _f4_idx}
					_f4_frq = {ix: stacked_df.loc[ix, 'freqs'] for ix in _f4_idx}
					_emg_audit.record(
						'Filter 4 (spectral freq window)', _f4_idx, _f4_pass,
						lambda ix: f"freq = {_f4_frq[ix]:.1f} MHz  "
						           f"(window {freq_min:.1f}-{freq_max:.1f})")
				rejected_spectral = stacked_df[~spectral_mask]
				stacked_df = stacked_df[spectral_mask]
				if SHOW_DIAGNOSTICS:
					print(f"\nFilter 4 (Stacked) - Spectral filter: {count_before_stacked_filter} → {len(stacked_df)} stacks ({count_before_stacked_filter - len(stacked_df)} rejected)")
					print(f"  Frequency range: {freq_min:.1f}-{freq_max:.1f} MHz")
					if len(rejected_spectral) > 0:
						for idx, row in rejected_spectral.iterrows():
							print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz (outside valid range)")
			else:
				if SHOW_DIAGNOSTICS:
					print(f"Warning: No valid subdf frequencies, keeping all stacked arrival times")
		# === END SPECTRAL FILTER FOR STACKED ===
		
		# === APPLY SPECTRAL FILTER TO GAUSSIAN STACKED ARRIVAL TIMES ===
		if USE_GAUSS_MIX and use_stacking_this_comp and 'stacked_gauss_df' in locals() and stacked_gauss_df is not None and len(stacked_gauss_df) > 0:
			count_before_stacked_filter = len(stacked_gauss_df)
			if _bw_co_this is not None:
				if SHOW_DIAGNOSTICS:
					print(f"\nFilter 4 (Gaussian Stacked) - Spectral filter: SKIPPED — bw_channel_override active "
					      f"({len(stacked_gauss_df)} stacks retained)")
			elif bwidth > 0 and np.isfinite(pkfreq):
				freq_min = pkfreq - bw_width_factor * bwidth
				freq_max = pkfreq + bw_width_factor * bwidth
				spectral_mask = (stacked_gauss_df['freqs'] >= freq_min) & (stacked_gauss_df['freqs'] <= freq_max)
				if '_gauss_audit' in dir():
					_g4_idx = list(stacked_gauss_df.index)
					_g4_pass = {ix: bool(spectral_mask.loc[ix]) for ix in _g4_idx}
					_g4_frq = {ix: stacked_gauss_df.loc[ix, 'freqs'] for ix in _g4_idx}
					_gauss_audit.record(
						'Filter 4 (spectral freq window)', _g4_idx, _g4_pass,
						lambda ix: f"freq = {_g4_frq[ix]:.1f} MHz  "
						           f"(window {freq_min:.1f}-{freq_max:.1f})")
				rejected_spectral = stacked_gauss_df[~spectral_mask]
				stacked_gauss_df = stacked_gauss_df[spectral_mask]
				if SHOW_DIAGNOSTICS:
					print(f"\nFilter 4 (Gaussian Stacked) - Spectral filter: {count_before_stacked_filter} → {len(stacked_gauss_df)} stacks ({count_before_stacked_filter - len(stacked_gauss_df)} rejected)")
					print(f"  Frequency range: {freq_min:.1f}-{freq_max:.1f} MHz")
					if len(rejected_spectral) > 0:
						for idx, row in rejected_spectral.iterrows():
							print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz (outside valid range)")
			else:
				if SHOW_DIAGNOSTICS:
					print(f"Warning: No valid bandwidth for spectral filter, keeping all Gaussian stacked arrival times")
		# === END SPECTRAL FILTER FOR GAUSSIAN STACKED ===
		
		
		# === FILTER 5 (Stacked) - Scattering timescale filter ===
		# Reject stacks whose fitted τ is implausibly larger than the
		# τ of THIS component's joint intensity-profile fit. Previously
		# this used `tmix_lambdas[0]` (component a's τ) as the reference
		# for every component, so e.g. component d's stacks — which
		# legitimately have τ ≈ 0.2 ms at low frequency — were rejected
		# against component a's much smaller τ ≈ 0.04 ms threshold.
		# Use the current component's τ instead, and clamp it to at
		# least one time-bin: if the joint profile fit gave a
		# Gaussian-like component (τ → 0 with huge error, as for 11A
		# component b), `10 × tau_profile` can collapse below the time
		# resolution and reject every plausible stack. The filter's
		# purpose is to catch *pathologically wide* stacks, not to
		# enforce sub-bin τ precision from the noisy joint fit.
		if USE_EMG_MIX and use_stacking_this_comp and 'stacked_df' in locals() and stacked_df is not None and len(stacked_df) > 0:
			_lam_for_filter5 = (
				tmix_lambdas[comp_idx_loop]
				if (comp_idx_loop < len(tmix_lambdas)
				    and np.isfinite(tmix_lambdas[comp_idx_loop])
				    and tmix_lambdas[comp_idx_loop] > 0)
				else None
			)
			if _lam_for_filter5 is not None:
				_tau_profile_raw = 1.0 / _lam_for_filter5
				# When the intensity-profile fit's τ pegs near the bin
				# floor (within 5% of dt_ms), the joint fit has converged
				# to a Gaussian-equivalent EMG and τ is not meaningfully
				# determined. Using `10 × τ_profile` as the threshold in
				# that case erroneously rejects every per-stack fit with
				# a legitimately resolved scattering tail (as seen on
				# B29: joint τ pegged at 64 µs, per-stack τ ~1–2 ms, so
				# every stack failed `τ ≤ 0.64 ms`).
				#
				# In that regime, fall back to σ_profile as the
				# threshold scale: a per-stack τ comparable to or larger
				# than the burst width is the pathology this filter is
				# meant to catch, regardless of what the joint τ says.
				# Otherwise (joint fit found a real, non-pegged τ),
				# trust the joint τ as before.
				_pegged_to_bin = (_tau_profile_raw < 1.05 * float(res_time_ms))
				if _pegged_to_bin:
					_sig_for_filter5 = (
						tmix_sigmas[comp_idx_loop]
						if comp_idx_loop < len(tmix_sigmas)
						   and np.isfinite(tmix_sigmas[comp_idx_loop])
						   and tmix_sigmas[comp_idx_loop] > 0
						else None
					)
					if _sig_for_filter5 is not None:
						tau_intensity_profile = abs(float(_sig_for_filter5))
					else:
						tau_intensity_profile = max(_tau_profile_raw, float(res_time_ms))
				else:
					# Floor at one bin — joint τ below dt_ms is meaningless.
					tau_intensity_profile = max(_tau_profile_raw, float(res_time_ms))
				# Allow stacks to have up to 10× the (clamped or
				# σ-scaled) profile τ.
				tau_upper_bound = 10.0 * tau_intensity_profile
				count_before_tau_filter = len(stacked_df)
				tau_mask = stacked_df['tau'] <= tau_upper_bound
				if '_emg_audit' in dir():
					_f5_idx = list(stacked_df.index)
					_f5_pass = {ix: bool(tau_mask.loc[ix]) for ix in _f5_idx}
					_f5_tau = {ix: stacked_df.loc[ix, 'tau'] for ix in _f5_idx}
					_emg_audit.record(
						'Filter 5 (tau <= 10x profile tau)', _f5_idx, _f5_pass,
						lambda ix: f"tau = {_f5_tau[ix]:.4f} ms  "
						           f"(need <= {tau_upper_bound:.4f})")
				rejected_tau = stacked_df[~tau_mask]
				stacked_df = stacked_df[tau_mask]
				if SHOW_DIAGNOSTICS:
					if _pegged_to_bin:
						_clamp_note = (
							f" (joint τ={_tau_profile_raw:.6f} ms pegged near bin floor "
							f"{float(res_time_ms):.6f} ms — using σ_profile={tau_intensity_profile:.6f} ms "
							f"as threshold scale instead)"
						)
					elif _tau_profile_raw < res_time_ms:
						_clamp_note = f" (clamped from raw τ={_tau_profile_raw:.6f} ms — joint fit gave sub-bin value)"
					else:
						_clamp_note = ""
					print(f"Filter 5 (Stacked) - Tau filter (tau <= 10×tau_profile of comp {subburst_suffixes[comp_idx_loop]}): "
					      f"{count_before_tau_filter} → {len(stacked_df)} stacks "
					      f"({count_before_tau_filter - len(stacked_df)} rejected)")
					print(f"  tau_intensity_profile (component {subburst_suffixes[comp_idx_loop]}) = "
					      f"{tau_intensity_profile:.6f} ms{_clamp_note}, upper bound = {tau_upper_bound:.6f} ms")
					if len(rejected_tau) > 0:
						for idx, row in rejected_tau.iterrows():
							print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, tau={row['tau']:.6f} ms (> {tau_upper_bound:.6f} ms)")
		# === END TAU FILTER FOR STACKED ===


		# === FILTER 5B (Stacked) - Reject unphysically narrow EMG fits ===
		# Catches stacks where the EMG converged to a delta-function-
		# like fit (σ or τ << time resolution) — usually triggered by
		# the optimiser locking onto a high noise spike rather than the
		# true broad signal. These fits have artificially tiny χ²_red
		# (the model is a vertical line that misses every bin except one),
		# and they slip through Filter 5 because their τ is below
		# tau_profile, not above it. The threshold is one time-bin —
		# nothing physical can be narrower than the sampling resolution.
		if USE_EMG_MIX and use_stacking_this_comp and 'stacked_df' in locals() and stacked_df is not None and len(stacked_df) > 0:
			_min_phys_ms = float(res_time_ms)  # one bin
			count_before_phys = len(stacked_df)
			# Require σ AND τ to both exceed one time-bin.
			# Use abs() defensively — σ should always be ≥ 0 from the
			# bounded fit, but a degenerate covariance can produce
			# nonsense values.
			_sig_col = 'sigma' if 'sigma' in stacked_df.columns else None
			_tau_col = 'tau' if 'tau' in stacked_df.columns else None
			phys_mask = pd.Series(True, index=stacked_df.index)
			if _sig_col is not None:
				phys_mask &= (stacked_df[_sig_col].abs() >= _min_phys_ms)
			if _tau_col is not None:
				phys_mask &= (stacked_df[_tau_col].abs() >= _min_phys_ms)
			rejected_phys = stacked_df[~phys_mask]
			if '_emg_audit' in dir():
				_f5b_idx = list(stacked_df.index)
				_f5b_pass = {ix: bool(phys_mask.loc[ix]) for ix in _f5b_idx}
				def _f5b_detail(ix, _sc=_sig_col, _tc=_tau_col):
					_s = stacked_df.loc[ix, _sc] if _sc else float('nan')
					_t = stacked_df.loc[ix, _tc] if _tc else float('nan')
					return f"sigma = {_s:.4f}, tau = {_t:.4f} ms  (both need >= {_min_phys_ms:.4f})"
				_emg_audit.record(
					'Filter 5B (physical width >= 1 bin)', _f5b_idx, _f5b_pass,
					_f5b_detail)
			stacked_df = stacked_df[phys_mask]
			if SHOW_DIAGNOSTICS:
				print(f"Filter 5B (Stacked) - Physical width (σ AND τ >= {_min_phys_ms:.4f} ms = 1 time-bin): "
				      f"{count_before_phys} → {len(stacked_df)} stacks "
				      f"({count_before_phys - len(stacked_df)} rejected)")
				if len(rejected_phys) > 0:
					for idx, row in rejected_phys.iterrows():
						_sig_v = row[_sig_col] if _sig_col else np.nan
						_tau_v = row[_tau_col] if _tau_col else np.nan
						print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, "
						      f"σ={_sig_v:.5f} ms, τ={_tau_v:.5f} ms (delta-function pathology)")
		# === END FILTER 5B ===


		# === STORE COUNT BEFORE AMPLITUDE FILTER ===
		count_before_amp = len(subdf)

		subdf = subdf[(subdf.amp > 0)]

		# === DIAGNOSTIC 3: After amplitude filter ===
		if SHOW_DIAGNOSTICS:
			print(f"After amplitude filter (amp>0): {len(subdf)} rows (lost {count_before_amp - len(subdf)})")
		printd(f"Debug: after fit amplitude filter {len(subdf) = }")

		# === STORE COUNT BEFORE UNCERTAINTY FILTER ===
		count_before_uncert = len(subdf)

		# --- Arrival-time uncertainty ratio filter ---
		# EMG path: use t_emg_hwhm_err / |t_emg_hwhm|
		# Gaussian/legacy path: keep tstart_err / tstart
		if USE_EMG_MIX and ('t_emg_hwhm' in subdf.columns) and ('t_emg_hwhm_err' in subdf.columns):
			tiny = max(1e-6, 0.1 * res_time_ms)  # avoid zero/near-zero denom
			denom = subdf['t_emg_hwhm'].abs().clip(lower=tiny)
			ratio = subdf['t_emg_hwhm_err'] / denom
			# === DIAGNOSTIC 3.5: Show uncertainty ratio stats ===
			if SHOW_DIAGNOSTICS and len(ratio) > 0:
				print(f"  Uncertainty ratio stats: median={np.nanmedian(ratio):.2f}, max={np.nanmax(ratio):.2f}, "
					  f">10: {(ratio >= 10).sum()} rows, NaN: {(~np.isfinite(ratio)).sum()}")
			mask = np.isfinite(ratio) & (ratio < 10)
			subdf = subdf[mask]
		else:
			legacy_ratio = subdf['tstart_err'] / subdf['tstart']
			# === DIAGNOSTIC 3.5: Show uncertainty ratio stats ===
			if SHOW_DIAGNOSTICS and len(legacy_ratio) > 0:
				print(f"  Uncertainty ratio stats: median={np.nanmedian(legacy_ratio):.2f}, max={np.nanmax(legacy_ratio):.2f}, "
					  f">10: {(legacy_ratio >= 10).sum()} rows")
			mask = np.isfinite(legacy_ratio) & (legacy_ratio < 10)
			subdf = subdf[mask]

		# === DIAGNOSTIC 4: After uncertainty filter ===
		if SHOW_DIAGNOSTICS:
			print(f"After uncertainty filter (ratio<10): {len(subdf)} rows (lost {count_before_uncert - len(subdf)})")
		printd(f"Debug: after tstart_err filter {len(subdf) = }")

		# Choose the time column used as "arrival time" for slope/arr-times:
		#  - EMG mode: use left-HWHM crossing (leading edge)
		#  - Gaussian mode: respect the existing tpoint ('tstart' by default)
		if USE_EMG_MIX:
			time_col = 't_emg_hwhm'
		else:
			time_col = tpoint  # 'tstart' | 'tend' | 'xo' (existing behavior)

		# Use tail-aware width for EMG mode
		width_for_filter = tmix_sigmas_std[xos.index(xosi)] if USE_EMG_MIX else sigma
# 		width_for_filter = sigma

		# === STORE COUNT BEFORE TIME FILTER ===
		count_before_time = len(subdf)

		# === TIME FILTER DIAGNOSTIC ===
		if diag_enabled('TIME_FILTER'):
			print(f"\n=== TIME FILTER DIAGNOSTIC for {rowname} ===")
			print(f"  Before time window filter: {len(subdf)} rows")
			print(f"  time_col = '{time_col}'")
			print(f"  sigma (core) = {sigma:.4f}, width_for_filter (sigma_std) = {width_for_filter:.4f}")
			print(f"  subpktime (expected peak in LOCAL coords) = {subpktime:.4f} ms")
			if len(subdf) > 0 and time_col in subdf.columns:
				valid_times = subdf[time_col].dropna()
				print(f"  Actual {time_col} range: [{valid_times.min():.4f}, {valid_times.max():.4f}] ms")
				print(f"  Valid values: {len(valid_times)}, NaN: {subdf[time_col].isna().sum()}")

		# Make it conditional:
		if USE_EMG_MIX:
			t_filter_factor_actual = 2.0  # More lenient for EMG leading-edge arrivals
		else:
			t_filter_factor_actual = t_filter_factor  # Keep user's setting

		# === TIME FILTER WINDOW ===
		filter_low = subpktime - t_filter_factor_actual * width_for_filter
		filter_high = subpktime + t_filter_factor_actual * width_for_filter
		if diag_enabled('TIME_FILTER'):
			print(f"  Filter window ({t_filter_factor_actual}×σ): [{filter_low:.4f}, {filter_high:.4f}] ms")
			if len(subdf) > 0 and time_col in subdf.columns:
				in_window = ((subdf[time_col] > filter_low) & (subdf[time_col] < filter_high)).sum()
				below_window = (subdf[time_col] <= filter_low).sum()
				above_window = (subdf[time_col] >= filter_high).sum()
				print(f"  Points in window: {in_window}, below: {below_window}, above: {above_window}")

		if SHOW_DIAGNOSTICS:
			print(f"  Filter range ({t_filter_factor_actual}×σ): [{subpktime-t_filter_factor_actual*width_for_filter:.3f}, "
				  f"{subpktime+t_filter_factor_actual*width_for_filter:.3f}]")

		subdf = subdf[
			(subpktime - t_filter_factor_actual*width_for_filter < subdf[time_col]) &
			(subdf[time_col] < subpktime + t_filter_factor_actual*width_for_filter)
		]

		# === FILTER CHAIN SUMMARY ===
		filter_tracking['after_spectral'] = count_before_amp  # The count after spectral filter is the count before amp filter
		filter_tracking['after_amplitude'] = count_before_uncert  # The count after amp filter is count before uncert filter
		filter_tracking['after_uncertainty'] = count_before_time  # The count after uncert filter is count before time filter
		filter_tracking['final'] = len(subdf)
		
		if diag_enabled('TIME_FILTER'):
			print(f"  After time filter: {len(subdf)} rows (lost {count_before_time - len(subdf)})")
			print(f"\n  *** FILTER CHAIN SUMMARY for {rowname} ***")
			print(f"    Initial:           {filter_tracking['initial']:4d} rows")
			print(f"    After spectral:    {filter_tracking['after_spectral']:4d} rows (lost {filter_tracking['initial'] - filter_tracking['after_spectral']:4d})")
			print(f"    After amplitude:   {filter_tracking['after_amplitude']:4d} rows (lost {filter_tracking['after_spectral'] - filter_tracking['after_amplitude']:4d})")
			print(f"    After uncertainty: {filter_tracking['after_uncertainty']:4d} rows (lost {filter_tracking['after_amplitude'] - filter_tracking['after_uncertainty']:4d})")
			print(f"    After time window: {filter_tracking['final']:4d} rows (lost {filter_tracking['after_uncertainty'] - filter_tracking['final']:4d})")
			print(f"    TOTAL RETAINED:    {filter_tracking['final']:4d} / {filter_tracking['initial']:4d} ({100*filter_tracking['final']/max(filter_tracking['initial'],1):.1f}%)")
			print(f"=== END TIME FILTER for {rowname}: {len(subdf)} TOAs retained ===\n")
		# === END TIME FILTER DIAGNOSTIC ===

		printd(f"Debug: after time window filter {len(subdf) = }")
		printd(f"Debug: post-filters {len(subdf) = }")

		## Measure dt/dnu using stacked arrival times when stacking is enabled
		if USE_EMG_MIX and use_stacking_this_comp and 'stacked_df' in locals() and stacked_df is not None and len(stacked_df) > 0:
			# Use STACKED arrival times for the line fit
			fit_df = stacked_df.copy()
			if SHOW_DIAGNOSTICS:
				print(f"\n=== Using STACKED EMG arrival times for line fit ({len(fit_df)} points) ===")
		elif USE_GAUSS_MIX and use_stacking_this_comp and 'stacked_gauss_df' in locals() and stacked_gauss_df is not None and len(stacked_gauss_df) > 0:
			# Use STACKED Gaussian arrival times for the line fit
			fit_df = stacked_gauss_df.copy()
			if SHOW_DIAGNOSTICS:
				print(f"\n=== Using STACKED Gaussian arrival times for line fit ({len(fit_df)} points) ===")
		else:
			# Use individual row arrival times
			fit_df = subdf.copy()
			if SHOW_DIAGNOSTICS:
				print(f"\n=== Using individual arrival times for line fit ({len(fit_df)} points) ===")

		# Apply time window filter to the fitting dataframe
		width_for_filter = tmix_sigmas_std[xos.index(xosi)] if USE_EMG_MIX else sigma
# 		width_for_filter = sigma       
		t_filter_factor_actual = 2.0 if USE_EMG_MIX else t_filter_factor
		
		fit_df = fit_df[
			(subpktime - t_filter_factor_actual*width_for_filter < fit_df[time_col]) &
			(fit_df[time_col] < subpktime + t_filter_factor_actual*width_for_filter)
		]
		
		# Also filter stacked_df for consistent plotting
		if USE_EMG_MIX and use_stacking_this_comp and 'stacked_df' in locals() and stacked_df is not None:
			count_before_time_filter = len(stacked_df)
			time_window_min = subpktime - t_filter_factor_actual*width_for_filter
			time_window_max = subpktime + t_filter_factor_actual*width_for_filter
			time_mask = (time_window_min < stacked_df[time_col]) & (stacked_df[time_col] < time_window_max)
			if '_emg_audit' in dir():
				_f6_idx = list(stacked_df.index)
				_f6_pass = {ix: bool(time_mask.loc[ix]) for ix in _f6_idx}
				_f6_t = {ix: stacked_df.loc[ix, time_col] for ix in _f6_idx}
				_emg_audit.record(
					f'Filter 6 (time window {t_filter_factor_actual:g}x sigma)',
					_f6_idx, _f6_pass,
					lambda ix: f"t_arr = {_f6_t[ix]:.4f} ms  "
					           f"(window {time_window_min:.3f} to {time_window_max:.3f})")
			rejected_time = stacked_df[~time_mask]
			stacked_df = stacked_df[time_mask]
# 		    print(f"\nFilter 5 (Stacked) - Time window ({t_filter_factor_actual}×σ): {count_before_time_filter} → {len(stacked_df)} stacks ({count_before_time_filter - len(stacked_df)} rejected)")
			print(f"\nFilter 6 (Stacked) - Time window ({t_filter_factor_actual}×σ): {count_before_time_filter} → {len(stacked_df)} stacks ({count_before_time_filter - len(stacked_df)} rejected)")            
			print(f"  Time window: [{time_window_min:.4f}, {time_window_max:.4f}] ms (centered on subpktime={subpktime:.4f})")
			if len(rejected_time) > 0:
				for idx, row in rejected_time.iterrows():
					print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, t_arr={row[time_col]:.4f} ms (outside window)")
			print(f"\n=== FINAL Stacked TOAs for line fit: {len(stacked_df)} ===\n")

		# Also filter stacked_gauss_df for consistent plotting (Gaussian stacking)
		if USE_GAUSS_MIX and use_stacking_this_comp and 'stacked_gauss_df' in locals() and stacked_gauss_df is not None:
			count_before_time_filter = len(stacked_gauss_df)
			time_window_min = subpktime - t_filter_factor_actual*width_for_filter
			time_window_max = subpktime + t_filter_factor_actual*width_for_filter
			time_mask = (time_window_min < stacked_gauss_df[tpoint]) & (stacked_gauss_df[tpoint] < time_window_max)
			if '_gauss_audit' in dir():
				_g5_idx = list(stacked_gauss_df.index)
				_g5_pass = {ix: bool(time_mask.loc[ix]) for ix in _g5_idx}
				_g5_t = {ix: stacked_gauss_df.loc[ix, tpoint] for ix in _g5_idx}
				_gauss_audit.record(
					f'Filter 5 (time window {t_filter_factor_actual:g}x sigma)',
					_g5_idx, _g5_pass,
					lambda ix: f"t_arr = {_g5_t[ix]:.4f} ms  "
					           f"(window {time_window_min:.3f} to {time_window_max:.3f})")
			rejected_time = stacked_gauss_df[~time_mask]
			stacked_gauss_df = stacked_gauss_df[time_mask]
			print(f"\nFilter 5 (Gaussian Stacked) - Time window ({t_filter_factor_actual}×σ): {count_before_time_filter} → {len(stacked_gauss_df)} stacks ({count_before_time_filter - len(stacked_gauss_df)} rejected)")
			print(f"  Time window: [{time_window_min:.4f}, {time_window_max:.4f}] ms (centered on subpktime={subpktime:.4f})")
			if len(rejected_time) > 0:
				for idx, row in rejected_time.iterrows():
					print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, tstart={row[tpoint]:.4f} ms (outside window)")
			print(f"\n=== FINAL Gaussian Stacked TOAs for line fit: {len(stacked_gauss_df)} ===\n")

		# === IGNORE FILTER: Remove stacks/channels containing ignored channels ===
		# This filter removes arrival times from stacks that contain any channel in the ignore list
		# For non-stacked data, it directly removes the specified channel indices
		# Applied to both stacked_df and fit_df for consistency
		# Initialize ignored dataframes for transparent plotting
		ignored_stacked_df = None
		ignored_stacked_gauss_df = None
		ignored_subdf = None
		
		if ignore and len(ignore) > 0:
			n_freq_for_ignore = len(freqs)  # Total number of frequency channels
			stack_size_for_ignore = stack_size_this_comp  # Current stack size
			
			# Expand any range objects in ignore list for direct channel comparison
			expanded_ignore = []
			for item in ignore:
				if hasattr(item, '__iter__') and not isinstance(item, (int, np.integer)):
					expanded_ignore.extend(list(item))
				else:
					expanded_ignore.append(item)
			
			# Filter stacked_df (EMG stacking)
			if USE_EMG_MIX and use_stacking_this_comp and 'stacked_df' in locals() and stacked_df is not None and len(stacked_df) > 0:
				count_before_ignore = len(stacked_df)
				# Use the dataframe index which corresponds to stack indices
				ignore_mask = stacked_df.index.map(
					lambda idx: not should_ignore_stack(idx, stack_size_for_ignore, n_freq_for_ignore, ignore)
				)
				if '_emg_audit' in dir():
					_f7_idx = list(stacked_df.index)
					_f7_mask_d = dict(zip(_f7_idx, list(ignore_mask)))
					_f7_pass = {ix: bool(_f7_mask_d[ix]) for ix in _f7_idx}
					_emg_audit.record(
						'Filter 7 (ignore channels)', _f7_idx, _f7_pass,
						lambda ix: ("stack clear of ignore list" if _f7_pass[ix]
						            else "stack contains an ignored channel"))
				rejected_ignore = stacked_df[~ignore_mask]
				ignored_stacked_df = rejected_ignore.copy()  # Save for plotting
				stacked_df = stacked_df[ignore_mask]
				print(f"\nFilter 7 (Stacked) - Ignore channels {ignore}: {count_before_ignore} → {len(stacked_df)} stacks ({count_before_ignore - len(stacked_df)} rejected)")
				if len(rejected_ignore) > 0:
					for idx, row in rejected_ignore.iterrows():
						stack_chs = get_stack_channels(idx, stack_size_for_ignore, n_freq_for_ignore)
						print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, channels={stack_chs}")
				print(f"\n=== FINAL Stacked TOAs after ignore filter: {len(stacked_df)} ===\n")
			
			# Filter stacked_gauss_df (Gaussian stacking)
			if USE_GAUSS_MIX and use_stacking_this_comp and 'stacked_gauss_df' in locals() and stacked_gauss_df is not None and len(stacked_gauss_df) > 0:
				count_before_ignore = len(stacked_gauss_df)
				ignore_mask = stacked_gauss_df.index.map(
					lambda idx: not should_ignore_stack(idx, stack_size_for_ignore, n_freq_for_ignore, ignore)
				)
				if '_gauss_audit' in dir():
					_g7_idx = list(stacked_gauss_df.index)
					_g7_mask_d = dict(zip(_g7_idx, list(ignore_mask)))
					_g7_pass = {ix: bool(_g7_mask_d[ix]) for ix in _g7_idx}
					_gauss_audit.record(
						'Filter 6 (ignore channels)', _g7_idx, _g7_pass,
						lambda ix: ("stack clear of ignore list" if _g7_pass[ix]
						            else "stack contains an ignored channel"))
				rejected_ignore = stacked_gauss_df[~ignore_mask]
				ignored_stacked_gauss_df = rejected_ignore.copy()  # Save for plotting
				stacked_gauss_df = stacked_gauss_df[ignore_mask]
				print(f"\nFilter 6 (Gaussian Stacked) - Ignore channels {ignore}: {count_before_ignore} → {len(stacked_gauss_df)} stacks ({count_before_ignore - len(stacked_gauss_df)} rejected)")
				if len(rejected_ignore) > 0:
					for idx, row in rejected_ignore.iterrows():
						stack_chs = get_stack_channels(idx, stack_size_for_ignore, n_freq_for_ignore)
						print(f"  Rejected stack {idx}: freq={row['freqs']:.2f} MHz, channels={stack_chs}")
				print(f"\n=== FINAL Gaussian Stacked TOAs after ignore filter: {len(stacked_gauss_df)} ===\n")
			
			# Filter fit_df (used for the actual slope calculation)
			if 'fit_df' in locals() and fit_df is not None and len(fit_df) > 0:
				count_before_ignore_fit = len(fit_df)
				if use_stacking_this_comp:
					# Stacked data: filter by stack index
					ignore_mask_fit = fit_df.index.map(
						lambda idx: not should_ignore_stack(idx, stack_size_for_ignore, n_freq_for_ignore, ignore)
					)
				else:
					# Non-stacked data: filter by channel index directly
					ignore_mask_fit = ~fit_df.index.isin(expanded_ignore)
				fit_df = fit_df[ignore_mask_fit]
				if count_before_ignore_fit != len(fit_df):
					print(f"  Applied ignore filter to fit_df: {count_before_ignore_fit} → {len(fit_df)} points")
			
			# Also filter subdf for non-stacked plotting consistency
			if not use_stacking_this_comp and 'subdf' in locals() and subdf is not None and len(subdf) > 0:
				count_before_ignore_subdf = len(subdf)
				ignore_mask_subdf = ~subdf.index.isin(expanded_ignore)
				rejected_ignore_subdf = subdf[~ignore_mask_subdf]
				ignored_subdf = rejected_ignore_subdf.copy()  # Save for plotting
				subdf = subdf[ignore_mask_subdf]
				if count_before_ignore_subdf != len(subdf):
					print(f"\nFilter - Ignore channels {ignore}: {count_before_ignore_subdf} → {len(subdf)} rows ({count_before_ignore_subdf - len(subdf)} rejected)")
					if len(rejected_ignore_subdf) > 0:
						for idx, row in rejected_ignore_subdf.iterrows():
							print(f"  Rejected channel {idx}: freq={row['freqs']:.2f} MHz")
		# === END IGNORE FILTER ===

		# === bw_ToA_OVERRIDE FILTER ===
		# Hard-coded channel-range restriction for the slope-fit ToA
		# list. When set, channels outside [i_lo, i_hi] are diverted
		# into `ignored_*` dataframes so they render as grey hollow
		# circles (same treatment as the `ignore` filter). Stacks
		# whose channels straddle the range boundary are also
		# diverted — a stack is kept only if ALL its constituent
		# channels fall inside the override range.
		#
		# This stacks with `ignore` (which usually targets a few
		# one-off bad channels); together you get both effects.
		if _bw_toa_co_this is not None:
			_toa_lo, _toa_hi = int(_bw_toa_co_this[0]), int(_bw_toa_co_this[1])

			# Filter unstacked subdf by direct channel index.
			if not use_stacking_this_comp and 'subdf' in locals() and subdf is not None and len(subdf) > 0:
				_n_before = len(subdf)
				_keep_mask = (subdf.index >= _toa_lo) & (subdf.index <= _toa_hi)
				_dropped = subdf[~_keep_mask]
				if len(_dropped) > 0:
					# Merge into existing ignored_subdf (set above by
					# `ignore`) so both filters' rejected rows render
					# together as grey hollow circles.
					if ignored_subdf is None or len(ignored_subdf) == 0:
						ignored_subdf = _dropped.copy()
					else:
						ignored_subdf = pd.concat([ignored_subdf, _dropped])
				subdf = subdf[_keep_mask]
				if _n_before != len(subdf):
					print(f"\nFilter - bw_ToA_override channels [{_toa_lo}, {_toa_hi}]: "
					      f"{_n_before} → {len(subdf)} rows ({_n_before - len(subdf)} rejected)")

			# Filter fit_df. For stacked: keep stacks whose ALL
			# constituent channels are in [i_lo, i_hi]. For
			# non-stacked: filter by channel index directly.
			if 'fit_df' in locals() and fit_df is not None and len(fit_df) > 0:
				_n_before_fit = len(fit_df)
				if use_stacking_this_comp:
					_n_freq_fit = len(freqs)
					_ss_fit = stack_size_this_comp

					def _stack_fully_inside_toa_range(stack_idx, _ss=_ss_fit, _nf=_n_freq_fit,
					                                   _lo=_toa_lo, _hi=_toa_hi):
						_chs = get_stack_channels(stack_idx, _ss, _nf)
						return all((_lo <= ch <= _hi) for ch in _chs)
					_keep_mask_fit = fit_df.index.map(_stack_fully_inside_toa_range)
				else:
					_keep_mask_fit = (fit_df.index >= _toa_lo) & (fit_df.index <= _toa_hi)
				fit_df = fit_df[_keep_mask_fit]
				if _n_before_fit != len(fit_df):
					print(f"  Applied bw_ToA_override filter to fit_df: "
					      f"{_n_before_fit} → {len(fit_df)} points")

			# Filter stacked_df (EMG) — same all-channels-inside rule.
			if USE_EMG_MIX and use_stacking_this_comp and 'stacked_df' in locals() and stacked_df is not None and len(stacked_df) > 0:
				_n_before_sd = len(stacked_df)
				_n_freq_sd = len(freqs)
				_ss_sd = stack_size_this_comp

				def _stack_fully_inside_emg(stack_idx, _ss=_ss_sd, _nf=_n_freq_sd,
				                             _lo=_toa_lo, _hi=_toa_hi):
					_chs = get_stack_channels(stack_idx, _ss, _nf)
					return all((_lo <= ch <= _hi) for ch in _chs)
				_keep_mask_sd = stacked_df.index.map(_stack_fully_inside_emg)
				_dropped_sd = stacked_df[~_keep_mask_sd]
				if len(_dropped_sd) > 0:
					if ignored_stacked_df is None or len(ignored_stacked_df) == 0:
						ignored_stacked_df = _dropped_sd.copy()
					else:
						ignored_stacked_df = pd.concat([ignored_stacked_df, _dropped_sd])
				stacked_df = stacked_df[_keep_mask_sd]
				if _n_before_sd != len(stacked_df):
					print(f"  Filter (Stacked EMG) - bw_ToA_override channels "
					      f"[{_toa_lo}, {_toa_hi}]: {_n_before_sd} → "
					      f"{len(stacked_df)} stacks ({_n_before_sd - len(stacked_df)} rejected)")

			# Filter stacked_gauss_df (Gaussian) — same rule.
			if USE_GAUSS_MIX and use_stacking_this_comp and 'stacked_gauss_df' in locals() and stacked_gauss_df is not None and len(stacked_gauss_df) > 0:
				_n_before_sg = len(stacked_gauss_df)
				_n_freq_sg = len(freqs)
				_ss_sg = stack_size_this_comp

				def _stack_fully_inside_gauss(stack_idx, _ss=_ss_sg, _nf=_n_freq_sg,
				                               _lo=_toa_lo, _hi=_toa_hi):
					_chs = get_stack_channels(stack_idx, _ss, _nf)
					return all((_lo <= ch <= _hi) for ch in _chs)
				_keep_mask_sg = stacked_gauss_df.index.map(_stack_fully_inside_gauss)
				_dropped_sg = stacked_gauss_df[~_keep_mask_sg]
				if len(_dropped_sg) > 0:
					if ignored_stacked_gauss_df is None or len(ignored_stacked_gauss_df) == 0:
						ignored_stacked_gauss_df = _dropped_sg.copy()
					else:
						ignored_stacked_gauss_df = pd.concat([ignored_stacked_gauss_df, _dropped_sg])
				stacked_gauss_df = stacked_gauss_df[_keep_mask_sg]
				if _n_before_sg != len(stacked_gauss_df):
					print(f"  Filter (Stacked Gaussian) - bw_ToA_override channels "
					      f"[{_toa_lo}, {_toa_hi}]: {_n_before_sg} → "
					      f"{len(stacked_gauss_df)} stacks ({_n_before_sg - len(stacked_gauss_df)} rejected)")
		# === END bw_ToA_OVERRIDE FILTER ===

		# Prepare uncertainty array
		if USE_EMG_MIX and 't_emg_hwhm_err' in fit_df.columns:
			sigma_arr = fit_df['t_emg_hwhm_err'].to_numpy()
			# guard against zeros/NaNs - use stronger replacement logic
			valid_mask = np.isfinite(sigma_arr) & (sigma_arr > 1e-12)
			n_invalid = np.sum(~valid_mask)
			if n_invalid > 0:
				print(f"  WARNING: {n_invalid} uncertainty values are zero/invalid, replacing with median")
			if np.any(valid_mask):
				replacement = np.nanmedian(sigma_arr[valid_mask])
			else:
				replacement = 1e-6  # fallback if all invalid
				print(f"  WARNING: All uncertainties invalid, using fallback={replacement}")
			sigma_arr = np.where(valid_mask, sigma_arr, replacement)
			
			# Floor suspiciously small uncertainties at 20% of median
			median_err = np.nanmedian(sigma_arr[sigma_arr > 1e-12])
			err_floor = 0.2 * median_err  # Floor at 20% of median
# 			err_floor = 0.4 * median_err  # Floor at 40% of median
			n_floored = np.sum(sigma_arr < err_floor)
			if n_floored > 0:
				print(f"  WARNING: {n_floored} uncertainties below floor ({err_floor:.6f} ms), raising to floor")
				sigma_arr = np.maximum(sigma_arr, err_floor)            
			
			# === OPTION 1: Scale uncertainties for stacked data ===
			if use_stacking_this_comp and 'stacked_df' in locals() and stacked_df is not None and len(stacked_df) > 0:
				# Stacked uncertainties are often underestimated; scale by empirical factor
				# This factor can be tuned based on observed scatter vs. error bars
				scaling_factor = 1.0  # A
				sigma_arr = sigma_arr * scaling_factor
				print(f"  Applied stacking uncertainty scaling factor: {scaling_factor}")
			# === END OPTION 1 ===
		elif USE_GAUSS_MIX and f'{tpoint}_err' in fit_df.columns:
			sigma_arr = fit_df[f'{tpoint}_err'].to_numpy()
			# guard against zeros/NaNs - use stronger replacement logic
			valid_mask = np.isfinite(sigma_arr) & (sigma_arr > 1e-12)
			n_invalid = np.sum(~valid_mask)
			if n_invalid > 0:
				print(f"  WARNING: {n_invalid} uncertainty values are zero/invalid, replacing with median")
			if np.any(valid_mask):
				replacement = np.nanmedian(sigma_arr[valid_mask])
			else:
				replacement = 1e-6  # fallback if all invalid
				print(f"  WARNING: All uncertainties invalid, using fallback={replacement}")
			sigma_arr = np.where(valid_mask, sigma_arr, replacement)
			
			# Floor suspiciously small uncertainties at 20% of median
			median_err = np.nanmedian(sigma_arr[sigma_arr > 1e-12])
			if np.isfinite(median_err) and median_err > 0:
				err_floor = 0.2 * median_err  # Floor at 20% of median
				n_floored = np.sum(sigma_arr < err_floor)
				if n_floored > 0:
					print(f"  WARNING: {n_floored} uncertainties below floor ({err_floor:.6f} ms), raising to floor")
					sigma_arr = np.maximum(sigma_arr, err_floor)
			
			# Scale uncertainties for Gaussian stacked data
			if use_stacking_this_comp and 'stacked_gauss_df' in locals() and stacked_gauss_df is not None and len(stacked_gauss_df) > 0:
				# With normalization applied during fitting, uncertainties should be properly scaled
				# Set to 1.0 (no additional scaling needed)
				scaling_factor = 1.0  # Normalization fix applied - no additional scaling needed
				sigma_arr = sigma_arr * scaling_factor
				print(f"  Applied Gaussian stacking uncertainty scaling factor: {scaling_factor}")
		else:
			sigma_arr = fit_df[f'{tpoint}_err'].to_numpy()

		if len(fit_df) > 1:  # only fit a line if more than 1 point
			tdata = fit_df[time_col] - subpktime
			
			# === LINE FIT DIAGNOSTIC ===
			if diag_enabled('LINE_FIT'):
				print(f"\n=== LINE FIT DIAGNOSTIC for {rowname} ===")
				print(f"  fit_df has {len(fit_df)} points")
				print(f"  time_col = '{time_col}'")
				print(f"  subpktime (subtracted from times) = {subpktime:.4f} ms")
				print(f"  fit_df[time_col] range (LOCAL): [{fit_df[time_col].min():.4f}, {fit_df[time_col].max():.4f}] ms")
				print(f"  tdata range (LOCAL - subpktime): [{tdata.min():.4f}, {tdata.max():.4f}] ms")
				print(f"  xosi (component center, GLOBAL): {xosi:.4f} ms")
			
			popt, pcov = scipy.optimize.curve_fit(
				line_model,
				fit_df['freqs'],
				tdata,
				sigma=sigma_arr,
			)
			perr = np.sqrt(np.diag(pcov))
			residuals = (tdata - line_model(fit_df['freqs'], *popt)) / sigma_arr
			chisq = np.sum((residuals) ** 2)
			dtdnu_redchisq = chisq / (len(tdata)- len(popt))
			dtdnu, dtdnu_err = popt[0], perr[0]
			t_b, tb_err = popt[1], perr[1]

			# === LINE FIT RESULTS DIAGNOSTIC ===
			if diag_enabled('LINE_FIT'):
				print(f"  LINE FIT RESULTS:")
				print(f"    dtdnu = {dtdnu:.5e} ms/MHz")
				print(f"    t_b (intercept) = {t_b:.4f} ms")
				print(f"    dtdnu_redchisq = {dtdnu_redchisq:.2f}")

			print(f"{dtdnu = :.5e} +/- {dtdnu_err:.5e} ms/MHz {dtdnu_redchisq = :.1f}")
			
			
			
			# === CHI-SQUARED DIAGNOSTIC ===
			if diag_enabled('CHI_SQUARED'):
				print("\n" + "="*60)
				print("CHI-SQUARED DIAGNOSTIC REPORT")
				print("="*60)

				# 1. Basic statistics
				print(f"\n[1] FIT SUMMARY:")
				print(f"    N points: {len(tdata)}")
				print(f"    DOF: {len(tdata) - len(popt)}")
				print(f"    Total χ²: {chisq:.2f}")
				print(f"    Reduced χ²: {dtdnu_redchisq:.2f}")

				# 2. Arrival time statistics
				print(f"\n[2] ARRIVAL TIME (tdata) STATISTICS:")
				print(f"    Range: [{tdata.min():.6f}, {tdata.max():.6f}] ms")
				print(f"    Mean: {tdata.mean():.6f} ms")
				print(f"    Std dev: {tdata.std():.6f} ms")
				print(f"    Median: {np.median(tdata):.6f} ms")

				# 3. Uncertainty statistics  
				print(f"\n[3] UNCERTAINTY (sigma_arr) STATISTICS:")
				print(f"    Range: [{sigma_arr.min():.6f}, {sigma_arr.max():.6f}] ms")
				print(f"    Mean: {sigma_arr.mean():.6f} ms")
				print(f"    Median: {np.median(sigma_arr):.6f} ms")

				# 4. Compare scatter to uncertainties
				print(f"\n[4] SCATTER VS UNCERTAINTY COMPARISON:")
				model_predictions = line_model(fit_df['freqs'].to_numpy(), *popt)
				raw_residuals = tdata.to_numpy() - model_predictions
				print(f"    RMS of residuals: {np.sqrt(np.mean(raw_residuals**2)):.6f} ms")
				print(f"    Mean uncertainty: {sigma_arr.mean():.6f} ms")
				print(f"    Ratio (RMS/mean_err): {np.sqrt(np.mean(raw_residuals**2))/sigma_arr.mean():.2f}")
				print(f"    --> If ratio >> 1, uncertainties are underestimated")

				# 5. Worst offenders (points contributing most to chi-squared)
				print(f"\n[5] TOP 5 WORST RESIDUALS (highest χ² contribution):")
				chi_contributions = (residuals**2).to_numpy() if hasattr(residuals, 'to_numpy') else np.array(residuals**2)
				sorted_idx = np.argsort(chi_contributions)[::-1]
				freq_arr_diag = fit_df['freqs'].to_numpy()
				tdata_arr_diag = tdata.to_numpy() if hasattr(tdata, 'to_numpy') else np.array(tdata)
				for i, idx in enumerate(sorted_idx[:5]):
					print(f"    #{i+1}: freq={freq_arr_diag[idx]:.1f} MHz, t={tdata_arr_diag[idx]:.6f} ms, "
						  f"err={sigma_arr[idx]:.6f} ms, |resid|={abs(raw_residuals[idx]):.6f} ms, "
						  f"χ²_contrib={chi_contributions[idx]:.2f}")

				# 6. What scaling factor would give χ²_red = 1?
				implied_scale = np.sqrt(dtdnu_redchisq)
				print(f"\n[6] SCALING ANALYSIS:")
				print(f"    Current scaling factor: {scaling_factor if 'scaling_factor' in dir() else 1.0}")
				print(f"    Implied scaling to get χ²_red=1: {implied_scale:.2f}")
				print(f"    --> Multiply current uncertainties by {implied_scale:.2f}")

				# 7. Check for systematic trends in residuals
				print(f"\n[7] RESIDUAL TREND CHECK:")
				freq_arr = fit_df['freqs'].to_numpy()
				freq_mid = np.median(freq_arr)
				low_freq_resid = raw_residuals[freq_arr < freq_mid]
				high_freq_resid = raw_residuals[freq_arr >= freq_mid]
				print(f"    Mean residual (low freq): {low_freq_resid.mean():.6f} ms")
				print(f"    Mean residual (high freq): {high_freq_resid.mean():.6f} ms")
				print(f"    --> If different signs, may indicate non-linear drift")

				print("="*60 + "\n")
			# === END DIAGNOSTIC BLOCK ===
			
			
		else:  # no measurement
			dtdnu, dtdnu_err, dtdnu_redchisq = 0, 0, 0
			t_b, tb_err = 0, 0

		# === ACF MODE: override the slope with the 2D-ACF measurement ===
		# For ACF components the per-channel ToA line fit above is not the
		# slope source. Replace dtdnu/dtdnu_err/dtdnu_redchisq with the ACF
		# values (pipeline dt/dnu convention). t_b has no ACF analogue and
		# is left as set above. On ACF fit failure the values are NaN and a
		# warning was already printed by acf_measure_component().
		if _comp_acf_on:
			if _comp_acf_result is not None and _comp_acf_result['ok']:
				dtdnu          = _comp_acf_result['dtdnu']
				dtdnu_err      = _comp_acf_result['dtdnu_err']
				dtdnu_redchisq = _comp_acf_result['redchisq']
			else:
				dtdnu, dtdnu_err, dtdnu_redchisq = np.nan, np.nan, np.nan
		# === END ACF MODE override ===

		# Sub-burst plot
		if show_components:
			subfig, subaxs = plotburst(
				subfall,
				subband,#.reshape(-1, 4).mean(axis=1),
				retfig=True,
				extent=[
					0,
					res_time_ms*subfall.shape[1],
					freqs_bin0,
					freqs_bin0 + res_freq*wfall.shape[0]
				]
			)
			subcolors = [(1, 1, 1, alpha) for alpha in subdf['amp']/subdf['amp'].max()]
			subaxs['W'].scatter(
				(subdf[time_col]),
				(subdf['freqs']),
				c=subcolors,
				edgecolor='r',
				marker='o',
				s=25
			)
			subaxs['W'].set_xlim(0, res_time_ms*subfall.shape[1])
			subaxs['W'].set_ylim(freqs_bin0, freqs_bin0 + res_freq*wfall.shape[0])
			subtimes = np.linspace(0, res_time_ms*subfall.shape[1], num=1000)
			
			if dtdnu != 0:
				subaxs['W'].plot(
					subtimes,
					(1.0/dtdnu)*(subtimes-subpktime),
# 					(1/dtdnu)*(subtimes-(subpktime + t_b)),
					'w--',
					label=f'{dtdnu=:.2e} ms/MHz'
				)
			subaxs['W'].legend()
			
# 			# guard against divide-by-zero & include fitted intercept in the anchor
# 			if np.isfinite(dtdnu) and dtdnu != 0:
# 				subaxs['W'].plot(
# 					subtimes,
# 					(1.0/dtdnu) * (subtimes - (subpktime + t_b)),
# 					'w--',
# 					label=f'dtdnu={dtdnu:.2e} ms/MHz'
# 				)


			subaxs['B'].plot(
				gauss_model(freqs, *subband_popt),
				freqs
			)
			subaxs['B'].plot(noisesmpl,freqs)
			if bandmask_thresi:
				plt.axvline(x=bandmask_thresi, c='r', ls='--')
			plt.title(rowname)
			plt.show(block=True)
			plt.close()
		subbandmodels.append(gauss_model(freqs, *subband_popt))


		if len(cuts) == 0:
			# Compute the shift that converts subfall-local coordinates
			# (where t=0 is the subfall's left edge) to raw-waterfall
			# coordinates (where t=0 is the waterfall's left edge).
			#
			# The shift IS the subfall's left edge in raw ms, plain and
			# simple. Earlier versions computed it as `xc - 4*width_used`
			# (the *uncliped* expression for the proposed left edge);
			# that's correct as long as the proposed window fits inside
			# the waterfall, but silently over-shifts whenever the
			# subfall got clipped to the waterfall boundary at line 5964.
			# L741965_b component a is a clear case: the proposed left
			# edge sat at raw_ms ≈ -21 (i.e. xc=93 ms minus 4σ_std=114 ms),
			# the actual subfall began at raw_ms = 0, and every stacked
			# TOA picked up a spurious -21 ms offset.
			_comp_idx_for_shift = xos.index(xosi)
			_lefe_chan = _subfall_left_chan_per_comp.get(_comp_idx_for_shift, None)
			if _lefe_chan is not None:
				shift = float(_lefe_chan) * res_time_ms
			else:
				# Fallback: only fires if the prep loop never ran for
				# this component (shouldn't happen on a normal path).
				# Keeps the old expression as a defensive last resort.
				width_used = tmix_sigmas_std[xos.index(xosi)] if USE_EMG_MIX else abs(sigma)
				if USE_EMG_MIX:
					xc_i = xcs[xos.index(xosi)]
					shift = xc_i - 4*width_used
				else:
					shift = xosi - 4*width_used
				print(f"  Warning: subfall_left_chan unavailable for component "
				      f"{_comp_idx_for_shift}; falling back to legacy shift expression "
				      f"({shift:.3f} ms). TOAs may be off by up to a few sigma_emg_std "
				      f"if the proposed window was clipped at the waterfall edge.")
			subdf[tpoint] = subdf[tpoint] + shift
			if USE_EMG_MIX and 't_emg_hwhm' in subdf.columns:
				subdf['t_emg_hwhm'] = subdf['t_emg_hwhm'] + shift
			# Also shift ignored_subdf for correct plotting (non-stacked mode)
			if 'ignored_subdf' in locals() and ignored_subdf is not None and len(ignored_subdf) > 0:
				ignored_subdf[tpoint] = ignored_subdf[tpoint] + shift
				if USE_EMG_MIX and 't_emg_hwhm' in ignored_subdf.columns:
					ignored_subdf['t_emg_hwhm'] = ignored_subdf['t_emg_hwhm'] + shift
			
			# === SHIFT STACKED TIMES TO MATCH SUBDF ===
			# Apply the same shift to stacked_df so arrival times align.
			# The shift global is stored unconditionally (see the Gaussian
			# twin below for the rationale): an all-rejected component
			# must still record its subfall geometry so the click-popup
			# places the stacked fit correctly.
			if USE_EMG_MIX and use_stacking_this_comp:
				_global_subfall_shift = shift
			if USE_EMG_MIX and use_stacking_this_comp and 'stacked_df' in locals() and stacked_df is not None and len(stacked_df) > 0:
				if 't_emg_hwhm' in stacked_df.columns:
					stacked_df['t_emg_hwhm'] = stacked_df['t_emg_hwhm'] + shift
				if 'xo' in stacked_df.columns:
					stacked_df['xo'] = stacked_df['xo'] + shift
			# Also shift ignored_stacked_df for correct plotting
			if USE_EMG_MIX and use_stacking_this_comp and 'ignored_stacked_df' in locals() and ignored_stacked_df is not None and len(ignored_stacked_df) > 0:
				if 't_emg_hwhm' in ignored_stacked_df.columns:
					ignored_stacked_df['t_emg_hwhm'] = ignored_stacked_df['t_emg_hwhm'] + shift
				if 'xo' in ignored_stacked_df.columns:
					ignored_stacked_df['xo'] = ignored_stacked_df['xo'] + shift
			# === END STACKED SHIFT ===
			
			# === SHIFT GAUSSIAN STACKED TIMES TO MATCH SUBDF ===
			# The shift global is pure subfall geometry, so it must be
			# stored whenever Gaussian stacking ran — even if every stack
			# was later filtered out (empty stacked_gauss_df). Previously
			# this assignment lived inside the `len(stacked_gauss_df) > 0`
			# gate below, so an all-rejected component left
			# _global_subfall_gauss_shift unset; the click-popup then fell
			# back to subfall_shift=0 and drew the stacked fit shifted by
			# the full subfall-left-edge offset (the spurious −70 ms).
			if USE_GAUSS_MIX and use_stacking_this_comp:
				_global_subfall_gauss_shift = shift
			if USE_GAUSS_MIX and use_stacking_this_comp and 'stacked_gauss_df' in locals() and stacked_gauss_df is not None and len(stacked_gauss_df) > 0:
				if 'tstart' in stacked_gauss_df.columns:
					stacked_gauss_df['tstart'] = stacked_gauss_df['tstart'] + shift
				if 'tend' in stacked_gauss_df.columns:
					stacked_gauss_df['tend'] = stacked_gauss_df['tend'] + shift
				if 'xo' in stacked_gauss_df.columns:
					stacked_gauss_df['xo'] = stacked_gauss_df['xo'] + shift
			# Also shift ignored_stacked_gauss_df for correct plotting
			if USE_GAUSS_MIX and use_stacking_this_comp and 'ignored_stacked_gauss_df' in locals() and ignored_stacked_gauss_df is not None and len(ignored_stacked_gauss_df) > 0:
				if 'tstart' in ignored_stacked_gauss_df.columns:
					ignored_stacked_gauss_df['tstart'] = ignored_stacked_gauss_df['tstart'] + shift
				if 'tend' in ignored_stacked_gauss_df.columns:
					ignored_stacked_gauss_df['tend'] = ignored_stacked_gauss_df['tend'] + shift
				if 'xo' in ignored_stacked_gauss_df.columns:
					ignored_stacked_gauss_df['xo'] = ignored_stacked_gauss_df['xo'] + shift
			# === END GAUSSIAN STACKED SHIFT ===

#         # transform times to full waterfall times
# 		if len(cuts) == 0:
# 			subdf[tpoint] = subdf[tpoint] + (xosi - 4*sigma)
# 			if USE_EMG_MIX and 't_emg_hwhm' in subdf.columns:
# 				subdf['t_emg_hwhm'] = subdf['t_emg_hwhm'] + (xosi - 4*sigma)
		
		elif len(cuts) > 0:
			# For EMG, use xcs (mu + 1/lambda) for comparison since subfall was centered around xcs
			xc_compare = xcs[xos.index(xosi)] if USE_EMG_MIX else xosi
			
			if xc_compare < cuts[0]:  # left edge
				pass  # times are already good
				# Store shift globally for click visualization (shift is 0 for left edge)
				if USE_EMG_MIX and use_stacking_this_comp:
					_global_subfall_shift = 0
				# Store shift for Gaussian stacking (shift is 0 for left edge)
				if USE_GAUSS_MIX and use_stacking_this_comp:
					_global_subfall_gauss_shift = 0
		
			elif xc_compare > cuts[-1]:  # right edge
				subdf[tpoint] = subdf[tpoint] + cuts[-1]
				if USE_EMG_MIX and 't_emg_hwhm' in subdf.columns:
					subdf['t_emg_hwhm'] = subdf['t_emg_hwhm'] + cuts[-1]
				# Also shift ignored_subdf for correct plotting (non-stacked mode)
				if 'ignored_subdf' in locals() and ignored_subdf is not None and len(ignored_subdf) > 0:
					ignored_subdf[tpoint] = ignored_subdf[tpoint] + cuts[-1]
					if USE_EMG_MIX and 't_emg_hwhm' in ignored_subdf.columns:
						ignored_subdf['t_emg_hwhm'] = ignored_subdf['t_emg_hwhm'] + cuts[-1]
				
				# === SHIFT STACKED TIMES TO MATCH SUBDF ===
				if USE_EMG_MIX and use_stacking_this_comp:
					_global_subfall_shift = cuts[-1]
				if USE_EMG_MIX and use_stacking_this_comp and 'stacked_df' in locals() and stacked_df is not None and len(stacked_df) > 0:
					if 't_emg_hwhm' in stacked_df.columns:
						stacked_df['t_emg_hwhm'] = stacked_df['t_emg_hwhm'] + cuts[-1]
					if 'xo' in stacked_df.columns:
						stacked_df['xo'] = stacked_df['xo'] + cuts[-1]
				# Also shift ignored_stacked_df for correct plotting
				if USE_EMG_MIX and use_stacking_this_comp and 'ignored_stacked_df' in locals() and ignored_stacked_df is not None and len(ignored_stacked_df) > 0:
					if 't_emg_hwhm' in ignored_stacked_df.columns:
						ignored_stacked_df['t_emg_hwhm'] = ignored_stacked_df['t_emg_hwhm'] + cuts[-1]
					if 'xo' in ignored_stacked_df.columns:
						ignored_stacked_df['xo'] = ignored_stacked_df['xo'] + cuts[-1]
				# === END STACKED SHIFT ===
				
				# === SHIFT GAUSSIAN STACKED TIMES FOR RIGHT EDGE ===
				if USE_GAUSS_MIX and use_stacking_this_comp:
					_global_subfall_gauss_shift = cuts[-1]
				if USE_GAUSS_MIX and use_stacking_this_comp and 'stacked_gauss_df' in locals() and stacked_gauss_df is not None and len(stacked_gauss_df) > 0:
					if 'tstart' in stacked_gauss_df.columns:
						stacked_gauss_df['tstart'] = stacked_gauss_df['tstart'] + cuts[-1]
					if 'tend' in stacked_gauss_df.columns:
						stacked_gauss_df['tend'] = stacked_gauss_df['tend'] + cuts[-1]
					if 'xo' in stacked_gauss_df.columns:
						stacked_gauss_df['xo'] = stacked_gauss_df['xo'] + cuts[-1]
				# Also shift ignored_stacked_gauss_df for correct plotting
				if USE_GAUSS_MIX and use_stacking_this_comp and 'ignored_stacked_gauss_df' in locals() and ignored_stacked_gauss_df is not None and len(ignored_stacked_gauss_df) > 0:
					if 'tstart' in ignored_stacked_gauss_df.columns:
						ignored_stacked_gauss_df['tstart'] = ignored_stacked_gauss_df['tstart'] + cuts[-1]
					if 'tend' in ignored_stacked_gauss_df.columns:
						ignored_stacked_gauss_df['tend'] = ignored_stacked_gauss_df['tend'] + cuts[-1]
					if 'xo' in ignored_stacked_gauss_df.columns:
						ignored_stacked_gauss_df['xo'] = ignored_stacked_gauss_df['xo'] + cuts[-1]
				# === END GAUSSIAN STACKED SHIFT ===
		
			else:  # middle
				ci = 0
				while xc_compare > cuts[ci]:
					ci += 1
				prev_ci = ci - 1
				subdf[tpoint] = subdf[tpoint] + cuts[prev_ci]
				if USE_EMG_MIX and 't_emg_hwhm' in subdf.columns:
					subdf['t_emg_hwhm'] = subdf['t_emg_hwhm'] + cuts[prev_ci]
				# Also shift ignored_subdf for correct plotting (non-stacked mode)
				if 'ignored_subdf' in locals() and ignored_subdf is not None and len(ignored_subdf) > 0:
					ignored_subdf[tpoint] = ignored_subdf[tpoint] + cuts[prev_ci]
					if USE_EMG_MIX and 't_emg_hwhm' in ignored_subdf.columns:
						ignored_subdf['t_emg_hwhm'] = ignored_subdf['t_emg_hwhm'] + cuts[prev_ci]
				
				# === SHIFT STACKED TIMES TO MATCH SUBDF ===
				if USE_EMG_MIX and use_stacking_this_comp:
					_global_subfall_shift = cuts[prev_ci]
				if USE_EMG_MIX and use_stacking_this_comp and 'stacked_df' in locals() and stacked_df is not None and len(stacked_df) > 0:
					if 't_emg_hwhm' in stacked_df.columns:
						stacked_df['t_emg_hwhm'] = stacked_df['t_emg_hwhm'] + cuts[prev_ci]
					if 'xo' in stacked_df.columns:
						stacked_df['xo'] = stacked_df['xo'] + cuts[prev_ci]
				# Also shift ignored_stacked_df for correct plotting
				if USE_EMG_MIX and use_stacking_this_comp and 'ignored_stacked_df' in locals() and ignored_stacked_df is not None and len(ignored_stacked_df) > 0:
					if 't_emg_hwhm' in ignored_stacked_df.columns:
						ignored_stacked_df['t_emg_hwhm'] = ignored_stacked_df['t_emg_hwhm'] + cuts[prev_ci]
					if 'xo' in ignored_stacked_df.columns:
						ignored_stacked_df['xo'] = ignored_stacked_df['xo'] + cuts[prev_ci]
				# === END STACKED SHIFT ===
				
				# === SHIFT GAUSSIAN STACKED TIMES FOR MIDDLE ===
				if USE_GAUSS_MIX and use_stacking_this_comp:
					_global_subfall_gauss_shift = cuts[prev_ci]
				if USE_GAUSS_MIX and use_stacking_this_comp and 'stacked_gauss_df' in locals() and stacked_gauss_df is not None and len(stacked_gauss_df) > 0:
					if 'tstart' in stacked_gauss_df.columns:
						stacked_gauss_df['tstart'] = stacked_gauss_df['tstart'] + cuts[prev_ci]
					if 'tend' in stacked_gauss_df.columns:
						stacked_gauss_df['tend'] = stacked_gauss_df['tend'] + cuts[prev_ci]
					if 'xo' in stacked_gauss_df.columns:
						stacked_gauss_df['xo'] = stacked_gauss_df['xo'] + cuts[prev_ci]
					_global_subfall_gauss_shift = cuts[prev_ci]
				# Also shift ignored_stacked_gauss_df for correct plotting
				if USE_GAUSS_MIX and use_stacking_this_comp and 'ignored_stacked_gauss_df' in locals() and ignored_stacked_gauss_df is not None and len(ignored_stacked_gauss_df) > 0:
					if 'tstart' in ignored_stacked_gauss_df.columns:
						ignored_stacked_gauss_df['tstart'] = ignored_stacked_gauss_df['tstart'] + cuts[prev_ci]
					if 'tend' in ignored_stacked_gauss_df.columns:
						ignored_stacked_gauss_df['tend'] = ignored_stacked_gauss_df['tend'] + cuts[prev_ci]
					if 'xo' in ignored_stacked_gauss_df.columns:
						ignored_stacked_gauss_df['xo'] = ignored_stacked_gauss_df['xo'] + cuts[prev_ci]
				# === END GAUSSIAN STACKED SHIFT ===



		subdf['color'] = next(colors) # assign color to points

		dtdnus.append((dtdnu, dtdnu_err, dtdnu_redchisq))
		intercepts.append((t_b, tb_err))
		subbandpopts.append(subband_popt)
		subbandperrs.append(subband_perr)
		subbandredchisqs.append(subband_redchisq)
		subdfs.append(subdf)
		# === PER-COMPONENT STACKED-DF ACCUMULATION ===
		# Capture this iteration's stacked fit results (which would otherwise
		# be overwritten when the loop moves to the next component).
		if use_stacking_this_comp:
			if USE_EMG_MIX and 'stacked_df' in locals() and stacked_df is not None:
				stacked_dfs_per_comp[comp_idx_loop] = stacked_df.copy()
			if USE_GAUSS_MIX and 'stacked_gauss_df' in locals() and stacked_gauss_df is not None:
				stacked_gauss_dfs_per_comp[comp_idx_loop] = stacked_gauss_df.copy()
		# === PER-COMPONENT IGNORED-DF ACCUMULATION ===
		# Same problem as for the stacked dfs: the loop-local
		# `ignored_*` names get overwritten on every iteration. Save
		# this iteration's values into per-comp arrays so the slope
		# panel and any other post-loop consumer can look them up by
		# component index instead of seeing only the last iteration's
		# values (which would, e.g., show component b's ignored stacked
		# X markers on component a's slope panel).
		if 'ignored_stacked_df' in locals() and ignored_stacked_df is not None:
			ignored_stacked_dfs_per_comp[comp_idx_loop] = ignored_stacked_df.copy()
		if 'ignored_stacked_gauss_df' in locals() and ignored_stacked_gauss_df is not None:
			ignored_stacked_gauss_dfs_per_comp[comp_idx_loop] = ignored_stacked_gauss_df.copy()
		if 'ignored_subdf' in locals() and ignored_subdf is not None:
			ignored_subdfs_per_comp[comp_idx_loop] = ignored_subdf.copy()
		# Also capture the global per-component state (profiles, fitdata,
		# center indices, subfall geometry) so the popup machinery can
		# resolve per-channel clicks back to the right component's stacked
		# profile, even when components have different stack_size values
		# or only some are stacked.
		if use_stacking_this_comp and USE_EMG_MIX and '_global_stacked_profiles' in globals() and _global_stacked_profiles is not None:
			# Finalize the per-stack filter audit: mark which stacks survived
			# all filters (final stacked_df index) so the click-popup can show
			# each stack's full pass/fail ladder.
			_emg_audit_obj = None
			if '_emg_audit' in dir() and '_emg_audit_all_idx' in dir():
				try:
					_retained_emg = (list(stacked_df.index)
					                 if ('stacked_df' in dir() and stacked_df is not None)
					                 else [])
					_emg_audit.finalize(_retained_emg, _emg_audit_all_idx)
					_emg_audit_obj = _emg_audit
				except Exception as _ae:
					print(f"  Warning: EMG filter audit finalize failed: {_ae}")
			_stacked_state_per_comp[comp_idx_loop] = {
				'profiles':         _global_stacked_profiles.copy(),
				'fitdata':          _global_stacked_fitdata.copy() if _global_stacked_fitdata is not None else None,
				'centers':          _global_stack_center_indices.copy() if _global_stack_center_indices is not None else None,
				'df':               _global_stacked_df.copy() if _global_stacked_df is not None else None,
				'stack_size':       stack_size_this_comp,
				'subfall_duration': _global_subfall_duration,
				'subfall_shift':    _global_subfall_shift if ('_global_subfall_shift' in globals() and _global_subfall_shift is not None) else 0,
				'wfall':            _global_wfall.copy() if ('_global_wfall' in globals() and _global_wfall is not None) else None,
				'adaptive_window_params': dict(_global_adaptive_window_params) if ('_global_adaptive_window_params' in globals() and _global_adaptive_window_params is not None) else None,
				'filter_audit':     _emg_audit_obj,
			}
		if use_stacking_this_comp and USE_GAUSS_MIX and '_global_stacked_gauss_profiles' in globals() and _global_stacked_gauss_profiles is not None:
			_gauss_audit_obj = None
			if '_gauss_audit' in dir() and '_gauss_audit_all_idx' in dir():
				try:
					_retained_g = (list(stacked_gauss_df.index)
					               if ('stacked_gauss_df' in dir() and stacked_gauss_df is not None)
					               else [])
					_gauss_audit.finalize(_retained_g, _gauss_audit_all_idx)
					_gauss_audit_obj = _gauss_audit
				except Exception as _ae:
					print(f"  Warning: Gaussian filter audit finalize failed: {_ae}")
			_stacked_gauss_state_per_comp[comp_idx_loop] = {
				'profiles':         _global_stacked_gauss_profiles.copy(),
				'fitdata':          _global_stacked_gauss_fitdata.copy() if _global_stacked_gauss_fitdata is not None else None,
				'centers':          _global_stack_gauss_center_indices.copy() if _global_stack_gauss_center_indices is not None else None,
				'df':               _global_stacked_gauss_df.copy() if _global_stacked_gauss_df is not None else None,
				'stack_size':       stack_size_this_comp,
				'subfall_duration': _global_subfall_gauss_duration if ('_global_subfall_gauss_duration' in globals()) else None,
				'subfall_shift':    _global_subfall_gauss_shift if ('_global_subfall_gauss_shift' in globals() and _global_subfall_gauss_shift is not None) else 0,
				'wfall':            _global_wfall_gauss.copy() if ('_global_wfall_gauss' in globals() and _global_wfall_gauss is not None) else None,
				'filter_audit':     _gauss_audit_obj,
			}
		# === END PER-COMPONENT STACKED-DF ACCUMULATION ===
		# print(f"{dtdnu = } +/- {dtdnu_err = }")
		# print(f"{rowname} number of arrival times: {len(subdf) = }")
		print(f"{bwidth = :.3f} +/- {bwidth_err:.3f} MHz")

		# Compute tau_sc (scattering timescale) for EMG mode
		if USE_EMG_MIX:
			comp_idx = xos.index(xosi)
			lam_slice = tmix_popt[3*n_bursts:4*n_bursts]
			lam_err_slice = tmix_perr[3*n_bursts:4*n_bursts] if isinstance(tmix_perr, (list, np.ndarray)) else []
			lam = np.atleast_1d(lam_slice)[comp_idx] if comp_idx < len(np.atleast_1d(lam_slice)) else np.nan
			if np.isfinite(lam) and lam > 0:
				tau_sc = 1.0 / lam
				tau_sc_err = (lam_err_slice[comp_idx] / (lam**2)) if (comp_idx < len(lam_err_slice) and np.isfinite(lam_err_slice[comp_idx])) else 0.0
			else:
				# lam was NaN'd by the boundary-pile-up detector (or never
				# fit). Write NaN to the CSV so downstream filtering can
				# distinguish "undetermined" from a real small value.
				tau_sc = np.nan
				tau_sc_err = np.nan
		else:
			tau_sc = 0.0
			tau_sc_err = 0.0

		# Determine the number of arrival times used (stacked vs individual)
		if use_stacking_this_comp:
			if USE_EMG_MIX and 'stacked_df' in locals() and stacked_df is not None:
				num_arrtimes = len(stacked_df)
			elif USE_GAUSS_MIX and 'stacked_gauss_df' in locals() and stacked_gauss_df is not None:
				num_arrtimes = len(stacked_gauss_df)
			else:
				num_arrtimes = len(subdf)
		else:
			num_arrtimes = len(subdf)

		# === DIAGNOSTIC: ERROR BAR COMPARISON for this component ===
		# (Commented out - moved to summary at end)
		# print(f"\n{'$'*80}")
		# print(f"$ ERROR BAR COMPARISON for component {rowname}")
		# print(f"{'$'*80}")
		# print(f"\n  HORIZONTAL ERROR BAR (t0_err = arrival time uncertainty):")
		# print(f"    Source: EMG/Gaussian mixture fit to integrated time profile")
		# print(f"    Value: xosi_err = {xosi_err:.6f} ms")
		# print(f"    This is sqrt(pcov[mu_index, mu_index]) from the mixture fit")
		# 
		# print(f"\n  VERTICAL ERROR BAR (center_f_err = center frequency uncertainty):")
		# print(f"    Source: Gaussian fit to frequency-integrated spectrum")
		# print(f"    Value: pkfreq_err = {pkfreq_err:.6f} MHz")
		# print(f"    This is sqrt(pcov[1,1]) from the spectral fit")
		# 
		# print(f"\n  COMPARISON:")
		# # Relative uncertainties
		# rel_t0_err = 100 * xosi_err / abs(xosi) if xosi != 0 else np.inf
		# rel_freq_err = 100 * pkfreq_err / abs(pkfreq) if pkfreq != 0 else np.inf
		# print(f"    t0 = {xosi:.4f} ± {xosi_err:.4f} ms (relative: {rel_t0_err:.1f}%)")
		# print(f"    center_f = {pkfreq:.2f} ± {pkfreq_err:.2f} MHz (relative: {rel_freq_err:.1f}%)")
		# 
		# # Ratio of uncertainties
		# if pkfreq_err > 0:
		# 	err_ratio = xosi_err / pkfreq_err
		# 	print(f"\n    Ratio of uncertainties (t0_err / center_f_err): {err_ratio:.2f}")
		# 	if err_ratio > 10:
		# 		print(f"    -> LARGE HORIZONTAL ERROR BAR: t0_err is {err_ratio:.1f}x larger than center_f_err")
		# 		print(f"    -> Likely causes:")
		# 		print(f"       1. EMG parameter degeneracy (mu-sigma-lambda correlation)")
		# 		print(f"       2. Low SNR in time profile")
		# 		print(f"       3. Overlapping components making mu poorly constrained")
		# 		print(f"       4. Ill-conditioned covariance matrix from curve_fit")
		# 	elif err_ratio < 0.1:
		# 		print(f"    -> LARGE VERTICAL ERROR BAR: center_f_err is {1/err_ratio:.1f}x larger than t0_err")
		# 
		# # Physical scale comparison
		# duration_scale = res_time_ms * wfall.shape[1]  # total time duration
		# bandwidth_scale = res_freq * wfall.shape[0]    # total bandwidth
		# print(f"\n    Physical scales:")
		# print(f"      t0_err as fraction of total duration: {100*xosi_err/duration_scale:.2f}%")
		# print(f"      center_f_err as fraction of bandwidth: {100*pkfreq_err/bandwidth_scale:.2f}%")
		# 
		# print(f"{'$'*80}\n")
		# === END ERROR BAR COMPARISON ===

		# === SCATTERING REGIME CLASSIFICATION (full-band EMG only) ===
		if USE_EMG_MIX and np.isfinite(tau_sc) and tau_sc > 0:
			# sigma_core for this component from the mixture fit
			comp_idx_regime = xos.index(xosi) if xosi in xos else 0
			sig_regime = float(np.atleast_1d(tmix_sigmas)[comp_idx_regime]) if comp_idx_regime < len(np.atleast_1d(tmix_sigmas)) else np.nan
			# Per-component (σ, λ) errors + their off-diagonal covariance,
			# for analytic propagation through emg_fwhm. Layout of
			# tmix_popt/tmix_pcov: [A_1..A_n, mu_1..mu_n, σ_1..σ_n, λ_1..λ_n, C]
			# so σ_i is at index (2*n_bursts + i) and λ_i at (3*n_bursts + i).
			_sig_err_regime = (
				float(np.atleast_1d(tmix_sigma_errs)[comp_idx_regime])
				if comp_idx_regime < len(np.atleast_1d(tmix_sigma_errs))
				else np.nan
			)
			_cov_sig_lam = 0.0
			try:
				if (isinstance(tmix_pcov, np.ndarray)
				        and tmix_pcov.ndim == 2
				        and tmix_pcov.shape[0] >= 4*n_bursts + 1):
					_i_sig = 2*n_bursts + comp_idx_regime
					_i_lam = 3*n_bursts + comp_idx_regime
					_cov_sig_lam = float(tmix_pcov[_i_sig, _i_lam])
			except Exception:
				_cov_sig_lam = 0.0
			regime_info = classify_scattering_regime(
				sig_regime, tau_sc,
				sigma_err=_sig_err_regime,
				tau_err=tau_sc_err if np.isfinite(tau_sc_err) else None,
				cov_sigma_lamb=_cov_sig_lam,
			)
			
			# Telescope resolution floor check
			_telescope = identify_telescope(bname)
			if _telescope is None and 'telescope' in data:
				_telescope = identify_telescope(str(data['telescope']))
			_native_res = get_native_resolution(_telescope)
			if _native_res is not None and np.isfinite(sig_regime) and np.isfinite(tau_sc):
				below_res = (sig_regime < _native_res) or (tau_sc < _native_res)
			else:
				below_res = False
			
			if _telescope is not None:
				print(f"  Scattering regime ({rowname}): {regime_info['regime']} "
					  f"(tau/sigma = {regime_info['tau_over_sigma']:.3f}), "
					  f"telescope = {_telescope} (native res = {_native_res:.4f} ms)"
					  f"{' *** BELOW NATIVE RES ***' if below_res else ''}")
			else:
				print(f"  Scattering regime ({rowname}): {regime_info['regime']} "
					  f"(tau/sigma = {regime_info['tau_over_sigma']:.3f}), "
					  f"telescope = unknown")
		else:
			regime_info = classify_scattering_regime(np.nan, np.nan)
			below_res = False
		# === END SCATTERING REGIME ===

		# === SUB-BAND SCATTERING-INDEX FIT + OPTIONAL DIAGNOSTIC PLOT (per component) ===
		# Run the sub-band α fit for every component when either
		# USE_SUBBAND_ALPHA_FIT or this component's subband_plot flag is
		# enabled, gated to EMG-mix mode. Per-component values for
		# n_subbands / constrain_index / bw_width_factor are pulled from
		# the lists built at the top of the function (see
		# "PER-COMPONENT SUB-BAND-FIT NORMALIZATION").
		_comp_sb_result = None
		_comp_subband_plot_flag = bool(_subband_plot_per_comp[comp_idx_loop]) if comp_idx_loop < len(_subband_plot_per_comp) else False
		_comp_n_subbands       = int(_n_subbands_per_comp[comp_idx_loop]) if comp_idx_loop < len(_n_subbands_per_comp) else _n_subbands
		_comp_constrain_index  = bool(_constrain_index_per_comp[comp_idx_loop]) if comp_idx_loop < len(_constrain_index_per_comp) else False
		_comp_bw_width_factor  = float(_bw_width_factor_per_comp[comp_idx_loop]) if comp_idx_loop < len(_bw_width_factor_per_comp) else 2.0
		_comp_ignore_subindex  = _ignore_subindex_per_comp[comp_idx_loop] if comp_idx_loop < len(_ignore_subindex_per_comp) else None
		_comp_constrain_sigma  = _constrain_sigma_per_comp[comp_idx_loop] if comp_idx_loop < len(_constrain_sigma_per_comp) else None

		if (USE_SUBBAND_ALPHA_FIT or _comp_subband_plot_flag) and USE_EMG_MIX:
			# Sanity check: subfall and freqs must agree on the frequency axis.
			# If they don't, channel_override indices (and the internal sub-band
			# splitting) refer to different things than the caller expects.
			if subfall.shape[0] != len(freqs):
				print(f"  Warning: subfall.shape[0]={subfall.shape[0]} != len(freqs)={len(freqs)} "
				      f"for component {rowname}. Sub-band channel indices may be misaligned. "
				      f"This usually means a frequency-axis slice was applied to subfall — "
				      f"please report so the per-component sub-band call can be fixed.")
			try:
				# === v56_10: FOUR-VARIATION SUB-BAND FITTING ====================
				# Always compute all four (index × σ) sub-band fits regardless
				# of the per-component constrain_index / constrain_sigma kwargs
				# (those now only choose which variation is flagged PRIMARY in
				# the terminal print). fit_subband_alpha returns, in ONE call,
				# both its σ-free pass ('alpha'…) and its σ-fixed pass
				# ('csig_alpha'…), each respecting constrain_alpha. So two calls
				# — constrain_alpha False then True — yield the full 2×2 grid:
				#   callA(False): alpha=(idx_free, σ_free)  csig_alpha=(idx_free, σ_fix)
				#   callB(True):  alpha=(idx_fix,  σ_free)  csig_alpha=(idx_fix,  σ_fix)
				# Per-sub-band POINTS depend only on σ (not on constrain_alpha),
				# so they are identical across the two calls; we take them from
				# callA. Cost: the per-sub-band EMG fits run twice (cheap); the
				# slow per-channel ToA step is untouched.
				#
				# σ-fixed value: explicit config value if given, else fall back
				# to the full-band σ so the σ-fixed pass ALWAYS runs.
				_sigfix_resolved = _comp_constrain_sigma
				if _sigfix_resolved is None or not (np.isfinite(_sigfix_resolved) and _sigfix_resolved > 0):
					_sigfix_resolved = sigma if (np.isfinite(sigma) and sigma > 0) else None
				_sb_common_kwargs = dict(
					n_sub=_comp_n_subbands,
					fb_sigma=sigma if np.isfinite(sigma) else None,
					fb_tau=tau_sc if (np.isfinite(tau_sc) and tau_sc > 0) else None,
					bname=rowname,
					bw_width_factor=_comp_bw_width_factor,
					channel_override=_bw_co_this,
					spec_center_outer=pkfreq if np.isfinite(pkfreq) else None,
					spec_sigma_outer=bwidth if (np.isfinite(bwidth) and bwidth > 0) else None,
					ignore_subindex=_comp_ignore_subindex,
					sigma_fixed=_sigfix_resolved,
				)
				_sb_callA = fit_subband_alpha(
					subfall, freqs, subfall.shape[1] * res_time_ms, res_time_ms,
					constrain_alpha=False, **_sb_common_kwargs)   # index-free
				_sb_callB = fit_subband_alpha(
					subfall, freqs, subfall.shape[1] * res_time_ms, res_time_ms,
					constrain_alpha=True, **_sb_common_kwargs)    # index-fixed (~-4)
				# Primary structure/points come from the index-free call; stash
				# the index-fixed call so the CSV row and second pop-up can reach it.
				_comp_sb_result = _sb_callA
				if _comp_sb_result is not None:
					_comp_sb_result['_callB_idxfixed'] = _sb_callB
				# --- legacy single-call form (pre-v56_10), kept for reference ---
				# _comp_sb_result = fit_subband_alpha(
				#     subfall, freqs, subfall.shape[1]*res_time_ms, res_time_ms,
				#     constrain_alpha=_comp_constrain_index,
				#     sigma_fixed=_comp_constrain_sigma, **<other kwargs>)
				# === END FOUR-VARIATION SUB-BAND FITTING ========================
				# Stash the exact (cropped + bg-subtracted) subfall and its
				# left-channel offset in wfall, so the interactive 'q'-key
				# sub-band popup can reconstruct the time series the fit
				# actually saw and align the fit's μ correctly on the
				# waterfall axis. (popt[1] from fit_subband_alpha is in
				# subfall-local time coordinates, not wfall coordinates.)
				if _comp_sb_result is not None:
					_comp_sb_result['_subfall_for_popup'] = subfall
					_comp_sb_result['_subfall_left_chan'] = int(subfall_left_chan)
					_comp_sb_result['_subfall_duration_ms'] = float(subfall.shape[1] * res_time_ms)
					# Also stash the per-component constrain_index used for the
					# fit; the popup needs this to label the fit correctly.
					_comp_sb_result['_constrain_index_used'] = _comp_constrain_index
				if _comp_subband_plot_flag and _comp_sb_result and _comp_sb_result['n_good_subbands'] >= 2:
					# v56_10: two separate 2×3 pop-ups instead of one crammed 4×3.
					# Split on the INDEX dimension — each pop-up keeps the existing
					# layout (σ-free top row, σ-fixed bottom row) and differs only
					# in the τ(ν) power-law fit line drawn through the points:
					#   pop-up 1 → index-free  fit (_sb_callA)
					#   pop-up 2 → index-fixed fit (_sb_callB, α pinned to ~-4)
					# The left/right columns (profiles, σ(ν), degeneracy) are
					# identical between the two windows by construction.
					for _ci_flag, _sb_res_plot, _nsfx in (
							(False, _sb_callA, 'idxfree'),
							(True,  _sb_callB, 'idxfixed')):
						if not (_sb_res_plot and _sb_res_plot.get('n_good_subbands', 0) >= 2):
							continue
						make_subband_diagnostic_plots(
							rowname, subfall, freqs,
							subfall.shape[1] * res_time_ms,
							_sb_res_plot,
							constrain_index=_ci_flag,
							fb_tau=tau_sc if (np.isfinite(tau_sc) and tau_sc > 0) else None,
							fb_sigma=sigma if np.isfinite(sigma) else None,
							res_time_ms=res_time_ms,
							save=save,
							outdir=outdir,
							show=show,
							outfmt=outfmt,
							comp_label=comp_idx_loop,
							model_tag=model,
							name_suffix=_nsfx,
						)
			except Exception as e:
				print(f"  Warning: Sub-band fit/plot failed for {rowname}: {e}")
				_comp_sb_result = None
		# Append even on failure so positions align with components.
		_comp_sb_results.append(_comp_sb_result)
		_comp_acf_results.append(_comp_acf_result)
		# === END SUB-BAND SCATTERING-INDEX FIT ===

		# === v56_10: encode the four sub-band variations for the CSV row ======
		# TWO list AXES are used and they are NOT interchangeable:
		#   • α / τ_ref columns  → 2-list over the INDEX axis:
		#         [0] index-free (constrain_alpha=False)
		#         [1] index-fixed (constrain_alpha=True, α≈-4)
		#     The plain columns (sb_alpha …) are the σ-FREE family; the
		#     *_sigma_constrained columns are the σ-FIXED family.
		#   • per-sub-band POINT columns (sb_tau_per_sb, sb_nu_ratio_kept)
		#     → 2-list over the σ axis: [0]=σ-free points, [1]=σ-fixed points.
		#   • sb_nu_per_sb stays a single list (centres are σ-independent).
		# Pairing for the τ(ν) toggle: σ-free fit (sb_alpha[0|1]) ↔ point-set[0];
		# σ-fixed fit (sb_alpha_sigma_constrained[0|1]) ↔ point-set[1].
		# Non-finite entries render as `nan`, so parse with a NaN-aware
		# evaluator, not bare ast.literal_eval.
		_sbA = _comp_sb_result                                      # index-free (primary)
		_sbB = _comp_sb_result.get('_callB_idxfixed') if _comp_sb_result else None  # index-fixed
		_have_sb = _comp_sb_result is not None
		_have_AB = _have_sb and (_sbB is not None)
		_fl2 = lambda a, b: f"[{a:.6g}, {b:.6g}]"
		def _flist(arr, nd=4):
			if arr is None or len(arr) == 0:
				return "[]"
			return "[" + ", ".join(f"{_x:.{nd}f}" for _x in arr) + "]"
		def _nu_ratio(_sbres, _maskkey, _freqkey):
			if not _sbres:
				return np.nan
			_f = _sbres.get(_freqkey); _m = _sbres.get(_maskkey)
			if _f is None or _m is None:
				return np.nan
			_f = np.asarray(_f, dtype=float); _m = np.asarray(_m, dtype=bool)
			if len(_f) < 2 or _m.shape != _f.shape or int(np.sum(~_m)) < 2:
				return np.nan
			return float(np.max(_f[~_m]) / np.min(_f[~_m]))
		# INDEX-axis 2-lists — σ-free family
		_sb_alpha_2       = _fl2(_sbA.get('alpha', np.nan),       _sbB.get('alpha', np.nan))       if _have_AB else np.nan
		_sb_alpha_err_2   = _fl2(_sbA.get('alpha_err', np.nan),   _sbB.get('alpha_err', np.nan))   if _have_AB else np.nan
		_sb_tau_ref_2     = _fl2(_sbA.get('tau_ref', np.nan),     _sbB.get('tau_ref', np.nan))     if _have_AB else np.nan
		_sb_tau_ref_err_2 = _fl2(_sbA.get('tau_ref_err', np.nan), _sbB.get('tau_ref_err', np.nan)) if _have_AB else np.nan
		# INDEX-axis 2-lists — σ-fixed family
		_sb_csig_alpha_2     = _fl2(_sbA.get('csig_alpha', np.nan),     _sbB.get('csig_alpha', np.nan))     if _have_AB else np.nan
		_sb_csig_alpha_err_2 = _fl2(_sbA.get('csig_alpha_err', np.nan), _sbB.get('csig_alpha_err', np.nan)) if _have_AB else np.nan
		_sb_csig_tau_ref_2   = _fl2(_sbA.get('csig_tau_ref', np.nan),   _sbB.get('csig_tau_ref', np.nan))   if _have_AB else np.nan
		# σ-axis 2-list of POINT lists: [σ-free, σ-fixed]
		_sb_tau_per_sb_2 = ("[" + _flist(_sbA.get('sub_taus'), 4) + ", "
		                    + _flist(_sbA.get('csig_sub_taus'), 4) + "]") if _have_sb else ""
		# σ-axis 2-list of per-point τ ERRORS [σ-free, σ-fixed], parallel to
		# _sb_tau_per_sb_2 — needed for weighted τ(ν) fits and error bars.
		_sb_tau_err_per_sb_2 = ("[" + _flist(_sbA.get('sub_tau_errs'), 4) + ", "
		                        + _flist(_sbA.get('csig_sub_tau_errs'), 4) + "]") if _have_sb else ""
		# σ-axis 2-list of ν-leverage ratios: [σ-free, σ-fixed]
		_sb_nu_ratio_2 = _fl2(_nu_ratio(_sbA, '_ignore_mask', 'sub_freqs'),
		                      _nu_ratio(_sbA, 'csig_ignore_mask', 'csig_sub_freqs')) if _have_sb else np.nan
		# PRIMARY selection — kwargs choose which of the 4 is flagged/printed first
		_sb_primary_tag = (f"{'idx_fixed' if _comp_constrain_index else 'idx_free'},"
		                   f"{'sig_fixed' if (_comp_constrain_sigma is not None) else 'sig_free'}")
		# Terminal echo: all four, primary first.
		if _have_AB and (_sbA.get('n_good_subbands', 0) >= 2):
			_var_tbl = [
				('idx_free,sig_free',  _sbA.get('alpha', np.nan),      _sbA.get('alpha_err', np.nan)),
				('idx_free,sig_fixed', _sbA.get('csig_alpha', np.nan), _sbA.get('csig_alpha_err', np.nan)),
				('idx_fixed,sig_free', _sbB.get('alpha', np.nan),      _sbB.get('alpha_err', np.nan)),
				('idx_fixed,sig_fixed',_sbB.get('csig_alpha', np.nan), _sbB.get('csig_alpha_err', np.nan)),
			]
			_var_tbl.sort(key=lambda r: 0 if r[0] == _sb_primary_tag else 1)
			print(f"  [{rowname}] sub-band α — 4 variations (primary first):")
			for _vt, _va, _ve in _var_tbl:
				_flag = "  <-- PRIMARY" if _vt == _sb_primary_tag else ""
				print(f"    {_vt:22s} α = {_va:+.3f} ± {_ve:.3f}{_flag}")
		# === END four-variation encoding ======================================

		results.append([ # see `results_columns`
			rowname,	      # 'name',
			targetDM,	      # 'DM',
			xosi,		      # 't0 (ms)',
			xosi_err,	      # 't0_err'
			pkfreq,		      # 'center_f (MHz)',
			pkfreq_err,	      # 'center_f_err',
			sigma,		      # 'duration (ms)',
			sigma_err,	      # 'duration_err',
			bwidth,		      # 'bandwidth (MHz)',
			bwidth_err,	      # 'bandwidth_err',
			dtdnu,		      # 'dtdnu (ms/MHz)',
			dtdnu_err,	      # 'dtdnu_err',
			tmix_redchisq,    # 'redchisq_t',
			subband_redchisq, # 'redchisq_nu',
			dtdnu_redchisq ,  # 'redchisq_dtdnu',
			t_b,		      # 'tb (ms)',
			tb_err,		      # 'tb_err'
			num_arrtimes,     # 'num_arrtimes'
			"EMG" if USE_EMG_MIX else "Gaussian",  # 'model'
			"Yes" if use_stacking_this_comp else "No", # 'signal_stacking'
			tau_sc,           # 'tau_sc (ms)'
			tau_sc_err,       # 'tau_sc_err'
			# --- Scattering regime columns ---
			regime_info['regime'],            # 'scattering_regime'
			regime_info['tau_over_sigma'],     # 'tau_over_sigma'
			regime_info['fwhm_weak'],          # 'fwhm_weak (ms)'
			regime_info['fwhm_intermediate'],  # 'fwhm_intermediate (ms)'
			regime_info['fwhm_strong'],        # 'fwhm_strong (ms)'
			regime_info['fwhm_emg'],           # 'fwhm_emg (ms)'
			regime_info['fwhm_emg_err'],       # 'fwhm_emg_err (ms)'
			regime_info['fwhm_reported'],      # 'fwhm_reported (ms)'
			below_res,                         # 'below_native_res'
			# --- Sub-band scattering index columns (per-component values) ---
			_sb_alpha_2,            # 'sb_alpha' — 2-list [idx_free, idx_fixed] @ σ-free
			_sb_alpha_err_2,        # 'sb_alpha_err' — 2-list [idx_free, idx_fixed] @ σ-free
			_sb_tau_ref_2,          # 'sb_tau_ref (ms)' — 2-list [idx_free, idx_fixed] @ σ-free
			_sb_tau_ref_err_2,      # 'sb_tau_ref_err' — 2-list [idx_free, idx_fixed] @ σ-free
			_comp_sb_result['best_sigma'] if _comp_sb_result else np.nan,       # 'sb_sigma_hi (ms)'
			_comp_sb_result['freq_center'] if _comp_sb_result else np.nan,      # 'sb_freq_center (MHz)'
			_comp_sb_result['n_good_subbands'] if _comp_sb_result else 0,       # 'sb_n_subbands'
			_comp_sb_result['regime_grid']['regime'] if _comp_sb_result else 'unknown',  # 'sb_regime_grid'
			_comp_sb_result['regime_grid']['fwhm_grid'] if _comp_sb_result else np.nan,  # 'sb_fwhm_grid (ms)'
			_comp_sb_result['regime_grid']['fwhm_reported'] if _comp_sb_result else np.nan,  # 'sb_fwhm_reported (ms)'
			_sb_primary_tag,  # 'sb_alpha_restricted' — repurposed: PRIMARY variation tag (e.g. 'idx_free,sig_free')
			"Yes" if _bw_co_this is not None else "No",  # 'bw_override_used'
			"Yes" if _bw_toa_co_this is not None else "No",  # 'bw_ToA_override_used'
			# 'sb_ignored_subindex': comma-separated 1-indexed list of
			# excluded sub-bands (e.g. "6,7"), or "None" if no sub-bands
			# were ignored (clearer than an empty string when scanning
			# the CSV by eye).
			(",".join(str(_x) for _x in _comp_sb_result['ignore_subindex_used'])
			 if _comp_sb_result and _comp_sb_result.get('ignore_subindex_used')
			 else "None"),
			# --- Smoothed σ(ν) power-law fit (ε) per-component values ---
			_comp_sb_result['eps_sigma_amp']      if _comp_sb_result else np.nan,  # 'eps_sigma_amp (ms)'
			_comp_sb_result['eps_sigma_amp_err']  if _comp_sb_result else np.nan,  # 'eps_sigma_amp_err'
			_comp_sb_result['eps_sigma_beta']     if _comp_sb_result else np.nan,  # 'eps_sigma_beta'
			_comp_sb_result['eps_sigma_beta_err'] if _comp_sb_result else np.nan,  # 'eps_sigma_beta_err'
			# --- σ-constrained sub-band diagnostic per-component values ---
			# All NaN / "" when constrain_sigma was None for this component.
			# Keys come from fit_subband_alpha's return dict; see comments
			# there for what each one means. The per-sb list is formatted
			# as a Python-list-ish string so a downstream consumer can do
			# `ast.literal_eval(row['dtau_over_sigma_tau_per_sb'])` and
			# get back a real list of floats. NaN-padded entries (sub-bands
			# that failed the constrained fit but not the free fit, or
			# vice-versa) are simply absent from the list — so the list's
			# length is ≤ n_subbands.
			(_comp_sb_result['sigma_constrained_value_ms']
			 if _comp_sb_result else np.nan),                                              # 'sigma_constrained_value (ms)'
			_sb_csig_alpha_2,                                                              # 'sb_alpha_sigma_constrained' — 2-list [idx_free, idx_fixed] @ σ-fixed
			_sb_csig_alpha_err_2,                                                          # 'sb_alpha_err_sigma_constrained' — 2-list [idx_free, idx_fixed] @ σ-fixed
			_sb_csig_tau_ref_2,                                                            # 'sb_tau_ref_sigma_constrained (ms)' — 2-list [idx_free, idx_fixed] @ σ-fixed
			(_comp_sb_result['csig_dtau_over_sigma_tau_median']
			 if _comp_sb_result else np.nan),                                              # 'dtau_over_sigma_tau_median'
			(_comp_sb_result['csig_dtau_over_sigma_tau_max']
			 if _comp_sb_result else np.nan),                                              # 'dtau_over_sigma_tau_max'
			# The aligned per-sb array may have NaNs for sub-bands where
			# Δτ couldn't be computed (e.g. one of the two passes failed
			# at that sub-band). Stringify those as 'nan' so the column
			# remains literal-eval'able.
			((lambda _arr: ("[" + ", ".join(f"{_x:.3f}" for _x in _arr) + "]"))
			 (_comp_sb_result['csig_dtau_over_sigma_tau_per_sb'])
			 if (_comp_sb_result is not None
			     and _comp_sb_result.get('csig_dtau_over_sigma_tau_per_sb') is not None
			     and len(_comp_sb_result['csig_dtau_over_sigma_tau_per_sb']) > 0)
			 else ""),                                                                     # 'dtau_over_sigma_tau_per_sb'
			# --- Per-sub-band τ(ν) raw values + tight ν ratio ---
			# Stringify the per-sub-band τ list (4-decimal places to
			# preserve enough resolution for downstream monotonicity
			# checks, since CHIME τ values commonly span 0.1–3 ms).
			_sb_tau_per_sb_2,                                                             # 'sb_tau_per_sb' — 2-list [σ-free pts, σ-fixed pts]
			_sb_tau_err_per_sb_2,                                                          # 'sb_tau_err_per_sb' — 2-list [σ-free errs, σ-fixed errs]
			((lambda _arr: ("[" + ", ".join(f"{_x:.2f}" for _x in _arr) + "]"))
			 (_comp_sb_result['sub_freqs'])
			 if (_comp_sb_result is not None
			     and _comp_sb_result.get('sub_freqs') is not None
			     and len(_comp_sb_result['sub_freqs']) > 0)
			 else ""),                                                                     # 'sb_nu_per_sb'
			# Tight ν leverage: max/min over the sub-bands that actually
			# fed the α fit (i.e. surviving both quality cuts AND
			# `ignore_subindex`). NaN when fewer than 2 sub-bands
			# remained, since the ratio is meaningless on a single point.
			_sb_nu_ratio_2,                                                                # 'sb_nu_ratio_kept' — 2-list [σ-free, σ-fixed]
			# --- Intensity-profile fit diagnostics (BIC/AIC) ---
			int(len(tseries)),                     # 'n_profile_bins'
			int(len(tmix_popt)),                   # 'n_params_t'
			# --- ACF (2D-autocorrelation) columns ---
			# Order MUST match the acf_* entries appended to
			# `results_columns`. NaN/False when ACF mode was off or the
			# ACF fit failed for this component.
			bool(_comp_acf_on),                                                              # 'ACF'
			(_comp_acf_result['dtdnu']        if _comp_acf_result else np.nan),               # 'acf_dtdnu (ms/MHz)'
			(_comp_acf_result['dtdnu_err']    if _comp_acf_result else np.nan),               # 'acf_dtdnu_err'
			(_comp_acf_result['slope_mhz_ms'] if _comp_acf_result else np.nan),               # 'acf_slope (MHz/ms)'
			(_comp_acf_result['theta']        if _comp_acf_result else np.nan),               # 'acf_theta'
			(_comp_acf_result['theta_err']    if _comp_acf_result else np.nan),               # 'acf_theta_err'
			(_comp_acf_result['sigma_t']      if _comp_acf_result else np.nan),               # 'acf_sigma_t (ms)'
			(_comp_acf_result['sigma_nu']     if _comp_acf_result else np.nan),               # 'acf_sigma_nu (MHz)'
			(_comp_acf_result['redchisq']     if _comp_acf_result else np.nan),               # 'acf_redchisq'
			(_comp_acf_result['redchisq_norm'] if _comp_acf_result else np.nan),              # 'acf_redchisq_norm'
			# --- Ridge (centroid) drift block; `ridge` is absent when ACF_ridge
			#     is off or the ACF fit failed, hence the (.. or {}).get(..) guard ---
			((_comp_acf_result.get('ridge') or {}).get('dtdnu', np.nan)         if _comp_acf_result else np.nan),  # 'acf_ridge_dtdnu (ms/MHz)'
			((_comp_acf_result.get('ridge') or {}).get('slope', np.nan)         if _comp_acf_result else np.nan),  # 'acf_ridge_drift (MHz/ms)'
			((_comp_acf_result.get('ridge') or {}).get('err_dtdnu', np.nan)     if _comp_acf_result else np.nan),  # 'acf_ridge_dtdnu_err'
			((_comp_acf_result.get('ridge') or {}).get('err_slope', np.nan)     if _comp_acf_result else np.nan),  # 'acf_ridge_drift_err'
			(_comp_acf_result.get('effective_dtdnu', np.nan)     if _comp_acf_result else np.nan),  # 'acf_effective_dtdnu (ms/MHz)'
			(_comp_acf_result.get('effective_dtdnu_err', np.nan) if _comp_acf_result else np.nan),  # 'acf_effective_dtdnu_err'
			(_comp_acf_result.get('effective_source', 'none')    if _comp_acf_result else 'none'),  # 'acf_effective_source'
			# --- Scintillation columns (order MUST match the scint_* entries
			#     appended to `results_columns`). NaN/False when scint mode was
			#     off or the Lorentzian fit failed for this component. ---
			bool(_comp_scint_on),                                                              # 'scint'
			(_comp_scint_result['scint_bw']            if _comp_scint_result else np.nan),      # 'scint_bw (MHz)'
			(_comp_scint_result['scint_bw_err']        if _comp_scint_result else np.nan),      # 'scint_bw_err'
			(_comp_scint_result['scint_tau_recip']     if _comp_scint_result else np.nan),      # 'scint_tau_recip (ms)'
			(_comp_scint_result['scint_tau_recip_err'] if _comp_scint_result else np.nan),      # 'scint_tau_recip_err'
			(_comp_scint_result['scint_redchisq']      if _comp_scint_result else np.nan),      # 'scint_redchisq'
			(_comp_scint_result['scint_mod_index']     if _comp_scint_result else np.nan),      # 'scint_mod_index'
			(_comp_scint_result['scint_n_scintles']    if _comp_scint_result else np.nan),      # 'scint_n_scintles'
			(bool(_comp_scint_result['scint_resolved']) if _comp_scint_result else False),      # 'scint_resolved'
		])

	# === ASSIGN LEGACY ALIAS: REPRESENTATIVE BURST-LEVEL SUB-BAND RESULT ===
	# Historical alias kept for back-compat with any unrefactored callers
	# or downstream scripts. The waterfall sub-band/fiducial overlays
	# now use the per-component list `_comp_sb_results[ci]` directly
	# (via _render_subband_overlays), and the 'q'-key popup resolves
	# the active component through `_toggle_state['current']`. This
	# scalar alias points at the first component's result so any
	# remaining legacy reader still sees something sensible.
	_sb_alpha_result = _comp_sb_results[0] if _comp_sb_results else None
	# === END LEGACY ALIAS ===

	# === DIAGNOSTIC 4: FINAL SUMMARY ACROSS ALL COMPONENTS ===
	# print(f"\n{'='*80}")
	# print(f"[DIAG 4] FINAL SUMMARY: {'EMG' if USE_EMG_MIX else 'GAUSSIAN'} MODE")
	# print(f"{'='*80}")
	# print(f"  Total components: {len(subdfs)}")
	# for i, (sdf, suffix) in enumerate(zip(subdfs, subburst_suffixes[:len(subdfs)])):
	# 	dtdnu_val, dtdnu_e, dtdnu_chi = dtdnus[i]
	# 	print(f"  Component {suffix}: {len(sdf):3d} TOAs, dt/dν = {dtdnu_val:.2e} ± {dtdnu_e:.2e} ms/MHz, χ²_red = {dtdnu_chi:.2f}")
	# 
	# total_toas = sum(len(sdf) for sdf in subdfs)
	# components_with_toas = sum(1 for sdf in subdfs if len(sdf) > 0)
	# print(f"\n  TOTAL TOAs across all components: {total_toas}")
	# print(f"  Components with TOAs: {components_with_toas} / {len(subdfs)}")
	# print(f"{'='*80}\n")
	# === END DIAGNOSTIC 4 ===

	# ============================================================================
	# CONSOLIDATED ERROR BAR DIAGNOSTIC SUMMARY
	# ============================================================================
	if diag_enabled('ERROR_BAR_SUMMARY'):
		print(f"\n{'#'*80}")
		print(f"# ERROR BAR DIAGNOSTIC SUMMARY")
		print(f"# Why are horizontal error bars (t0_err) large vs vertical (center_f_err)?")
		print(f"{'#'*80}")
		
		# --- Section 1: EMG/Gaussian Mixture Fit Analysis ---
		print(f"\n{'='*60}")
		print(f"[1] EMG/GAUSSIAN MIXTURE FIT TO INTEGRATED TIME PROFILE")
		print(f"{'='*60}")
		print(f"  This fit determines t0_err (horizontal error bars)")
		print(f"  Number of components: {n_bursts}")
		print(f"  Reduced chi-squared: {tmix_redchisq:.3f}")
		
		print(f"\n  Parameter values and uncertainties:")
		print(f"  {'Parameter':<15} {'Value':>12} {'Uncertainty':>12} {'Rel.Err (%)':>12}")
		print(f"  {'-'*51}")
		for i, pname in enumerate(_diag_param_names):
			val = tmix_popt[i]
			err = tmix_perr[i]
			rel_err = 100 * err / abs(val) if val != 0 else np.inf
			print(f"  {pname:<15} {val:>12.6f} {err:>12.6f} {rel_err:>12.1f}%")
		
		print(f"\n  Covariance matrix condition number: {_diag_cond_num:.2e}")
		if _diag_cond_num > 1e10:
			print(f"    *** WARNING: Very high condition number - fit is ill-conditioned! ***")
		elif _diag_cond_num > 1e6:
			print(f"    ** CAUTION: Moderately high - some parameter degeneracy **")
		
		# Show high correlations
		print(f"\n  Parameter correlations (|corr| > 0.5 indicate degeneracy):")
		high_corr_found = False
		for i in range(len(_diag_param_names)):
			for j in range(i+1, len(_diag_param_names)):
				if abs(_diag_tmix_corr[i,j]) > 0.5:
					high_corr_found = True
					print(f"    {_diag_param_names[i]} <-> {_diag_param_names[j]}: corr = {_diag_tmix_corr[i,j]:+.3f}")
		if not high_corr_found:
			print(f"    No high correlations found")
		
		# Focus on mu uncertainties
		print(f"\n  ARRIVAL TIME (mu) UNCERTAINTIES -> t0_err:")
		for idx in range(n_bursts):
			mu_idx = n_bursts + idx
			mu_val = tmix_popt[mu_idx]
			mu_err = tmix_perr[mu_idx]
			variance = tmix_pcov[mu_idx, mu_idx]
			print(f"    Component {subburst_suffixes[idx]}:")
			print(f"      mu = {mu_val:.6f} ms")
			print(f"      Variance tmix_pcov[{mu_idx},{mu_idx}] = {variance:.6e}")
			print(f"      t0_err = sqrt(variance) = {mu_err:.6f} ms")
			# Show correlations for this mu
			corr_str = ""
			for j, pname in enumerate(_diag_param_names):
				if j != mu_idx and abs(_diag_tmix_corr[mu_idx,j]) > 0.3:
					corr_str += f"{pname}({_diag_tmix_corr[mu_idx,j]:+.2f}) "
			if corr_str:
				print(f"      Correlations: {corr_str}")
		
		# --- Section 2: Per-component comparison ---
		print(f"\n{'='*60}")
		print(f"[2] ERROR BAR COMPARISON BY COMPONENT")
		print(f"{'='*60}")
		print(f"  {'Component':<12} {'t0_err (ms)':>12} {'center_f_err':>14} {'Ratio':>10} {'Issue?':<20}")
		print(f"  {'-'*70}")
		
		for row in results:
			comp_name = row[0]
			t0 = row[2]       # t0 (ms)
			t0_err = row[3]   # t0_err
			cf = row[4]       # center_f (MHz)
			cf_err = row[5]   # center_f_err
			
			if cf_err > 0:
				ratio = t0_err / cf_err
				issue = ""
				if ratio > 10:
					issue = "LARGE HORIZ ERR"
				elif ratio < 0.1:
					issue = "LARGE VERT ERR"
			else:
				ratio = np.inf
				issue = "cf_err=0"
			
			# Extract just component suffix from name
			suffix = comp_name.split('_')[-1] if '_' in comp_name else comp_name
			print(f"  {suffix:<12} {t0_err:>12.6f} {cf_err:>14.4f} {ratio:>10.2f} {issue:<20}")
		
		# --- Section 3: Explanation ---
		print(f"\n{'='*60}")
		print(f"[3] INTERPRETATION")
		print(f"{'='*60}")
		print(f"  HORIZONTAL ERROR BAR (t0_err):")
		print(f"    - Source: EMG mixture fit covariance matrix")
		print(f"    - Large values indicate:")
		print(f"      * mu-sigma-lambda degeneracy (EMG parameters trade off)")
		print(f"      * Low SNR in time profile")
		print(f"      * Overlapping components")
		print(f"      * Ill-conditioned fit (check condition number above)")
		print(f"")
		print(f"  VERTICAL ERROR BAR (center_f_err):")
		print(f"    - Source: Gaussian fit to frequency spectrum")
		print(f"    - Usually smaller because:")
		print(f"      * Only 3 parameters (A, center_f, sigma_f)")
		print(f"      * Many frequency channels -> well-constrained")
		print(f"      * No scattering tail to model")
		
		print(f"{'#'*80}\n")
	# ============================================================================
	# END CONSOLIDATED DIAGNOSTIC SUMMARY
	# ============================================================================

	subdf = pd.concat(subdfs)
	# INSTRUMENTATION: dump per-channel arrival times and window geometry
	import json as _json
	subdf.to_csv(f"{outdir}/instr_subdf.csv")
	with open(f"{outdir}/instr_geom.json", "w") as _f:
		_json.dump(dict(subfall=globals().get('_INSTR_SUBFALL', {}), res_time_ms=float(res_time_ms),
		                nt=int(wfall.shape[1]), nf=int(wfall.shape[0]), freqs=[float(v) for v in freqs],
		                dtdnus=[[float(v) for v in d] for d in dtdnus], tmix_popt=[float(v) for v in tmix_popt]), _f)
	if save_solutions:
		if outdir == '' or outdir[-1] == '/':
			solname = f"{outdir}{bname}-{datetime.now().strftime('%b-%d-%Y')}.sols.npz"
		else:
			solname = f"{outdir}/{bname}-{datetime.now().strftime('%b-%d-%Y')}.sols.npz"
		np.savez(
			solname,
			tmix_popt=tmix_popt,
			tmix_perr=tmix_perr,
			tmix_redchisq=tmix_redchisq,
			subbandpopts=subbandpopts,
			subbandperrs=subbandperrs,
			subbandredchisqs=subbandredchisqs
		)
		print(f'Info: Saved {solname} solutions file')

	##### Plotting
	extent = [
		-pktime,
		res_time_ms*wfall.shape[1]-pktime,
		freqs_bin0,
		freqs_bin0 + res_freq*wfall.shape[0]
	]

	# === EARLY PER-COMPONENT BOUNDARY / TIME-WINDOW COMPUTATION ===
	# These structures are needed by both the waterfall plotting code
	# (sub-band horizontal lines, fiducial scattering tail) and the
	# on-pulse/off-pulse indicator bar that's set up later. Compute
	# them once here, in a small fault-isolated try block, before
	# anything that consumes them.
	#
	# Outputs:
	#   _on_segments              — list[(lo, hi)] per component, unshifted ms
	#   _on_segment_records       — list[dict] parallel, with A/μ/σ/λ/model
	#   _on_segments_plot_full    — merged segments in plot-frame ms
	#   _on_segments_per_comp_plot — per-comp segments in plot-frame ms
	#   _any_overlap              — bool, True iff any on-pulse windows overlap
	#   _component_boundaries_ms  — per-pair boundary (unshifted ms)
	#   _comp_time_windows_ms     — per-component (t_lo, t_hi) (unshifted ms)
	_DECAY_FRAC = 0.10
	_on_segments = []
	_on_segment_records = []
	_on_segments_plot_full = []
	_on_segments_per_comp_plot = []
	_any_overlap = False
	_component_boundaries_ms = []
	_comp_time_windows_ms = []

	def _decay_x_for_record(rec):
		"""Trailing-edge time at which the component's profile crosses
		`_DECAY_FRAC × peak`. Returns None on degenerate/failed input."""
		A_i  = rec.get('A', np.nan)
		mu   = rec.get('mu', np.nan)
		sg   = rec.get('sigma', np.nan)
		lam  = rec.get('lam', None)
		model = rec.get('model', None)
		if not (np.isfinite(mu) and np.isfinite(sg) and sg > 0):
			return None
		if model == 'gauss':
			return float(mu + sg * np.sqrt(-2.0 * np.log(_DECAY_FRAC)))
		elif model == 'emg':
			if lam is None or not np.isfinite(lam) or lam <= 0:
				return float(mu + sg * np.sqrt(-2.0 * np.log(_DECAY_FRAC)))
			if not np.isfinite(A_i) or A_i == 0:
				A_i = 1.0
			tau = 1.0 / lam
			_t_max = mu + max(8.0 * sg, 10.0 * tau)
			_grid = np.linspace(mu - 2.0 * sg, _t_max, 4096)
			try:
				_prof = emg_model(_grid, A_i, mu, sg, lam, 0.0)
			except Exception:
				return None
			if not np.all(np.isfinite(_prof)):
				return None
			_peak = float(np.max(_prof))
			if _peak <= 0:
				return None
			_thr = _DECAY_FRAC * _peak
			_argmax = int(np.argmax(_prof))
			_tail = _prof[_argmax:]
			_tail_t = _grid[_argmax:]
			_below = np.where(_tail <= _thr)[0]
			if len(_below) == 0:
				return None
			return float(_tail_t[int(_below[0])])
		return None

	try:
		_n_b = int(n_bursts) if 'n_bursts' in dir() else 0
		_have_tmix = (_n_b >= 1
		              and isinstance(tmix_popt, (list, tuple, np.ndarray)))
		# ── Use the SAME on-pulse intervals as the noise mask ──
		# `_on_segments_global_ms` was computed before the prep loop
		# (in the GLOBAL OFF-PULSE MASK + ROBUST PER-CHANNEL NOISE
		# ESTIMATE block) using the adaptive 1% → 5% → 10% → 20%
		# threshold cascade with a leading-edge sigma pad. Using
		# those same intervals here means the green/orange visual
		# bar is a faithful diagnostic of what the noise computation
		# saw as on-pulse — when components overlap, the bar shows
		# it, because they actually overlapped during noise masking.
		#
		# We still build `_on_segment_records` (with per-component
		# A/μ/σ/λ/model) because `_decay_x_for_record` (used for
		# component-boundary fallback when `cuts` is empty) needs
		# the per-component parameters, not just the merged intervals.
		_on_segments = list(_on_segments_global_ms)  # copy
		if _have_tmix and USE_EMG_MIX:
			_amps = np.atleast_1d(tmix_popt[0:_n_b])
			_mus = np.atleast_1d(tmix_popt[_n_b:2*_n_b])
			_sgs = np.atleast_1d(tmix_popt[2*_n_b:3*_n_b])
			_lms = np.atleast_1d(tmix_popt[3*_n_b:4*_n_b])
			# Build records in the same order as _on_segments. The
			# helper filters out components with non-finite μ/σ, so
			# we apply the same filter here to keep the lists aligned.
			for _A_i, _mu_i, _sg_i, _lm_i in zip(_amps, _mus, _sgs, _lms):
				if not (np.isfinite(_mu_i) and np.isfinite(_sg_i) and _sg_i > 0):
					continue
				_on_segment_records.append({
					'A':     float(_A_i) if np.isfinite(_A_i) else np.nan,
					'mu':    float(_mu_i),
					'sigma': float(abs(_sg_i)),
					'lam':   float(_lm_i) if (np.isfinite(_lm_i) and _lm_i > 0) else None,
					'model': 'emg',
				})
		elif _have_tmix:
			_amps = np.atleast_1d(tmix_popt[0:_n_b])
			_mus = np.atleast_1d(tmix_popt[_n_b:2*_n_b])
			_sgs = np.atleast_1d(tmix_popt[2*_n_b:3*_n_b])
			for _A_i, _mu_i, _sg_i in zip(_amps, _mus, _sgs):
				if not (np.isfinite(_mu_i) and np.isfinite(_sg_i) and _sg_i > 0):
					continue
				_on_segment_records.append({
					'A':     float(_A_i) if np.isfinite(_A_i) else np.nan,
					'mu':    float(_mu_i),
					'sigma': float(abs(_sg_i)),
					'lam':   None,
					'model': 'gauss',
				})

		# Merged segments → plot frame (for bar drawing).
		if _on_segments:
			_segs_sorted = sorted(
				[(float(a), float(b)) for a, b in _on_segments
				 if np.isfinite(a) and np.isfinite(b)]
			)
			_merged = []
			for _seg in _segs_sorted:
				if _merged and _seg[0] <= _merged[-1][1]:
					_merged[-1] = (_merged[-1][0], max(_merged[-1][1], _seg[1]))
				else:
					_merged.append(list(_seg))
			for (a, b) in _merged:
				_on_segments_plot_full.append((a - pktime, b - pktime))

		# Per-component segments in plot frame (for per-component shading).
		if _on_segments and len(_on_segments) >= 2:
			_paired_sort_early = sorted(
				zip(_on_segments, _on_segment_records),
				key=lambda p: p[1].get('mu', np.inf),
			)
			for _i in range(len(_paired_sort_early) - 1):
				if _paired_sort_early[_i + 1][0][0] <= _paired_sort_early[_i][0][1]:
					_any_overlap = True
					break
			for (_seg, _rec) in _paired_sort_early:
				_on_segments_per_comp_plot.append(
					(float(_seg[0]) - pktime, float(_seg[1]) - pktime)
				)

		# Boundaries (per-pair, unshifted ms): explicit `cuts` win, else
		# well-resolved → midpoint of off-pulse gap; not-well-resolved →
		# midpoint of (decay-to-10%, right_start) with fallback to
		# right_start.
		if len(cuts) > 0:
			_component_boundaries_ms = [float(c) for c in sorted(cuts)]
		elif len(xos) > 1 and len(_on_segments) >= 2:
			_paired_b = sorted(
				zip(_on_segments, _on_segment_records),
				key=lambda p: p[1].get('mu', np.inf),
			)
			for _i in range(len(_paired_b) - 1):
				(_, _seg_hi),  _rec_left = _paired_b[_i]
				(_next_lo, _), _         = _paired_b[_i + 1]
				_left_end    = float(_seg_hi)
				_right_start = float(_next_lo)
				if _left_end < _right_start:
					_component_boundaries_ms.append(0.5 * (_left_end + _right_start))
				else:
					_decay_x = _decay_x_for_record(_rec_left)
					if (_decay_x is not None
						and np.isfinite(_decay_x)
						and _decay_x < _right_start):
						_component_boundaries_ms.append(0.5 * (_decay_x + _right_start))
					else:
						_component_boundaries_ms.append(_right_start)

		# Per-component time windows (unshifted ms). xos is sorted in
		# μ-order at function entry, so component i ↔ (boundary[i-1],
		# boundary[i]) with -inf / +inf at the ends. Clamped to the
		# observation time range.
		_full_lo = float(times_ms[0])
		_full_hi = float(times_ms[-1])
		for _i in range(len(xos)):
			_lo = _component_boundaries_ms[_i - 1] if _i > 0 else _full_lo
			_hi = _component_boundaries_ms[_i]     if _i < len(_component_boundaries_ms) else _full_hi
			_lo = max(_full_lo, float(_lo))
			_hi = min(_full_hi, float(_hi))
			if _hi <= _lo:
				_lo, _hi = _full_lo, _full_hi
			_comp_time_windows_ms.append((_lo, _hi))
	except Exception as _bdy_e:
		# Non-fatal: if boundary computation fails the downstream
		# plotting falls back to legacy single-component visuals.
		print(f"  [boundary] computation failed: {_bdy_e}")
	# === END EARLY PER-COMPONENT BOUNDARY / TIME-WINDOW COMPUTATION ===

	fig, axs = plt.subplot_mosaic(
		'''
		T.
		AS
		AS
		AS
		EE
		''',
		figsize=figsize,
		width_ratios=[3,1],
		# gridspec_kw={'hspace':0.464}
	)

	# === RESOLVE COMPONENT INDEX FOR SLOPE PANEL / TOGGLE ===
	# Moved up from its old location near the slope-panel plotting so
	# that the intensity-profile legend can render the per-component
	# σ_t/τ label for the initially-selected component. The toggle
	# button installed later overrides this at runtime.
	pidx = pslopeidx  # Plot last component by default

	def _comp_has_toas(i):
		# A component has usable TOAs if EITHER its per-channel subdf
		# has rows OR (when that component was stacked) its stacked
		# dataframe has rows.
		if i < len(subdfs) and len(subdfs[i]) > 0:
			return True
		if i < len(_stacking_per_comp) and _stacking_per_comp[i]:
			if USE_EMG_MIX:
				sdf = stacked_dfs_per_comp[i] if i < len(stacked_dfs_per_comp) else None
				if sdf is not None and len(sdf) > 0:
					return True
			if USE_GAUSS_MIX:
				sdf = stacked_gauss_dfs_per_comp[i] if i < len(stacked_gauss_dfs_per_comp) else None
				if sdf is not None and len(sdf) > 0:
					return True
		return False

	def _comp_has_data(i):
		# A component has displayable per-component data either when it
		# has usable per-channel/stacked TOAs (handled by `_comp_has_toas`)
		# OR when it produced a successful ACF result (the bottom slope
		# panel will show that ACF's autocorrelation + fit, and the
		# waterfall legend already has its ACF slope entry registered in
		# `_wfall_slope_lines`). Both are valid reasons to include the
		# component in the Toggle-component cycle and as a render target.
		if _comp_has_toas(i):
			return True
		if (i < len(_comp_acf_results)
		        and _comp_acf_results[i] is not None):
			return True
		return False

	original_pidx = pidx
	if not _comp_has_data(pidx):
		for i in range(len(subdfs)):
			if _comp_has_data(i):
				pidx = i
				print(f"  Note: Component {subburst_suffixes[original_pidx]} has 0 TOAs, using component {subburst_suffixes[pidx]} instead")
				break
	# === END pidx RESOLUTION ===

	### Waterfall
	ax_wfall = axs['A']
	ax_wfall.imshow(
		wfall,
		aspect='auto',
		origin='lower',
		interpolation='none',
		cmap=cmap,
		extent=extent,
		norm=cmap_norm,
		vmax=np.quantile(wfall, 0.999),
		# vmin=5, # hewitt microshots
	)
	# === Sub-band overlays (boundaries + fiducial scattering tail) ===
	# Per-component drawing of the cyan ±bw·σ boundary box, the white
	# dashed sub-band frequency dividers, and the fiducial scattering
	# tail. Defined as artists registered against `ax_wfall` so they
	# can be cleared and re-rendered when the user clicks the
	# Toggle-component button. Initial render: ALL components shown,
	# each clipped to its own time window. Post-toggle: only the
	# active component's overlays shown, unclipped in time.
	#
	# The renderer is defined right here so it has closure access to
	# all the local plot state (_comp_sb_results, _comp_time_windows_ms,
	# freqs, pktime, etc.). The fiducial-tail label-text path
	# (slope-driven arrival time × τ(ν) power law) requires `dtdnus`
	# and `intercepts` which are not in scope until the per-component
	# loop has finished — but the renderer is *called* later in the
	# function (after those are populated), so this works.
	_sb_overlay_artists = []  # list of matplotlib artists (Line2D, etc.)

	def _clear_sb_overlays():
		"""Remove any artists drawn by `_render_subband_overlays`."""
		for _a in _sb_overlay_artists:
			try:
				_a.remove()
			except (ValueError, NotImplementedError, AttributeError):
				pass
		_sb_overlay_artists.clear()

	def _render_subband_overlays(active_comp):
		"""Draw sub-band frequency dividers + fiducial scattering tail.

		Parameters
		----------
		active_comp : int or None
			If None, draw overlays for ALL components, each clipped to
			its own time window (`_comp_time_windows_ms[i]`). This is
			the initial-render mode.
			If int, draw overlays ONLY for that component, unclipped
			in time (spanning the full x-axis). This is the
			post-toggle mode.

		Reads (closure):
			_comp_sb_results (list of dict)
			_comp_time_windows_ms (list of (t_lo, t_hi) in unshifted ms)
			_n_subbands_per_comp (list of int)
			freqs, pktime, dtdnus, intercepts, xcs, xos
			subband_plot (bool — "any component requested sub-band overlay")
		"""
		_clear_sb_overlays()
		if not subband_plot:
			return
		if not _comp_sb_results:
			return

		# Determine which components to draw, and their clipping windows.
		if active_comp is None:
			_comps_to_draw = list(range(len(_comp_sb_results)))
			_use_curve_clip = True
		else:
			_comps_to_draw = [int(active_comp)] if (0 <= active_comp < len(_comp_sb_results)) else []
			_use_curve_clip = False

		_x_full_lo = float(times_ms[0]) - pktime
		_x_full_hi = float(times_ms[-1]) - pktime

		for _ci in _comps_to_draw:
			_sb = _comp_sb_results[_ci]
			if _sb is None or not _sb.get('n_good_subbands', 0) > 0:
				continue

			# Component-specific time window in plot-frame coords. Used
			# UNCONDITIONALLY for the horizontal sub-band lines (cyan
			# ±σ box and white-dashed dividers) — they're a
			# "this-burst's measurement-region" indicator and so
			# always live inside the component's [t_lo, t_hi]. The
			# fiducial tail curve uses a different clip rule (see
			# `_use_curve_clip` below): clipped when in the
			# all-components view, unclipped when toggled to a single
			# component (the curve is a predicted envelope and you
			# asked it to extend past the boundary).
			if _ci < len(_comp_time_windows_ms):
				_t_lo_ms, _t_hi_ms = _comp_time_windows_ms[_ci]
				_x_lo_h = float(_t_lo_ms) - pktime
				_x_hi_h = float(_t_hi_ms) - pktime
			else:
				_x_lo_h, _x_hi_h = _x_full_lo, _x_full_hi

			# Curve clip window: same as horizontals when None, full
			# axis when toggled to a single component.
			if _use_curve_clip:
				_x_lo_c, _x_hi_c = _x_lo_h, _x_hi_h
			else:
				_x_lo_c, _x_hi_c = _x_full_lo, _x_full_hi

			# Cyan ±σ boundary box at this component's freq_lo / freq_hi.
			# Always read the values (needed below for the τ(ν) curve
			# and for the dashed dividers' frequency range).
			_sb_freq_lo = _sb.get('freq_lo', freqs.min())
			_sb_freq_hi = _sb.get('freq_hi', freqs.max())

			# Horizontals (cyan ±σ box + white dashed dividers) are
			# drawn in both views, time-clipped to the component's own
			# window so they read as "component i's measurement region"
			# regardless of toggle state.
			_l1 = ax_wfall.plot(
				[_x_lo_h, _x_hi_h], [_sb_freq_lo, _sb_freq_lo],
				color='cyan', ls='-', alpha=0.6, lw=1.0,
			)[0]
			_l2 = ax_wfall.plot(
				[_x_lo_h, _x_hi_h], [_sb_freq_hi, _sb_freq_hi],
				color='cyan', ls='-', alpha=0.6, lw=1.0,
			)[0]
			_sb_overlay_artists.extend([_l1, _l2])

			# White dashed sub-band frequency dividers (per-component
			# `n_sub` from _n_subbands_per_comp), clipped to the same
			# component window.
			_n_sub_this = int(_n_subbands_per_comp[_ci]) if _ci < len(_n_subbands_per_comp) else _n_subbands
			_sb_mask = (freqs >= _sb_freq_lo) & (freqs <= _sb_freq_hi)
			_sb_n_chan = int(np.sum(_sb_mask))
			if _sb_n_chan > 0 and _n_sub_this > 1:
				_sb_chan_per = _sb_n_chan // _n_sub_this
				_sb_indices = np.where(_sb_mask)[0]
				for _j in range(1, _n_sub_this):
					_boundary_idx = _sb_indices[0] + _j * _sb_chan_per
					if _boundary_idx < len(freqs):
						_ld = ax_wfall.plot(
							[_x_lo_h, _x_hi_h], [freqs[_boundary_idx], freqs[_boundary_idx]],
							color='white', ls='--', alpha=0.4, lw=0.8,
						)[0]
						_sb_overlay_artists.append(_ld)

			# Fiducial scattering tail (arrival-time line + arrival+τ envelope).
			# Uses this component's slope/intercept and τ(ν) power law from
			# its own fit. Drawn as Line2D segments clipped (or not) to
			# the time window.
			if not _sb.get('success', False):
				continue
			if not (len(dtdnus) > _ci and len(intercepts) > _ci):
				continue
			_dtdnu_i, _, _ = dtdnus[_ci]
			_tb_i, _ = intercepts[_ci]
			# Per-component reference time, mirroring the waterfall
			# slope-line and bottom-panel logic: xcs only when EMG
			# AND no cuts; otherwise xos. See the slope-line plot
			# comment around line ~6877 for the derivation.
			if USE_EMG_MIX and len(cuts) == 0:
				_ref_time_i = xcs[_ci]
			else:
				_ref_time_i = xos[_ci]
			if _dtdnu_i == 0:
				continue
			# Curves are parameterised in ν; we'll mask out points whose
			# arrival-time falls outside the clip window.
			_nu_curve = np.linspace(freqs.min(), freqs.max(), 300)
			_tau_curve = _sb['tau_ref'] * (_nu_curve / _sb['freq_center']) ** _sb['alpha']
			# === ARRIVAL-TIME LINE ANCHORING ===
			# In ToA mode the per-channel line fit produces (dtdnu, t_b)
			# such that line_model(ν) = dtdnu·ν + t_b passes through the
			# arrival-time cloud in subfall-local coordinates. Adding
			# ref_time shifts it into absolute (waterfall-frame) time:
			#     t_arr_abs = dtdnu·ν + t_b + ref_time
			#
			# In ACF mode `dtdnu` comes from the 2D autocorrelation of
			# the sub-fall — a measurement of the burst's *shape* — and
			# has no intercept analogue. `t_b` in this branch is still
			# whatever the per-channel ToA line fit happened to land on
			# (usually meaningless), so pairing the ACF slope with that
			# intercept produces a line with the right slope but an
			# arbitrary location (typically several ms off the burst).
			#
			# The right anchoring for the ACF case is to make the line
			# pass through the burst centroid (ν = freq_center,
			# t = ref_time):
			#     t_arr_abs = dtdnu·(ν − freq_center) + ref_time
			# which uses the ACF slope plus a physically meaningful
			# anchor and reduces to the ToA form when t_b happens to
			# equal −dtdnu·freq_center (which is what a well-behaved
			# ToA fit through the centroid would produce).
			_comp_acf_active = (
				_ci < len(_acf_per_comp) and bool(_acf_per_comp[_ci])
			)
			if _comp_acf_active:
				_t_arr_abs = _dtdnu_i * (_nu_curve - _sb['freq_center']) + _ref_time_i
			else:
				_t_arr_abs = _dtdnu_i * _nu_curve + _tb_i + _ref_time_i
			# === END ANCHORING ===
			_t_scat_abs  = _t_arr_abs + _tau_curve
			_t_arr_plot  = _t_arr_abs  - pktime
			_t_scat_plot = _t_scat_abs - pktime
			if _use_curve_clip:
				# Mask: keep only points where arrival AND envelope sit
				# inside [_x_lo_c, _x_hi_c]. Clipping is per-point; the
				# resulting Line2D will be drawn as a (possibly
				# disjoint) polyline.
				_mask = (_t_arr_plot >= _x_lo_c) & (_t_scat_plot <= _x_hi_c)
				if not np.any(_mask):
					continue
				_t_arr_plot_use = np.where(_mask, _t_arr_plot, np.nan)
				_t_scat_plot_use = np.where(_mask, _t_scat_plot, np.nan)
			else:
				_t_arr_plot_use = _t_arr_plot
				_t_scat_plot_use = _t_scat_plot

			_la = ax_wfall.plot(
				_t_arr_plot_use, _nu_curve,
				color='#66CCFF', ls=':', lw=1.5, alpha=0.7,
				zorder=5,
			)[0]
			_sfx_i = subburst_suffixes[_ci] if _ci < len(subburst_suffixes) else str(_ci)
			_prefix_i = f'{_sfx_i}. ' if len(_comp_sb_results) > 1 else ''
			_label_i = (f'{_prefix_i}$\\tau(\\nu)$: '
			            f'$\\alpha$={_sb["alpha"]:.2f}±{_sb["alpha_err"]:.2f}, '
			            f'$\\tau_{{ref}}$={_sb["tau_ref"]:.3f} ms')
			_le = ax_wfall.plot(
				_t_scat_plot_use, _nu_curve,
				color='#FF6B6B', ls='--', lw=2, alpha=0.85,
				label=_label_i,
				zorder=5,
			)[0]
			_sb_overlay_artists.extend([_la, _le])

			# === Combined 2·σ_smooth(ν) + τ(ν) envelope (light grey dashed) ===
			# Traces where the visible trailing edge of the burst sits:
			#   t_edge(ν) = t_arr(ν) + 2·σ_smooth(ν) + τ(ν)
			# Serial addition (not quadrature) — σ + τ are the two
			# characteristic widths of the EMG-convolved profile, and
			# their sum approximates where signal drops to noise. The
			# factor of 2 on σ is empirical: k = 1 places the envelope
			# at ~60% of peak for an unscattered Gaussian, which sits
			# inside the visible burst; k = 2 places it at ~14% of
			# peak, matching the visual extent of the trailing edge.
			#
			# σ_smooth(ν) source: the ε power-law fit from
			# fit_subband_alpha (σ(ν) = A · (ν/ν_ref)^β, β ≤ 0).
			# Fallback if ε fit failed: constant σ = mean(sub_sigmas)
			# (frequency-independent).
			#
			# τ(ν) is the same as for the orange curve. If the τ fit
			# failed we've already `continue`-ed above, so reaching
			# this point guarantees τ is usable.
			#
			# Clipping: standard time-window clip AND restriction to
			# the sub-band frequency range [sb_freq_lo, sb_freq_hi]
			# — the ε fit is defined only there.
			_eps_success = bool(_sb.get('eps_sigma_success', False))
			_eps_amp_i   = _sb.get('eps_sigma_amp',  np.nan)
			_eps_beta_i  = _sb.get('eps_sigma_beta', np.nan)
			if _eps_success and np.isfinite(_eps_amp_i) and np.isfinite(_eps_beta_i):
				_sig_smooth_curve = (_eps_amp_i
				                     * (_nu_curve / _sb['freq_center']) ** _eps_beta_i)
			else:
				# Fallback: flat mean σ across the sub-bands. No
				# ν-dependence; the curve still bows because of τ(ν).
				_sub_sig_i = np.asarray(_sb.get('sub_sigmas', []), dtype=float)
				_sub_sig_i = _sub_sig_i[np.isfinite(_sub_sig_i) & (_sub_sig_i > 0)]
				if _sub_sig_i.size == 0:
					_sig_smooth_curve = None  # cannot draw
				else:
					_sig_smooth_curve = np.full_like(_nu_curve, float(np.mean(_sub_sig_i)))

			if _sig_smooth_curve is not None:
				# k = 2 multiplier on σ_smooth: the grey envelope traces
				# 2σ past the arrival (roughly the 14%-of-peak contour
				# of the underlying Gaussian) plus τ, which visually
				# matches the trailing edge of the burst on the
				# dynamic spectrum better than k = 1.
				_t_env_abs  = _t_arr_abs + 2.0 * _sig_smooth_curve + _tau_curve
				_t_env_plot = _t_env_abs - pktime

				# Frequency clip: keep only the sub-band region.
				# Falling back to the curve's own ν range if the
				# sub-band bounds are missing (defensive — they
				# should always be set when _sb['success'] is True).
				_sb_lo_i = _sb.get('freq_lo', np.nan)
				_sb_hi_i = _sb.get('freq_hi', np.nan)
				if np.isfinite(_sb_lo_i) and np.isfinite(_sb_hi_i) and _sb_hi_i > _sb_lo_i:
					_freq_mask_env = (_nu_curve >= _sb_lo_i) & (_nu_curve <= _sb_hi_i)
				else:
					_freq_mask_env = np.ones_like(_nu_curve, dtype=bool)

				# Time clip: same logic as the orange envelope —
				# require arrival inside [_x_lo_c, _x_hi_c] AND
				# the envelope itself inside [_x_lo_c, _x_hi_c].
				if _use_curve_clip:
					_time_mask_env = ((_t_arr_plot >= _x_lo_c)
					                  & (_t_env_plot <= _x_hi_c))
				else:
					_time_mask_env = np.ones_like(_nu_curve, dtype=bool)

				_env_mask = _freq_mask_env & _time_mask_env
				if np.any(_env_mask):
					_t_env_plot_use = np.where(_env_mask, _t_env_plot, np.nan)
					_lp = ax_wfall.plot(
						_t_env_plot_use, _nu_curve,
						color='#9A9A9A', ls='--', lw=1.4, alpha=0.85,
						zorder=4,  # below the orange/cyan curves
					)[0]
					_sb_overlay_artists.append(_lp)
			# === END Combined 2·σ_smooth(ν) + τ(ν) envelope ===
	# === END Sub-band overlays renderer ===

	dmlbl = f"DM = {targetDM:.3f} pc/cm$^3$" if not recalculateDM else f"DM = {targetDM:.3f} +/- {ddmopt_std:.3f} pc/cm$^3$"
	ax_wfall.annotate(
		dmlbl,
		xy=(0.05, 0.925),
		xycoords='axes fraction',
		color='white',
		weight='black',
		size=10,
		bbox={"boxstyle":"round"}
	)
	# Plot individual arrival times (white circles) for components that
	# were NOT stacked. Previously this block was gated on
	#   `not (USE_EMG_MIX and USE_SIGNAL_STACKING)`,
	# but USE_SIGNAL_STACKING is `any(_stacking_per_comp)` — so in the
	# mixed case (e.g. stacking=[False, True]) the gate read True and
	# the entire white-circle plot was silently skipped, even for the
	# unstacked component a. Fix: build a filtered subdf containing
	# only rows from components whose per-component stacking flag is
	# False, and plot those circles. Stacked components keep getting
	# their red-X markers from the dedicated stacked-plot loops further
	# down (one for EMG, one for Gaussian). Both EMG and Gaussian
	# code paths need the white-circle plot when there are non-stacked
	# components — the earlier version only restored it for EMG.
	#
	# Additional rule (v56_30): for STACKED components, if their
	# stacking pipeline ended up with zero surviving stacks (e.g.
	# every stack rejected by Filter 1B's peak-SNR threshold because
	# stack_size was too small to boost SNR above the cutoff), fall
	# back to the per-channel subdf so the per-channel ToAs that
	# survived their own filters still get drawn. Without this
	# fallback, weak components disappeared entirely from the
	# dynamic spectrum — no red X's (stacks filtered out) AND no
	# colored circles (gated out by stacking flag).
	def _stacked_dfs_yielded_zero(i):
		"""True if component i was stacked but ended up with 0 surviving stacks."""
		if i >= len(_stacking_per_comp) or not _stacking_per_comp[i]:
			return False
		if USE_EMG_MIX:
			sdf = stacked_dfs_per_comp[i] if i < len(stacked_dfs_per_comp) else None
		elif USE_GAUSS_MIX:
			sdf = stacked_gauss_dfs_per_comp[i] if i < len(stacked_gauss_dfs_per_comp) else None
		else:
			sdf = None
		return sdf is None or len(sdf) == 0

	_nonstacked_subdfs = []
	for i in range(len(subdfs)):
		if i >= len(_stacking_per_comp):
			continue
		if len(subdfs[i]) == 0:
			continue
		# Components in ACF mode get their slope from the 2D
		# autocorrelation of the sub-fall, not from per-channel ToAs.
		# The per-channel arrival times still exist in subdfs[i] (they
		# are computed for reference) but they are NOT used in the
		# slope fit, and showing them on the dynamic spectrum would
		# falsely suggest they were. Skip them at the source so every
		# downstream scatter call (component fit points, red SNR-failed
		# markers, gray ignored markers, hollow ignored-stacked
		# markers) automatically has nothing to plot for ACF
		# components.
		if i < len(_acf_per_comp) and _acf_per_comp[i]:
			continue
		# Draw per-channel circles either when this component wasn't
		# stacked at all, OR when stacking filtered it down to zero
		# (so the user still sees the channel-level ToAs that DID survive).
		if not _stacking_per_comp[i] or _stacked_dfs_yielded_zero(i):
			_nonstacked_subdfs.append(subdfs[i])
			if _stacked_dfs_yielded_zero(i):
				_sfx = subburst_suffixes[i] if i < len(subburst_suffixes) else str(i)
				print(f"  Note: component {_sfx} was stacked but all stacks were "
				      f"filtered out (likely by Filter 1B peak-SNR cutoff). Falling "
				      f"back to non-stacked per-channel ToAs for the waterfall "
				      f"overlay. Consider increasing stack_size for this component.")
	if _nonstacked_subdfs:
		subdf_nonstacked = pd.concat(_nonstacked_subdfs)
	else:
		subdf_nonstacked = subdf.iloc[0:0]  # empty same-schema df

	if len(subdf_nonstacked) > 0:
		# === ARRIVAL TIME PLOTTING DIAGNOSTIC ===
		if diag_enabled('ARRIVAL_TIME_PLOT'):
			print(f"\n=== ARRIVAL TIME PLOTTING DIAGNOSTIC ===")
			print(f"  Non-stacked subdf has {len(subdf_nonstacked)} rows "
			      f"(from {len(_nonstacked_subdfs)} non-stacked components)")
			print(f"  time_col = '{time_col}'")
			print(f"  subdf[time_col] range: [{subdf_nonstacked[time_col].min():.4f}, {subdf_nonstacked[time_col].max():.4f}] ms")
			print(f"  pktime (subtracted for plotting) = {pktime:.4f} ms")
			print(f"  Plotted X coords range: [{(subdf_nonstacked[time_col]-pktime).min():.4f}, {(subdf_nonstacked[time_col]-pktime).max():.4f}] ms")
			print(f"  Waterfall extent[0:2] (x-axis): [{extent[0]:.4f}, {extent[1]:.4f}] ms")
			unique_colors = subdf_nonstacked['color'].unique()
			print(f"  Unique component colors: {len(unique_colors)}")
			for color in unique_colors:
				color_df = subdf_nonstacked[subdf_nonstacked['color'] == color]
				print(f"    Color {color}: {len(color_df)} points, time range: [{color_df[time_col].min():.4f}, {color_df[time_col].max():.4f}]")
		# === END ARRIVAL TIME PLOTTING DIAGNOSTIC ===

		ax_wfall.scatter( # component fit points
			subdf_nonstacked[time_col]-pktime,
			subdf_nonstacked['freqs'],
			# Filled circles in each component's color, with a thin
			# black outline for contrast on light signal regions and
			# the colored fill standing out against dark noise
			# regions. Previously this drew `c='w'` (white fill) with
			# edges in `subdf['color']`, but since the colors cycle
			# starts with 'white', component a's markers were
			# white-on-white (invisible on signal) and other
			# components' colors blended into either the bright
			# signal (red, yellow) or the dark noise (black, blue,
			# green) depending on where they landed. Filling with
			# the component color and outlining black gives reliable
			# visibility on both backgrounds.
			#
			# Alpha is normalised to the in-component max amplitude,
			# but with a floor of 0.4 to keep low-amp markers
			# visible. Without the floor, channels whose EMG fit
			# returned a tiny amplitude (often 1–3 orders of
			# magnitude below the peak channel's amplitude) became
			# essentially transparent — making 8–9 of the 10
			# per-component ToAs disappear in the rendered figure
			# even though they were present in the underlying scatter.
			c=subdf_nonstacked['color'],
			edgecolors='black',
			marker='o',
			s=25,
			linewidths=0.8,
			alpha=np.clip(
				subdf_nonstacked['amp']/subdf_nonstacked['amp'].max(),
				0.4, 1.0
			),
		)
	
	# === PLOT STACKED ARRIVAL TIMES (red X markers) ===
	# Per-component aware: iterate over each component that was stacked and
	# plot its surviving stacked TOAs on the waterfall. Without this loop,
	# only the last loop iteration's `stacked_df` is visible (and it's None
	# whenever the last component was not stacked), so mixed
	# per-component stacking would silently drop overlays from earlier
	# stacked components.
	if USE_EMG_MIX and USE_SIGNAL_STACKING:
		for _ci, _sdf in enumerate(stacked_dfs_per_comp):
			if _sdf is None or len(_sdf) == 0:
				continue
			# Skip components in ACF mode -- their slope comes from the 2D
			# autocorrelation of the sub-fall, not per-channel/stacked ToAs.
			if _ci < len(_acf_per_comp) and _acf_per_comp[_ci]:
				continue
			ax_wfall.scatter(
				_sdf[time_col]-pktime,
				_sdf['freqs'],
				c='r',                    # Red color
				marker='x',               # X-shaped marker
				s=25,                    # Larger size for visibility
				linewidths=2,             # Thicker lines
				label=('Stacked arrival times' if _ci == 0 else None),
				zorder=10                 # Plot on top
			)
	# === END STACKED PLOTTING ===
	
	# === PLOT IGNORED STACKED EMG ARRIVAL TIMES (transparent X markers) ===
	# Per-component aware: iterate over each component's saved ignored
	# stacked-EMG df. Previously this read the loop-leaked variable
	# `ignored_stacked_df`, which always held the LAST iteration's
	# value — so only the last component's ignored stacked X markers
	# rendered on the waterfall regardless of which component they
	# belonged to. Mirrors the per-component pattern used by the
	# (non-ignored) stacked-X plot above.
	for _ci, _isd in enumerate(ignored_stacked_dfs_per_comp):
		if _isd is None or len(_isd) == 0:
			continue
		# Skip components in ACF mode -- their slope comes from the 2D
		# autocorrelation of the sub-fall, not per-channel/stacked ToAs.
		if _ci < len(_acf_per_comp) and _acf_per_comp[_ci]:
			continue
		ax_wfall.scatter(
			_isd[time_col]-pktime,
			_isd['freqs'],
			c='#333333',                 # Gray color for ignored
			marker='x',               # X-shaped marker (same as stacked)
			s=30,                     # Slightly larger for visibility
			linewidths=1.5,             # Same linewidth as stacked
			alpha=0.7,                # More visible (was 0.5)
			zorder=9
		)
	
	# === PLOT IGNORED GAUSSIAN STACKED ARRIVAL TIMES (transparent X markers) ===
	# Per-component aware: same fix as the EMG case above.
	for _ci, _isgd in enumerate(ignored_stacked_gauss_dfs_per_comp):
		if _isgd is None or len(_isgd) == 0:
			continue
		# Skip components in ACF mode -- their slope comes from the 2D
		# autocorrelation of the sub-fall, not per-channel/stacked ToAs.
		if _ci < len(_acf_per_comp) and _acf_per_comp[_ci]:
			continue
		ax_wfall.scatter(
			_isgd[tpoint]-pktime,
			_isgd['freqs'],
			c='#333333',                 # Gray color for ignored
			marker='x',               # X-shaped marker (same as stacked)
			s=30,                     # Slightly larger for visibility
			linewidths=1.5,             # Same linewidth as stacked
			alpha=0.7,                # More visible (was 0.5)
			zorder=9
		)

	# === PLOT IGNORED NON-STACKED ARRIVAL TIMES (transparent circles) ===
	# Per-component aware: iterate over each component's saved ignored
	# non-stacked subdf. Same loop-leak fix as the stacked cases above.
	time_col_ignored = time_col if USE_EMG_MIX else tpoint
	for _ci, _isub in enumerate(ignored_subdfs_per_comp):
		if _isub is None or len(_isub) == 0:
			continue
		# Skip components in ACF mode -- their slope comes from the 2D
		# autocorrelation of the sub-fall, not per-channel/stacked ToAs.
		if _ci < len(_acf_per_comp) and _acf_per_comp[_ci]:
			continue
		ax_wfall.scatter(
			_isub[time_col_ignored]-pktime,
			_isub['freqs'],
			facecolors='none',        # Hollow circle
			edgecolors='#333333',        # Gray outline
			marker='o',               # Circle marker (same as non-stacked)
			s=25,                     # Slightly larger for visibility
			linewidths=1.5,             # Thicker lines
			alpha=0.7,                # More visible (was 0.5)
			zorder=9
		)
	
	# === PLOT GAUSSIAN STACKED ARRIVAL TIMES (red X markers) ===
	# Per-component aware: see EMG case above.
	if USE_GAUSS_MIX and USE_SIGNAL_STACKING:
		for _ci, _sdf in enumerate(stacked_gauss_dfs_per_comp):
			if _sdf is None or len(_sdf) == 0:
				continue
			# Skip components in ACF mode -- their slope comes from the 2D
			# autocorrelation of the sub-fall, not per-channel/stacked ToAs.
			if _ci < len(_acf_per_comp) and _acf_per_comp[_ci]:
				continue
			ax_wfall.scatter(
				_sdf[tpoint]-pktime,
				_sdf['freqs'],
				c='r',                    # Red color
				marker='x',               # X-shaped marker
				s=25,                     # Larger size for visibility
				linewidths=2,             # Thicker lines
				label=('Stacked Gaussian arrival times' if _ci == 0 else None),
				zorder=10                 # Plot on top
			)
	# === END GAUSSIAN STACKED PLOTTING ===
	

# 	# Component lines
# 	for (dtdnu, dtdnu_err, _), (tb, tb_err), xoi in zip(dtdnus, intercepts, xos):
# 		if dtdnu != 0:
# 			ax_wfall.plot(
# 				times_ms-pktime,
# 				(1/dtdnu)*(times_ms-xoi) - tb/dtdnu,
# 				'w-.',
# 				alpha=0.75,
# 				# label=f'$dt/d\\nu = $ {dtdnu:.2e} $\\pm$ {dtdnu_err:.2e}'
# 				label=f'{subburst_suffixes[xos.index(xoi)]}. $dt/d\\nu =$ {scilabel(dtdnu, dtdnu_err)} ms/MHz'
# 			)
				   
		
	# Per-component slope lines on the waterfall. Captured by index so the
	# toggle button can hide non-selected components' legend entries
	# (the LINES themselves remain visible — only the labels are
	# toggled, by setting label=None on non-active components and then
	# re-rendering the legend).
	_wfall_slope_lines = {}
	_wfall_slope_labels = {}  # full text per component, for restoring on toggle

	for idx, ((dtdnu, dtdnu_err, _), (tb, tb_err), xoi) in enumerate(zip(dtdnus, intercepts, xos)):
		# === SLOPE LINE PLOTTING DIAGNOSTIC ===
		if diag_enabled('SLOPE_PLOT'):
			print(f"\n=== SLOPE LINE PLOTTING DIAGNOSTIC (component {subburst_suffixes[idx]}) ===")
			print(f"  dtdnu = {dtdnu:.5e} ms/MHz")
			print(f"  tb (intercept from fit) = {tb:.4f} ms")
			print(f"  xoi (center for this component, from xos) = {xoi:.4f} ms")
			print(f"  pktime (global peak time) = {pktime:.4f} ms")
			print(f"  Line equation: freq = (1/dtdnu) * (time - xoi) - tb/dtdnu")
			if dtdnu != 0:
				print(f"  At time=xoi: freq = -tb/dtdnu = {-tb/dtdnu:.2f} MHz")
			else:
				print(f"  At time=xoi: freq = -tb/dtdnu = inf MHz")
		# === END SLOPE LINE PLOTTING DIAGNOSTIC ===
		
# 		if dtdnu != 0:
# 			ax_wfall.plot(
#     			times_ms-pktime,
#     			(1/dtdnu)*(times_ms-xoi) - tb/dtdnu,
#     			'w-.',
#     			alpha=0.75,
#     			label=f'{subburst_suffixes[idx]}. $dt/d\\nu =$ {scilabel(dtdnu, dtdnu_err)} ms/MHz'
#     		)
			
##FIX

		# For ACF components the per-channel intercept t_b does not
		# exist, and the slope line must pass through the component's
		# (centroid time, centre frequency); anchoring at freq=0 (as the
		# per-channel formula below does) puts the line off-screen. So
		# the ACF slope line is drawn separately in the ACF overlay
		# block; here we draw only the per-channel ToA line.
		_idx_acf_on = (idx < len(_comp_acf_results)
		               and _comp_acf_results[idx] is not None
		               and _comp_acf_results[idx].get('ok', False))
		if dtdnu != 0 and np.isfinite(dtdnu) and not _idx_acf_on:
			# Reference time for the per-channel slope line, mirroring
			# the bottom-panel `_intercept_offset` logic:
			#
			#   no cuts, EMG  : `xcs[idx]` (= μ + τ). The per-channel
			#                    TOAs were shifted by `xcs - 4·width_used`,
			#                    so the line's anchor in the unshifted
			#                    plot frame is `xcs[idx] + dtdnu·ν + tb`.
			#   no cuts, Gauss: `xoi` (= μ); `xcs == xos`.
			#   with cuts    : `xoi` always — the per-component shift
			#                    (whether 0, cuts[prev_ci], or cuts[-1])
			#                    is chosen so the post-shift TOAs read
			#                    as `xosi + dtdnu·ν + tb` in unshifted
			#                    ms, regardless of EMG/Gauss.
			#
			# The previous unconditional `xcs[idx] if USE_EMG_MIX else xoi`
			# was correct only in the no-cuts case; with cuts it placed
			# the slope line τ ms above the TOAs (matching the visible
			# offset you flagged).
			if USE_EMG_MIX and len(cuts) == 0:
				ref_time = xcs[idx]
			else:
				ref_time = xoi
			_label_text = (f'{subburst_suffixes[idx]}. '
			               f'$dt/d\\nu =$ {scilabel(dtdnu, dtdnu_err)} ms/MHz')
			_lines = ax_wfall.plot(
				times_ms-pktime,
				(1/dtdnu)*(times_ms-ref_time) - tb/dtdnu,
				'w-.',
				alpha=0.75,
				label=_label_text,
			)
			if _lines:
				_wfall_slope_lines[idx] = _lines[0]
				_wfall_slope_labels[idx] = _label_text

		# === ACF ellipse overlay on the waterfall (per ACF component) ===
		# draw_acf_on_waterfall draws the fitted-ellipse shape (it also
		# draws a slope line, but we suppress that here since the per-
		# component slope line above already covers it — pass a no-op by
		# drawing only the ellipse via the dedicated helper). The helper
		# is a no-op when the ACF result is missing or the fit failed.
		if (idx < len(_comp_acf_results)
		        and _comp_acf_results[idx] is not None
		        and _comp_acf_results[idx].get('ok', False)):
			from matplotlib.patches import Ellipse as _ACFEllipse
			_acf_r = _comp_acf_results[idx]
			# Component centre in the plot frame (waterfall x is t-pktime).
			_acf_t0 = xoi - pktime
			# Per-component centre frequency: column 'center_f (MHz)' of the
			# results row for this component. results_columns index 4 is
			# 'center_f (MHz)' (name, DM, t0, t0_err, center_f, ...).
			if idx < len(results) and len(results[idx]) > 4:
				_acf_cf = results[idx][4]
			else:
				_acf_cf = float(np.mean(freqs))
			if not np.isfinite(_acf_cf):
				_acf_cf = float(np.mean(freqs))

			# --- ACF slope line through (centroid time, centre freq) ---
			# Line equation: t - t0 = dtdnu * (nu - center_f). Drawn
			# parametrically in nu (spanning the panel's frequency axis)
			# so the anchor point is exact regardless of slope magnitude.
			# Anchoring at freq=0 (the per-channel formula) would put the
			# line off-screen for the steep ACF slopes typical at LOFAR.
			# Use the EFFECTIVE ACF slope for the on-waterfall line: the ridge
			# (centroid) slope where the major axis has blown up
			# (effective_source == 'ridge'), the major-axis "core" otherwise.
			# Falls back to the major axis if the effective readout is absent
			# (e.g. ACF_ridge disabled). The line is coloured + labelled by
			# which readout is shown, so it is obvious at a glance which the
			# effective selection preferred.
			_acf_eff_src = _acf_r.get('effective_source', 'major')
			_acf_eff_val = _acf_r.get('effective_dtdnu', np.nan)
			if np.isfinite(_acf_eff_val):
				_acf_dtdnu     = _acf_eff_val
				_acf_dtdnu_err = _acf_r.get('effective_dtdnu_err', np.nan)
			else:
				_acf_dtdnu     = _acf_r['dtdnu']
				_acf_dtdnu_err = _acf_r.get('dtdnu_err', np.nan)
				_acf_eff_src   = 'major'
			_acf_src_is_ridge = (_acf_eff_src == 'ridge')
			_acf_src_word     = 'ridge' if _acf_src_is_ridge else 'core'
			_acf_line_color   = 'orange' if _acf_src_is_ridge else 'white'
			_nu_lo, _nu_hi = ax_wfall.get_ylim()
			_t_lo, _t_hi = ax_wfall.get_xlim()
			# Build the line as nu vs t, then clip to the panel's TIME
			# limits. LOFAR ACF slopes are steep (dt/dnu ~ -19 ms/MHz),
			# so a line drawn across the full 78-MHz band sweeps ~1.5 s
			# in time -- far beyond the ~200 ms time axis. Solving the
			# line equation for nu at the panel's t-edges keeps only the
			# on-screen segment:  nu = center_f + (t - t0)/dtdnu.
			if _acf_dtdnu != 0 and np.isfinite(_acf_dtdnu):
				_nu_at_tlo = _acf_cf + (_t_lo - _acf_t0) / _acf_dtdnu
				_nu_at_thi = _acf_cf + (_t_hi - _acf_t0) / _acf_dtdnu
				# Intersect [nu_at_tlo, nu_at_thi] with the panel freq range
				# so the drawn segment is inside BOTH axes.
				_nu_seg_lo = max(min(_nu_at_tlo, _nu_at_thi), min(_nu_lo, _nu_hi))
				_nu_seg_hi = min(max(_nu_at_tlo, _nu_at_thi), max(_nu_lo, _nu_hi))
				if _nu_seg_hi > _nu_seg_lo:
					_nu_line = np.linspace(_nu_seg_lo, _nu_seg_hi, 200)
				else:
					# Line does not cross the panel; fall back to full range.
					_nu_line = np.linspace(_nu_lo, _nu_hi, 200)
			else:
				_nu_line = np.linspace(_nu_lo, _nu_hi, 200)
			_t_line = _acf_t0 + _acf_dtdnu * (_nu_line - _acf_cf)
			_acf_label = (f'{subburst_suffixes[idx]}. ACF ({_acf_src_word}): '
			              f'$dt/d\\nu =$ '
			              f'{scilabel(_acf_dtdnu, _acf_dtdnu_err)} ms/MHz')
			_acf_lines = ax_wfall.plot(
				_t_line, _nu_line,
				ls='--', color=_acf_line_color, lw=1.8, alpha=0.95,
				label=_acf_label, zorder=8,
			)
			if _acf_lines:
				_wfall_slope_lines[idx] = _acf_lines[0]
				_wfall_slope_labels[idx] = _acf_label

			# Fitted-ellipse geometry: use the ellipse_* fields computed
			# in acf_measure_component. These carry the RAW fit angle
			# (popt[5], measured from +time axis -- the correct matplotlib
			# Ellipse angle) and the burst-frame (/sqrt2) width/height, so
			# the ellipse tilt matches the slope line. Using the resolved
			# `theta` (the slope angle) here was the bug that made the
			# oval point the wrong way. Drawn at 2-sigma (n_sigma=2: the
			# stored ellipse_* widths are 2*sigma/sqrt2, i.e. 1-sigma-ish,
			# so x2 gives the 2-sigma ellipse).
			_acf_w  = 2.0 * _acf_r['ellipse_width_t']    # 2-sigma full width (time)
			_acf_h  = 2.0 * _acf_r['ellipse_height_nu']  # 2-sigma full height (freq)
			_acf_ell = _ACFEllipse(
				(_acf_t0, _acf_cf), width=_acf_w, height=_acf_h,
				angle=_acf_r['ellipse_angle_deg'],
				fill=False, edgecolor='white', lw=1.4, ls='--',
				alpha=0.9, zorder=7,
			)
			ax_wfall.add_patch(_acf_ell)
		# === END ACF ellipse overlay ===

	# Noise sample lines
	# s1 = np.floor(1*np.abs(s)/res_time_ms)
	for ns in noise_edges:
		for n in ns:
			ax_wfall.axvline(
				x=n*res_time_ms-pktime,
				c='r',
				ls='--'
			)

	if len(xos) > 1 and measure_drift:
		targetdf = pd.DataFrame(
			data=results,
			columns=results_columns
		).set_index('name')
		# --- NEW: drop components whose centroid measurement failed (NaN t0 /
		# center_f -- e.g. heavily-overlapping components the intensity fit could
		# not isolate). A single NaN anywhere in scipy.odr.RealData stops the ODR
		# from converging, so it silently returned the beta0=[-1,0] seed as
		# "Drift = (-1.0 +/- 0.0)". Fit only the valid components; skip the drift
		# line entirely if fewer than 2 survive (a 1-point drift is meaningless).
		_drift_cols = ['center_f (MHz)', 't0 (ms)', 'center_f_err', 't0_err']
		targetdf = targetdf.dropna(subset=_drift_cols)
		if len(targetdf) < 2:
			print(f"  [drift] {len(targetdf)} component(s) with valid centroids "
			      f"(need >=2); skipping Delta_t/Delta_nu drift fit.")
		else:
			odrjob = scipy.odr.ODR(
				scipy.odr.RealData(
					targetdf['center_f (MHz)'],
					targetdf['t0 (ms)']-pktime,
					sx=targetdf['center_f_err'],
					sy=targetdf['t0_err'],
				),
				scipy.odr.Model(lambda B, x: B[0]*x + B[1]),
				beta0=[-1, 0]
			)
			odrjob.set_job(fit_type=0)
			odrfit = odrjob.run()
			drift_redchisq = odrfit.res_var # https://stackoverflow.com/a/21406281/3133399
			drift, drift_err = odrfit.beta[0], np.sqrt(np.diag(odrfit.cov_beta))[0]

			ax_wfall.plot(
				times_ms-pktime,
				(1/drift)*(times_ms-pktime)+(-odrfit.beta[1]/drift),
				'r-.',
				label=f'Drift: $\\Delta t / \\Delta \\nu = ${scilabel(drift, drift_err)} ms/MHz',
			)
			ax_wfall.errorbar(
				targetdf['t0 (ms)']-pktime,
				targetdf['center_f (MHz)'],
				xerr=targetdf['t0_err'],
				yerr=targetdf['center_f_err'],
				fmt='rX',
				markeredgecolor='k'
			)

	# Test line for 1 parameter line model:
	# ax_wfall.plot(
	# 	times_ms-pktime,
	# 	(1/dtdnu2)*(times_ms-xoi)+pkfreq,
	# 	'y-.',
	# 	alpha=0.75,
	# 	# label=f'$dt/d\\nu = $ {dtdnu2:.2e} $\\pm$ {dtdnu_err:.2e}'
	# 	label=f'{subburst_suffixes[xos.index(xoi)]}. $dt/d\\nu =$ {scilabel(dtdnu2, dtdnu_err2)} ms/MHz'
	# )

	# === FIDUCIAL SCATTERING CURVES (per-component, clipped to time windows) ===
	# Initial render: all components' sub-band frequency dividers AND
	# fiducial scattering tails, each clipped to its own time region.
	# The toggle button (installed later) re-calls this with the
	# active component index for unclipped single-component view.
	_render_subband_overlays(None)
	# === END FIDUCIAL SCATTERING CURVES ===

	ax_wfall.set_title(f"{bname}")
	# Initial waterfall legend: ALL components' slope labels visible.
	# This matches the initial all-components-clipped sub-band overlay
	# view (toggle state = None). When the user clicks Toggle, the
	# handler filters to only the active component's label.
	# Drift line and scattering envelope are drawn outside
	# _wfall_slope_lines and are independent of this filter.
	for _ci, _ln in _wfall_slope_lines.items():
		_ln.set_label(_wfall_slope_labels[_ci])
	if not hide_legend: ax_wfall.legend(loc=legendloc, handlelength=0, fontsize=7)
	ax_wfall.set_xlabel('Time (ms)')
	ax_wfall.set_ylabel('Frequency (MHz)')

	# === Panel labels (top-left, just above each subplot) ===
	ax_wfall.text(
		0.0, 1.01, 'Dynamic spectrum',
		transform=ax_wfall.transAxes,
		ha='left', va='bottom', fontsize=9, color='black',
	)
	axs['T'].text(
		0.0, 1.02, 'Full-band intensity profile',
		transform=axs['T'].transAxes,
		ha='left', va='bottom', fontsize=9, color='black',
	)
	axs['S'].text(
		0.0, 1.01, 'Spectral profile',
		transform=axs['S'].transAxes,
		ha='left', va='bottom', fontsize=9, color='black',
	)
	# Slope-fitting label is added inside _render_slope_panel() below,
	# after ax_slope.clear() — otherwise the toggle-driven re-render
	# would wipe it. See the corresponding ax_slope.text() call near
	# the start of _render_slope_panel.
	# === END Panel labels ===

	ax_tseries = axs['T']
	ax_tseries.plot(times_ms-pktime, tseries)

	
	# plot filter windows (time): draw two underlines per component
	sp = 0
	if USE_EMG_MIX:
		# Purple underline at x_c with tail-aware halfwidth
		for s_std, xc in zip(tmix_sigmas_std, xcs):
			w_tail = t_filter_factor * np.abs(s_std)
			ax_tseries.add_patch(Rectangle(
				(xc - pktime - w_tail, ax_tseries.get_ylim()[0] + sp*(np.max(tseries)*0.075)),
				width=2*w_tail,
				height=np.max(tseries)*0.075,
				color='purple',
				alpha=0.5
			))
			if label_components:
				ax_tseries.annotate(
					f"{subburst_suffixes[xcs.index(xc)]}$_{{c}}$",
					(xc - pktime, ax_tseries.get_ylim()[0] + sp*(np.max(tseries)*0.075)),
				)
			sp += 1
	# Red underline at x_o (mu) with core halfwidth (works for both Gaussian and EMG)
	for s_core, xo in zip(tmix_sigmas, xos):
		w_core = t_filter_factor * np.abs(s_core)
		ax_tseries.add_patch(Rectangle(
			(xo - pktime - w_core, ax_tseries.get_ylim()[0] + sp*(np.max(tseries)*0.075)),
			width=2*w_core,
			height=np.max(tseries)*0.075,
			color='red',
			alpha=0.5
		))
		if label_components:
			ax_tseries.annotate(
				f"{subburst_suffixes[xos.index(xo)]}$_{{o}}$",
				(xo - pktime, ax_tseries.get_ylim()[0] + sp*(np.max(tseries)*0.075)),
			)
		sp += 1
	ax_tseries.set_ylim(ax_tseries.get_ylim()[0], np.max(tseries)*1.1)

	# === ON-PULSE / OFF-PULSE INDICATOR BAR ===
	# A thin coloured band beneath the x-axis of the full-band intensity
	# panel making the regions explicit:
	#   • Lime green = on-pulse window. Per-component intervals come
	#                  from the adaptive decay-fraction algorithm
	#                  (1% → 5% → 10% → 20%) in
	#                  `compute_onpulse_intervals_and_offpulse_mask`,
	#                  with a leading-edge σ pad. These are the SAME
	#                  intervals used to mask out signal when estimating
	#                  per-channel noise — so the bar faithfully shows
	#                  what the noise computation saw as on-pulse.
	#   • Bright orange = off-pulse. These bins fed the rescaled-MAD
	#                  per-channel noise estimate.
	# When per-component on-pulse intervals overlap (`_any_overlap`),
	# the bar uses cycling shades of green to make the overlap visible,
	# giving you a direct read on whether components are too tightly
	# packed for clean noise estimation.
	# Drawn with clip_on=False in (data-x, axes-y) coords so the bar sits
	# below the axes visible region but moves with the plot's x-axis. The
	# point is to make it qualitatively obvious why the off-pulse-noise
	# χ²_red can be inflated when the burst dominates many bins: any
	# signal-dependent extra scatter inside the lime-green window is
	# weighted by the much smaller orange-window σ_off, and that mismatch
	# shows up as χ²_red >> 1 even when the fit visually tracks the data.
	# === COMPONENT BOUNDARIES (for popup region-restriction and visual cues) ===
	# Component boundaries split the time axis into one region per sub-burst,
	# so the double-click popup can restrict its row slice to a single
	# component. Two sources:
	#   * Explicit `cuts` parameter (if supplied) — used verbatim.
	#   * Implicit fallback (this section) — midpoints between adjacent
	#     on-pulse segments of the multi-component intensity-profile fit.
	# Stored in UNSHIFTED ms coords (same frame as `xos` and `cuts`) so
	# downstream code that already works in the unshifted frame can use
	# them without further conversion. The popup converts to plot frame
	# (subtracts pktime) and to channel index (divides by res_time_ms) at
	# point of use.
	#
	# `_component_boundaries_ms` and `_comp_time_windows_ms` are now
	# computed once in the "EARLY PER-COMPONENT BOUNDARY /
	# TIME-WINDOW COMPUTATION" block above (before the figure is
	# created), so both the waterfall plot and the bar block read the
	# same lists.
	try:
		from matplotlib.transforms import blended_transform_factory
		_xlo_data = float(times_ms[0] - pktime)
		_xhi_data = float(times_ms[-1] - pktime)
		# Off-pulse x-window used by fitemg for σ estimation
		_off_n = max(len(tseries) // NOISE_WIDTH_FACTOR, 1)
		_off_lead_lo = _xlo_data
		_off_lead_hi = float(times_ms[_off_n - 1] - pktime) if _off_n - 1 < len(times_ms) else _xhi_data

		# Build on-pulse window from the fitted single/mixture EMG params
		# (or Gaussian if EMG is disabled). We source the parameters from
		# the multi-component time fit (tmix_popt) — same parameters that
		# generate the dashed red curve overlaid on this panel at line
		# ~6711, and the σ_t/τ values quoted in the title label. The
		# earlier 1D fit (_t1d_popt) sometimes converges to a degenerate
		# local minimum (especially for heavily-scattered single bursts
		# where σ-τ degeneracy is severe), which placed the green band
		# at the wrong location.
		#
		# Per-component on-pulse segments, merged segments, boundaries,
		# the decay helper, and time-windows are all computed in the
		# "EARLY PER-COMPONENT BOUNDARY / TIME-WINDOW COMPUTATION" block
		# before the main figure is created. This block just reads them.

		# === COMPONENT BOUNDARY LINES (on the waterfall) ===
		# Faint grey dashed vertical lines on the dynamic spectrum,
		# matching the style of the white sub-band boundary lines above
		# but vertical (axvline). Same boundaries that gate the popup,
		# so the user can see where clicks split between components.
		for _b_ms in _component_boundaries_ms:
			ax_wfall.axvline(
				float(_b_ms) - pktime,
				color='white', ls='--', alpha=0.4, lw=0.8,
			)
		# === END COMPONENT BOUNDARY LINES (waterfall) ===

		# Bar geometry: thin strip just below the x-axis spine, in
		# (data-x, axes-y) coords so it tracks the plot's x-axis but its
		# vertical position is independent of data y-limits.
		_bar_y_top    = -0.07    # top of bar, axes fraction (below tick marks)
		_bar_height   = 0.08     # bar height, axes fraction
		_bar_y_bottom = _bar_y_top - _bar_height
		_blended = blended_transform_factory(ax_tseries.transData, ax_tseries.transAxes)
		_off_color = '#FF8C1A'
		_on_color  = '#7CFC00'

		# On-pulse colour palette. When the on-pulse windows of adjacent
		# components overlap (`_any_overlap`), each component gets its
		# own shade of green so the user can see where one ends and the
		# next begins inside the merged region. Components cycle through
		# the palette in μ-order. Otherwise we keep the historical
		# single-lime look. Two shades are enough for the common
		# 2-component case; longer palettes are appended in case of
		# 3+ overlapping components.
		_on_color_palette = ['#7CFC00', '#4FA800', '#A6E84A', '#2F8B0F']

		# Track patches on the axes so the redraw callback can remove and
		# re-add them on every xlim change. Using a per-axes attribute
		# (rather than a closure-local list) lets the callback be picklable
		# / re-entrant-safe and survives any axes refresh round-trip.
		ax_tseries._pulse_bar_patches = []

		def _redraw_pulse_bars(ax):
			"""
			Redraw the on/off-pulse indicator bars clipped to the axes'
			current x-limits. Called once at setup and on every
			'xlim_changed' event (i.e. on every zoom/pan/set_xlim).
			"""
			# Remove any previously-drawn bar patches (avoid accumulation).
			for _p in getattr(ax, '_pulse_bar_patches', []):
				try:
					_p.remove()
				except (ValueError, NotImplementedError):
					# Patch already removed or detached; ignore.
					pass
			ax._pulse_bar_patches = []

			# Current view in plot-frame coords.
			_xv_lo, _xv_hi = ax.get_xlim()
			if _xv_lo > _xv_hi:
				_xv_lo, _xv_hi = _xv_hi, _xv_lo
			if _xv_hi <= _xv_lo:
				return

			# Clip MERGED on-pulse segments to the current view (used
			# for off-pulse coverage; orange = visible − union of
			# on-pulse, and the merged form already represents the
			# union).
			_on_vis = []
			for (a, b) in _on_segments_plot_full:
				_a = max(_xv_lo, a)
				_b = min(_xv_hi, b)
				if _b > _a:
					_on_vis.append((_a, _b))

			# Off-pulse coverage = visible window minus on-pulse union.
			_off_vis = []
			_cursor = _xv_lo
			for (a, b) in _on_vis:
				if a > _cursor:
					_off_vis.append((_cursor, a))
				_cursor = max(_cursor, b)
			if _cursor < _xv_hi:
				_off_vis.append((_cursor, _xv_hi))
			# Defensive: if no on-pulse segments survived (e.g. fit failed
			# or window doesn't overlap any segment), show entire visible
			# range as off-pulse so the bar is still visible.
			if not _off_vis and not _on_vis:
				_off_vis = [(_xv_lo, _xv_hi)]

			# Draw off-pulse (orange) first, then on-pulse green(s) on top.
			for (a, b) in _off_vis:
				_rect = Rectangle(
					(a, _bar_y_bottom), width=(b - a), height=_bar_height,
					transform=_blended, clip_on=False,
					facecolor=_off_color, edgecolor='none', alpha=0.95, zorder=5,
				)
				ax.add_patch(_rect)
				ax._pulse_bar_patches.append(_rect)

			if _any_overlap and _on_segments_per_comp_plot:
				# Overlapping case: draw one rectangle per component in
				# μ-order with cycling shades of green. Each rectangle
				# is fully opaque and drawn at an increasing zorder so
				# the later (higher-μ) component visibly takes over the
				# overlap region. The colour split lands at the later
				# component's on-pulse start, telling the user "this is
				# where component i+1 starts being on-pulse". The
				# black popup-boundary tick (drawn later, higher zorder)
				# may sit slightly to the LEFT of that colour split —
				# that's expected: the bar tracks the FIT's on-pulse
				# windows, the tick tracks the POPUP's region split
				# (which uses a decay-midpoint rule, not the on-pulse
				# start).
				for _ci, (a, b) in enumerate(_on_segments_per_comp_plot):
					_a = max(_xv_lo, a)
					_b = min(_xv_hi, b)
					if _b <= _a:
						continue
					_col = _on_color_palette[_ci % len(_on_color_palette)]
					_rect = Rectangle(
						(_a, _bar_y_bottom), width=(_b - _a), height=_bar_height,
						transform=_blended, clip_on=False,
						facecolor=_col, edgecolor='none', alpha=1.0,
						zorder=6 + _ci,
					)
					ax.add_patch(_rect)
					ax._pulse_bar_patches.append(_rect)
			else:
				# Well-resolved (no overlap) or single-component case:
				# uniform lime-green rectangles per merged segment.
				for (a, b) in _on_vis:
					_rect = Rectangle(
						(a, _bar_y_bottom), width=(b - a), height=_bar_height,
						transform=_blended, clip_on=False,
						facecolor=_on_color, edgecolor='none', alpha=0.95, zorder=6,
					)
					ax.add_patch(_rect)
					ax._pulse_bar_patches.append(_rect)

			# === COMPONENT BOUNDARY LINES (on the bar) ===
			# A thin black vertical tick spanning the bar height at each
			# component boundary, so the user can see exactly where the
			# popup will split clicks between sub-bursts.
			for _b_ms in _component_boundaries_ms:
				_b_plot = float(_b_ms) - pktime
				if _xv_lo <= _b_plot <= _xv_hi:
					_line = ax.plot(
						[_b_plot, _b_plot],
						[_bar_y_bottom, _bar_y_top],
						transform=_blended, clip_on=False,
						color='black', linewidth=1.2, zorder=20,
					)[0]
					ax._pulse_bar_patches.append(_line)
			# === END COMPONENT BOUNDARY LINES ===

		# Initial draw, then hook into x-limit changes.
		_redraw_pulse_bars(ax_tseries)
		ax_tseries.callbacks.connect('xlim_changed', _redraw_pulse_bars)
	except Exception as _bar_e:
		# Non-fatal: the indicator bar is purely informational. Skip
		# silently if anything in this block fails (e.g. tmix_popt is
		# None for a degenerate fit).
		pass
	# === END ON-PULSE / OFF-PULSE INDICATOR BAR ===


#Gaussian plot
# 	ax_tseries.plot(
# 		times_ms-pktime,
# 		gauss_model(
# 			times_ms-pktime,
# 			np.max(tseries)*t_popt[0],
# 			t_popt[1]-pktime,
# 			t_popt[2]
# 		),
# 		'k--',
# 		alpha=0.1
# 	)

	# === MCMC FIT PLOT ===
	if USE_MCMC and _mcmc_results is not None and _mcmc_results["status"] == "success":
		try:
			_mcmc_A, _mcmc_mu, _mcmc_sigma, _mcmc_lamb = _mcmc_results["median"]
			_mcmc_baseline = _mcmc_results["baseline"]
			_mcmc_tau = _mcmc_results["tau"]
			_mcmc_tau_err_low = _mcmc_results["tau_err_low"]
			_mcmc_tau_err_high = _mcmc_results["tau_err_high"]
			_mcmc_sigma_err_low = _mcmc_results["err_low"][2]
			_mcmc_sigma_err_high = _mcmc_results["err_high"][2]
			
			# Shift for plotting
			_t_plot = times_ms - pktime
			_mcmc_curve = emg_model(_t_plot, _mcmc_A, _mcmc_mu - pktime, _mcmc_sigma, _mcmc_lamb, _mcmc_baseline)
			
			# Plot MCMC fit (blue dashed)
			ax_tseries.plot(
				_t_plot, _mcmc_curve,
				linestyle="--", alpha=0.8, color="blue",
				label=(
					f"MCMC: $\\sigma_t$ = {scilabel(_mcmc_sigma, (_mcmc_sigma_err_low + _mcmc_sigma_err_high)/2)} ms, "
					f"$\\tau$ = {scilabel(_mcmc_tau, (_mcmc_tau_err_low + _mcmc_tau_err_high)/2)} ms"
				)
			)
			
			# Plot 3-sigma region (light blue fill)
			_mask_3sigma = np.abs(_t_plot - (_mcmc_mu - pktime)) <= 3 * _mcmc_sigma
			ax_tseries.fill_between(
				_t_plot, 0, _mcmc_curve,
				where=_mask_3sigma, alpha=0.3, color="lightblue",
				label="MCMC ±3σ region"
			)
		except Exception as e:
			print(f"[DEBUG] MCMC plot failed: {e}")
	# === END MCMC PLOT ===

	
# 	# --- Reconstituted Gaussian (green) ---
# 	if USE_EMG_MIX and (n_bursts == 1) and hasattr(_t1d_popt, "__len__") and len(_t1d_popt) >= 5:
# 		try:
# 			A, mu, sig_core, lam, C = _t1d_popt[:5]
# 			_t = times_ms - pktime
# 			_emg_curve  = emg_model(_t, A, mu - pktime, sig_core, lam, C)
# 			_peak_above = float(np.nanmax(_emg_curve - C))
# 			if np.isfinite(_peak_above) and _peak_above > 0 and np.isfinite(sig_core):
# 				reconG = C + gauss_model(_t, _peak_above, mu - pktime, abs(sig_core))
# 				ax_tseries.plot(_t, reconG, linestyle=':', color='green', linewidth=1.0,
#                                 zorder=20, label='reconstituted Gaussian')
# 		except Exception as e:
# 			print("[DEBUG] reconG(single):", repr(e))

	

#  	# Gaussian mix model
#  	tmix_popt[:n_bursts] = [a*np.max(tseries) for a in tmix_amps]
#  	tmix_popt[n_bursts:n_bursts*2] = [x-pktime for x in tmix_xos]
#  	ax_tseries.plot(
# 		times_ms-pktime,
# 		gaussmix_model(
#  			times_ms-pktime,
#  			*tmix_popt
# 		),
# 		'k--',
# 		alpha=0.8,
# 		label=( # because of the loop above, this is implicitly for the last component
#  			f"$\\sigma_t =$ {scilabel(sigma, sigma_err)} ms, $\\chi^2_{{\\text{{red}}}} =${tmix_redchisq:.2f}"
# 		),
#  	)
#  	ax_tseries.legend(frameon=False, fontsize=12, handlelength=0)


	# --- (A) Gaussian-mix overlay (unchanged behavior) ---
	tmix_popt_g = tmix_popt.copy()
	# rescale amplitudes and shift centres to match your plot axis
	tmix_popt_g[:n_bursts] = [a*np.max(tseries) for a in tmix_amps]
	tmix_popt_g[n_bursts:2*n_bursts] = [x - pktime for x in tmix_xos]
	
# 	ax_tseries.plot(
#         times_ms - pktime,
#         gaussmix_model(times_ms - pktime, *tmix_popt_g),
#         linestyle='--', color='gray', alpha=0.9, label=f"Gauss-mix  $\\chi^2_{{\\rm red}}={tmix_redchisq:.2f}$"
#     )
	
	# --- (B) EMG-mix overlay (fit on the fly; plot only) ---
# 	try:
#         # reuse the same component centres/sigmas as seeds; keep xo optionally fixed
# 		emg_popt, emg_pcov, emg_redchisq = fitemgmix(
#             tseries, duration,
#             xos=(tmix_xos.tolist() if hasattr(tmix_xos, 'tolist') else list(tmix_xos)),
#             sigmas=(tmix_sigmas.tolist() if hasattr(tmix_sigmas, 'tolist') else list(tmix_sigmas)),
#             fix_xos=fix_xos, tol=tolms
#         )
	
#         # rescale like your Gaussian path: A and baseline C back to tseries units; shift xo by pktime
# 		emg_plot = emg_popt.copy()
# 		emg_plot[:n_bursts] = [A*np.max(tseries) for A in emg_plot[:n_bursts]]
# 		emg_plot[n_bursts:2*n_bursts] = [xo - pktime for xo in emg_plot[n_bursts:2*n_bursts]]
# 		idx_C = 4 * n_bursts                         # C is the last term in EMG-mix param vector
# 		if idx_C < len(emg_plot):
# 			emg_plot[idx_C] = emg_plot[idx_C] * np.max(tseries)
	
# 		ax_tseries.plot(
#             times_ms - pktime,
#             emgmix_model(times_ms - pktime, *emg_plot),
#             linestyle='--', color='red', alpha=0.9, label=f"EMG-mix  $\\chi^2_{{\\rm red}}={emg_redchisq:.2f}$"
#         )
# 	except Exception as _:
#         # If emgmix code isn't present, we silently skip the EMG overlay
# 		pass
	
	ax_tseries.legend(frameon=False, fontsize=12, handlelength=0)


	# EMG mix model (plot)
	tmix_popt_plot = tmix_popt.copy()
	# rescale amplitudes and baseline back to original tseries units
	tmix_popt_plot[:n_bursts] = [a*np.max(tseries) for a in tmix_amps]
	tmix_popt_plot[n_bursts:2*n_bursts] = [x - pktime for x in tmix_xos]  # shift xo
	# C (shared baseline) sits at the end of the parameter vector. Its
	# index depends on the model: EMG layout is [A..A, μ..μ, σ..σ, λ..λ, C]
	# (length 4n+1), Gaussian layout is [A..A, μ..μ, σ..σ, C] (length 3n+1).
	# Previously this was hard-coded to 4*n_bursts, which silently no-op'd
	# for the Gaussian case (condition idx_C < len() was False) and left
	# C in normalized units when displayed downstream — so the dashed
	# fit line sat at a small but visibly non-zero baseline.
	idx_C = (4 if USE_EMG_MIX else 3) * n_bursts
	if idx_C < len(tmix_popt_plot):
		tmix_popt_plot[idx_C] = tmix_popt_plot[idx_C] * np.max(tseries)
		
	# Build per-component σ_t / τ label data for the toggle button. The
	# label rendered in the intensity-profile legend depends on which
	# component is currently selected; we capture the per-component
	# values once here so the button's click handler can reconstruct
	# the label string without re-running the fit.
	#
	# σ_t per component comes from tmix_popt[2*n_bursts : 3*n_bursts];
	# σ_err from tmix_perr similarly. τ per component from λ (EMG only).
	# χ²_red stays global (per user spec).
	_tmix_sigmas_arr = np.atleast_1d(tmix_popt[2*n_bursts:3*n_bursts])
	_tmix_sigma_errs_arr = (
		np.atleast_1d(tmix_perr[2*n_bursts:3*n_bursts])
		if isinstance(tmix_perr, (list, np.ndarray))
		else np.full(n_bursts, np.nan)
	)
	if USE_EMG_MIX:
		_tmix_lams_arr = np.atleast_1d(tmix_popt[3*n_bursts:4*n_bursts])
		_tmix_lam_errs_arr = (
			np.atleast_1d(tmix_perr[3*n_bursts:4*n_bursts])
			if isinstance(tmix_perr, (list, np.ndarray))
			else np.full(n_bursts, np.nan)
		)
	else:
		_tmix_lams_arr = None
		_tmix_lam_errs_arr = None

	def _intensity_label_for_comp(ci):
		"""Build the σ_t / τ / χ²_red legend string.

		ci=None ⇒ multi-component "all" view: every σ_t and every τ
		comma-separated. ci=int ⇒ single component's values.
		χ²_red is the global mixture-fit value in both cases.

		When the bounded EMG fit pinned λ at 1/dt_ms (boundary-pile-up
		detected upstream → λ NaN'd), the component's τ is "unresolved"
		— the data couldn't constrain it. We annotate the τ field with
		"unresolved" rather than silently dropping it, so the legend
		honestly tells the user the EMG collapsed to Gaussian-like.
		"""
		def _tau_field_for(_i):
			"""Return one of: scilabel string, 'unresolved', or None.
			   None means "this isn't an EMG fit, omit τ entirely"."""
			if not (USE_EMG_MIX and _tmix_lams_arr is not None
			        and _i < len(_tmix_lams_arr)):
				return None
			_lam = _tmix_lams_arr[_i]
			if np.isfinite(_lam) and _lam > 0:
				_tau = 1.0 / _lam
				_lam_err = (_tmix_lam_errs_arr[_i]
				            if _i < len(_tmix_lam_errs_arr) else np.nan)
				_tau_err = (_lam_err / (_lam**2)) if np.isfinite(_lam_err) else np.nan
				return scilabel(_tau, _tau_err)
			# NaN λ → boundary-pile-up flagged it. τ is unresolved.
			return "unresolved"

		if ci is None:
			# All-components view: comma-separated lists.
			_sig_strs = []
			for _i in range(len(_tmix_sigmas_arr)):
				_s = float(abs(_tmix_sigmas_arr[_i]))
				_se = float(abs(_tmix_sigma_errs_arr[_i])) if _i < len(_tmix_sigma_errs_arr) else np.nan
				_sig_strs.append(scilabel(_s, _se))
			_tau_strs = []
			for _i in range(len(_tmix_sigmas_arr)):
				_field = _tau_field_for(_i)
				if _field is not None:
					_tau_strs.append(_field)
			_sig_part = (r"$\sigma_t =$ " + ", ".join(_sig_strs) + " ms") if _sig_strs else ""
			# Only the EMG branch emits a τ part. If every τ is "unresolved",
			# write the label without the per-value " ms" suffix (it would
			# look like "τ = unresolved ms"); if any τ has a real value, the
			# " ms" applies — accept the cosmetic imperfection on mixed cases.
			if _tau_strs:
				_all_unresolved = all(t == "unresolved" for t in _tau_strs)
				_tau_suffix = "" if _all_unresolved else " ms"
				_tau_part = ",  " + r"$\tau$ = " + ", ".join(_tau_strs) + _tau_suffix
			else:
				_tau_part = ""
			return f"{_sig_part}{_tau_part}, $\\chi^2_{{\\text{{red}}}} =${tmix_redchisq:.2f}"
		# Single-component view
		_sig = float(abs(_tmix_sigmas_arr[ci])) if ci < len(_tmix_sigmas_arr) else np.nan
		_sig_err = float(abs(_tmix_sigma_errs_arr[ci])) if ci < len(_tmix_sigma_errs_arr) else np.nan
		_tau_str = ""
		_field = _tau_field_for(ci)
		if _field == "unresolved":
			_tau_str = ",  " + r"$\tau$ = unresolved"
		elif _field is not None:
			_tau_str = ",  " + r"$\tau$ = " + _field + " ms"
		return (f"$\\sigma_t =$ {scilabel(_sig, _sig_err)} ms{_tau_str}, "
		        f"$\\chi^2_{{\\text{{red}}}} =${tmix_redchisq:.2f}")

	# Capture the mixture-fit Line2D so the toggle handler can rewrite
	# its label and re-render the legend. Initial render uses the
	# multi-component "all" view (ci=None), which matches the initial
	# all-components-clipped sub-band overlay rendering. The toggle
	# button switches to single-component labels and cycles back to
	# this all-components label.
	#
	# === HARD-CLIPPED STITCHED-LINE DISPLAY (HARD-ISOLATION MODE) ===
	# When `cuts` is supplied the per-region fits use σ/τ bounded by
	# region width, so each component lives strictly inside its own
	# region. The displayed model evaluates each component on the
	# full time grid and then NaN-clips outside its region, so the
	# matplotlib line breaks cleanly at the cut boundaries — i.e. the
	# fit doesn't visually bleed across the hard-isolation cuts. The
	# single dashed line carries the σ_t / τ / χ²_red legend (one
	# entry, matching the legacy joint-fit display) and serves as
	# the toggle handler's target.
	#
	# Falls back to plain `MIX_MODEL_FUNC(...)` when cuts is empty
	# (legacy joint mixture fit — no per-region clipping needed).

	# Per-component intensity line handles (kept for toggle handler
	# API compatibility — empty in this single-dashed-line design,
	# but the toggle iterates over this dict so we keep it as an
	# empty dict for the no-op branch).
	_per_comp_intensity_lines = {}

	# Hard-isolation mode active iff cuts was non-empty.
	_hard_iso_for_plot = (cuts is not None and len(cuts) > 0)

	if _hard_iso_for_plot:
		# Build region edges in plot frame and a single NaN-clipped
		# stitched model that matches the fit's hard isolation but
		# renders as one dashed line.
		_t_eval_shift = times_ms - pktime
		_t_min_plot = float(_t_eval_shift[0])
		_t_max_plot = float(_t_eval_shift[-1])
		_cuts_plot = sorted(float(c) - pktime for c in cuts)
		_region_edges_plot = [_t_min_plot] + _cuts_plot + [_t_max_plot]

		# Unpack parameters from tmix_popt_plot (display units, μ
		# already shifted to plot frame by the earlier rescaling
		# block — see line ~9760).
		_As_p   = np.atleast_1d(tmix_popt_plot[0:n_bursts])
		_mus_p  = np.atleast_1d(tmix_popt_plot[n_bursts:2*n_bursts])
		_sgs_p  = np.atleast_1d(tmix_popt_plot[2*n_bursts:3*n_bursts])
		if USE_EMG_MIX:
			_lms_p = np.atleast_1d(tmix_popt_plot[3*n_bursts:4*n_bursts])
			_C_idx = 4 * n_bursts
		else:
			_lms_p = None
			_C_idx = 3 * n_bursts
		_C_p    = float(tmix_popt_plot[_C_idx]) if (_C_idx < len(tmix_popt_plot)) else 0.0

		# Sort components by μ to assign each one a region position.
		_sort_idx = sorted(range(n_bursts), key=lambda i: float(_mus_p[i]))
		_region_of_orig = [None] * n_bursts
		for j, orig_i in enumerate(_sort_idx):
			_region_of_orig[orig_i] = j

		# Stitched-fit accumulator. NaN outside all regions; inside a
		# region it holds (component value at that t) + C_p, so the
		# line lives at the correct absolute level.
		_stitched = np.full_like(_t_eval_shift, np.nan, dtype=float)

		for i in range(n_bursts):
			j = _region_of_orig[i]
			if j is None or j >= len(_region_edges_plot) - 1:
				continue
			_r_lo = _region_edges_plot[j]
			_r_hi = _region_edges_plot[j + 1]
			_sg_i = float(_sgs_p[i]) if (np.isfinite(_sgs_p[i]) and _sgs_p[i] > 0) else None
			if _sg_i is None:
				continue

			if USE_EMG_MIX:
				_lm_i = float(_lms_p[i]) if (np.isfinite(_lms_p[i]) and _lms_p[i] > 0) else None
				if _lm_i is None:
					continue
				_comp_vals_full = emg_model(_t_eval_shift, float(_As_p[i]),
				                              float(_mus_p[i]), _sg_i, _lm_i, 0.0)
			else:
				_comp_vals_full = gauss_model(_t_eval_shift, float(_As_p[i]),
				                                float(_mus_p[i]), _sg_i)

			_in_region = (_t_eval_shift >= _r_lo) & (_t_eval_shift <= _r_hi)
			# Regions don't overlap by construction, so the nan-aware
			# update is equivalent to a sum over regions.
			_stitched = np.where(_in_region, _comp_vals_full + _C_p, _stitched)

		_intensity_model_vals = _stitched
		_intensity_model_t   = times_ms - pktime
	else:
		# Smooth fine-grid evaluation (20× the data resolution) so the
		# dashed model line is visually smooth even when σ is narrow
		# enough to render as line segments on the data grid.
		_fine_factor_mix = 20
		_intensity_model_t = np.linspace(
			times_ms[0] - pktime, times_ms[-1] - pktime,
			_fine_factor_mix * len(times_ms),
		)
		# --- Per-component curve build, with τ-unresolved fallback ---
		# If any EMG component had its λ NaN'd upstream (boundary-pile-up:
		# the bounded fit pinned λ at 1/dt_ms because τ was unresolved by
		# the data), MIX_MODEL_FUNC would evaluate to NaN at every point
		# and the dashed red curve would disappear entirely. Instead, for
		# any such component we draw the EMG at its ORIGINAL (pre-NaN)
		# bounded λ — i.e. the curve the optimizer actually found. That's
		# the right shape to show alongside the "τ = unresolved"
		# annotation: σ is reliable, τ is at the resolution limit, so the
		# EMG collapses to a near-Gaussian peak with a one-bin scattering
		# tail. The pre-NaN λ values live in `_tmix_lams_orig` (set by
		# the boundary-pile-up detector upstream); when that's None (e.g.
		# load_solutions branch, or no NaN-ing happened), we fall back to
		# the standard MIX_MODEL_FUNC path.
		_have_nan_lam = (
			USE_EMG_MIX and len(tmix_popt_plot) == 4*n_bursts + 1
			and np.any(~np.isfinite(np.atleast_1d(tmix_popt_plot[3*n_bursts:4*n_bursts])))
		)
		if _have_nan_lam and _tmix_lams_orig is not None:
			_t_A_p     = np.atleast_1d(tmix_popt_plot[0          : n_bursts  ])
			_t_mu_p    = np.atleast_1d(tmix_popt_plot[n_bursts   : 2*n_bursts])
			_t_sig_p   = np.atleast_1d(tmix_popt_plot[2*n_bursts : 3*n_bursts])
			_t_lam_p   = np.atleast_1d(tmix_popt_plot[3*n_bursts : 4*n_bursts])
			_t_C_p     = float(tmix_popt_plot[4*n_bursts])
			_intensity_model_vals = np.full_like(_intensity_model_t, _t_C_p, dtype=float)
			for _i in range(n_bursts):
				_Ai = float(_t_A_p[_i])
				_mui = float(_t_mu_p[_i])
				_si = float(_t_sig_p[_i])
				_li = float(_t_lam_p[_i])
				# Recover bounded λ when the current one was NaN'd.
				if not (np.isfinite(_li) and _li > 0):
					if _i < len(_tmix_lams_orig):
						_li = float(_tmix_lams_orig[_i])
				if np.isfinite(_li) and _li > 0 and np.isfinite(_si) and _si > 0:
					_intensity_model_vals = _intensity_model_vals + emg_model(
						_intensity_model_t, _Ai, _mui, _si, _li, 0.0
					)
				# else: σ also invalid, skip this component entirely.
		else:
			_intensity_model_vals = MIX_MODEL_FUNC(_intensity_model_t, *tmix_popt_plot)
		# --- end τ-unresolved fallback ---

	# One dashed line for the full model (red for EMG, black for
	# Gaussian) with the legacy σ_t / τ / χ²_red legend. The toggle
	# handler updates this line's label when the user clicks a
	# component (single-component view).
	_mix_line_handles = ax_tseries.plot(
		_intensity_model_t,
		_intensity_model_vals,
		linestyle='--', alpha=0.8, color=('red' if USE_EMG_MIX else 'black'),
		label=_intensity_label_for_comp(None),
	)
	_mix_line = _mix_line_handles[0] if _mix_line_handles else None
	
	# --- Reconstituted Gaussian per EMG component (green, dotted) ---
# 	try:
# 		if USE_EMG_MIX and n_bursts >= 1 and isinstance(tmix_popt, (list, tuple, np.ndarray)):
# 			_t = times_ms - pktime
	
#             # fitemgmix() normalized the time series to max=1; rescale A_i and C back
#             # to the original time-series units like elsewhere in the code.
#             # A_i -> A_i * max(tseries); Cmix is the last term in the EMG-mix vector.
# 			A_slice   = np.atleast_1d(tmix_amps) * np.max(tseries)
# 			mu_slice  = np.atleast_1d(tmix_xos)          # absolute time (ms)
# 			sig_slice = np.atleast_1d(tmix_sigmas)
# 			lam_slice = np.atleast_1d(tmix_lambdas)
# 			C_idx     = 4 * n_bursts
# 			Cmix      = float(tmix_popt[C_idx]) * np.max(tseries) if (C_idx < len(tmix_popt)) else 0.0
	
# 			first_label_done = False
# 			for Ai, mui, si, li in zip(A_slice, mu_slice, sig_slice, lam_slice):
# 				if not (np.isfinite(Ai) and np.isfinite(mui) and np.isfinite(si) and np.isfinite(li)):
# 					continue
	
# 				mu_shift = float(mui - pktime)
	
#                 # Component-only EMG above baseline (set C=0 so we get the pure component).
# 				comp_emg = emg_model(_t, Ai, mu_shift, si, li, 0.0)
# 				peak_above = float(np.nanmax(comp_emg))
# 				if not (np.isfinite(peak_above) and peak_above > 0.0):
# 					continue
	
#                 # Reconstituted Gaussian for this component:
#                 # centered at mu (x_o), width = sigma (core), peak height = EMG component peak above baseline
# 				recon_i = Cmix + gauss_model(_t, peak_above, mu_shift, abs(si))
	
# 				ax_tseries.plot(
#                     _t, recon_i,
#                     linestyle=':', linewidth=1.0, color='green', alpha=0.95, zorder=20,
#                     label=('reconstituted Gaussian' if not first_label_done else None)
#                 )
# 				first_label_done = True
# 	except Exception as e:
# 		print("[DEBUG] reconG-mix: exception while plotting:", repr(e))

	
	
	ax_tseries.legend(frameon=False, fontsize=6, handlelength=0)

	_snr_off_n = max(len(tseries) // NOISE_WIDTH_FACTOR, 1)
	_snr_noise = np.std(tseries[:_snr_off_n])
	_snr_val   = float(np.max(tseries)) / _snr_noise if _snr_noise > 0 else np.nan
	ax_tseries.text(
		0.02, 0.97,
		f"S/N = {_snr_val:.1f}",
		transform=ax_tseries.transAxes,
		ha='left', va='top', fontsize=7, color='black',
	)

	### Summed Spectrum (summed over burst widths). Total and individual
	downband = 1
	if len(bandpass) % downband != 0:
		downband = smallestdivisor(len(bandpass))
	bandpass_down = bandpass.reshape(-1, downband).mean(axis=1)
	axs['S'].stairs(
		bandpass_down,
		np.linspace(*extent[2:], num=len(bandpass_down)+1),
		orientation='horizontal',
		label=( # because of the loop above, this is implicitly for the last component
			# f"$\\sigma_{{\\nu}} =$ {scilabel(bwidth, bwidth_err)} MHz"
			f"\n$\\chi^2_{{\\text{{red}}}} =${subband_redchisq:.2f}"
		),
		# lw=2
	)
	axs['S'].legend(frameon=False, fontsize=12, handlelength=0)
	for subbandmodel in subbandmodels:
		axs['S'].plot(
			subbandmodel,
			freqs,
			'k--',
			alpha=0.5,
			zorder=-1,
		)

	# DIAGNOSTIC OVERLAY (disabled): the inner spectral Gaussian fit from
	# fit_subband_alpha, plotted as a magenta dashed curve. Used during
	# development to confirm that the inner and outer spectral fits can
	# diverge for broadband bursts. Re-enable by uncommenting the block
	# below if you need to visually compare the two fits again.
	# if (_sb_alpha_result is not None
	#         and _sb_alpha_result.get('spec_popt_inner') is not None):
	# 	_inner_a, _inner_c, _inner_s = _sb_alpha_result['spec_popt_inner']
	# 	axs['S'].plot(
	# 		gauss_model(freqs, _inner_a, _inner_c, _inner_s),
	# 		freqs,
	# 		color='magenta',
	# 		ls='--',
	# 		alpha=0.7,
	# 		zorder=-1,
	# 		label=f'inner fit μ={_inner_c:.1f}, σ={_inner_s:.1f}',
	# 	)

	for bandmask_thresi, subband in zip(bandmask_thres, subbands):
		if bandmask_thresi:
			mask_freqs = freqs[(subband > bandmask_thresi).nonzero()]
			axs['S'].scatter(
				bandpass_down[subband > bandmask_thresi],
				mask_freqs,
				color='r',
				marker='x',
				s=10,
			)
			if len(mask_freqs) > 0: # technically region where SOME channels are masked
				tmplims = axs['S'].get_xlim()
				axs['S'].fill_betweenx(
					mask_freqs[[0,-1]],
					*(10*np.array(axs['S'].get_xlim())),
					hatch='x',
					color='r',
					alpha=0.05
				)
				axs['S'].set_xlim(tmplims)

	# plot 1 sigma burst bandwidth region
	for subbandpopt in subbandpopts:
		pkfreq, bw_1σ = subbandpopt[1], subbandpopt[2]

		tmplims = axs['S'].get_xlim() # hack for region not extending all the way
		axs['S'].fill_betweenx(
			[pkfreq-bw_1σ, pkfreq+bw_1σ],
			*(10*np.array(axs['S'].get_xlim())),
			zorder=-20,
			alpha=0.75,
			color='cyan'
		)
		axs['S'].axhline(
			y=pkfreq,
			ls='-.',
			zorder=-19
		)
		axs['S'].set_xlim(tmplims)

	# === bw_manual BAND HIGHLIGHT ON SPECTRAL PROFILE ===============
	# When the full-band intensity profile was integrated over a manually
	# pinned frequency band (bw_manual=(freq_lo, freq_hi) MHz), shade that
	# band in cyan on the spectral profile so it's visually obvious which
	# channels fed the duration/tau fit. Mirrors the cyan 1σ spectral-
	# Gaussian region drawn just above, but is bounded by dash-dot cyan
	# edge lines to mark it as a MANUAL band (not a fitted σ). `_bw_manual_band`
	# is resolved up in the tseries-integration block (None when bw_manual
	# is not supplied), so this is a no-op for the default full-band path.
	if _bw_manual_band is not None:
		_bm_lo_plot, _bm_hi_plot = _bw_manual_band
		_bm_tmplims = axs['S'].get_xlim()  # preserve x-limits (fill spans 10× the range)
		axs['S'].fill_betweenx(
			[_bm_lo_plot, _bm_hi_plot],
			*(10*np.array(axs['S'].get_xlim())),
			zorder=-18,
			alpha=0.5,
			color='cyan',
			label=f'bw_manual [{_bm_lo_plot:.0f}, {_bm_hi_plot:.0f}] MHz',
		)
		for _bm_edge in (_bm_lo_plot, _bm_hi_plot):
			axs['S'].axhline(
				y=_bm_edge,
				ls='-.',
				color='cyan',
				lw=1.2,
				zorder=-17,
			)
		axs['S'].set_xlim(_bm_tmplims)
	# === END bw_manual BAND HIGHLIGHT ===============================

	# plot filter windows (frequency). One rectangle per component on
	# the spectral profile panel, suppressed for any component whose
	# bandwidth was overridden manually (those don't have a
	# pkfreq±bw·σ window — the override IS the window).
	if bw_filter == 'model_width':
		for i, subbandpopt in enumerate(subbandpopts):
			_co_i = (
				_bw_channel_override_per_comp[i]
				if i < len(_bw_channel_override_per_comp) else None
			)
			if _co_i is not None:
				continue
			pkfreq, bw = subbandpopt[1], bw_width_factor*subbandpopt[2]
			rectwidth = np.max(bandpass_down)*0.035
			axs['S'].add_patch(Rectangle(
				(0.025+axs['S'].get_xlim()[0] + i*(rectwidth+0.025), pkfreq-bw),
				height=2*bw,
				width=rectwidth,
				color='tomato',
				alpha=0.5
			))

	axs['T'].sharex(axs['A'])
	axs['A'].sharey(axs['S'])
	axs['S'].set_xlabel('Intensity (arb.)')
	ax_wfall.set_xlim(extent[0], extent[1])
	ax_wfall.set_ylim(extent[2], extent[3])

	plt.setp(ax_tseries.get_xticklabels(), visible=False)
	plt.setp(axs['S'].get_yticklabels(), visible=False) # For paper

#SLOPE
	### Slope measurement plot.
	# pidx is already resolved earlier (after fig creation) so the
	# intensity-profile legend can use it for the per-component σ_t/τ
	# label. Toggle button installs a click handler that updates pidx
	# at runtime.

	# === BOTTOM SUBPLOT DIAGNOSTIC ===
	if diag_enabled('BOTTOM_SUBPLOT'):
		print(f"\n=== BOTTOM SUBPLOT (SLOPE) DIAGNOSTIC ===")
		print(f"  pslopeidx (input) = {pslopeidx}")
		print(f"  pidx (after fallback) = {pidx}")
		print(f"  len(xos) = {len(xos)}, len(subdfs) = {len(subdfs)}")
		print(f"  xos = {xos}")
		print(f"  xos[pidx] = {xos[pidx]}")
		print(f"  subdfs[pidx] has {len(subdfs[pidx])} rows")
		for i, sdf in enumerate(subdfs):
			print(f"    subdfs[{i}] ({subburst_suffixes[i]}): {len(sdf)} TOAs")
	# === END BOTTOM SUBPLOT DIAGNOSTIC ===
	
	ax_slope = axs['E']

	def _render_slope_panel(ci):
		"""(Re)render the slope-fitting panel for component index `ci`.

		Clears `ax_slope` first so the function is safe to call from
		the toggle-button handler. All per-component state is derived
		from `ci`; nothing on the slope panel survives between calls.
		"""
		# Clear the panel completely. ax_slope.clear() resets axes
		# limits, title, labels, ticks, all artists — we re-establish
		# what we need below.
		ax_slope.clear()

		# Panel label "Slope fitting" in the top-left corner of the
		# bottom subplot (re-added after every clear so it survives
		# the toggle-driven re-render).
		ax_slope.text(
			0.0, 1.05, 'Slope fitting',
			transform=ax_slope.transAxes,
			ha='left', va='bottom', fontsize=9, color='black',
		)

		# === ACF MODE: replace the slope panel with the ACF view ===
		# When ACF mode ran for this component, the bottom panel shows
		# the 2D autocorrelation with the fitted-Gaussian contours
		# instead of the per-channel ToA scatter + line fit. The helper
		# also handles the fit-failure case (prints a message in-panel).
		if (ci < len(_comp_acf_results)
		        and _comp_acf_results[ci] is not None):
			_acf_panel_lbl = (subburst_suffixes[ci]
			                  if ci < len(subburst_suffixes) else str(ci))
			acf_slope.draw_acf_panel(
				ax_slope, _comp_acf_results[ci],
				comp_label=_acf_panel_lbl,
			)
			# Re-add the "Slope fitting" panel label (draw_acf_panel
			# calls ax.clear()).
			ax_slope.text(
				0.0, 1.05, 'Slope fitting (ACF mode)',
				transform=ax_slope.transAxes,
				ha='left', va='bottom', fontsize=9, color='black',
			)
			return
		# === END ACF MODE ===

		# === DEFENSIVE EMPTY-COMPONENT GUARD ===
		# When `subband_diagnostic` skips a component, its entry in
		# `subdfs` is the empty `pd.DataFrame()` stub pushed by the
		# diagnostic skip block — no columns. Same applies to any
		# other code path that pushes empty stubs (e.g. a component
		# whose per-channel ToA fits all failed quality cuts so
		# `subdfs[ci]` came back empty from `fitrowsemg`). Indexing
		# `_plotdf['freqs']` on such a stub raises KeyError. We can't
		# render a slope panel without data; paint an explanatory
		# message and return.
		_has_subdf_rows = (
			ci < len(subdfs)
			and len(subdfs[ci]) > 0
			and 'freqs' in subdfs[ci].columns
		)
		_has_stacked_rows = False
		if ci < len(_stacking_per_comp) and _stacking_per_comp[ci]:
			if USE_EMG_MIX and ci < len(stacked_dfs_per_comp):
				_sdf = stacked_dfs_per_comp[ci]
				if _sdf is not None and len(_sdf) > 0 and 'freqs' in _sdf.columns:
					_has_stacked_rows = True
			if USE_GAUSS_MIX and ci < len(stacked_gauss_dfs_per_comp):
				_sgdf = stacked_gauss_dfs_per_comp[ci]
				if _sgdf is not None and len(_sgdf) > 0 and 'freqs' in _sgdf.columns:
					_has_stacked_rows = True
		if not (_has_subdf_rows or _has_stacked_rows):
			# Paint a centered explanatory message and bail out.
			_skipped_sfx = (subburst_suffixes[ci]
			                if ci < len(subburst_suffixes) else str(ci))
			ax_slope.text(
				0.5, 0.5,
				f"No TOA data for component {_skipped_sfx}\n"
				f"(skipped by subband_diagnostic, or all per-channel fits rejected)",
				transform=ax_slope.transAxes,
				ha='center', va='center', fontsize=9,
				color='gray', style='italic',
			)
			ax_slope.set_xlabel("Frequency (MHz)")
			ax_slope.set_ylabel("Time (ms)")
			ax_slope.set_xlim(min(freqs), max(freqs))
			return
		# === END DEFENSIVE GUARD ===

		_psfx = subburst_suffixes[xos.index(xos[ci])]
		_pdtdnu, _pdtdnu_err, _pdtdnu_redchisq = dtdnus[ci]
		_pt_b, _ = intercepts[ci]

		# Use stacked arrival times when this component was stacked.
		_ci_stacked = bool(_stacking_per_comp[ci]) if ci < len(_stacking_per_comp) else False
		_ci_emg_stacked_df = (
			stacked_dfs_per_comp[ci]
			if (USE_EMG_MIX and _ci_stacked and ci < len(stacked_dfs_per_comp))
			else None
		)
		_ci_gauss_stacked_df = (
			stacked_gauss_dfs_per_comp[ci]
			if (USE_GAUSS_MIX and _ci_stacked and ci < len(stacked_gauss_dfs_per_comp))
			else None
		)
		if USE_EMG_MIX and _ci_emg_stacked_df is not None and len(_ci_emg_stacked_df) > 0:
			_plotdf = _ci_emg_stacked_df
			_plot_marker = 'x'
			_toa_label = "Stacked EMG TOAs"
			_time_col_plot = time_col
			_uncert_col_plot = 't_emg_hwhm_err'
		elif USE_GAUSS_MIX and _ci_gauss_stacked_df is not None and len(_ci_gauss_stacked_df) > 0:
			_plotdf = _ci_gauss_stacked_df
			_plot_marker = 'x'
			_toa_label = "Stacked Gaussian TOAs"
			_time_col_plot = tpoint
			_uncert_col_plot = f'{tpoint}_err'
		else:
			_plotdf = subdfs[ci]
			_plot_marker = 'o'
			_toa_label = "TOAs"
			_time_col_plot = time_col if USE_EMG_MIX else tpoint
			_uncert_col_plot = 't_emg_hwhm_err' if USE_EMG_MIX else f'{tpoint}_err'

		# Component label in top-left corner
		ax_slope.text(
			0.02, 0.98, f"multicomponent {_psfx}",
			transform=ax_slope.transAxes,
			fontsize=6, verticalalignment='top', horizontalalignment='left',
			color='gray',
		)

		# Main scatter (TOAs vs frequency, offset by xos[ci])
		ax_slope.scatter(
			_plotdf['freqs'], _plotdf[_time_col_plot] - xos[ci],
			c='k', s=20, marker=_plot_marker,
		)

		# Ignored TOAs per component. Earlier this code read the
		# loop-local `ignored_stacked_df` / `ignored_stacked_gauss_df`
		# / `ignored_subdf` via closure, which meant the slope panel
		# always rendered the LAST iteration's ignored points
		# regardless of which component (ci) we were rendering for.
		# That produced bugs like component a's panel showing
		# component b's stacked-X ignored markers. Fix: look up by
		# component index from the per-comp accumulators populated
		# at the end of each iteration of the per-component loop.
		_isd = (ignored_stacked_dfs_per_comp[ci]
		        if ci < len(ignored_stacked_dfs_per_comp) else None)
		_isgd = (ignored_stacked_gauss_dfs_per_comp[ci]
		         if ci < len(ignored_stacked_gauss_dfs_per_comp) else None)
		_isub = (ignored_subdfs_per_comp[ci]
		         if ci < len(ignored_subdfs_per_comp) else None)

		if _isd is not None and len(_isd) > 0:
			ax_slope.scatter(
				_isd['freqs'],
				_isd[time_col] - xos[ci],
				c='#333333', marker='x', s=25,
				linewidths=1.5, alpha=0.7, zorder=5,
			)
		if _isgd is not None and len(_isgd) > 0:
			ax_slope.scatter(
				_isgd['freqs'],
				_isgd[tpoint] - xos[ci],
				c='#333333', marker='x', s=25,
				linewidths=1.5, alpha=0.7, zorder=5,
			)
		if _isub is not None and len(_isub) > 0:
			_time_col_ignored = time_col if USE_EMG_MIX else tpoint
			ax_slope.scatter(
				_isub['freqs'],
				_isub[_time_col_ignored] - xos[ci],
				facecolors='none', edgecolors='#333333',
				marker='o', s=25,
				linewidths=1.5, alpha=0.7, zorder=5,
			)

		# Error bars
		ax_slope.errorbar(
			_plotdf['freqs'],
			_plotdf[_time_col_plot] - xos[ci],
			yerr=np.sqrt(_pdtdnu_redchisq) * _plotdf[_uncert_col_plot],
			xerr=None,
			fmt='none',
			zorder=-1,
			color='#888888',
		)

		# Fit line. Adjust intercept for coordinate system. The data is
		# plotted as `_plotdf[_time_col_plot] - xos[ci]`. For the dashed
		# line `_pdtdnu * freqs + _pt_b + _intercept_offset` to pass
		# through that data, the offset must equal whatever extra
		# (constant-in-frequency) shift was applied between the
		# per-channel fit's local frame and the plotted (unshifted)
		# frame. That depends on whether `cuts` were supplied:
		#
		#   no cuts, EMG  : per-channel TOAs were shifted by
		#                   `xcs[ci] - 4*width_used`, and the fit's
		#                   internal reference (`subpktime`) was
		#                   `4*width_used`. Net constant in
		#                   (TOA - xos[ci]) − (dtdnu*ν + tb) is
		#                   xcs[ci] - xos[ci] = τ. So offset = τ.
		#   no cuts, Gauss: xcs == xos, offset = 0.
		#   with cuts    : the shift applied to subdf[time_col] is
		#                   exactly chosen to bring TOAs back into the
		#                   unshifted-global frame anchored at xosi
		#                   (left/right/middle branches all converge
		#                   to: global_time = xosi + dtdnu*ν + tb).
		#                   So offset = 0 regardless of EMG/Gauss.
		#
		# Previously this branch always used `xcs[ci] - xos[ci]` for
		# EMG, which is correct only in the no-cuts case. When `cuts`
		# was supplied (your 9-04 config), the fit line was offset
		# upward by τ relative to the data, visible as the ~2-ms
		# vertical gap in the bottom panel.
		if USE_EMG_MIX and len(cuts) == 0:
			_intercept_offset = xcs[ci] - xos[ci]
		else:
			_intercept_offset = 0
		ax_slope.plot(freqs, _pdtdnu * freqs + _pt_b + _intercept_offset, 'k--')

		# Slope/χ² annotation (right side)
		ax_slope.annotate(
			f"{_psfx}. $dt/d\\nu =$ {scilabel(_pdtdnu, _pdtdnu_err)} ms/MHz"
			f"\n$\\chi^2_{{\\text{{red}}}} =$ {_pdtdnu_redchisq:.2f}",
			xy=(0.99, 0.475),
			ha='right',
			xycoords='axes fraction',
			color='k',
			weight='black',
			size=12,
			bbox={"boxstyle":"round", 'alpha': 0.},
		)

		# TOA count (bottom-left)
		ax_slope.text(
			0.02, 0.02, f"{_toa_label}: {len(_plotdf)}",
			transform=ax_slope.transAxes,
			ha="left", va="bottom",
		)

		ax_slope.set_xlabel("Frequency (MHz)")
		ax_slope.set_ylabel("Time (ms)")
		ax_slope.set_xlim(min(freqs), max(freqs))

	# Initial render
	_render_slope_panel(pidx)
	plt.tight_layout()

	# === TOGGLE BUTTON: cycle through components for slope panel and
	# legends. Lives at the top-right corner of the figure as a small
	# text-as-button. Cycles through:
	#   None  → all-components view (initial-render state)
	#   0..n  → single-component views (only components with TOAs)
	# After the last single-component view, cycling returns to the
	# None all-components state.
	_toggle_state = {'current': None}

	# Build the cycle order. None first (initial state), then each
	# component that has either usable TOAs or a successful ACF
	# result (so the toggle can reach ACF-only components).
	_cycle_order = [None] + [i for i in range(len(xos)) if _comp_has_data(i)]

	def _redraw_for_component(new_ci):
		# Slope panel: full re-render. For new_ci=None (all-components
		# view) we have no aggregate slope plot, so show pidx's panel
		# as the resting state.
		_slope_panel_ci = pidx if new_ci is None else new_ci
		_render_slope_panel(_slope_panel_ci)

		# Intensity-profile mix-fit label: update and re-legend. The
		# label function accepts None and returns the multi-component
		# all-σ_t/all-τ list, or single-component σ_t/τ when ci is int.
		# Single dashed line carries the legend in both hard-isolation
		# mode and the legacy joint-fit case — no per-component lines
		# to dim.
		if _mix_line is not None:
			try:
				_mix_line.set_label(_intensity_label_for_comp(new_ci))
				ax_tseries.legend(frameon=False, fontsize=6, handlelength=0)
			except Exception as _e:
				print(f"[toggle] intensity-profile legend update failed: {_e}")

		# Sub-band overlays on waterfall: re-render. new_ci=None goes
		# back to the all-components clipped initial view; an int goes
		# to single-component unclipped view (with horizontal sub-band
		# frequency dividers and cyan ±σ box suppressed — see renderer).
		try:
			_render_subband_overlays(new_ci)
		except Exception as _e:
			print(f"[toggle] sub-band overlay update failed: {_e}")

		# Waterfall legend: for new_ci=None show ALL slope labels; for
		# an int show only the active component's. Drift line is
		# unaffected (drawn outside _wfall_slope_lines).
		for _ci, _ln in _wfall_slope_lines.items():
			if new_ci is None or _ci == new_ci:
				_ln.set_label(_wfall_slope_labels[_ci])
			else:
				_ln.set_label(None)
		try:
			if not hide_legend:
				ax_wfall.legend(loc=legendloc, handlelength=0, fontsize=7)
		except Exception as _e:
			print(f"[toggle] waterfall legend update failed: {_e}")

		_toggle_state['current'] = new_ci
		try:
			fig.canvas.draw_idle()
		except Exception:
			pass

	# Toggle button for cycling through components. Implemented as a
	# matplotlib.widgets.Button rather than a fig.text-with-picker so
	# it remains clickable when the zoom/pan navigation tools in the
	# toolbar are active. Pick events get blocked while the toolbar is
	# in navigation mode (documented matplotlib behavior), which used
	# to make the toggle unresponsive after zooming. The Button widget
	# works regardless of toolbar mode.
	#
	# We anchor the button axes at roughly the same figure-fraction
	# === SUBBAND DIAGNOSTIC MODE BANNER ===
	# A bright red banner across the top-center of the figure when
	# diagnostic mode is active, so the user can't forget they're in
	# tuning mode (where some components have empty ToA data and CSV
	# writing is disabled).
	if _diag_mode_active:
		_diag_comps_str = ', '.join(
			subburst_suffixes[i] for i in sorted(_diag_comp_set)
			if i < len(subburst_suffixes)
		)
		fig.text(
			0.5, 0.985,
			f'⚠ DIAGNOSTIC MODE — only processing component(s): {_diag_comps_str}  |  CSV writing disabled',
			ha='center', va='top',
			fontsize=10, color='white', fontweight='bold',
			bbox=dict(boxstyle='round,pad=0.5', facecolor='#CC0000',
			          edgecolor='#660000', linewidth=1.2),
			zorder=100,
		)
	# === END DIAGNOSTIC BANNER ===

	# position as the old text artist (top-right corner) so the visual
	# placement is identical. Width is set so the "Toggle component"
	# label has comfortable horizontal padding (was 0.10, too tight —
	# the leading T and trailing t overlapped the edges).
	_button_ax = fig.add_axes([0.855, 0.965, 0.135, 0.028])
	_button = Button(
		_button_ax, 'Toggle component',
		color='#EEEEEE', hovercolor='#DDDDDD',
	)
	_button.label.set_fontsize(9)
	# Match the rounded look of the previous text-bbox approach as
	# closely as the widget allows — Button uses a Rectangle patch
	# under the hood, so we just style its frame to match.
	for _spine in _button_ax.spines.values():
		_spine.set_edgecolor('#888888')
		_spine.set_linewidth(0.8)

	def _on_toggle_click(event):
		# Called by Button.on_clicked with the matplotlib event. We
		# don't need to verify the artist here — the Button widget
		# only fires for clicks inside its own axes.
		if not _cycle_order:
			return
		_cur = _toggle_state['current']
		try:
			_pos = _cycle_order.index(_cur)
		except ValueError:
			_pos = -1
		_new = _cycle_order[(_pos + 1) % len(_cycle_order)]
		_redraw_for_component(_new)

	_button.on_clicked(_on_toggle_click)
	# Keep a reference to the Button alive — without this it can be
	# garbage-collected, severing the callback wiring. Attaching it to
	# the figure object is the matplotlib-idiomatic way.
	fig._toggle_button_widget_ref = _button
	# === END TOGGLE BUTTON ===

	for t in xos:
		ax_tseries.axvline(x=t-pktime, ls='--')
	for cut in cuts:
		ax_tseries.axvline(
			x=cut-pktime,
			ls='-.',
			c='k',
			zorder=-1,
		)

	# 1 sigma regions used for spectrum
	for s, xoi in zip(tmix_sigmas, xos):
		# for xsig in [xoi-pktime-np.abs(s), xoi-pktime+np.abs(s)]:
		# 	ax_tseries.axvline(x=xsig, alpha=1, color='beige', zorder=-20)
		tmplims = ax_tseries.get_ylim() # hack for region not extending all the way
		ax_tseries.fill_between(
			[xoi-pktime-np.abs(s), xoi-pktime+np.abs(s)],
			# 0, 0.1,
			*(10*np.array(ax_tseries.get_ylim())),
			zorder=-20,
			alpha=0.75,
			color='cyan'
		)
		ax_tseries.set_ylim(tmplims)
			
			
		def printinfo(event):
# 			global _global_subfall_shift  # Access the global shift variable
			global _global_subfall_shift, _global_wfall  # Access global variables for EMG stacking
			global _global_subfall_gauss_shift, _global_wfall_gauss  # Access global variables for Gaussian stacking
			global _global_stacked_gauss_profiles, _global_stacked_gauss_fitdata, _global_subfall_gauss_duration  # Gaussian stacking data
			global _global_adaptive_window_params  # Access adaptive window parameters for visualization            
			# Only react when clicking on the dynamic spectrum (waterfall) axes
			if event.inaxes is not ax_wfall or event.xdata is None or event.ydata is None:
				return
		
			# Map click to (row, col) once, using the same extent used in imshow
			# x-axis is [-pktime, res_time_ms*ncol - pktime]; y-axis is [freqs_bin0, freqs_bin0 + res_freq*nrow]
			x_ms = float(event.xdata)                  # time coordinate in ms (already shifted by -pktime)
			y_mhz = float(event.ydata)
			ychan = int((y_mhz - freqs_bin0) / res_freq)
			xchan = int((x_ms + pktime) / res_time_ms)
			if not (0 <= ychan < wfall.shape[0] and 0 <= xchan < wfall.shape[1]):
				return
		
			duration = wfall.shape[1] * res_time_ms  # total time window of a row
		
			# --- Left click (single): print pixel info (freq chan, time chan, amp) ---
			if event.button == 1 and not event.dblclick:
				freq_mhz = freqs_bin0 + (ychan + 0.5) * res_freq
				# Map displayed channel back to pre-downsampled indices for 'masks' param
				df0 = downfactors[0]
				raw_lo = ychan * df0
				raw_hi = (ychan + 1) * df0 - 1
				if df0 == 1:
					mask_str = f"{raw_lo}"
				else:
					mask_str = f"range({raw_lo},{raw_hi+1})"
				print(f"freq chan: {ychan} time chan: {xchan} amp: {wfall[ychan, xchan]:.4f}  |  {freq_mhz:.1f} MHz  |  masks: [{mask_str}]")
				return
		


			# --- Left click (double): open per-channel time plot with model fit & arrival time(s) ---
			if event.button == 1 and event.dblclick:
				row_full = wfall[ychan, :]
				
				# === REGION-BASED VISUALIZATION FOR MULTICOMPONENT BURSTS ===
				# When cuts are present, determine which region the click is in and only show that region
				click_time_ms = x_ms + pktime  # time in ms (0-based, not shifted)
				
				# Determine region boundaries.
				# `_component_boundaries_ms` is populated upstream in the
				# pulse-bar block and unifies the two boundary sources:
				#   * explicit `cuts` parameter, or
				#   * implicit midpoints between adjacent on-pulse segments
				#     of the multi-component intensity-profile fit.
				# So this branch fires whenever we have at least one
				# boundary, regardless of how it was derived.
				if len(_component_boundaries_ms) > 0 and len(xos) > 1:
					# Sort boundaries to ensure proper ordering
					sorted_cuts = sorted(_component_boundaries_ms)
					cutchans = np.floor(np.array(sorted_cuts) / res_time_ms).astype(int)
					
					# Determine which region the click is in
					# Region 0: [0, cuts[0])
					# Region i: [cuts[i-1], cuts[i]) for i > 0 and i < len(cuts)
					# Region n: [cuts[-1], end)
					region_idx = 0
					for i, cut_chan in enumerate(cutchans):
						if xchan >= cut_chan:
							region_idx = i + 1
					
					# Determine the boundaries for this region (in channel indices)
					if region_idx == 0:
						left_chan = 0
						right_chan = cutchans[0]
					elif region_idx >= len(cutchans):
						left_chan = cutchans[-1]
						right_chan = wfall.shape[1]
					else:
						left_chan = cutchans[region_idx - 1]
						right_chan = cutchans[region_idx]
					
					# Extract the row data for this region only
					row = row_full[left_chan:right_chan]
					region_duration = row.size * res_time_ms
					region_start_ms = left_chan * res_time_ms
					
					# Build fit axis (0 to region_duration) and plot axis (shifted for waterfall alignment)
					xs_fit = np.linspace(0, region_duration, num=row.size)
					xs_plot = xs_fit + region_start_ms - pktime  # shift to align with waterfall
					
					# Determine which component is in this region for seed values
					region_component_idx = None
					xos_sorted = sorted(xos)
					for i, xo in enumerate(xos_sorted):
						xo_chan = int(xo / res_time_ms)
						if left_chan <= xo_chan < right_chan:
							region_component_idx = xos.index(xo)  # get original index
							break
					
					# Get component-specific seeds if available
					if region_component_idx is not None:
						region_sigma = tmix_sigmas[region_component_idx] if region_component_idx < len(tmix_sigmas) else None
						region_lambda = tmix_lambdas[region_component_idx] if ('tmix_lambdas' in dir() and region_component_idx < len(tmix_lambdas)) else None
						# Convert xo to region-local coordinates
						region_xo = xos[region_component_idx] - region_start_ms
					else:
						region_sigma = None
						region_lambda = None
						region_xo = None
					
					# Flag to indicate we're in region mode (single component fit)
					use_single_fit = True
					region_label = f"Region {region_idx + 1}/{len(cutchans) + 1}"
					
					# Set plot limits to this region
					plot_xlim = (region_start_ms - pktime, region_start_ms + region_duration - pktime)
				else:
					# No cuts or single component: use full row
					row = row_full
					region_duration = duration
					xs_fit = np.linspace(0, duration, num=row.size)
					xs_plot = xs_fit - pktime
					use_single_fit = (len(xos) <= 1)
					region_label = None
					plot_xlim = (-pktime, duration - pktime)
					region_start_ms = 0
					# Seeds for no-cuts / single-component case
					region_sigma = tmix_sigmas[0] if len(tmix_sigmas) > 0 else None
					try:
						region_lambda = tmix_lambdas[0] if len(tmix_lambdas) > 0 else None
					except (NameError, TypeError):
						region_lambda = None
					region_xo = xos[0] if len(xos) > 0 else None
				# === END REGION-BASED SETUP ===
			
				# === CHECK IF WE SHOULD SHOW STACKED PLOT ===
				show_stacked_plot = False
				show_stacked_gauss_plot = False
				stack_idx = None
				stack_gauss_idx = None
				# Per-component popup resolution: pick the component whose xos
				# is closest to the click time, then use that component's
				# stacked state (stack_size, profiles, fitdata, df) if it was
				# stacked. This is correct for mixed per-component stacking
				# configurations where different components may have different
				# stack_size values or stacking on/off.
				_popup_click_time = x_ms + pktime
				_popup_comp_idx = 0
				if len(xos) > 0:
					_popup_comp_idx = int(np.argmin([abs(_xo - _popup_click_time) for _xo in xos]))
				_popup_stack_size = (
					int(_stack_size_per_comp[_popup_comp_idx])
					if _popup_comp_idx < len(_stack_size_per_comp)
					else STACK_SIZE
				)
				_popup_emg_state = (
					_stacked_state_per_comp[_popup_comp_idx]
					if _popup_comp_idx < len(_stacked_state_per_comp)
					else None
				)
				_popup_gauss_state = (
					_stacked_gauss_state_per_comp[_popup_comp_idx]
					if _popup_comp_idx < len(_stacked_gauss_state_per_comp)
					else None
				)

				# Check for EMG stacking (per-component)
				if (USE_EMG_MIX and _popup_emg_state is not None and
					_popup_emg_state.get('profiles') is not None):
					stack_idx = ychan // _popup_stack_size
					if stack_idx < len(_popup_emg_state['profiles']):
						show_stacked_plot = True

				# Check for Gaussian stacking (per-component)
				if (USE_GAUSS_MIX and _popup_gauss_state is not None and
					_popup_gauss_state.get('profiles') is not None):
					stack_gauss_idx = ychan // _popup_stack_size
					if stack_gauss_idx < len(_popup_gauss_state['profiles']):
						show_stacked_gauss_plot = True

				# Create figure with 1 or 2 subplots based on stacking
				if show_stacked_plot or show_stacked_gauss_plot:
					# Two stacked time-series panels in a left column, plus a
					# tall right-hand panel holding the per-stack filter audit
					# (which filters this stack passed/failed). 2 rows × 2 cols;
					# the right column spans both rows.
					fig2 = plt.figure(figsize=(12.5, 8.4))
					_popup_gs = fig2.add_gridspec(
						2, 2, width_ratios=[2.4, 1.4], hspace=0.32, wspace=0.06)
					ax2 = fig2.add_subplot(_popup_gs[0, 0])
					ax2_stacked = fig2.add_subplot(_popup_gs[1, 0])
					ax2_audit = fig2.add_subplot(_popup_gs[:, 1])
					# The audit panel is text-only — strip all axes furniture.
					ax2_audit.axis('off')
				else:
					fig2, ax2 = plt.subplots(1, 1, figsize=(7.5, 4.2))
					ax2_audit = None
				# === END STACKED PLOT CHECK ===
				
				ax2.plot(xs_plot, row, label="data")
			
				try:
					if USE_EMG_MIX:
						# Use single EMG fit when in region mode OR when only one component
						if use_single_fit:
							# Pass dt_ms for bin-integrated fitting and seeds from
							# the intensity-profile fit to avoid sigma-tau degeneracy
							_mu_seed = float(region_xo) if (region_xo is not None and np.isfinite(region_xo)) else None
							_sig_seed = float(region_sigma) if (region_sigma is not None and np.isfinite(region_sigma)) else None
							_lam_seed = float(region_lambda) if (region_lambda is not None and np.isfinite(region_lambda)) else None
							popt_row, pcov_row, red_row = fitemg(
								row, region_duration, dt_ms=res_time_ms,
								mu0_seed=_mu_seed, sigma0_seed=_sig_seed, lambda0_seed=_lam_seed
							)
							model = emg_model(xs_fit, *popt_row)
							A, mu, sig, lam, C = popt_row
							ax2.axvline(mu + region_start_ms - pktime, ls="--", c="k")
							if np.isfinite(lam) and lam > 0:
								ax2.axvline(mu + 1.0 / lam + region_start_ms - pktime, ls="--", c="purple")
							tau_ms = (1.0/lam) if np.isfinite(lam) and lam > 0 else np.nan
							# τ uncertainty from the curve_fit covariance: σ_τ = σ_λ / λ²
							# (delta method, since τ = 1/λ). pcov_row[3,3] is the
							# variance of the λ parameter. Non-finite or degenerate
							# pcov falls back to a NaN error in the label.
							_tau_err_ms = np.nan
							try:
								if (isinstance(pcov_row, np.ndarray)
										and pcov_row.shape == (5, 5)
										and np.isfinite(lam) and lam > 0
										and np.isfinite(pcov_row[3, 3]) and pcov_row[3, 3] > 0):
									_lam_err = float(np.sqrt(pcov_row[3, 3]))
									_tau_err_ms = _lam_err / (lam * lam)
							except Exception:
								_tau_err_ms = np.nan
							_n_bins_row = int(len(row))
							_tau_str = (f"{tau_ms:.3g} ± {_tau_err_ms:.2g}"
							            if np.isfinite(_tau_err_ms) else f"{tau_ms:.3g}")
							label_model = (f"singleEMG  χ²_red={red_row:.2f}\n"
							               f"τ = {_tau_str} ms\n"
							               f"bins used: {_n_bins_row}")
							if region_label:
								label_model = f"{region_label}: {label_model}"
						else:
							# Full mixture fit (no cuts, multiple components)
							_lams = tmix_lambdas if "tmix_lambdas" in globals() else None
							popt_row, pcov_row, red_row = fitemgmix(
								row, duration,
								xos=xos,
								sigmas=tmix_sigmas,
								lambdas=_lams,
								fix_xos=True,
								tol=0.25 * res_time_ms,
								dt_ms=res_time_ms
							)
							model = emgmix_model(xs_fit, *popt_row)
			
							n = len(xos)
							mus  = np.atleast_1d(popt_row[n:2*n])
							lams = np.atleast_1d(popt_row[3*n:4*n]) if len(popt_row) >= 4*n+1 else np.array([])
							for i, mu_i in enumerate(mus):
								ax2.axvline(mu_i - pktime, ls="--", c="k")
								if i < lams.size and np.isfinite(lams[i]) and lams[i] > 0:
									ax2.axvline(mu_i + 1.0 / lams[i] - pktime, ls="--", c="purple")
									
							# Per-component τ and τ_err in lockstep so degenerate
							# components don't scramble the error bars in the label.
							# Mixture EMG parameter layout: [A_1..A_n, μ_1..μ_n,
							# σ_1..σ_n, λ_1..λ_n, C]; λ_i lives at index 3n+i.
							_pcov_ok = (isinstance(pcov_row, np.ndarray)
							            and pcov_row.ndim == 2
							            and pcov_row.shape[0] >= 4 * n + 1)
							_tau_pairs = []  # list of (tau_i, tau_err_i_or_nan)
							for i, l in enumerate(np.atleast_1d(lams)):
								if not (np.isfinite(l) and l > 0):
									continue
								t_i = 1.0 / l
								t_err_i = np.nan
								if _pcov_ok:
									_idx = 3 * n + i
									_var = pcov_row[_idx, _idx]
									if np.isfinite(_var) and _var > 0:
										t_err_i = float(np.sqrt(_var)) / (l * l)
								_tau_pairs.append((t_i, t_err_i))

							_n_bins_row = int(len(row))
							_tau_strs = []
							for t_i, t_err in _tau_pairs:
								if np.isfinite(t_err):
									_tau_strs.append(f"{t_i:.3g} ± {t_err:.2g}")
								else:
									_tau_strs.append(f"{t_i:.3g}")
							tau_text = ", ".join(_tau_strs) if _tau_strs else "n/a"
							label_model = (f"mixEMG  χ²_red={red_row:.2f}\n"
							               f"τ = [{tau_text}] ms\n"
							               f"bins used: {_n_bins_row}")
					else:
						# Gaussian mode
						if use_single_fit:
							popt_row, pcov_row, red_row = fitgauss(row, region_duration)
							model = gauss_model(xs_fit, *popt_row)
							ax2.axvline(popt_row[1] + region_start_ms - pktime, ls="--", c="k")
							label_model = (f"singleGauss  χ²_red={red_row:.2f}\n"
							               f"bins used: {int(len(row))}")
							if region_label:
								label_model = f"{region_label}: {label_model}"
						else:
							# Full mixture fit (no cuts, multiple components)
							popt_row, pcov_row, red_row = fitgaussmix(
								row, duration,
								xos=xos,
								sigmas=tmix_sigmas,
								fix_xos=True,
								tol=0.25 * res_time_ms
							)
							model = gaussmix_model(xs_fit, *popt_row)
							n = len(xos)
							mus = np.atleast_1d(popt_row[n:2*n])
							for mu_i in mus:
								ax2.axvline(mu_i - pktime, ls="--", c="k")
							label_model = (f"mixGauss  χ²_red={red_row:.2f}\n"
							               f"bins used: {int(len(row))}")
			
					# Plot model on the shifted x-axis so it lines up with the waterfall.
					# Two grids matter (same logic as the sub-band 'q' popup):
					#   • Fine grid: smooth analytic curve. Avoids the jagged look at
					#     low resolution (σ ≲ dt) where line segments between data
					#     bin centres misrepresent the underlying smooth model.
					#   • Bin-centre grid: the bin-integrated EMG that the fit
					#     actually minimised against (only for EMG branches —
					#     fitgauss/fitgaussmix here use the instantaneous model).
					_fine_factor_row = 20
					xs_fit_fine = np.linspace(xs_fit[0], xs_fit[-1],
					                          _fine_factor_row * len(xs_fit))
					xs_plot_fine = xs_fit_fine + (xs_plot[0] - xs_fit[0])
					if USE_EMG_MIX:
						if use_single_fit:
							model_fine = emg_model(xs_fit_fine, *popt_row)
						else:
							model_fine = emgmix_model(xs_fit_fine, *popt_row)
					else:
						if use_single_fit:
							model_fine = gauss_model(xs_fit_fine, *popt_row)
						else:
							model_fine = gaussmix_model(xs_fit_fine, *popt_row)

					kwargs = {"color": "red"} if USE_EMG_MIX else {}
					ax2.plot(xs_plot_fine, model_fine, "--", label=label_model, **kwargs)

					# Bin-integrated dots at the actual data bin centres — only
					# for EMG, since fitemg/fitemgmix were called with
					# dt_ms=res_time_ms (i.e. the fit minimised against
					# bin_integrated_emg / bin_integrated_emgmix).
					if USE_EMG_MIX:
						try:
							if use_single_fit:
								_sig_p = popt_row[2]; _lam_p = popt_row[3]
								_tau_p = (1.0/_lam_p) if (np.isfinite(_lam_p) and _lam_p > 0) else np.nan
								_ovs = pick_oversamp(res_time_ms, _sig_p, _tau_p)
								model_bins = bin_integrated_emg(
									xs_fit, *popt_row,
									dt_ms=res_time_ms, oversamp=_ovs
								)
							else:
								# Mixture: pick conservative oversamp from the
								# narrowest component so even sharp pulses are
								# faithfully bin-integrated.
								n_comp = len(xos)
								_sigs_p = np.atleast_1d(popt_row[2*n_comp:3*n_comp])
								_lams_p = (np.atleast_1d(popt_row[3*n_comp:4*n_comp])
								           if len(popt_row) >= 4*n_comp+1 else np.array([]))
								_min_sig = float(np.nanmin(_sigs_p)) if _sigs_p.size else np.nan
								_taus_p = [1.0/l for l in _lams_p if np.isfinite(l) and l > 0]
								_min_tau = float(np.nanmin(_taus_p)) if _taus_p else np.nan
								_ovs = pick_oversamp(res_time_ms, _min_sig, _min_tau)
								model_bins = bin_integrated_emgmix(
									xs_fit, res_time_ms, _ovs, *popt_row
								)
							ax2.plot(xs_plot, model_bins, '.', color='red',
							         ms=3, alpha=0.6)
						except Exception:
							# Non-fatal: fall back to just the smooth curve.
							pass
					
					# --- EMG arrival point(s) - respects PEAK_INTENSITY_TOA toggle ---
					if USE_EMG_MIX:
						if use_single_fit:
							# Single EMG: popt_row = [A, mu, sigma, lambda, C]
							A, mu, sig, lam, C = popt_row
							try:
								if np.isfinite(mu) and np.isfinite(sig) and np.isfinite(lam) and (sig > 0) and (lam > 0):
									if PEAK_INTENSITY_TOA:
										# Peak mode: find the maximum of the EMG function
										t_arr = find_emg_peak(mu=float(mu), sigma=float(sig), lamb=float(lam))
									else:
										# Leading edge mode: use left HWHM crossing
										tau = 1.0/lam
										stats = emg_left_hwhm(mu=float(mu), sigma=float(sig), tau=float(tau))
										t_arr = float(stats["x_left"])
								else:
									t_arr = mu - np.sqrt(2.0) * np.abs(sig)
							except Exception:
								t_arr = np.nan
					
							if np.isfinite(t_arr):
								# Evaluate the model EXACTLY at t_arr instead of
								# snapping to the nearest data bin via argmin.
								# Snapping can land the marker at the pre-rise
								# baseline (y ≈ C) or post-rise peak when the
								# EMG's leading edge spans <2 bins — visible
								# as the marker appearing at the top or bottom
								# of the burst rather than near half-max.
								_y_marker = float(emg_model(
									np.atleast_1d(t_arr), *popt_row
								)[0])
								_x_marker_plot = float(t_arr) + (xs_plot[0] - xs_fit[0])
								ax2.plot(_x_marker_plot, _y_marker, marker='o', ms=7, mfc='white', mec='k', label="arrival time")
						else:
							# EMG mixture: parameters packed by components
							labeled = False
							n = len(xos)
							mus  = np.atleast_1d(popt_row[n:2*n])
							sigs = np.atleast_1d(popt_row[2*n:3*n])
							lams = np.atleast_1d(popt_row[3*n:4*n]) if len(popt_row) >= 4*n+1 else np.array([])
					
							for i, mu_i in enumerate(mus):
								sig_i = sigs[i] if i < sigs.size else np.nan
								lam_i = lams[i] if i < lams.size else np.nan
								try:
									if np.isfinite(mu_i) and np.isfinite(sig_i) and np.isfinite(lam_i) and (sig_i > 0) and (lam_i > 0):
										if PEAK_INTENSITY_TOA:
											# Peak mode: find the maximum of the EMG function
											t_arr = find_emg_peak(mu=float(mu_i), sigma=float(sig_i), lamb=float(lam_i))
										else:
											# Leading edge mode: use left HWHM crossing
											tau_i = 1.0/lam_i
											stats = emg_left_hwhm(mu=float(mu_i), sigma=float(sig_i), tau=float(tau_i))
											t_arr = float(stats["x_left"])
									else:
										t_arr = mu_i - np.sqrt(2.0) * np.abs(sig_i)
								except Exception:
									t_arr = np.nan
					
								if np.isfinite(t_arr):
									# Evaluate the mixture model EXACTLY at t_arr
									# (see comment in single-EMG branch above).
									_y_marker = float(emgmix_model(
										np.atleast_1d(t_arr), *popt_row
									)[0])
									_x_marker_plot = float(t_arr) + (xs_plot[0] - xs_fit[0])
									ax2.plot(_x_marker_plot, _y_marker, marker='o', ms=7, mfc='white', mec='k',
											 label=None if labeled else "arrival time")
									labeled = True

					# --- Arrival time point(s) for Gaussian path - respects PEAK_INTENSITY_TOA ---
					if not USE_EMG_MIX:
						if use_single_fit:
							# Single Gauss: popt_row = [A, mu, sigma]
							A, mu_g, sig_g = popt_row[:3]
							if PEAK_INTENSITY_TOA:
								t_arr_g = mu_g  # Peak mode: arrival at Gaussian center
							else:
								t_arr_g = mu_g - np.sqrt(2.0) * np.abs(sig_g)  # Leading edge mode
							# Evaluate the Gaussian model EXACTLY at t_arr_g
							# (see comment in single-EMG branch above).
							_y_marker = float(gauss_model(
								np.atleast_1d(t_arr_g), *popt_row[:3]
							)[0])
							_x_marker_plot = float(t_arr_g) + (xs_plot[0] - xs_fit[0])
							ax2.plot(_x_marker_plot, _y_marker, marker='o', ms=7, mfc='white', mec='k', label="arrival time")
						else:
							# Mix-Gaussian fit: popt_row packs [A(0..n-1), mu(0..n-1), sigma(0..n-1), C]
							n = len(xos)
							mus  = np.atleast_1d(popt_row[n:2*n])
							sigs = np.atleast_1d(popt_row[2*n:3*n])
							labeled = False
							for mu_i, sig_i in zip(mus, sigs):
								if PEAK_INTENSITY_TOA:
									t_arr_g = mu_i  # Peak mode: arrival at Gaussian center
								else:
									t_arr_g = mu_i - np.sqrt(2.0) * np.abs(sig_i)  # Leading edge mode
								# Evaluate the mixture Gaussian model EXACTLY at
								# t_arr_g (see comment in single-EMG branch above).
								_y_marker = float(gaussmix_model(
									np.atleast_1d(t_arr_g), *popt_row
								)[0])
								_x_marker_plot = float(t_arr_g) + (xs_plot[0] - xs_fit[0])
								ax2.plot(_x_marker_plot, _y_marker, marker='o', ms=7, mfc='white', mec='k',
										 label=None if labeled else "arrival time")
								labeled = True

					# --- Overlay a simple single-Gaussian on the EMG per-row plot ---
					if USE_EMG_MIX:
						g_popt, g_pcov, g_red = fitgauss(row, region_duration)
						# Use the same fine grid built above for the EMG curve
						g_curve_fine = gauss_model(xs_fit_fine, *g_popt)
						ax2.plot(xs_plot_fine, g_curve_fine, '--', c='k', lw=1.4,
								label=f"Gaussian (χ²_red={g_red:.2f})")

			
				except Exception as e:
					ax2.text(0.02, 0.95, f"fit failed: {e}", transform=ax2.transAxes, va="top")
			
				# Set limits to the appropriate region
				ax2.set_xlim(plot_xlim)
			
				title = f"Row @ ~{freqs[ychan]:.2f} MHz (chan {ychan})"
				if region_label:
					title = f"{title} - {region_label}"
				ax2.set_title(title)
				ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("Intensity")
				ax2.legend(frameon=False, loc='upper right', fontsize=7,
				           handlelength=1.4, labelspacing=0.3, borderpad=0.3)
				
				# === PLOT STACKED CHANNEL DATA (IF AVAILABLE) ===
				if show_stacked_plot and stack_idx is not None:
					try:
						# Get stacked profile and fit data for this stack
						# (per-component state resolved by click time)
						stacked_row = _popup_emg_state['profiles'][stack_idx, :]
						
						# Get the fit parameters from stacked_fitdata
						# Columns: freqs, tstart, tend, amp, xo, tstart_err, sigma, red_chisq, 
						#          amp_err, xo_err, sigma_err, lambda, tau, lambda_err, tau_err,
						#          sigma_emg_std, sigma_emg_std_err, t_emg_hwhm, t_emg_hwhm_err, C
						fit_params = _popup_emg_state['fitdata'][stack_idx, :]
						A_stack = fit_params[3]      # amp
						mu_stack = fit_params[4]     # xo (mu)
						sig_stack = fit_params[6]    # sigma
						lam_stack = fit_params[11]   # lambda
						# Use the stored baseline if present (column 19), else 0.
						# Previously this hard-coded C=0 which produced visibly
						# wrong popup curves when the fit converged to a
						# non-trivial baseline (e.g. degenerate fits with
						# A < 0 and large C cancelling out at the peak).
						C_stack = (
							float(fit_params[19])
							if (len(fit_params) > 19 and np.isfinite(fit_params[19]))
							else 0.0
						)
						redchisq_stack = fit_params[7]  # red_chisq
						t_arr_stack = fit_params[17]    # t_emg_hwhm (arrival time)
						duration_stack = _popup_emg_state['subfall_duration']
						
						
						# Calculate which channels are in this stack
						start_chan = stack_idx * _popup_stack_size
						end_chan = min((stack_idx + 1) * _popup_stack_size, wfall.shape[0]) - 1
						freq_low = freqs[start_chan]
						freq_high = freqs[end_chan]
						
						# Build time axes for plotting
						# Subfall coordinates for model evaluation (0 to duration_stack)
						# The EMG parameters were fit in these coordinates
						xs_subfall = np.linspace(0, duration_stack, num=stacked_row.size)
						
						# Get the shift that converts subfall coords to full waterfall coords
						subfall_shift = _popup_emg_state.get('subfall_shift', 0) or 0
						
						# Convert to plot coordinates (relative to pktime)
						xs_plot_stack = xs_subfall + subfall_shift - pktime
						
						# === PLOT FULL WATERFALL STACKED (lighter blue, for context) ===
						try:
							_popup_wfall_emg = _popup_emg_state.get('wfall', None)
							if _popup_wfall_emg is not None:
								# Stack the same channels from the FULL waterfall
								full_stacked_row = np.sum(_popup_wfall_emg[start_chan:end_chan+1, :], axis=0)

								# Build time axis for full waterfall (same as top subplot)
								xs_full = np.linspace(0, duration, num=full_stacked_row.size)
								xs_plot_full = xs_full - pktime

								# Plot as lighter blue, behind the subfall data
								ax2_stacked.plot(xs_plot_full, full_stacked_row, color='lightblue', 
												alpha=0.8, zorder=1)
						except Exception as e:
							print(f"[DEBUG] Full waterfall stacking failed: {e}")
						# === END FULL WATERFALL STACKED ===                        
						
					# Plot stacked data
						#ax2_stacked.plot(xs_plot_stack, stacked_row, label="stacked data", color='blue')
						ax2_stacked.plot(xs_plot_stack, stacked_row, label="stacked data (fit region)", color='blue', zorder=5)
						
						# Plot EMG model if fit was successful
						if np.isfinite([A_stack, mu_stack, sig_stack, lam_stack]).all():
							# Compute EMG model - parameters are in subfall coordinates.
							# Keep this sample-rate evaluation around for the
							# arrival-time marker logic below (uses argmin
							# against xs_subfall to pick a marker index).
							model_stack = emg_model(xs_subfall, A_stack, mu_stack, sig_stack, lam_stack, C_stack)
							tau_stack = 1.0/lam_stack if lam_stack > 0 else np.nan

							# Smooth analytic curve on a 20×-finer grid — avoids
							# the jagged look at narrow σ ≲ dt.
							_fine_factor_stk = 20
							xs_subfall_fine = np.linspace(xs_subfall[0], xs_subfall[-1],
							                              _fine_factor_stk * len(xs_subfall))
							xs_plot_stack_fine = xs_subfall_fine + (xs_plot_stack[0] - xs_subfall[0])
							model_stack_fine = emg_model(
								xs_subfall_fine,
								A_stack, mu_stack, sig_stack, lam_stack, C_stack
							)
							ax2_stacked.plot(xs_plot_stack_fine, model_stack_fine, '--', color='red', 
											label=f"EMG (χ²_red={redchisq_stack:.2f}, τ={tau_stack:.3g} ms)", zorder=6)

							# Bin-integrated EMG at the actual data bin centres —
							# the model the stacked fit minimised against (the
							# stacked fit uses dt_ms=res_time_ms, see
							# bin_integrated_emg call upstream). Drawn small so
							# it doesn't clutter the figure but reveals the
							# bin-integration offset for narrow pulses.
							try:
								_ovs_stk = pick_oversamp(res_time_ms, sig_stack, tau_stack)
								model_stack_bins = bin_integrated_emg(
									xs_subfall, A_stack, mu_stack, sig_stack, lam_stack, C_stack,
									dt_ms=res_time_ms, oversamp=_ovs_stk
								)
								ax2_stacked.plot(xs_plot_stack, model_stack_bins, '.',
								                 color='red', ms=3, alpha=0.6, zorder=6)
							except Exception:
								# Non-fatal: fall back to just the smooth curve.
								pass

							# Mark arrival time on the curve (not as vertical line).
							# Evaluate the model EXACTLY at t_arr_stack instead of
							# snapping to the nearest sample bin — same fix as
							# the per-channel popup. With narrow scattering
							# tails the rise spans only 1-2 bins, so snapping
							# can land the marker at the pre-rise baseline or
							# post-rise peak rather than at half-max.
							if np.isfinite(t_arr_stack):
								# t_arr_stack is in subfall coordinates
								_y_marker = float(emg_model(
									np.atleast_1d(t_arr_stack),
									A_stack, mu_stack, sig_stack, lam_stack, C_stack
								)[0])
								_x_marker_plot = float(t_arr_stack) + (xs_plot_stack[0] - xs_subfall[0])
								# Red X marker to match the stacked-TOA convention
								# used on the slope panel and main waterfall (where
								# red X = "this is a stacked arrival time"). The
								# previous yellow filled circle conflicted with
								# that convention and was confusing alongside the
								# colored, circle-shaped non-stacked TOA markers.
								ax2_stacked.plot(_x_marker_plot, _y_marker,
												marker='x', ms=12, mew=2.5, color='red',
												label=f"arrival time ({_x_marker_plot:.3f} ms)", zorder=10,
												linestyle='None')
							
							# === ADAPTIVE WINDOW VISUALIZATION ===
						# Show the boundary of the adaptive fit window for this frequency
						try:
							_popup_awp = _popup_emg_state.get('adaptive_window_params', None)
							if (_popup_awp is not None and 
								_popup_awp.get('enabled', False) and
								_popup_awp.get('tau_sc_ms', 0) > 0):
								
								awp = _popup_awp
								stack_freq = awp['stacked_freqs'][stack_idx]
								
								# Compute adaptive window for this stack
								extra_time_ms = awp['tau_sc_ms'] * (awp['nu_max'] - stack_freq) / awp['nu_range']
								adaptive_window_ms = awp['base_window_ms'] + extra_time_ms
								
								# Convert to plot coordinates
								# adaptive_window_ms is in subfall coordinates (starts at 0)
								adaptive_boundary_plot = adaptive_window_ms + subfall_shift - pktime
								
								# Draw vertical line at adaptive window boundary (full fit window)
								ax2_stacked.axvline(x=adaptive_boundary_plot, color='orange', 
												   linestyle='--', linewidth=2, alpha=0.8,
												   label=f"fit window: {adaptive_window_ms:.2f} ms", zorder=4)
								
								# Show base (symmetric) window boundary and scattering extension
								if extra_time_ms > 0.001:  # Only show if there's meaningful extension
									base_boundary_plot = awp['base_window_ms'] + subfall_shift - pktime
									ax2_stacked.axvline(x=base_boundary_plot, color='green', 
													   linestyle=':', linewidth=2, alpha=0.7,
													   label=f"base window: {awp['base_window_ms']:.2f} ms", zorder=4)
									# Shade the asymmetric extension region (scattering tail window)
									ax2_stacked.axvspan(base_boundary_plot, adaptive_boundary_plot, 
													   alpha=0.3, color='orange', zorder=0,
													   label=f'scattering extension (+{extra_time_ms:.2f} ms)')
								
								# Shade region beyond fit window (if any data exists there)
								right_edge_plot = xs_plot_stack[-1]
								if right_edge_plot > adaptive_boundary_plot:
									ax2_stacked.axvspan(adaptive_boundary_plot, right_edge_plot, 
													   alpha=0.2, color='gray', zorder=0,
													   label='outside fit window')
						except Exception as e:
							print(f"[DEBUG] Adaptive window visualization failed: {e}")
						# === END ADAPTIVE WINDOW VISUALIZATION ===
						
						# Match the upper subplot's x-limits so both panels
						# show the same time window for a multi-component
						# click. plot_xlim is the region when one was
						# selected, else the full burst window.
						ax2_stacked.set_xlim(plot_xlim)
						
						ax2_stacked.set_title(f"Stacked channels {start_chan}-{end_chan} @ {freq_low:.1f} - {freq_high:.1f} MHz")
						ax2_stacked.set_xlabel("Time (ms)")
						ax2_stacked.set_ylabel("Stacked Intensity")
						ax2_stacked.legend(frameon=False)
					
					except Exception as e:
						print(f"Warning: Could not plot stacked channels: {e}")
						ax2_stacked.text(0.5, 0.5, f"Stacked plot failed: {e}", 
									   transform=ax2_stacked.transAxes, ha='center', va='center')
				# === END STACKED PLOT ===
				
				# === PLOT GAUSSIAN STACKED CHANNEL DATA (IF AVAILABLE) ===
				if show_stacked_gauss_plot and stack_gauss_idx is not None:
					try:
						# Get stacked profile and fit data for this stack
						# (per-component state resolved by click time)
						stacked_gauss_row = _popup_gauss_state['profiles'][stack_gauss_idx, :]
						
						# Get the fit parameters from stacked_gauss_fitdata
						# Columns: freqs, tstart, tend, amp, xo, tstart_err, sigma, red_chisq, amp_err, xo_err, sigma_err
						fit_params_gauss = _popup_gauss_state['fitdata'][stack_gauss_idx, :]
						A_stack_g = fit_params_gauss[3]       # amp
						mu_stack_g = fit_params_gauss[4]      # xo (mu)
						sig_stack_g = fit_params_gauss[6]     # sigma
						redchisq_stack_g = fit_params_gauss[7]  # red_chisq
						t_arr_stack_g = fit_params_gauss[1]   # tstart (arrival time)
						duration_stack_g = _popup_gauss_state['subfall_duration']
						
						# Calculate which channels are in this stack
						start_chan_g = stack_gauss_idx * _popup_stack_size
						end_chan_g = min((stack_gauss_idx + 1) * _popup_stack_size, wfall.shape[0]) - 1
						freq_low_g = freqs[start_chan_g]
						freq_high_g = freqs[end_chan_g]
						
						# Build time axes for plotting
						xs_subfall_g = np.linspace(0, duration_stack_g, num=stacked_gauss_row.size)
						
						# Get the shift that converts subfall coords to full waterfall coords
						subfall_shift_g = _popup_gauss_state.get('subfall_shift', 0) or 0
						
						# Convert to plot coordinates (relative to pktime)
						xs_plot_stack_g = xs_subfall_g + subfall_shift_g - pktime
						
						# === PLOT FULL WATERFALL STACKED (lighter blue, for context) ===
						try:
							_popup_wfall_gauss = _popup_gauss_state.get('wfall', None)
							if _popup_wfall_gauss is not None:
								# Stack the same channels from the FULL waterfall
								full_stacked_row_g = np.sum(_popup_wfall_gauss[start_chan_g:end_chan_g+1, :], axis=0)
								
								# Build time axis for full waterfall
								xs_full_g = np.linspace(0, duration, num=full_stacked_row_g.size)
								xs_plot_full_g = xs_full_g - pktime
								
								# Plot as lighter blue, behind the subfall data
								ax2_stacked.plot(xs_plot_full_g, full_stacked_row_g, color='lightblue', 
												alpha=0.8, zorder=1)
						except Exception as e:
							print(f"[DEBUG] Full waterfall Gaussian stacking failed: {e}")
						# === END FULL WATERFALL STACKED ===
						
						# Plot stacked data
						ax2_stacked.plot(xs_plot_stack_g, stacked_gauss_row, label="stacked data (fit region)", color='blue', zorder=5)
						
						# Plot Gaussian model if fit was successful
						if np.isfinite([A_stack_g, mu_stack_g, sig_stack_g]).all():
							# Compute Gaussian model - parameters are in subfall coordinates.
							# Sample-rate evaluation is kept around for the
							# arrival-time markers below; the dashed curve uses
							# a 20×-finer grid to avoid jagged edges.
							model_stack_g = gauss_model(xs_subfall_g, A_stack_g, mu_stack_g, sig_stack_g)

							_fine_factor_stk_g = 20
							xs_subfall_g_fine = np.linspace(xs_subfall_g[0], xs_subfall_g[-1],
							                                _fine_factor_stk_g * len(xs_subfall_g))
							xs_plot_stack_g_fine = xs_subfall_g_fine + (xs_plot_stack_g[0] - xs_subfall_g[0])
							model_stack_g_fine = gauss_model(xs_subfall_g_fine, A_stack_g, mu_stack_g, sig_stack_g)
							ax2_stacked.plot(xs_plot_stack_g_fine, model_stack_g_fine, '--', color='red', 
											label=f"Gaussian (χ²_red={redchisq_stack_g:.2f}, σ={sig_stack_g:.3g} ms)", zorder=6)
							
							# Also plot mixture Gaussian if MULTIPLE components fall inside this
							# stack's cropped window [0, duration_stack_g]. Components outside it
							# (e.g. the other burst in a multi-component file) are excluded: they
							# have no data in this window, so fitgaussmix leaves their amplitude
							# unconstrained (curve_fit drives it huge) and the off-window arrival
							# marker then wrecks the y-autoscale (the flat-line / y=-250 panel).
							# OLD: if len(xos) > 1:
							_xos_win, _sig_win = [], []
							for _xo_i, _sg_i in zip(xos, tmix_sigmas):
								if 0.0 <= _xo_i <= duration_stack_g:
									_xos_win.append(_xo_i); _sig_win.append(_sg_i)
							# fitgaussmix normalises data to max=1 internally; rescale its model
							# back to raw stacked-intensity units so the green overlay + markers
							# sit on the blue data instead of ~1/max(stacked) too tall.
							_mix_scale = float(np.max(stacked_gauss_row))
							if not (np.isfinite(_mix_scale) and _mix_scale > 0):
								_mix_scale = 1.0
							if len(_xos_win) > 1:
								try:
									# Fit mixture model for visualization (in-window components only)
									popt_mix, pcov_mix, red_mix = fitgaussmix(
										stacked_gauss_row, duration_stack_g,
										# OLD: xos=xos, sigmas=tmix_sigmas,
										xos=_xos_win,
										sigmas=_sig_win,
										fix_xos=True,
										tol=0.25 * res_time_ms
									)
									# Smooth mixture overlay on the fine grid, rescaled to raw units.
									# OLD (normalised; off-window comp blew up the panel):
									#   model_mix_g_fine = gaussmix_model(xs_subfall_g_fine, *popt_mix)
									model_mix_g_fine = _mix_scale * gaussmix_model(xs_subfall_g_fine, *popt_mix)
									ax2_stacked.plot(xs_plot_stack_g_fine, model_mix_g_fine, ':', color='green', lw=1.5,
													label=f"mixGauss (χ²_red={red_mix:.2f})", zorder=5)
									
									# Mark arrival times for each component in mixture.
									# Evaluate the mixture model EXACTLY at t_arr_mix
									# instead of snapping (same fix as per-channel popup).
									# OLD: n_comp = len(xos)
									n_comp = len(_xos_win)
									mus_mix = np.atleast_1d(popt_mix[n_comp:2*n_comp])
									sigs_mix = np.atleast_1d(popt_mix[2*n_comp:3*n_comp])
									labeled_mix = False
									for mu_mix_i, sig_mix_i in zip(mus_mix, sigs_mix):
										t_arr_mix = mu_mix_i - np.sqrt(2.0) * np.abs(sig_mix_i)
										if np.isfinite(t_arr_mix):
											# OLD: _y_marker = float(gaussmix_model(np.atleast_1d(t_arr_mix), *popt_mix)[0])
											_y_marker = _mix_scale * float(gaussmix_model(
												np.atleast_1d(t_arr_mix), *popt_mix
											)[0])
											_x_marker_plot = float(t_arr_mix) + (xs_plot_stack_g[0] - xs_subfall_g[0])
											ax2_stacked.plot(_x_marker_plot, _y_marker,
															marker='s', ms=6, mfc='lime', mec='k',
															label=None if labeled_mix else "mix arrival times", zorder=11)
											labeled_mix = True
								except Exception as e:
									print(f"[DEBUG] Mixture Gaussian fit failed for visualization: {e}")
							
							# Mark arrival time on the single Gaussian curve.
							# Evaluate model EXACTLY at t_arr_stack_g (same fix
							# as per-channel popup).
							if np.isfinite(t_arr_stack_g):
								# t_arr_stack_g is in subfall coordinates
								_y_marker = float(gauss_model(
									np.atleast_1d(t_arr_stack_g),
									A_stack_g, mu_stack_g, sig_stack_g
								)[0])
								_x_marker_plot = float(t_arr_stack_g) + (xs_plot_stack_g[0] - xs_subfall_g[0])
								# Red X marker — see the EMG popup above for rationale.
								ax2_stacked.plot(_x_marker_plot, _y_marker,
												marker='x', ms=12, mew=2.5, color='red',
												label=f"arrival time ({_x_marker_plot:.3f} ms)", zorder=10,
												linestyle='None')
						
						# Match the upper subplot's x-limits so both panels
						# show the same time window for a multi-component
						# click. plot_xlim is the region when one was
						# selected, else the full burst window.
						ax2_stacked.set_xlim(plot_xlim)
						# Pin y-limits from the in-window stacked data so a stray off-window
						# artist can't blow up the panel's autoscale (set_xlim does NOT re-scale y).
						_ylo = min(0.0, float(np.nanmin(stacked_gauss_row)))
						_yhi = float(np.nanmax(stacked_gauss_row))
						if np.isfinite(_ylo) and np.isfinite(_yhi) and _yhi > _ylo:
							_pad = 0.15 * (_yhi - _ylo)
							ax2_stacked.set_ylim(_ylo - _pad, _yhi + 2.0 * _pad)
						
						ax2_stacked.set_title(f"Stacked channels {start_chan_g}-{end_chan_g} @ {freq_low_g:.1f} - {freq_high_g:.1f} MHz")
						ax2_stacked.set_xlabel("Time (ms)")
						ax2_stacked.set_ylabel("Stacked Intensity")
						ax2_stacked.legend(frameon=False)
					
					except Exception as e:
						print(f"Warning: Could not plot Gaussian stacked channels: {e}")
						import traceback
						traceback.print_exc()
						ax2_stacked.text(0.5, 0.5, f"Gaussian stacked plot failed: {e}", 
									   transform=ax2_stacked.transAxes, ha='center', va='center')
				# === END GAUSSIAN STACKED PLOT ===

				# --- Per-stack filter audit panel (right-hand column) ---
				# Render the pass/fail ladder for whichever stack was
				# clicked. Prefers the path actually shown: EMG audit if an
				# EMG stack was plotted, else Gaussian.
				if ax2_audit is not None:
					try:
						_audit_obj = None
						_audit_stack_idx = None
						if (show_stacked_plot and _popup_emg_state is not None
								and _popup_emg_state.get('filter_audit') is not None):
							_audit_obj = _popup_emg_state['filter_audit']
							_audit_stack_idx = stack_idx
						elif (show_stacked_gauss_plot and _popup_gauss_state is not None
								and _popup_gauss_state.get('filter_audit') is not None):
							_audit_obj = _popup_gauss_state['filter_audit']
							_audit_stack_idx = stack_gauss_idx

						if _audit_obj is not None and _audit_stack_idx is not None:
							_audit_lines = _audit_obj.render_lines(_audit_stack_idx)
						else:
							_audit_lines = ["Filter audit unavailable",
							                "(no audit recorded for this",
							                " stacking run)"]
						# Monospace so [PASS]/[FAIL] columns align.
						ax2_audit.text(
							0.02, 0.98, "\n".join(_audit_lines),
							transform=ax2_audit.transAxes,
							ha='left', va='top', family='monospace', fontsize=8.5,
							linespacing=1.35)
					except Exception as _aerr:
						ax2_audit.text(0.02, 0.98, f"audit render failed:\n{_aerr}",
						               transform=ax2_audit.transAxes,
						               ha='left', va='top', family='monospace',
						               fontsize=8)

				fig2.tight_layout()
				# non-blocking to avoid Qt "event loop already running" spam
				fig2.canvas.draw_idle(); plt.pause(0.001)
				return


			# --- Right click (single): add component center x0 (and x_c in EMG mode) ---
			if event.button == 3 and not event.dblclick:
				xos.append(x_ms + pktime)
				print(f"'xos' : {[round(xi, 2) for xi in xos]}")
				[ax_tseries.axvline(x=t - pktime, ls="--", c="k") for t in xos]
				([ax_tseries.axvline(x=xc - pktime, ls="--", c="purple") for xc in xcs] if USE_EMG_MIX else [])
				fig.canvas.draw()
				return

			# --- Right click (double): drop a cut time ---
			if event.button == 3 and event.dblclick:
				cuts.append(x_ms + pktime)
				print(f"'cuts' : {[round(xi, 2) for xi in cuts]}")
				[ax_tseries.axvline(x=t - pktime, ls="-.", c="k") for t in cuts]
				fig.canvas.draw()
				return


		def subband_keypress(event):
			"""
			'q'-key handler: when the cursor is hovering over a sub-band region of
			the dynamic spectrum (within ±2σ of the spectral profile, i.e. between
			the cyan boundary lines, and within one of the gray dashed sub-band
			divisions), pop up the cached EMG fit for that sub-band — same fit
			that contributes a point to the τ(ν) and σ diagnostic plots.

			Pulls from the currently TOGGLED component's `_comp_sb_results[ci]`
			(sub_popts, sub_labels, sub_freqs, sub_taus, sub_sigmas) so the
			plotted fit matches the diagnostics exactly. Reduced-χ² is
			recomputed from the cached popt against the reconstructed
			sub-band time-series for display.
			"""
			# Only react to the 'q' key on the waterfall axes
			if event.key != 'q':
				return
			if event.inaxes is not ax_wfall or event.ydata is None:
				return

			# Resolve the active component's sub-band result. The Toggle
			# button updates `_toggle_state['current']` to whichever
			# component the user has cycled to: an int (single component)
			# or None (all-components view).
			#
			# For the None case (stage 1, all-components view) we
			# resolve which component the cursor is over by mapping
			# event.xdata (plot-frame ms) back to the unshifted ms
			# frame and matching against `_comp_time_windows_ms`. This
			# lets the user 'q' any sub-burst's sub-band region in
			# stage 1 without first toggling to it.
			_active_ci = _toggle_state.get('current', None)
			if _active_ci is None:
				_active_ci = pidx  # safe default; refined below if possible
				if event.xdata is not None and _comp_time_windows_ms:
					_click_ms = float(event.xdata) + pktime
					for _i, (_lo_ms, _hi_ms) in enumerate(_comp_time_windows_ms):
						if _lo_ms <= _click_ms <= _hi_ms:
							_active_ci = _i
							break
			if not (0 <= _active_ci < len(_comp_sb_results)):
				print("[q] Toggled component index out of bounds for sub-band results.")
				return
			_active_sb_result = _comp_sb_results[_active_ci]

			# Need at minimum a populated geometric-slot list, even if
			# the power-law α fit failed. The fallback path below shows
			# raw subfall data for slots whose individual EMG fits
			# failed; the geometric bounds are populated by
			# fit_subband_alpha regardless of whether the overall α
			# fit converged. The only case we genuinely can't help with
			# is `_active_sb_result is None` (the call was skipped
			# entirely) or `all_sub_freq_los` empty (the per-slot
			# loop never even started — e.g. chan_per_sub < 2).
			_all_lo_check = _active_sb_result.get('all_sub_freq_los', None) if _active_sb_result is not None else None
			_overall_success = bool(_active_sb_result.get('success', False)) if _active_sb_result is not None else False
			_n_good = int(_active_sb_result.get('n_good_subbands', 0)) if _active_sb_result is not None else 0
			_have_geom_slots = (_all_lo_check is not None and len(_all_lo_check) > 0)
			if _active_sb_result is None:
				print(f"[q] No sub-band α fit was run for component "
				      f"{subburst_suffixes[_active_ci] if _active_ci < len(subburst_suffixes) else _active_ci}.")
				return
			if not (_overall_success or _have_geom_slots):
				print(f"[q] No cached sub-band data available for component "
				      f"{subburst_suffixes[_active_ci] if _active_ci < len(subburst_suffixes) else _active_ci} "
				      f"(power-law α fit failed AND no geometric slot bounds cached). "
				      f"This is the chan_per_sub<2 case in fit_subband_alpha.")
				return

			y_mhz = float(event.ydata)
			sb_freq_lo = _active_sb_result.get('freq_lo', freqs.min())
			sb_freq_hi = _active_sb_result.get('freq_hi', freqs.max())

			# Outside-region check (above/below cyan ±2σ band)
			if not (sb_freq_lo <= y_mhz <= sb_freq_hi):
				print(f"[q] Cursor at {y_mhz:.1f} MHz is outside the sub-band region "
				      f"[{sb_freq_lo:.1f}, {sb_freq_hi:.1f}] MHz "
				      f"(above/below the cyan ±2σ boundary).")
				return

			# Match cursor frequency against sub_labels (each label is "lo-hi" in MHz).
			# fit_subband_alpha skips bad sub-bands via quality cuts (continue), so
			# sub_labels only contains the *good* ones — matching by freq range
			# both finds the right cached fit and naturally skips bands that
			# weren't fit at all.
			sub_labels = _active_sb_result['sub_labels']
			sub_popts  = _active_sb_result['sub_popts']
			sub_freqs  = _active_sb_result['sub_freqs']
			sub_taus   = _active_sb_result['sub_taus']
			sub_sigmas = _active_sb_result['sub_sigmas']
			sub_tau_errs = _active_sb_result['sub_tau_errs']
			# Exact float bounds — preferred. sub_labels (e.g. "492-531") is
			# rounded for display only; using it to reconstruct the channel
			# mask drops or adds boundary channels and shifts the data
			# baseline by a fixed multiplicative factor versus the fit.
			sub_freq_los = _active_sb_result.get('sub_freq_los', None)
			sub_freq_his = _active_sb_result.get('sub_freq_his', None)
			alpha_fit  = _active_sb_result.get('alpha', np.nan)
			alpha_err  = _active_sb_result.get('alpha_err', np.nan)

			matched_idx = None
			matched_lo = matched_hi = None
			# Prefer exact float bounds when present; fall back to parsing
			# the rounded labels only if the result dict predates the fix.
			_have_exact = (sub_freq_los is not None and sub_freq_his is not None
			               and len(sub_freq_los) == len(sub_labels)
			               and len(sub_freq_his) == len(sub_labels))
			for j, lbl in enumerate(sub_labels):
				if _have_exact:
					f_lo = float(sub_freq_los[j])
					f_hi = float(sub_freq_his[j])
				else:
					try:
						f_lo, f_hi = [float(x) for x in lbl.split('-')]
					except (ValueError, AttributeError):
						continue
				if f_lo <= y_mhz <= f_hi:
					matched_idx = j
					matched_lo, matched_hi = f_lo, f_hi
					break

			# Track whether the cursor matched a successful fit (overlay
			# will be drawn) or only a geometric slot whose fit was
			# rejected (fall back to data-only view).
			_fit_unavailable = False

			if matched_idx is None:
				# Cursor fell in a geometric sub-band slot whose fit was rejected
				# by the quality cuts in fit_subband_alpha (σ too narrow, redχ²>5,
				# or non-finite τ). Rather than bailing, try to match the cursor
				# against the ALL-geometric-slots bounds and show the raw subfall
				# data with no fit overlay or residual subplot, so the user can
				# still inspect what the band looked like.
				_all_lo = _active_sb_result.get('all_sub_freq_los', None)
				_all_hi = _active_sb_result.get('all_sub_freq_his', None)
				if (_all_lo is not None and _all_hi is not None
						and len(_all_lo) == len(_all_hi) and len(_all_lo) > 0):
					for j in range(len(_all_lo)):
						f_lo = float(_all_lo[j])
						f_hi = float(_all_hi[j])
						if f_lo <= y_mhz <= f_hi:
							matched_idx = j   # 0-indexed slot within ALL slots
							matched_lo, matched_hi = f_lo, f_hi
							_fit_unavailable = True
							break
				if matched_idx is None:
					# Still no match — cursor isn't in any geometric slot at all.
					print(f"[q] Cursor at {y_mhz:.1f} MHz is inside the ±2σ region but no "
					      f"good cached fit covers this frequency — the corresponding "
					      f"sub-band was rejected by quality cuts (σ<0.001·duration, "
					      f"redχ²>5, or non-finite τ). Good sub-bands: {list(sub_labels)}.")
					return

			# ── Reconstruct the sub-band time series the fit actually saw ──
			# fit_subband_alpha was called with `subfall` (a time-cropped,
			# baseline-subtracted slice of wfall) and its `popt[1]` is in
			# *subfall-local* time, with t=0 at subfall channel 0. To plot the
			# fit on the same waterfall-shifted x-axis as the data, we need:
			#   1. The exact subfall the fit consumed (stashed by the call site)
			#   2. The subfall→wfall channel offset, to shift t=0 of subfall
			#      into wfall coordinates and then by -pktime.
			# Popt is only available for successful fits. For the
			# data-only fallback path (_fit_unavailable=True), skip
			# popt-dependent code below.
			if not _fit_unavailable:
				popt_sb = sub_popts[matched_idx]
			else:
				popt_sb = None
			subfall_cached = _active_sb_result.get('_subfall_for_popup', None)
			subfall_left_chan = _active_sb_result.get('_subfall_left_chan', None)
			subfall_dur_ms = _active_sb_result.get('_subfall_duration_ms', None)
			if subfall_cached is None or subfall_left_chan is None or subfall_dur_ms is None:
				print("[q] Cached subfall is unavailable — re-run the analysis with this "
				      "version of arrivaltimes_stacked to populate the popup cache.")
				return

			mask_sb = (freqs >= matched_lo) & (freqs <= matched_hi)
			sub_ts = subfall_cached[mask_sb, :].sum(axis=0)
			n_time_local = subfall_cached.shape[1]
			# t_axis_local: subfall-local time (matches the axis the fit was made on)
			t_axis_local = np.linspace(0, subfall_dur_ms, n_time_local)
			# t_axis_wfall_shifted: same samples expressed on the main waterfall's
			# shifted axis (so the popup's x-axis matches what you see in the
			# main figure when you hover).
			subfall_left_ms = subfall_left_chan * res_time_ms
			t_axis_plot = t_axis_local + subfall_left_ms - pktime

			# Evaluate cached EMG model — only when a successful fit is
			# available. In the data-only fallback path these stay unset.
			fit_y_fine = None
			t_axis_fine_plot = None
			fit_y_bins = None
			resid = None
			redchisq = np.nan
			oversamp = None
			if not _fit_unavailable:
				# TWO grids matter here:
				#   • Fine grid: smooth instantaneous EMG, for a clean visual curve.
				#     If we evaluate at the data sample rate (n_time_local points,
				#     typically ~ms-spaced for CHIME) and let matplotlib draw line
				#     segments, the model looks jagged for narrow bursts where
				#     σ ≲ dt — the underlying EMG is smooth, but its sampled
				#     values are sparse. So we evaluate on a 20×-finer grid here.
				#   • Bin-centre grid: bin-integrated EMG, evaluated at the same
				#     bin centres as the data. This is the model that
				#     _fit_emg_subband actually minimised against (it passes
				#     dt_ms=res_time_ms, triggering bin_integrated_emg with
				#     adaptive oversamp). Residuals and χ²_red MUST use this
				#     model — using the instantaneous emg_model here gave a
				#     biased χ²_red for narrow bursts where bin-integration
				#     materially flattens the modelled peak.
				fine_factor = 20
				t_axis_fine = np.linspace(0, subfall_dur_ms, fine_factor * n_time_local)
				t_axis_fine_plot = t_axis_fine + subfall_left_ms - pktime
				fit_y_fine = emg_model(t_axis_fine, *popt_sb)

				# Bin-integrated model at data bin centres (matches the fit)
				_sigma_pop = popt_sb[2]
				_lam_pop   = popt_sb[3]
				_tau_pop   = (1.0 / _lam_pop) if (np.isfinite(_lam_pop) and _lam_pop > 0) else np.nan
				oversamp = pick_oversamp(res_time_ms, _sigma_pop, _tau_pop)
				fit_y_bins = bin_integrated_emg(
					t_axis_local, *popt_sb,
					dt_ms=res_time_ms, oversamp=oversamp
				)
				resid = sub_ts - fit_y_bins
			# χ²_red, noise estimate — only meaningful when a fit exists.
			if not _fit_unavailable:
				n_params = 5  # (A, mu, sigma, lamb, C)
				dof = max(1, n_time_local - n_params)
				# Noise estimate: MAD of off-pulse residuals (more robust than full-array std).
				# Use a 3σ trim of |resid| against an initial std as the off-pulse mask.
				_init_std = np.std(resid) if np.std(resid) > 0 else 1.0
				_off_mask = np.abs(resid) < 3.0 * _init_std
				if _off_mask.sum() > 10:
					noise = 1.4826 * np.median(np.abs(resid[_off_mask] - np.median(resid[_off_mask])))
				else:
					noise = np.std(resid)
				noise = noise if noise > 0 else 1.0
				redchisq = float(np.sum((resid / noise) ** 2) / dof)

				# Cached scalars for this sub-band (successful-fit case)
				sigma_ms = float(sub_sigmas[matched_idx])
				tau_ms   = float(sub_taus[matched_idx])
				tau_err_ms = float(sub_tau_errs[matched_idx]) if matched_idx < len(sub_tau_errs) else np.nan
				nu_eff   = float(sub_freqs[matched_idx])
			else:
				# Data-only fallback: no fit scalars to report.
				sigma_ms = np.nan
				tau_ms = np.nan
				tau_err_ms = np.nan
				nu_eff = 0.5 * (matched_lo + matched_hi)

			# ── Popup figure layout depends on whether a fit is available ──
			# Successful fit → two-panel layout (profile+fit on top, residuals
			#                  on bottom).
			# No fit         → single-panel layout, just the raw sub-band data.
			if not _fit_unavailable:
				fig_sb, (ax_sb_top, ax_sb_bot) = plt.subplots(
					2, 1, figsize=(8, 6), height_ratios=[3, 1], sharex=True
				)
			else:
				fig_sb, ax_sb_top = plt.subplots(1, 1, figsize=(8, 5))
				ax_sb_bot = None

			# Top: data + EMG fit (when available) + arrival time
			ax_sb_top.plot(t_axis_plot, sub_ts, lw=1.0, label='Sub-band data (subfall)')
			if not _fit_unavailable:
				# Smooth instantaneous EMG on a fine grid — this is the analytic
				# curve the popt parameters describe.
				ax_sb_top.plot(t_axis_fine_plot, fit_y_fine, 'r--', lw=1.6,
				               label='EMG fit (continuous)')
				# Bin-integrated EMG at the data bin centres — this is what the
				# fit actually minimised against. Shown as small dots so that
				# the gap between continuous (red dashed) and bin-integrated
				# (red dots) is visible for narrow pulses where σ ≲ dt; for
				# well-resolved pulses the two will overlap exactly.
				ax_sb_top.plot(t_axis_plot, fit_y_bins, 'r.', ms=5, alpha=0.9,
				               label=f'EMG fit (bin-integrated, ×{oversamp})')
				# Mark the EMG mean μ — popt[1] is in subfall-local time, so convert
				# to the same waterfall-shifted axis used for plotting.
				mu_local = popt_sb[1]
				mu_plot = mu_local + subfall_left_ms - pktime
				ax_sb_top.axvline(mu_plot, ls='--', c='k', alpha=0.6,
				                  label=f'μ = {mu_plot:+.2f} ms')

			# Title — successful-fit titles report σ/τ/α/χ²_red; failed-fit
			# titles say "fit unavailable" so the user knows what they're
			# looking at.
			if not _fit_unavailable:
				alpha_str = (f"α = {alpha_fit:+.2f} ± {alpha_err:.2f}"
				             if (np.isfinite(alpha_fit) and np.isfinite(alpha_err))
				             else (f"α = {alpha_fit:+.2f}" if np.isfinite(alpha_fit) else "α = nan"))
				tau_str = (f"τ = {tau_ms:.3f} ± {tau_err_ms:.3f} ms"
				           if np.isfinite(tau_err_ms) else f"τ = {tau_ms:.3f} ms")
				ax_sb_top.set_title(
					f"{bname} — Sub-band {matched_lo:.0f}–{matched_hi:.0f} MHz "
					f"(ν_eff = {nu_eff:.1f} MHz)\n"
					f"σ = {sigma_ms:.3f} ms,  {tau_str},  {alpha_str},  "
					f"χ²_red = {redchisq:.2f}",
					fontsize=10
				)
			else:
				ax_sb_top.set_title(
					f"{bname} — Sub-band {matched_lo:.0f}–{matched_hi:.0f} MHz "
					f"(ν_eff = {nu_eff:.1f} MHz)\n"
					f"⚠ EMG fit unavailable (rejected by quality cuts) — showing raw data only",
					fontsize=10, color='#660000'
				)
			ax_sb_top.set_ylabel('Intensity (sum over sub-band)')
			ax_sb_top.legend(fontsize=8, loc='best')
			ax_sb_top.grid(alpha=0.25)
			# When there's no residual panel, the top panel owns the x-label.
			if _fit_unavailable:
				ax_sb_top.set_xlabel('Time (ms)')

			# Bottom: residuals — only drawn when a fit is available.
			if not _fit_unavailable:
				ax_sb_bot.axhline(0, color='gray', lw=0.8, alpha=0.6)
				ax_sb_bot.plot(t_axis_plot, resid, lw=0.8, color='C0')
				ax_sb_bot.set_xlabel('Time (ms)')
				ax_sb_bot.set_ylabel('Residual')
				ax_sb_bot.grid(alpha=0.25)

			fig_sb.tight_layout()
			fig_sb.show()

			if not _fit_unavailable:
				print(f"[q] Sub-band {matched_lo:.0f}-{matched_hi:.0f} MHz  "
				      f"(geometric idx {matched_idx} of {len(sub_labels)} good): "
				      f"σ={sigma_ms:.3f} ms, τ={tau_ms:.3f} ms, χ²_red={redchisq:.2f}")
			else:
				print(f"[q] Sub-band {matched_lo:.0f}-{matched_hi:.0f} MHz: "
				      f"showing raw subfall data only — EMG fit was rejected by "
				      f"quality cuts (σ<0.001·duration, redχ²>5, or non-finite τ)")


	cid = fig.canvas.mpl_connect('button_press_event', printinfo)
	# Free 'q' from Matplotlib's default keymap.quit (which would close the
	# figure before/after our handler fires) so the sub-band popup actually
	# shows. We mutate rcParams rather than the per-figure keymap because
	# matplotlib stores the active keymap on the rcParams at canvas-creation
	# time. Leaves Ctrl-W / Cmd-W as the close shortcut.
	try:
		_kq = list(plt.rcParams.get('keymap.quit', []))
		if 'q' in _kq:
			_kq.remove('q')
			plt.rcParams['keymap.quit'] = _kq
	except Exception as _e:
		print(f"[q] Note: could not strip 'q' from keymap.quit ({_e}); "
		      f"the figure may close when 'q' is pressed.")
	cid_q = fig.canvas.mpl_connect('key_press_event', subband_keypress)

	if show:
		plt.show(block=True)
		# Monitor hack
		# mngr = plt.get_current_fig_manager()
		# # to put it into the upper left corner for example:
		# # mngr.window.setGeometry(50,100,640, 545) # newX, newY, dx, dy
		# geom = mngr.window.geometry()
		# x,y,dx,dy = geom.getRect()
		# print(x, y, dx, dy)
	if save:
		if outdir == '' or outdir[-1] == '/':
			outname = f"{outdir}{bname}_DM{targetDM:.3f}{outfmt}"
		else:
			outname = f"{outdir}/{bname}_DM{targetDM:.3f}{outfmt}"
		fig.savefig(outname)
		print(f"Info: Saved {outname}.")

	# === BURST DURATION RESULT ===
	if diag_enabled('BURST_DURATION'):
		print(f"\n=== BURST DURATION RESULT ===\nσ_t (intrinsic duration) = {results[-1][6]:.4f} ± {results[-1][7]:.4f} ms\nτ (scattering timescale) = {results[-1][20]:.4f} ± {results[-1][21]:.4f} ms\n")

	if return_arrivaltimes and not return_fig:
		plt.close()
		return results, subdf
	elif not return_arrivaltimes and return_fig:
		return results, fig
	elif return_arrivaltimes and return_fig:
		return results, subdf, fig
	else:
		plt.close()
		return results

drift_columns = [
	'name',
	'DM',
	'center_f (MHz)',
	'center_f_err',
	'duration (ms)',
	'duration_err',
	'drift (ms/MHz)',
	'drift_err',
	'redchisq_drift',
	'source'
]
def measuredrifts(
	df,
	show_plot=True,
	verbose=False,
	search_paths=[]
):
	"""
	Using a results spreadsheet created from `measureburst`, find bursts with multiple components and compute their drift rate
	using the t0 and center_f measurement data.

	Args:
		df
		show_plot (bool, optional): If True, show a plot of the drift rate measurement. False by default.
		search_paths (list[str], optional): If provided, and if show_plot is True,
			will search for a corresponding .npz file of the burst waterfall and load
			it in order to display a plot of the waterfall with the measured drift rate overlaid.
		verbose (bool, optional): If True, print a lot of information about the drift measurement, including the components used.
	"""
	df = df[df['t0 (ms)'].notna()] # ignore rows that don't have t0 data
	seldf = df.loc[df.index.str.endswith('_a')]
	# bursts with components as identified by the '_a' suffix.
	# If source is a column, then the spreadsheet is multi-source data

	if 'source' in seldf.columns:
		targetbursts = zip(
			[s[:-2] for s in seldf.index],
			seldf['source']
		)
	else:
		targetbursts = [s[:-2] for s in seldf.index]

	if verbose: print(f"Info: {targetbursts = }")
	rows = [] # name, drift (ms/MHz), drift_err, duration (ms), duration_err
	for burst in targetbursts:
		if 'source' in seldf.columns:
			burst, source = burst
			targetdf = df.loc[(df.index.str.startswith(burst+'_')) & (df.source == source)]
		else:
			targetdf = df.loc[df.index.str.startswith(burst+'_')]
			source = ''
		if verbose: print(targetdf)

		duration = targetdf['t0 (ms)'][-1] - targetdf['t0 (ms)'][0] # ÃŽâ€t between first and last components
		duration_err = np.sqrt(targetdf['t0_err'][-1]**2 + targetdf['t0_err'][0]**2)
		mean_freq = targetdf['center_f (MHz)'].mean()
		mean_freq_err = np.sqrt((targetdf['center_f_err']**2).sum())/len(targetdf)
		odrjob = scipy.odr.ODR(
			scipy.odr.RealData(
				targetdf['center_f (MHz)'],
				targetdf['t0 (ms)'],
				sx=targetdf['center_f_err'],
				sy=targetdf['t0_err'],
			),
			scipy.odr.Model(lambda B, x: B[0]*x + B[1]),
			beta0=[-1, 0]
		)
		odrjob.set_job(fit_type=0)
		odrfit = odrjob.run()
		# odrfit.pprint()
		drift_redchisq = odrfit.res_var # https://stackoverflow.com/a/21406281/3133399
		# drift, drift_err = odrfit.beta[0], odrfit.sd_beta[0] # sd_beta is scaled by reduced chisq
		drift, drift_err = odrfit.beta[0], np.sqrt(np.diag(odrfit.cov_beta))[0]

		rows.append([
			burst,
			targetdf['DM'][0],
			mean_freq,
			mean_freq_err,
			duration,
			duration_err,
			drift,
			drift_err,
			drift_redchisq,
			source
		])

		if show_plot:
			fig, ax = plt.subplots(1,1, figsize=(6,5))
			times = np.linspace(targetdf['t0 (ms)'].min()*0.9, targetdf['t0 (ms)'].max()*1.1, num=100)
			ax.errorbar(
				targetdf['t0 (ms)'],
				targetdf['center_f (MHz)'],
				xerr=targetdf['t0_err'],
				yerr=targetdf['center_f_err'],
				fmt='ko',
			)
			ax.plot(
				times,
				(1/drift)*times+(-odrfit.beta[1]/drift),
				label=f'ODR: $\\Delta t / \\Delta \\nu = ${scilabel(drift, drift_err)} ms/MHz',
			)
			ax.set_title(f"{burst} $\\sigma_T = $ {duration:.2f} ms")
			plt.legend()
			plt.show(block=True)

	return rows

if __name__ == 'n__main__':
	df = pd.read_csv('measurements/frb20121102A/gajjar2018/results_Aug-16-2024-560.105.csv').set_index('name')
	measuredrifts(
		df,
		search_paths=['/Users/mchamma/dev/SurveyFRB20121102A/data/gajjar2018/']
	)


def measure_allmethods(
	filename,
	show=True,
	p0tw=0.01,
	p0bw=100,
	outdir='measurements/collected/',
	**kwargs
	):
	"""
	Collect spectro-temporal measurements of a burst using multiple techniques.
	Utility for comparing the result of spectro-temporal measurements of a burst obtained from the following
	techniques:
	1. arrival times method
	2. ACF method
	3. direct 2d gaussian with three gaussian forms

	Args:
		filename (str): path to .npz of burst in frbgui burst format
		kwargs: keywords to pass to `measureburst` for loading/processing the waterfall.
			`measureburst` with `loadonly=True` is used to load the burst.
	"""
	model_results = []
	precalc_results = []
	print(f"{kwargs =}")
	(
		wfall,
		freqs,
		times_ms,
		res_freq,
		res_time_ms,
		targetDM,
		t_popt
	) = measureburst(
		filename,
		loadonly=True,
		**kwargs
	)

	freqs_bin0 = min(freqs)
	tseries = np.nanmean(wfall, axis=0)
	pktime = np.nanargmax(tseries)*res_time_ms
	extent, corrext = driftrate.getExtents(wfall, res_freq, res_time_ms, freqs_bin0)

	plt.close()
	fig, axs = plt.subplots(1,2, figsize=(12,5.5))
	axs[0].imshow(
		wfall,
		aspect='auto',
		origin='lower',
		extent=extent,
		interpolation='none',
		norm='linear',
		vmax=np.quantile(wfall, 0.999),
	)
	axs[0].annotate(
		f"$DM =$ {targetDM:.3f} pc/cm$^3$",
		xy=(0.05, 0.925),
		xycoords='axes fraction',
		color='white',
		weight='black',
		size=10,
		bbox={"boxstyle":"round"}
	)
	axs[0].set_xlabel("Time (ms)")
	axs[0].set_ylabel("Frequency (MHz)")

	## Arrival times measurement
	arr_result, arrtimesdf = measureburst(
		filename,
		save=False,
		show=False,
		return_arrivaltimes=True,
		**kwargs
	)

	arrdf = pd.DataFrame(
		data=arr_result,
		columns=results_columns
	).set_index('name')
	print(arrdf[['dtdnu (ms/MHz)']], 'ms/MHz', arrdf[['dtdnu_err']], 'ms/MHz')

	print("Arrival Times method:")
	print(f"\t{arrdf.iloc[0]['dtdnu (ms/MHz)']:.4e} +/- {arrdf.iloc[0]['dtdnu_err']:.4e} ms/MHz")



	if len(arrtimesdf) > 0:
		# Use EMG HWHM arrivals when available; otherwise fall back to tstart
		arr_col = 't_emg_hwhm' if (USE_EMG_MIX and 't_emg_hwhm' in arrtimesdf.columns) else 'tstart'
	
		axs[0].scatter(  # component fit points
			arrtimesdf[arr_col],
			arrtimesdf['freqs'],
			# See ax_wfall.scatter in measureburst for rationale: fill
			# in the component color, black outline for contrast, and
			# clip alpha to [0.4, 1.0] to keep low-amplitude markers
			# visible.
			c=arrtimesdf['color'],
			edgecolors='black',
			marker='o',
			s=25,
			linewidths=0.8,
			alpha=np.clip(
				arrtimesdf['amp'] / arrtimesdf['amp'].max(),
				0.4, 1.0
			),
			label=f"$dt/d\\nu =$ {scilabel(arrdf.iloc[0]['dtdnu (ms/MHz)'], arrdf.iloc[0]['dtdnu_err'])} ms/MHz"
		)

		
		

		axs[0].plot( # <--- This line does not appear in the correct spot
# 			times_ms-pktime,
			(1/arrdf.iloc[0]['dtdnu (ms/MHz)'])*(times_ms-pktime) - arrdf.iloc[0]['tb (ms)']/arrdf.iloc[0]['dtdnu (ms/MHz)'],
			'w--',
			alpha=0.75,
			# label=f'$dt/d\\nu = $ {dtdnu:.2e} $\\pm$ {dtdnu_err:.2e}'
			label='arrival times fit'
		)
	axs[0].set_xlim(extent[0], extent[1])
	axs[0].set_ylim(extent[2], extent[3])

	## ACF measurement (-3470 mhz/ms in frbgui)
	p0s = [
		[1, pktime, freqs[len(freqs)//2], p0tw, p0bw, 0],  # amp, xo, yo, sigma_x, sigma_y, theta
		[1, pktime, 0., freqs[len(freqs)//2], p0tw, p0bw], # amp, t0, dt, nu0, sigma_t, sigma_nu
		[1, pktime, 0., freqs[len(freqs)//2], p0tw, p0bw]  # amp, t0, dnu, nu0, w_t, w_nu
	]

	(
		slope,
		slope_error,
		cpopt,
		cperr,
		_, # theta,
		_, # red_chisq,
		_, # center_f,
		_, # center_f_err,
		fitmap,
	) = driftrate.processBurst(
		wfall/np.max(wfall),
		res_freq,
		res_time_ms,
		freqs_bin0,
		p0=[1, 0, 0, p0tw, p0bw, 0],
		verbose=False,
		plot=False
	)
	print(f"ACF method: \n\t{slope = } +/- {slope_error} MHz/ms")
	print(f"\t{1/slope:.4e} +/- {slope_error/slope**2:.4e} ms/MHz")

	precalc_results += [1/slope, slope_error/slope**2]
	model_results += list(cpopt)+list(cperr)

	corr = driftrate.autocorr2d(wfall)

	corrplot = axs[1].imshow(
		corr,
		interpolation='none',
		aspect='auto',
		cmap='gray',
		extent=corrext,
		origin='lower',
	)
	corrplot.set_clim(0, np.max(corr))
	if cpopt[0] > 0:
		c = axs[1].contour(
			fitmap,
			[cpopt[0]/4, cpopt[0]*0.9],
			colors='b',
			alpha=0.75,
			extent=corrext,
		)
		c.collections[0].set_label(
			f"ACF: $dt/d\\nu =$ {scilabel(1/slope, slope_error/slope**2)} ms/MHz"
		)
	axs[1].set_xlabel("Time lag (ms)")
	axs[1].set_ylabel("Frequency lag (MHz)")

	## ACF with constant floor
	print(f"{cpopt = }")
	print("ACF with floor:")
	(
		slope2,
		slope_error2,
		cpopt2,
		cperr2,
		_, # theta,
		_, # red_chisq,
		_, # center_f,
		_, # center_f_err,
		fitmap2,
	) = driftrate.processBurst(
		wfall/np.max(wfall),
		res_freq,
		res_time_ms,
		freqs_bin0,
		p0=[1, 0, 0, p0tw, p0bw, 0, 0],
		verbose=False,
		plot=False,
		usefloor=True
	)
	print(f"ACF with floor method: \n\t{slope2 = } +/- {slope_error2} MHz/ms, floor = {cpopt2[6]}")
	print(f"\t{1/slope2:.4e} +/- {slope_error2/slope2**2:.4e} ms/MHz")

	if cpopt2[0] > 0:
		c = axs[1].contour(
			fitmap2,
			[cpopt2[0]/4, cpopt2[0]*0.9],
			colors='g',
			alpha=0.75,
			extent=corrext,
		)
		c.collections[0].set_label(
			f"ACF floored: $dt/d\\nu =$ {scilabel(1/slope2, slope_error2/slope2**2)} ms/MHz \nfloor = {cpopt2[6]:.2e}"
		)
	axs[0].legend()

	## Gaussian models measurements direct to waterfall
	models = [driftrate.twoD_Gaussian, driftrate.gaussian_dt, driftrate.gaussian_dnu] # preserve order
	sigma = np.std( wfall[:, 0:50])
	for model, p0 in zip(models, p0s):
		try:
			popt, pcov = driftrate.fitdatagaussiannlsq(
				wfall,
				extent,
				p0=p0,
				sigma=sigma,
				model=model
			)
		except RuntimeError as e:
			print(f"Error: {e}")
			popt = [0]*len(p0)
			pcov = np.diag([0]*len(p0))

		# print(popt)
		r = wfall - driftrate.makeDataFitmap(popt, wfall, extent, model=model)
		chisq = np.sum((r / sigma) ** 2)
		redchisq = chisq / (wfall.shape[0]*wfall.shape[1] - len(popt))

		perr = np.sqrt(np.diag(pcov))

		print(f"{model.__name__}: {sigma = } {redchisq = }")

		if model == driftrate.twoD_Gaussian: # amp, xo, yo, sigma_x, sigma_y, theta
			units = ['']*len(p0)
			lbls = ['$A$', '$x_0$', '$y_0$', '$\\sigma_x$', '$\\sigma_y$', '$\\theta$']

			theta = popt[5] if abs(popt[3]) > abs(popt[4]) else popt[5] - np.pi/2 # defined via eqs. A2 and A3
			slope = np.tan(theta) # MHz/ms
			theta_err = perr[-1]
			slope_error = (theta_err * (1/np.cos(theta))**2)

			print(f"\t{slope:.4e} MHz/ms +/- {slope_error}")
			print(f"\t{1/slope:.4e} +/- {slope_error/slope**2:.4e} ms/MHz")
			precalc_results += [1/slope, slope_error/slope**2]
			legend_lbl = f"G: $dt/d\\nu =$ {scilabel(1/slope, slope_error/slope**2)} ms/MHz"
		elif model == driftrate.gaussian_dt: # amp, t0, dt, nu0, sigma_t, sigma_nu
			units = ['', 'ms', 'ms/MHz', 'MHz', 'ms', 'MHz']
			lbls = ['$A_{dt}$', '$t_0$', '$d_t$', '$\\nu_0$', '$\\sigma_t$', '$\\sigma_\\nu$']

			print(f"\t{popt[2]:.4e} +/- {perr[2]:.4e} ms/MHz")
			precalc_results += [popt[2], perr[2]]
			legend_lbl = f"$d_t =$ {scilabel(popt[2], perr[2])} ms/MHz"
		elif model == driftrate.gaussian_dnu: # amp, t0, dnu, nu0, w_t, w_nu
			units = ['', 'ms', 'MHz/ms', 'MHz', 'ms', 'MHz']
			lbls = ['$A_{dnu}$', '$t_0$', '$d_\\nu$', '$\\nu_0$', '$w_t$', '$w_\\nu$']

			if popt[0] > 0:
				gnu_dt = popt[2] / (popt[2]**2 + popt[5]**2/popt[4]**2)
			else:
				gnu_dt = 0

			print(f"\t {gnu_dt = :.4e} ms/MHz")
			precalc_results += [gnu_dt, -1] # uncertainty needs to be derived from eq. A4 in Jahns+2023
			legend_lbl = ''#f"$d_{{t,g\\nu}} =$ {scilabel(gnu_dt, -1)} ms/MHz"
			# if perr[0] == np.inf:
			# 	perr = [-1]*len(perr)

		# Output
		model_results += list(popt)+list(perr)
		resultstr = ', '.join([f"{lbl} = {f:.4e} {unit}" for lbl, f, unit in zip(lbls, popt, units)])
		print('\t', resultstr)
		errstr = ', '.join([f"d{lbl} = {f:.4e} {unit}" for lbl, f, unit in zip(lbls, perr, units)])
		print('\t', errstr)

		poptmap = driftrate.makeDataFitmap(
			popt,
			wfall,
			extent,
			model=model
		)

		contours, cr = [], 'w'
		if popt[0] > 0:
			contours, cr = [popt[0]/4, popt[0]*0.9], 'w'
		elif popt[0] < 0:
			contours, cr = [-popt[0]/4, -popt[0]*0.9], 'r'
		c = axs[0].contour(
			poptmap,
			contours,
			colors=cr,
			alpha=0.33,
			extent=extent,
		)
		if contours != []:
			c.collections[0].set_label(legend_lbl)
		bname = filename.split('/')[-1].split('.')[0]
		axs[0].set_title(bname)

	# axs[0].legend(handlelength=0)
	try:
		axs[0].legend(handlelength=0, fontsize='small', loc=4)
		axs[1].legend(ncols=1, handlelength=0, fontsize='small', loc=4)
	except IndexError as e:
		print("error: weird legend bug")


	# === MCMC RESULTS COMPILATION ===
	mcmc_results = []
	if '_mcmc_results' not in dir():
		_mcmc_results = None
# 	if USE_MCMC and "_mcmc_results" in locals() and _mcmc_results is not None:
		if _mcmc_results["status"] == "success":
			_A, _mu, _sigma, _lamb = _mcmc_results["median"]
			_err_low = _mcmc_results["err_low"]
			_err_high = _mcmc_results["err_high"]
			mcmc_results = [
				_mcmc_results["status"],  # mcmc_status
				_A, _err_low[0], _err_high[0],  # mcmc_A, mcmc_A_err_low, mcmc_A_err_high
				_mu, _err_low[1], _err_high[1],  # mcmc_toa, mcmc_toa_err_low, mcmc_toa_err_high
				_sigma, _err_low[2], _err_high[2],  # mcmc_sigma, mcmc_sigma_err_low, mcmc_sigma_err_high
				_lamb, _err_low[3], _err_high[3],  # mcmc_lambda, mcmc_lambda_err_low, mcmc_lambda_err_high
				_mcmc_results["tau"], _mcmc_results["tau_err_low"], _mcmc_results["tau_err_high"],  # mcmc_tau
				_mcmc_results["baseline"],  # mcmc_C
			]
		else:
			# Fallback case - all NaN except status
			mcmc_results = ["fallback"] + [np.nan] * 15
	else:
		# MCMC not used - all NaN
		mcmc_results = ["not_used"] + [np.nan] * 15
	# === END MCMC RESULTS ===

		results_allmethods = list(arrdf.reset_index().iloc[0]) + precalc_results + model_results
	results_allmethods = results_allmethods + mcmc_results

	plt.tight_layout()
	if show: plt.show(block=True)
	plt.savefig(f"{outdir}{bname}")
	plt.close()
	return results_allmethods

allmethods_columns = [
	## precalc columns
	'dtdnu_ACF (ms/MHz)',
	'dtdnu_ACF_err',
	'dtdnu_g (ms/MHz)',
	'dtdnu_g_err',
	'dt_gdt (ms/MHz)',
	'dt_gdt_err',
	'dt_gdnu (ms/MHz)',
	'dt_gdnu_err',
	## Raw parameter columns with uncertainties
	'A_ACF',
	'x0_ACF',
	'y0_ACF',
	'sigma_x_ACF',
	'sigma_y_ACF',
	'theta_ACF',
	'A_ACF_err',
	'x0_ACF_err',
	'y0_ACF_err',
	'sigma_x_ACF_err',
	'sigma_y_ACF_err',
	'theta_ACF_err',
	##
	'A_g',
	'x0',
	'y0',
	'sigma_x',
	'sigma_y',
	'theta',
	'A_g_err',
	'x0_err',
	'y0_err',
	'sigma_x_err',
	'sigma_y_err',
	'theta_err',
	##
	'A_dt',
	't0_gdt (ms)',
	'd_t (ms/MHz)',
	'nu0_gdt (MHz)',
	'sigma_t (ms)',
	'sigma_nu (MHz)',
	'A_dt_err',
	't0_gdt_err',
	'd_t_err',
	'nu0_gdt_err',
	'sigma_t_err',
	'sigma_nu_err',
	##
	'A_dnu',
	't0_gdnu (ms)',
	'd_nu (MHz/ms)',
	'nu0_gdnu (MHz)',
	'w_t (ms)',
	'w_nu (MHz)',
	'A_dnu_err',
	't0_gdnu_err',
	'd_nu_err',
	'nu0_gdnu_err',
	'w_t_err',
	'w_nu_err',
	## MCMC columns
	"mcmc_status",
	"mcmc_A",
	"mcmc_A_err_low",
	"mcmc_A_err_high",
	"mcmc_toa",
	"mcmc_toa_err_low",
	"mcmc_toa_err_high",
	"mcmc_sigma",
	"mcmc_sigma_err_low",
	"mcmc_sigma_err_high",
	"mcmc_lambda",
	"mcmc_lambda_err_low",
	"mcmc_lambda_err_high",
	"mcmc_tau",
	"mcmc_tau_err_low",
	"mcmc_tau_err_high",
	"mcmc_C",
]


# if __name__ == '__main__':
# 	files = glob.glob('/Users/mchamma/dev/SurveyFRB20121102A/data/snelders2023/figure_1/*.npz')
# 	files = glob.glob('/Users/mchamma/dev/frbdata/FRB20220912A/sheikh2023/npzs/*.npz')
#	files = glob.glob(prefix+'*.npz')
# 	[print(f) for f in sorted(files)]
# 	exit()s