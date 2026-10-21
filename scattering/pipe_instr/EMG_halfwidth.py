"""
EMG_halfwidth.py -- mode and leading-edge half-maximum point of an
exponentially modified Gaussian, found numerically (scipy minimize_scalar
for the mode, brentq for the half-maximum crossing). Used by
arrivaltimes_stacked_v56_9.py to define per-channel arrival times.

Author: Victor Pop
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import erfc
from scipy.optimize import minimize_scalar, brentq
import argparse

# ---------- EMG math ----------
def emg_pdf(x, mu, sigma, tau):
    lam = 1.0 / tau
    z = (mu + (lam * sigma**2) - x) / (np.sqrt(2) * sigma)
    return 0.5 * lam * np.exp(lam*(mu - x) + 0.5*(lam*sigma)**2) * erfc(z)

def emg_left_hwhm(mu, sigma, tau):
    """Return mode, f_mode, left crossing x_left, and left half-width only."""
    def neglogpdf(x):
        fx = emg_pdf(x, mu, sigma, tau)
        return np.inf if fx <= 0 else -np.log(fx)

    # Find the mode (peak) numerically
    width = 10.0 * max(sigma, tau)
    res = minimize_scalar(neglogpdf, bounds=(mu - width, mu + width), method='bounded')
    x_mode = res.x
    f_mode = emg_pdf(x_mode, mu, sigma, tau)
    half = 0.5 * f_mode

    # Solve f(x) = half only on the LEFT of the mode
    def g(x): return emg_pdf(x, mu, sigma, tau) - half

    L_lo, L_hi = x_mode - width, x_mode - 1e-9
    for _ in range(5):
        if g(L_lo) < 0 and g(L_hi) > 0:
            break
        L_lo -= width

    x_left = brentq(g, L_lo, L_hi)
    HWHM_left = x_mode - x_left
    return dict(x_mode=x_mode, f_mode=f_mode, x_left=x_left, HWHM_left=HWHM_left)

# ---------- Randomization ----------
def random_emg_params(seed=None,
                      mu_range=(-2.0, 2.0),
                      sigma_range=(0.3, 2.0),
                      tau_range=(0.3, 5.0)):
    """Draw a random (mu, sigma, tau). sigma and tau ~ log-uniform."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(*mu_range)
    sigma = np.exp(rng.uniform(np.log(sigma_range[0]), np.log(sigma_range[1])))
    tau   = np.exp(rng.uniform(np.log(tau_range[0]),   np.log(tau_range[1])))
    return mu, sigma, tau

# ---------- Plotting ----------
def plot_emg_with_left_hwhm(mu, sigma, tau, save_path=None, show=True):
    stats = emg_left_hwhm(mu, sigma, tau)

    # x-range scaled to parameters
    span = 12 * max(sigma, tau)
    x = np.linspace(mu - 0.5*span, mu + 1.5*span, 2400)
    y = emg_pdf(x, mu, sigma, tau)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, label="EMG pdf")
    ax.axvline(stats["x_mode"], linestyle="--", label="Mode")
    ax.axhline(0.5*stats["f_mode"], linestyle=":", label="Half-maximum")

    # Left half-maximum point only
    ax.plot([stats["x_left"]], [0.5*stats["f_mode"]], marker="o", label="Left HWHM point")

    ax.annotate(f"HWHM_L ≈ {stats['HWHM_left']:.3g}",
                xy=(stats["x_left"], 0.5*stats["f_mode"]),
                xytext=(stats["x_mode"] - 0.6*(stats["x_mode"]-stats["x_left"]), 0.8*stats["f_mode"]),
                arrowprops=dict(arrowstyle="->"), ha="center")

    ax.set_xlabel("x")
    ax.set_ylabel("EMG density")
    ax.set_title(f"EMG with Left HWHM (leading edge)\n"
                 f"(mu={mu:.3g}, sigma={sigma:.3g}, tau={tau:.3g})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)  # Windows-safe
        plt.savefig(save_path, dpi=160)
        print(f"Saved to: {save_path.resolve()}")
    if show:
        plt.show()
    plt.close(fig)

    return stats

# ---------- CLI / main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random EMG left half-width demo")
    parser.add_argument("--n", type=int, default=1, help="number of random plots to generate")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="directory to save PNGs; if omitted, just shows the plots")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        mu, sigma, tau = random_emg_params(seed=rng.integers(0, 2**31-1))
        save_path = None
        if args.save_dir:
            save_path = Path(args.save_dir) / f"emg_left_hwhm_{i+1:03d}.png"
        stats = plot_emg_with_left_hwhm(mu, sigma, tau, save_path=save_path, show=(args.save_dir is None))
        print(f"Run {i+1}: mu={mu:.4g}, sigma={sigma:.4g}, tau={tau:.4g}, "
              f"HWHM_L={stats['HWHM_left']:.4g}, x_mode={stats['x_mode']:.4g}, x_left={stats['x_left']:.4g}")
