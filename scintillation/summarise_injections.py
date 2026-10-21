#!/usr/bin/env python
"""
summarise_injections.py -- tables and figure from inject_results.npz.

Per (bandwidth, m) it reports, over trials:
  P_det  : fraction of trials with a formal detection in >= 1 block
           (acf_scintillation.py criterion: nu_s and m both > 3 sigma and
           nu_s > 1.5 channels)
  P_mod  : fraction with a > 3 sigma modulation in >= 1 block, width ignored
  P_comb : fraction where the error-weighted mean m over the four blocks
           (the script's combined statistic) is > 3 sigma from zero
  P_rec  : fraction recovered: a detection in >= 1 block with the fitted
           nu_s within a factor of two of the injected value
  m_chan : median modulation index present in the channelised injected gain
  m_rec  : median recovered m, and median recovered nu_s
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = np.load(os.path.join(HERE, "inject_results.npz"))["results"]
FIG = os.path.join(HERE, "scint_injection_recovery.pdf")

LABELS = ["YMW16", "NE2001", "4", "8", "12", "18", "25", "35", "50", "70",
          "100", "150"]


def per_trial(sub):
    """Collapse block rows into per-trial outcomes."""
    out = []
    for k in np.unique(sub["trial"]):
        t = sub[sub["trial"] == k]
        m, e = t["m_rec"], t["m_err"]
        ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
        if ok.any():
            w = 1.0 / e[ok] ** 2
            comb = (m[ok] * w).sum() / w.sum() * np.sqrt(w.sum())
        else:
            comb = 0.0
        match = (t["detected"] & (t["nus_rec"] > 0.5 * t["nus_inj"])
                 & (t["nus_rec"] < 2.0 * t["nus_inj"]))
        out.append((t["detected"].any(), (t["sig_m"] > 3).any(), comb > 3,
                    match.any()))
    return np.array(out)


def table():
    print(f"{'nu_s':>7} {'m':>4} {'P_det':>6} {'P_rec':>6} {'P_mod':>6} {'P_comb':>6} "
          f"{'m_chan':>6} {'m_rec':>6} {'nu_rec':>7} {'kind=two':>8}")
    summary = {}
    for lab in LABELS:
        for m in np.unique(R["m_inj"][R["label"] == lab]):
            sub = R[(R["label"] == lab) & (R["m_inj"] == m)]
            pt = per_trial(sub)
            row = dict(P_det=pt[:, 0].mean(), P_mod=pt[:, 1].mean(),
                       P_comb=pt[:, 2].mean(), P_rec=pt[:, 3].mean(),
                       m_chan=np.nanmedian(sub["m_chan"]),
                       m_rec=np.nanmedian(sub["m_rec"]),
                       nu_rec=np.nanmedian(sub["nus_rec"]),
                       two=np.mean(sub["kind"] == "two"))
            summary[(lab, m)] = row
            print(f"{lab:>7} {m:4.1f} {row['P_det']:6.2f} {row['P_rec']:6.2f} {row['P_mod']:6.2f} "
                  f"{row['P_comb']:6.2f} {row['m_chan']:6.2f} "
                  f"{row['m_rec']:6.2f} {row['nu_rec']:7.1f} {row['two']:8.2f}")
        print()
    null = R[R["label"] == "none"]
    print("real data (no injection): m_rec per block",
          np.round(null["m_rec"], 2), " sig_m", np.round(null["sig_m"], 2),
          " detected", null["detected"])
    return summary


def threshold(summary, lab, key, level):
    ms = sorted(m for (l, m) in summary if l == lab)
    for m in ms:
        if summary[(lab, m)][key] >= level:
            return m
    return np.nan


def figure(summary):
    grid = [l for l in LABELS if l not in ("YMW16", "NE2001")]
    cmap = plt.get_cmap("Blues")
    cols = {l: cmap(0.3 + 0.65 * i / (len(grid) - 1)) for i, l in enumerate(grid)}

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)

    for ax, key, title in (
            (axs[0], "P_rec", "(a) detected, $\\nu_{\\rm s}$ within ×2"),
            (axs[1], "P_det", "(b) any formal detection")):
        for l in grid:
            ms = sorted(m for (x, m) in summary if x == l)
            ax.plot(ms, [summary[(l, m)][key] for m in ms], "-", lw=1.2,
                    color=cols[l], label=f"{l} kHz")
        for l, mk, ls in (("YMW16", "o", "--"), ("NE2001", "s", ":")):
            ms = sorted(m for (x, m) in summary if x == l)
            ax.plot(ms, [summary[(l, m)][key] for m in ms], ls, marker=mk,
                    ms=5, lw=1.4, color="k", mfc="white", label=l)
        ax.set(xlabel="injected modulation index $m$",
               ylabel="fraction of trials", ylim=(-0.03, 1.03), title=title)
        ax.grid(alpha=0.2, lw=0.5)

    ax = axs[2]
    for l in grid:
        ms = sorted(m for (x, m) in summary if x == l)
        ax.plot(ms, [summary[(l, m)]["m_rec"] for m in ms], "-", lw=1.2,
                color=cols[l])
    for l, mk, ls in (("YMW16", "o", "--"), ("NE2001", "s", ":")):
        ms = sorted(m for (x, m) in summary if x == l)
        ax.plot(ms, [summary[(l, m)]["m_rec"] for m in ms], ls, marker=mk,
                ms=5, lw=1.4, color="k", mfc="white")
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="-.")
    ax.set(xlabel="injected modulation index $m$",
           ylabel="median recovered $m$", title="(c) recovered modulation",
           xlim=(0, 1.02), ylim=(0, None))
    ax.grid(alpha=0.2, lw=0.5)
    h, lbl = axs[0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="outside right center", fontsize=7, frameon=False,
               title="injected $\\nu_{\\rm s}$", title_fontsize=7)
    fig.savefig(FIG)
    fig.savefig(FIG.replace(".pdf", ".png"), dpi=150)
    print(f"figure -> {FIG}")


def main():
    s = table()
    print("thresholds (smallest injected m reaching the level):")
    for lab in LABELS:
        print(f"  {lab:>7}: P_det>=0.5 at m={threshold(s, lab, 'P_det', 0.5)}, "
              f">=0.9 at m={threshold(s, lab, 'P_det', 0.9)} | "
              f"P_comb>=0.5 at m={threshold(s, lab, 'P_comb', 0.5)}, "
              f">=0.9 at m={threshold(s, lab, 'P_comb', 0.9)}")
    figure(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
