"""
figures.py -- Figures 1 and 2 of the final draft (vector PDF + PNG preview).

Figure 1 includes the advisor's final Figure-1 requests (email of 11 Oct):
  * panel (b) dashed-curve label  t - mu = tau_ref (nu/nu_ref)^alpha
  * panel (c): masked intervals drawn as the same grey bands as panel (b),
    legend says "unmasked spectrum", excluded sub-band marked
  * "(c)" label moved off the blue curve (top right)
Figure 2 applies the final draft's red note: the dotted reference curves are
bold, drawn above the fits, and the "dof" label is removed from chi2_red,PL.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle

import scatlaw as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
TOA_CSV = os.path.join(HERE, "pipe_instr", "out", "instr_subdf.csv")
TOA_GEOM = os.path.join(HERE, "pipe_instr", "out", "instr_geom.json")

DISPLAY_NATIVE = 256           # 6.25-MHz display channels
DISPLAY_MASK_FRAC = 0.10       # display row grey if < 10% of its channels unmasked
C_FREE, C_FIXED, C_WIDTH = "#2f7fc1", "#ef6c3a", "#2aa889"
C_GREY = "#bdbdbd"
FIG1_LEGEND_FS = 10.0           # Figure 1 legend font size (all panels)
DOTS = (0, (0.1, 1.9))         # round-capped dots for reference curves (Figure 2)

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "pdf.fonttype": 42,
})


def panel_label(ax, s, x=0.025, y=0.95, ha="left", **kw):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top", fontweight="bold",
            fontsize=12, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            zorder=20, **kw)


def load_toas(burst):
    """Pipeline per-channel arrival times, converted to true sample time.
    The pipeline fits each row on np.linspace(0, n*dt, n) inside a sub-window
    starting at sample `left`, so t_true = left*dt + (t - left*dt)(n-1)/n."""
    d = pd.read_csv(TOA_CSV)
    g = json.load(open(TOA_GEOM))
    sf = g["subfall"]["0"]
    dt = g["res_time_ms"]
    t0 = sf["left"] * dt
    fac = (sf["nt"] - 1) / sf["nt"]
    t = t0 + (d["t_emg_hwhm"].values - t0) * fac
    terr = d["t_emg_hwhm_err"].values * fac
    sl, sle, rchi = g["dtdnus"][0][:3]
    return d["freqs"].values, t, terr, (sl * fac, sle * fac, rchi)   # same stretch removed


def display_dynspec(burst):
    """Average unmasked native channels into 6.25-MHz display rows."""
    n = burst.freqs.size // DISPLAY_NATIVE
    g = burst.good[: n * DISPLAY_NATIVE].reshape(n, DISPLAY_NATIVE)
    d = burst.data[: n * DISPLAY_NATIVE].reshape(n, DISPLAY_NATIVE, -1)
    fr = burst.freqs[: n * DISPLAY_NATIVE].reshape(n, DISPLAY_NATIVE)
    with np.errstate(invalid="ignore"):
        dyn = np.nansum(d * g[..., None], axis=1) / g.sum(1)[:, None]
    frac = g.mean(1)
    edges = np.r_[fr[:, 0], fr[-1, -1] + L.NATIVE_RES_MHZ] - 0.5 * L.NATIVE_RES_MHZ
    return dyn, frac, edges


def masked_intervals(frac, edges):
    bad = frac < DISPLAY_MASK_FRAC
    out, i = [], 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j + 1 < len(bad) and bad[j + 1]:
                j += 1
            out.append((edges[i], edges[j + 1]))
            i = j + 1
        else:
            i += 1
    return out


def subband_boundaries(sbs):
    b = [sbs[0]["f_lo"]]
    for s0, s1 in zip(sbs[:-1], sbs[1:]):
        b.append(0.5 * (s0["f_hi"] + s1["f_lo"]))
    b.append(sbs[-1]["f_hi"])
    return np.array(b)


def figure1(burst, fb, res, s):
    mu = fb["emg"]["mu"]
    trel = burst.t - mu
    dyn, frac, edges = display_dynspec(burst)
    bad_iv = masked_intervals(frac, edges)
    sbs = res["sbs"]
    bnd = subband_boundaries(sbs)
    ex_lo, ex_hi = bnd[-2], bnd[-1]
    nu_t, t_toa, terr, (slope, slope_err, _) = load_toas(burst)
    pf = res["pl_free"]

    fig = plt.figure(figsize=(7.3, 7.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 3.2, 1.15], width_ratios=[3.2, 1.0],
                          hspace=0.12, wspace=0.06, left=0.10, right=0.985, top=0.985, bottom=0.07)
    gs_b = gs[1:, :]
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[1, 0], sharex=axa)
    axc = fig.add_subplot(gs[1, 1], sharey=axb)
    gsd = gs[2, :].subgridspec(2, 1, height_ratios=[0.32, 1.0], hspace=0)   # (d) spans (b) and (c)
    axd = fig.add_subplot(gsd[1])

    # (a) full-band profile
    e = fb["emg"]
    norm = np.max(e["model"] - e["C"])
    axa.plot(trel, (fb["profile"] - e["C"]) / norm, color="0.45", lw=1.0, label="data")
    tf = np.linspace(trel[0], trel[-1], 4000)
    model_fine = L.emg_binned(tf + mu, e["A"], e["mu"], e["sigma"], e["tau"], 0.0, burst.dt)
    axa.plot(tf, model_fine / norm, color=C_FREE, lw=1.4, label="EMG fit")
    axa.set_ylabel("Norm. flux")
    axa.legend(loc="upper left", bbox_to_anchor=(0.06, 1.0), frameon=False, ncol=2, fontsize=FIG1_LEGEND_FS)
    axa.text(0.985, 0.88, rf"$\sigma_{{\rm int}}={e['sigma']:.2f}\pm{e['perr_scaled'][2]:.2f}$ ms" "\n"
             rf"$\tau_{{\rm emp}}={e['tau']:.1f}\pm{e['perr_scaled'][3]:.1f}$ ms",
             transform=axa.transAxes, ha="right", va="top", fontsize=10)
    axa.set_ylim(-0.08, 1.12)
    panel_label(axa, "(a)")
    plt.setp(axa.get_xticklabels(), visible=False)

    # (b) dynamic spectrum
    vals = dyn[frac >= DISPLAY_MASK_FRAC]
    vmin, vmax = np.nanpercentile(vals, [5, 99.7])
    tedge = np.r_[trel - 0.5 * burst.dt, trel[-1] + 0.5 * burst.dt]
    show = np.where((frac >= DISPLAY_MASK_FRAC)[:, None], dyn, np.nan)
    axb.pcolormesh(tedge, edges, show, cmap="viridis", vmin=vmin, vmax=vmax,
                   shading="flat", rasterized=True)
    for lo, hi in bad_iv:
        axb.axhspan(lo, hi, color=C_GREY, zorder=2, lw=0)
    for f in bnd[1:]:
        axb.axhline(f, color="white", ls=(0, (4, 3)), lw=1.0, zorder=3)
    xl = (tedge[0], tedge[-1])
    hatch_x1 = -27.0
    axb.add_patch(Rectangle((xl[0], ex_lo), hatch_x1 - xl[0], ex_hi - ex_lo, fill=False,
                            hatch="///", ec="white", lw=0, zorder=4))
    axb.text(0.5 * (xl[0] + hatch_x1), 0.5 * (ex_lo + ex_hi) + 6, "excluded sub-band", ha="center",
             va="center", fontsize=9.5, zorder=5,
             bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="0.2", lw=0.6))
    nuc = np.linspace(400.0, 800.0, 400)
    axb.plot(pf["ref"] * (nuc / res["nu_ref"]) ** pf["index"], nuc, color=C_FIXED, ls=(0, (5, 3)),
             lw=1.6, zorder=6, label=r"$t-\mu=\tau_{\rm ref}(\nu/\nu_{\rm ref})^{\alpha}$")
    axb.plot(t_toa - mu, nu_t, "o", ms=3.4, mfc="black", mec="white", mew=0.6, zorder=7,
             label="arrival times")
    axb.set_xlim(*xl)
    axb.set_ylim(400.0, 800.0)
    axb.set_ylabel("Frequency (MHz)")
    axb.set_xlabel(r"Time relative to full-band $\mu$ (ms)")
    axb.legend(loc="upper right", fontsize=FIG1_LEGEND_FS, framealpha=0.9)
    panel_label(axb, "(b)", y=0.965)

    # (c) time-integrated spectrum
    t0 = burst.on_region[0] - 5
    base = burst.data[:, :t0].mean(1, keepdims=True)
    spec_n = (burst.data[:, t0:] - base).sum(1)
    n = burst.freqs.size // DISPLAY_NATIVE
    g = burst.good[: n * DISPLAY_NATIVE].reshape(n, DISPLAY_NATIVE)
    sp = spec_n[: n * DISPLAY_NATIVE].reshape(n, DISPLAY_NATIVE)
    with np.errstate(invalid="ignore"):
        spd = np.where(g.mean(1) >= DISPLAY_MASK_FRAC, (sp * g).sum(1) / g.sum(1), np.nan)
    spd = spd / np.nanmax(spd)
    # step plot with gaps at masked rows
    xs, ys = [], []
    for i in range(n):
        if np.isfinite(spd[i]):
            xs += [spd[i], spd[i]]
            ys += [edges[i], edges[i + 1]]
        else:
            xs.append(np.nan)
            ys.append(np.nan)
    axc.plot(xs, ys, color="0.35", lw=0.9, label="unmasked spectrum", zorder=3)
    nus = np.linspace(400.0, 800.0, 800)
    Sm = burst.S(nus)
    ok = np.isfinite(spd)
    cen = 0.5 * (edges[:-1] + edges[1:])
    k = np.nansum(spd[ok] * burst.S(cen[ok])) / np.nansum(burst.S(cen[ok]) ** 2)
    axc.plot(k * Sm, nus, color=C_FREE, lw=1.6, label=r"Catalog 2 model $S(\nu)$", zorder=4)
    for lo, hi in bad_iv:
        axc.axhspan(lo, hi, color=C_GREY, zorder=2, lw=0)
    axc.add_patch(Rectangle((0, ex_lo), 1.05, ex_hi - ex_lo, fill=False, hatch="//",
                            ec="0.45", lw=0, zorder=1, label="excluded sub-band"))
    axc.set_xlim(0, 1.05)
    axc.set_xlabel("Norm. flux")
    plt.setp(axc.get_yticklabels(), visible=False)
    hdl, lab = axc.get_legend_handles_labels()
    order = [lab.index("excluded sub-band"), lab.index("unmasked spectrum"),
             lab.index(r"Catalog 2 model $S(\nu)$")]
    # legend in the empty space directly above panel (c), same font size as panels (a), (b)
    axc.legend([hdl[i] for i in order], [lab[i] for i in order], loc="lower right",
               bbox_to_anchor=(1.0, 1.03), fontsize=FIG1_LEGEND_FS, framealpha=0.95, handlelength=1.5,
               handletextpad=0.5, borderpad=0.45, labelspacing=0.45, borderaxespad=0.0)
    panel_label(axc, "(c)", x=0.95, y=0.965, ha="right")

    # (d) per-channel arrival times relative to the linear relation at nu_ref
    w = 1.0 / terr ** 2
    nu_ref = res["nu_ref"]
    t_ref = np.sum(w * (t_toa - slope * (nu_t - nu_ref))) / np.sum(w)
    dtv = t_toa - t_ref
    axd.errorbar(nu_t, dtv, yerr=terr, fmt="D", ms=3.2, color="black", ecolor="0.4", elinewidth=0.8,
                 capsize=0, zorder=3)
    nl = np.array([400.0, 800.0])
    axd.plot(nl, slope * (nl - nu_ref), color=C_FREE, lw=1.4, zorder=2)
    axd.axhline(0, color="0.8", lw=0.6, zorder=1)
    axd.set_xlim(400, 800)
    axd.set_ylim(-2.2, 2.2)
    axd.set_xlabel("Frequency (MHz)")
    axd.set_ylabel(r"$\Delta t$ (ms)")
    mant, expo = slope / 1e-4, -4
    axd.text(0.985, 0.92, rf"$\mathrm{{d}}t/\mathrm{{d}}\nu=({mant:.1f}\pm{slope_err / 1e-4:.1f})\times10^{{{expo}}}$ ms MHz$^{{-1}}$",
             transform=axd.transAxes, ha="right", va="top", fontsize=9.5,
             bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="0.6", lw=0.6))
    axd.grid(alpha=0.25, lw=0.5)
    panel_label(axd, "(d)", y=0.93)
    return fig


def _tau_panel(ax, axr, res, which, color, marker, label_fit, panel):
    nu = res["nu"]
    keep = res["keep"]
    tau = res["tau"] if which == "free" else res["tau_fixed"]
    err = res["tau_err"] if which == "free" else res["tau_fixed_err"]
    pl = res["pl_free"] if which == "free" else res["pl_fixed"]
    nur = res["nu_ref"]
    xx = np.linspace(405, 760, 300)
    # alpha = -4 reference: bold round dots drawn above the fitted curve, which it
    # nearly coincides with in panel (a) (advisor's Figure 2 note)
    ax.plot(xx, pl["ref"] * (xx / nur) ** -4.0, color="black", lw=2.2, ls=DOTS,
            dash_capstyle="round", zorder=2.6, label=r"$\alpha=-4$")
    ax.plot(xx, pl["ref"] * (xx / nur) ** pl["index"], color=color, lw=1.6, zorder=2.2, label=label_fit)
    ax.errorbar(nu[keep], tau[keep], yerr=err[keep], fmt=marker, color=color, ms=5.5, capsize=0,
                label=r"sub-band $\tau_{\rm emp}$")
    ax.errorbar(nu[~keep], tau[~keep], yerr=err[~keep], fmt=marker, mfc="white", mec="0.45",
                ecolor="0.45", ms=7, label="excluded")
    ax.set_yscale("log")
    ax.set_ylim(2.5, 62)
    ax.set_yticks([3, 5, 10, 20, 30])
    ax.set_yticklabels(["3", "5", "10", "20", "30"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("Scattering time (ms)")
    ax.legend(loc="upper right", fontsize=8.6, framealpha=0.9)
    ax.text(0.04, 0.06, rf"$\chi^2_{{\rm red,PL}}={pl['chi2_red']:.2f}$",
            transform=ax.transAxes, fontsize=9.5)
    panel_label(ax, panel)
    plt.setp(ax.get_xticklabels(), visible=False)
    pred = pl["ref"] * (nu / nur) ** pl["index"]
    r = (tau - pred) / err
    axr.axhline(0, color="black", lw=0.8)
    axr.plot(nu[keep], r[keep], marker, color=color, ms=4)
    axr.plot(nu[~keep], r[~keep], marker, mfc="white", mec="0.45", ms=5.5)
    axr.set_ylim(-3.8, 3.8)
    axr.set_yticks([-2.5, 0, 2.5])
    axr.set_ylabel(r"Residual ($\sigma$)", fontsize=10)
    axr.set_xlabel("Frequency (MHz)")


def figure2(res, wl, s):
    fig = plt.figure(figsize=(7.3, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.30, left=0.10, right=0.985, top=0.985, bottom=0.075)
    gA = gs[0, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0)
    gC = gs[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0)
    axA, axAr = fig.add_subplot(gA[0]), fig.add_subplot(gA[1])
    axC, axCr = fig.add_subplot(gC[0]), fig.add_subplot(gC[1])
    axB, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    pf, px = res["pl_free"], res["pl_fixed"]
    _tau_panel(axA, axAr, res, "free", C_FREE, "o",
               rf"$\sigma_{{\rm int}}$ free: $\alpha={pf['index']:.2f}\pm{pf['index_err']:.2f}$", "(a)")
    _tau_panel(axC, axCr, res, "fixed", C_FIXED, "s",
               rf"$\sigma_{{\rm int}}={res['sigma_fixed']:.2f}$ ms: $\alpha={px['index']:.2f}\pm{px['index_err']:.2f}$", "(c)")
    for a in (axA, axAr, axC, axCr):
        a.set_xlim(398, 752)

    # (b) widths
    nu, sig, sige, keep = res["nu"], res["sigma"], res["sigma_err"], res["keep"]
    nur = res["nu_ref"]
    xx = np.linspace(405, 752, 300)
    a9, ab = wl["all"], wl["above"]
    axB.plot(xx, a9["ref"] * (xx / nur) ** a9["index"], color=C_WIDTH, lw=1.6,
             label=rf"all nine: $\beta={a9['index']:.2f}\pm{a9['index_err']:.2f}$")
    axB.plot(xx, ab["ref"] * (xx / nur) ** ab["index"], color=C_FREE, ls="--", lw=1.6,
             label=rf"$\nu_{{\rm eff}}>440$ MHz: $\beta={ab['index']:.2f}\pm{ab['index_err']:.2f}$")
    axB.plot(xx, ab["ref"] * (xx / nur) ** -1.0, color="black", ls=DOTS, lw=2.0, dash_capstyle="round",
             zorder=2.6, label=r"$\beta=-1$ (same normalisation)")
    axB.axhline(res["sigma_fixed"], color="#8c6d46", ls="-.", lw=1.2,
                label=f"fixed value {res['sigma_fixed']:.2f} ms")
    axB.errorbar(nu[keep], sig[keep], yerr=sige[keep], fmt="o", color=C_WIDTH, ms=5.5,
                 label=r"sub-band $\sigma_{\rm int}$")
    axB.errorbar(nu[~keep], sig[~keep], yerr=sige[~keep], fmt="o", mfc="white", mec="0.45",
                 ecolor="0.45", ms=7, label="excluded")
    axB.set_xlim(398, 752)
    axB.set_ylim(0.78, 2.55)
    axB.set_xlabel("Frequency (MHz)")
    axB.set_ylabel("Intrinsic width (ms)")
    axB.legend(loc="upper right", fontsize=8.3, framealpha=0.9)
    panel_label(axB, "(b)")

    # (d) |dtau| / mean error
    dtau = np.array([b["dtau_over_err"] for b in s["subbands"]])
    med = s["dtau_median_retained"]
    axD.axhline(1.0, color="black", ls=DOTS, lw=1.8, dash_capstyle="round",
                label=r"$|\Delta\tau_j|=\bar{\sigma}_{\tau,j}$")
    axD.axhline(med, color=C_FREE, ls="--", lw=1.4, label=f"median (retained) = {med:.2f}")
    axD.plot(nu[keep], dtau[keep], "o-", color="black", ms=5.5, lw=1.0, label="retained")
    axD.plot(nu[~keep], dtau[~keep], "o", mfc="white", mec="0.45", ms=7, label="excluded")
    axD.set_xlim(398, 752)
    axD.set_ylim(0, max(3.4, dtau.max() * 1.13))
    axD.set_xlabel("Frequency (MHz)")
    axD.set_ylabel(r"Scattering-time difference ($\sigma$)")
    axD.legend(loc="upper right", fontsize=8.8, framealpha=0.9)
    axD.grid(alpha=0.25, lw=0.5)
    panel_label(axD, "(d)")
    return fig


def make_all(burst, fb, res, wl, s):
    os.makedirs(OUT, exist_ok=True)
    for name, fig in (("frb20210415C_waterfall", figure1(burst, fb, res, s)),
                      ("frb20210415C_subband", figure2(res, wl, s))):
        fig.savefig(os.path.join(OUT, name + ".pdf"))
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=150)
        plt.close(fig)
    print(f"figures written to {OUT}")


def figure3(measured):
    """Injection-recovery figure (App. A.1)."""
    d = pd.read_csv(os.path.join(HERE, "inject_out", "trials_main.csv"))
    z = np.load(os.path.join(HERE, "inject_out", "taus_m4_main.npz"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.3, 3.25),
                                 gridspec_kw=dict(wspace=0.28, left=0.085, right=0.985, top=0.97, bottom=0.15))
    a0, ae = measured["index"], measured["index_err"]
    ax.axhspan(a0 - ae, a0 + ae, color=C_WIDTH, alpha=0.18, lw=0,
               label=rf"measured $\alpha={a0:.2f}\pm{ae:.2f}$")
    xx = np.array([-4.42, -3.58])
    ax.plot(xx, xx, color="0.45", ls=":", lw=1.0, label="one-to-one")
    g = d.groupby("alpha_inj")
    al = np.array(sorted(g.groups))
    for col, dx, kw, lab in (
            ("flux_ext", -0.012, dict(fmt="s", color=C_FREE, ms=5), r"flux-weighted $\nu_{\rm eff}$, extended window"),
            ("flux_obs", 0.0, dict(fmt="o", color=C_FIXED, ms=5), r"flux-weighted $\nu_{\rm eff}$, observed window"),
            ("arith_obs", 0.012, dict(fmt="^", mfc="white", mec="0.3", ecolor="0.3", ms=6),
             r"arithmetic-mean $\nu$, observed window")):
        m = g[col].mean().reindex(al).values
        s = g[col].std().reindex(al).values
        ax.errorbar(al + dx, m, yerr=s, capsize=0, elinewidth=0.9, label=lab, **kw)
    ax.set_xlim(-4.42, -3.58)
    ax.set_ylim(-4.42, -3.58)
    ax.set_xlabel(r"Injected $\alpha$")
    ax.set_ylabel(r"Recovered $\alpha$ (free-width)")
    ax.legend(loc="lower right", fontsize=7.6, framealpha=0.9)
    panel_label(ax, "(a)")

    nu = z["nu_flux"]
    xl = np.linspace(410, 760, 300)
    bx.plot(xl, float(z["tau_ref"]) * (xl / float(z["nu_ref"])) ** -4.0, color="black", lw=1.4,
            label=r"injected $\tau\propto\nu^{-4}$")
    for key, dx, kw, lab in (("ext", -3.0, dict(fmt="s", color=C_FREE, ms=5), "extended window"),
                             ("obs", 3.0, dict(fmt="o", color=C_FIXED, ms=5), "observed window")):
        arr = z[key]
        med = np.nanmedian(arr, 0)
        lo, hi = np.nanpercentile(arr, [16, 84], axis=0)
        bx.errorbar(nu + dx, med, yerr=[med - lo, hi - med], capsize=0, elinewidth=0.9, label=lab, **kw)
    bx.set_yscale("log")
    bx.set_yticks([3, 5, 10, 20, 30])
    bx.set_yticklabels(["3", "5", "10", "20", "30"])
    bx.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    bx.set_xlim(405, 765)
    bx.set_xlabel(r"$\nu_{\rm eff}$ (MHz)")
    bx.set_ylabel(r"Recovered $\tau_{\rm emp}$ (ms)")
    bx.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    panel_label(bx, "(b)", y=0.12)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "alpha_injection_recovery.pdf"))
    fig.savefig(os.path.join(OUT, "alpha_injection_recovery.png"), dpi=150)
    plt.close(fig)
