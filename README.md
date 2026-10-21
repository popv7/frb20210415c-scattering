# FRB 20210415C: frequency-resolved scattering law

Analysis code for

> V. Pop, M. A. Chamma, A. Kumar & F. Rajabi,
> *A frequency-resolved measurement of the scattering law in FRB 20210415C*,
> New Astronomy (submitted).

The scripts reproduce the numbers, tables and tests in the paper from the
public CHIME/FRB Catalog 2 Stokes-I dynamic spectrum of the burst.

## Data

The input is the Stokes-I dynamic spectrum of FRB 20210415C from the Second
CHIME/FRB Catalog data release (<https://doi.org/10.11570/25.0066>). Download it
and save it as

```
data/FRB20210415C_stokesi_dynamic_spectrum.h5
```

The file is not included in this repository.

## Installation

Python ≥ 3.10. Install the dependencies with

```
pip install -r requirements.txt
```

## Contents and usage

Run the scripts from inside their own folders. Each script writes its output
next to itself.

### `scattering/`: scattering law and profile tests (Sections 2–3, Appendix A)

| Script | What it does | Paper |
|---|---|---|
| `scatlaw.py` | Core library: data access, sub-bands, dedrift, bin-integrated EMG / Gaussian fits, power-law fits | Sec. 2 |
| `final_analysis.py` | Full-band fit and ΔBIC, nine sub-band fits, width-free and fixed-width scattering laws, width–frequency relation → `results.json` | Sec. 3, Table 1 |
| `derived_quantities.py` | Checks the core numbers against `results_paper.json`; BIC details, multi-start check, τ at 600 MHz, local width–tail correlations → `derived_quantities.json` | Sec. 3, App. A |
| `appendix_tests.py selection` | All nine sub-bands, leave-one-out, alternative band | App. A.2 |
| `appendix_tests.py dm` | DM grid: re-dedisperse, remeasure the drift slope, refit | App. A.2 |
| `appendix_tests.py joint` | Joint eight-profile MCMC fit (emcee) | App. A.3 |
| `appendix_tests.py foreground` | Galactic NE2001 / YMW16 scattering predictions | Sec. 4 |
| `joint_600.py` | Reruns the joint fit, keeps the chain, and gives τ at 600 MHz | Sec. 3 |
| `inject_final.py` | Injection–recovery tests: window truncation, effective frequency, Gaussian envelope, negative control (`main`, `gauss`, `null`, `width_nu1`, `width_measured`) | App. A.1, A.3, Fig. 3 |
| `width_tests.py` | Width law over 770 band/sub-band configurations (`grid`) and width-recovery injections (`inject`) | App. A.4 |
| `pbf_tests.py` | Injection tests with thin-screen, thick-screen and uniform-medium pulse-broadening functions | App. A.4 |
| `figures.py` | Figures 1–3 (called by `final_analysis.py --figures` and `run_all.sh`) | Figs. 1–3 |
| `run_all.sh` | Runs everything above in order | — |

```
cd scattering
./run_all.sh          # about 1 h on 2 cores, most of it the joint MCMC fit
```

The JSON files in `scattering/` (`results_paper.json`, `derived_quantities.json`,
`selection_tests.json`, `dm_grid.json`, `foreground.json`, `joint_fit_scaled_paper.json`,
`joint_600.json`, `width_grid.json`, `width_injections.json`, `pbf_tests.json`,
`inject_out/injection_config_*.json`) are the outputs used in the paper. A fresh
run reproduces them.

**Arrival times and drift slope (`scattering/pipe_instr/`).** The 53 per-channel
arrival times and the dedrift slope come from the arrival-time pipeline
(`arrivaltimes_stacked_v56_9.py`), based on the Arrival Times Pipeline by
Mohammed Chamma (`arrivaltimes.py` in [FRBGui](https://github.com/mef51/frbgui),
MIT licence), which implements the per-channel method of Chamma et al. (2025);
`driftrate.py` is from FRBGui.
`run_toas.py` runs it (about 10 s); `make_input.py`, called automatically,
builds its input file from the Catalog 2 data. `run_all.sh` does this first.
Three modules the pipeline imports but does not call in this analysis
(`DM_phase.py`, `acf_slope.py`, `scint_bw.py`) are stubs that raise if called;
`DM_phase.py` stands in for the DM_phase package of Seymour, Michilli & Pleunis.
If `pipe_instr/out/` is absent, `final_analysis.py` uses the slope stored in
`results_paper.json`, (−4.62 ± 2.6) × 10⁻⁴ ms MHz⁻¹, and Figure 1 and the DM
grid are skipped.

### `scintillation/`: scintillation search (Appendix B)

| Script | What it does |
|---|---|
| `acf_scintillation.py` | Spectral ACF in each clean channel block; Lorentzian fit for the scintillation bandwidth and modulation index; comparison with NE2001 / YMW16 |
| `inject_scintillation.py` | Injects simulated scintillation into the observed spectrum and reruns the same estimator → `inject_results.npz` |
| `summarise_injections.py` | Detection and recovery fractions, and the recovery figure |

`inject_scintillation.py` takes about 10 min on 2 cores.

```
cd scintillation
python acf_scintillation.py
python inject_scintillation.py --ntrial 100 --nproc 4
python summarise_injections.py
```

## Credits and methods

**Code from other projects**

- `scattering/pipe_instr/arrivaltimes_stacked_v56_9.py` is based on the Arrival
  Times Pipeline by Mohammed Chamma (`arrivaltimes.py` in
  [FRBGui](https://github.com/mef51/frbgui), originally `ArrivalTimes.ipynb` in
  [subdriftlaw](https://github.com/mef51/subdriftlaw)), extended by Victor Pop
  with EMG fits, EMG leading-edge arrival times (`EMG_halfwidth.py`), MCMC and
  channel stacking.
- `scattering/pipe_instr/driftrate.py` is from FRBGui, with one change:
  `dedisperse()` accepts the true channel frequencies of a masked waterfall.
- Both are © 2022 Mohammed Chamma under the MIT licence; the notice is in
  `scattering/pipe_instr/LICENSE-FRBGui`.
- `DM_phase.py`, `acf_slope.py` and `scint_bw.py` in `pipe_instr/` are stubs.
  `DM_phase.py` stands in for DM_phase (Seymour, Michilli & Pleunis 2019,
  [ascl:1910.004](https://ascl.net/1910.004)), which FRBGui uses to re-derive
  the DM; that option is off here.
- One line in the pipeline (reduced χ² of a `scipy.odr` fit) follows a
  [Stack Overflow answer](https://stackoverflow.com/a/21406281) (CC BY-SA).

**Data and models**

- Data: the Second CHIME/FRB Catalog (CHIME/FRB Collaboration 2026),
  doi:[10.11570/25.0066](https://doi.org/10.11570/25.0066). The spectral model
  S(ν) and the burst model used in the scintillation injections are the
  catalogue's fitburst models (Fonseca et al. 2024, ApJS;
  [arXiv:2311.05829](https://arxiv.org/abs/2311.05829)).
- Galactic foreground: NE2001 (Cordes & Lazio 2002) and YMW16 (Yao, Manchester
  & Wang 2017) via [PyGEDM](https://pygedm.readthedocs.io/) (Price et al. 2021,
  PASA; [arXiv:2106.15816](https://arxiv.org/abs/2106.15816)); the YMW16
  scattering time uses the τ–DM relation of Krishnakumar et al. (2015).

**Methods**

| Method | Reference | Where |
|---|---|---|
| Per-channel arrival times and drift slope | Chamma et al. (2025) | `pipe_instr/` |
| Drifting-Gaussian models | Jahns et al. (2023) | `pipe_instr/driftrate.py` |
| Model selection (BIC; ΔBIC > 6) | Schwarz (1978); Kass & Raftery (1995) | `final_analysis.py`, `derived_quantities.py` |
| MCMC sampling, autocorrelation time | emcee (Foreman-Mackey et al. 2013) | `appendix_tests.py`, `joint_600.py`, pipeline |
| Differential-evolution and snooker moves | ter Braak (2006); ter Braak & Vrugt (2008) | `appendix_tests.py` |
| Pulse-broadening functions | Williamson (1972); Bhat et al. (2004) | `pbf_tests.py` |
| Scintillation ACF and per-lag errors | Jahns-Schindler et al. (2026) | `acf_scintillation.py` |
| ACF normalisation (amplitude m²) | Masui et al. (2015) | `acf_scintillation.py` |
| Bandwidth–broadening relation 2πΔν_d τ = C₁ | Cordes & Rickett (1998) | `scintillation/` |

All other code, including the pipeline extensions above, is by Victor Pop.

## Citation

If you use this code, please cite the paper above.

## License

MIT (see `LICENSE`).
