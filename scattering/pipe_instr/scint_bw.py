"""Placeholder for scint_bw.py. Every call site in arrivaltimes_stacked_v56_9.py is
gated on scint=True, which is off for this analysis (the paper's scintillation
search is ../../scintillation/acf_scintillation.py). Any call raises, so a silent
dependency can't hide."""


def measure_scint_bw(*args, **kwargs):
    raise RuntimeError("scint_bw placeholder called: real module needed for scint mode")
