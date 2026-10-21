"""Placeholder for acf_slope.py (2D-ACF drift mode). Every call site in
arrivaltimes_stacked_v56_9.py is gated on ACF=True, which is off for this
analysis. Any call raises, so a silent dependency can't hide."""


def _unavailable(*args, **kwargs):
    raise RuntimeError("acf_slope placeholder called: real module needed for ACF mode")


acf_measure_component = _unavailable
acf_ridge_drift = _unavailable
effective_acf_dtdnu = _unavailable
draw_acf_panel = _unavailable
