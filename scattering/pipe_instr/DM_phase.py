"""Placeholder for DM_phase.py (Seymour, Michilli & Pleunis 2019, ascl:1910.004), which
arrivaltimes_stacked_v56_9.py imports but calls only with recalculateDM=True.
That option is off for this analysis: the catalogue DM is used and the DM is
varied explicitly in ../appendix_tests.py dm. Any call raises, so a silent
dependency can't hide. Install the original DM_phase to use recalculateDM."""


def get_dm(*args, **kwargs):
    raise RuntimeError("DM_phase placeholder called: real module needed for recalculateDM")
