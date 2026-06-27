"""
ml/eda/helios_eda
==================
Exploratory Data Analysis module for HEL1OS (Aditya-L1 hard X-ray) data.

Mirrors the structure of ml/eda/ for SoLEXS so that the two EDA modules
are interchangeable in run_eda.py and test suites.

Submodules
----------
helios_statistics : HEL1OSEDAStatistics, HEL1OSEDAReport
    Per-detector and cross-detector descriptive statistics, ACF, and
    detector-comparison plots for CdTe1/CdTe2/CZT1/CZT2.

helios_plotter : HEL1OSPlotter
    Light-curve, flare-overlay, multi-band, count-rate histogram,
    timing, and detector-comparison plots.

run_helios_eda : main entry point
    ``python -m ml.eda.helios_eda.run_helios_eda --date 20260621``
"""

from ml.eda.helios_eda.helios_statistics import HEL1OSEDAReport, HEL1OSEDAStatistics
from ml.eda.helios_eda.helios_plotter import HEL1OSPlotter

__all__ = [
    "HEL1OSEDAReport",
    "HEL1OSEDAStatistics",
    "HEL1OSPlotter",
]