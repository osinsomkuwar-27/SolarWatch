"""
ml/eda
=======
Exploratory Data Analysis module for the Aditya-L1 pipeline.

Submodules
----------
light_curve_plotter : LightCurvePlotter
    Publication-quality light curve, dual-band, Neupert, and spectrogram plots.

flare_detector : FlareDetector, FlareEvent
    Rule-based derivative-threshold flare detection and GOES-proxy labelling.

statistics : EDAStatistics, EDAReport
    Descriptive statistics, distribution plots, ACF, class imbalance report.

run_eda : main entry point
    ``python -m ml.eda.run_eda --date 20260621``
"""

from ml.eda.flare_detector import FlareDetector, FlareEvent
from ml.eda.light_curve_plotter import LightCurvePlotter
from ml.eda.statistics import EDAReport, EDAStatistics

__all__ = [
    "FlareDetector",
    "FlareEvent",
    "LightCurvePlotter",
    "EDAReport",
    "EDAStatistics",
]
