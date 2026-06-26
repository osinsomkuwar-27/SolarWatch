"""
ml/features
============
Feature engineering for the Aditya-L1 Solar Flare Prediction pipeline.

Repository layout
-----------------
ml/features/
    __init__.py          ← This file; exports the public API
    basic_features.py    ← Rolling statistics, log-transform, RMS, energy
    temporal_features.py ← Derivatives, lag features, EMA, rolling median
    flare_features.py    ← Peak detection, rise/decay time, SNR, flux ratio
    spectral_features.py ← Energy bands, hardness ratio, spectral entropy
    feature_pipeline.py  ← Orchestrator: runs all modules on a DataFrame

Data requirements per feature class
-------------------------------------
Module              | .lc | GTI | .pi | HEL1OS
--------------------|-----|-----|-----|-------
basic_features      |  ✓  |     |     |
temporal_features   |  ✓  |     |     |
flare_features      |  ✓  |  ✓  |     |
spectral_features   |     |     |  ✓  |   ✓

Scientific reference
---------------------
All feature definitions are grounded in:
    Benz, A.O. (2008), "Flare Observations",
    Living Reviews in Solar Physics, 5, 1.
    https://doi.org/10.12942/lrsp-2008-1
"""

from ml.features.basic_features import BasicFeatureExtractor

__all__ = [
    "BasicFeatureExtractor",
]