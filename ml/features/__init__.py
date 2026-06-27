"""
solar.ml.features
==================

Feature engineering pipeline for solar flare X-ray light curve data
(e.g. GOES soft X-ray, RHESSI hard X-ray count rates).

Scientific grounding: Benz, A.O., "Flare Observations",
Living Rev. Solar Phys., 5, (2008), 1.

Modules
-------
basic_features      Rolling statistics on raw count rate (log, mean, std, ...)
temporal_features   Derivatives, lags, EMA, rolling median (timing/dynamics)
flare_features      Soft-hard-soft index, Neupert effect, phase classification
spectral_features   Power-law / thermal spectral fit features
feature_pipeline    Orchestrator: chains all of the above on one DataFrame
"""

from .basic_features import BasicFeatures, BasicFeatureConfig
from .temporal_features import TemporalFeatures, TemporalFeatureConfig
from .flare_features import FlareFeatures, FlareFeatureConfig
from .spectral_features import SpectralFeatures, SpectralFeatureConfig
from .feature_pipeline import FeaturePipeline, PipelineConfig

__all__ = [
    "BasicFeatures",
    "BasicFeatureConfig",
    "TemporalFeatures",
    "TemporalFeatureConfig",
    "FlareFeatures",
    "FlareFeatureConfig",
    "SpectralFeatures",
    "SpectralFeatureConfig",
    "FeaturePipeline",
    "PipelineConfig",
]
