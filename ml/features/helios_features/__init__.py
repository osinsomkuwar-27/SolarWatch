"""
solar.ml.features.helios_features
===================================
HEL1OS-specific feature engineering modules.

Mirrors the structure of solar.ml.features (SoLEXS) so the two instrument
pipelines are interchangeable through FeaturePipeline.

Modules
-------
helios_basic_features    : CdTe/CZT broadband statistics, energy-band features
helios_temporal_features : HXR temporal dynamics, lag features, EMA
helios_flare_features    : Hardness ratios, cumulative fluence, phase labels
helios_spectral_features : Spectral slope, curvature, photon index
helios_pipeline          : HEL1OSFeaturePipeline orchestrator
"""

from .helios_basic_features import HEL1OSBasicFeatures, HEL1OSBasicFeatureConfig
from .helios_temporal_features import HEL1OSTemporalFeatures, HEL1OSTemporalFeatureConfig
from .helios_flare_features import HEL1OSFlareFeatures, HEL1OSFlareFeatureConfig
from .helios_spectral_features import HEL1OSSpectralFeatures, HEL1OSSpectralFeatureConfig
from .helios_pipeline import HEL1OSFeaturePipeline, HEL1OSPipelineConfig

__all__ = [
    "HEL1OSBasicFeatures",
    "HEL1OSBasicFeatureConfig",
    "HEL1OSTemporalFeatures",
    "HEL1OSTemporalFeatureConfig",
    "HEL1OSFlareFeatures",
    "HEL1OSFlareFeatureConfig",
    "HEL1OSSpectralFeatures",
    "HEL1OSSpectralFeatureConfig",
    "HEL1OSFeaturePipeline",
    "HEL1OSPipelineConfig",
]