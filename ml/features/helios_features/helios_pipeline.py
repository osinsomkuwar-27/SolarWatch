"""
helios_features/helios_pipeline.py
=====================================

HEL1OS feature pipeline: chains HEL1OSBasicFeatures, HEL1OSTemporalFeatures,
HEL1OSFlareFeatures, and HEL1OSSpectralFeatures in the exact same pattern
as FeaturePipeline (SoLEXS) so both can be used interchangeably by
run_features.py and the Dataset Builder.

Usage
-----
>>> from solar.ml.features.helios_features import HEL1OSFeaturePipeline, HEL1OSPipelineConfig
>>> pipe = HEL1OSFeaturePipeline(HEL1OSPipelineConfig())
>>> features_df = pipe.transform(raw_df)
>>> pipe.feature_names()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .helios_basic_features    import HEL1OSBasicFeatures,    HEL1OSBasicFeatureConfig
from .helios_temporal_features import HEL1OSTemporalFeatures, HEL1OSTemporalFeatureConfig
from .helios_flare_features    import HEL1OSFlareFeatures,    HEL1OSFlareFeatureConfig
from .helios_spectral_features import HEL1OSSpectralFeatures, HEL1OSSpectralFeatureConfig


@dataclass
class HEL1OSPipelineConfig:
    """Top-level configuration for the HEL1OS feature pipeline.

    Mirrors PipelineConfig (SoLEXS) in every respect — same field naming
    convention, same from_dict / to_dict pattern — so that run_features.py
    can construct either config from the same YAML structure with only a
    top-level 'instrument' key difference.
    """

    basic:    HEL1OSBasicFeatureConfig    = field(default_factory=HEL1OSBasicFeatureConfig)
    temporal: HEL1OSTemporalFeatureConfig = field(default_factory=HEL1OSTemporalFeatureConfig)
    flare:    HEL1OSFlareFeatureConfig    = field(default_factory=HEL1OSFlareFeatureConfig)
    spectral: HEL1OSSpectralFeatureConfig = field(default_factory=HEL1OSSpectralFeatureConfig)

    enable_basic:    bool = True
    enable_temporal: bool = True
    enable_flare:    bool = True
    enable_spectral: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HEL1OSPipelineConfig":
        """Build from a nested plain dict (e.g. parsed from YAML).

        Example input::

            {
              "basic":    {"windows_sec": [60, 300], "cdte_col": "cdte_CR"},
              "temporal": {"lags_sec": [12, 60]},
              "flare":    {},
              "spectral": {},
              "enable_spectral": False
            }
        """
        d = dict(d)
        basic    = HEL1OSBasicFeatureConfig.from_dict(d.pop("basic", {}))
        temporal = HEL1OSTemporalFeatureConfig.from_dict(d.pop("temporal", {}))
        flare    = HEL1OSFlareFeatureConfig.from_dict(d.pop("flare", {}))
        spectral = HEL1OSSpectralFeatureConfig.from_dict(d.pop("spectral", {}))
        return cls(basic=basic, temporal=temporal, flare=flare, spectral=spectral, **d)

    def to_dict(self) -> Dict[str, Any]:
        """Inverse of from_dict; useful for logging / serialising a run."""
        return asdict(self)


class HEL1OSFeaturePipeline:
    """Chains all HEL1OS feature stages over a single input DataFrame.

    Identical contract to FeaturePipeline (SoLEXS):
      - transform()           chains stages; each stage receives the previous output.
      - transform_separately() runs every stage independently on the raw input.
      - feature_names()        lists all columns this pipeline will add.
    """

    def __init__(self, config: Optional[HEL1OSPipelineConfig] = None):
        self.config = config or HEL1OSPipelineConfig()

        self._stages: List[Tuple[str, Any]] = []
        if self.config.enable_basic:
            self._stages.append(("basic",    HEL1OSBasicFeatures(self.config.basic)))
        if self.config.enable_temporal:
            self._stages.append(("temporal", HEL1OSTemporalFeatures(self.config.temporal)))
        if self.config.enable_flare:
            self._stages.append(("flare",    HEL1OSFlareFeatures(self.config.flare)))
        if self.config.enable_spectral:
            self._stages.append(("spectral", HEL1OSSpectralFeatures(self.config.spectral)))

    # ------------------------------------------------------------------
    # Public API  (mirrors FeaturePipeline exactly)
    # ------------------------------------------------------------------

    def feature_names(self) -> List[str]:
        """Full ordered list of feature columns this pipeline will add."""
        names: List[str] = []
        for _, stage in self._stages:
            names.extend(stage.feature_names())
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run every enabled stage in order and return the combined result.

        Raises
        ------
        RuntimeError
            If a stage fails, re-raised with the stage name attached.
        """
        out = df
        for stage_name, stage in self._stages:
            try:
                out = stage.transform(out)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"HEL1OSFeaturePipeline stage '{stage_name}' failed: {exc}"
                ) from exc
        return out

    def transform_separately(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run every enabled stage independently on the raw input.

        Returns
        -------
        dict mapping stage name → output DataFrame
        """
        results: Dict[str, pd.DataFrame] = {}
        for stage_name, stage in self._stages:
            try:
                results[stage_name] = stage.transform(df)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"HEL1OSFeaturePipeline stage '{stage_name}' failed: {exc}"
                ) from exc
        return results