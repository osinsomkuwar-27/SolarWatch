"""
feature_pipeline.py
====================

Orchestrator that chains basic_features, temporal_features,
flare_features, and spectral_features into a single transform, with
one combined configuration object that can be constructed from a
plain dict (e.g. parsed from YAML/JSON config files).

Usage
-----
>>> from ml.features import FeaturePipeline, PipelineConfig
>>> pipe = FeaturePipeline(PipelineConfig())
>>> features_df = pipe.transform(raw_df)
>>> pipe.feature_names()  # full list of generated columns, in order
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .basic_features import BasicFeatures, BasicFeatureConfig
from .temporal_features import TemporalFeatures, TemporalFeatureConfig
from .flare_features import FlareFeatures, FlareFeatureConfig
from .spectral_features import SpectralFeatures, SpectralFeatureConfig


@dataclass
class PipelineConfig:
    """Top-level configuration bundling all stage configs.

    Each stage can be configured independently; omitted stages fall
    back to that stage's own defaults. Stages can also be individually
    disabled via `enable_basic` / `enable_temporal` / `enable_flare` /
    `enable_spectral`, e.g. to run a cheap basic-only pass first.
    """

    basic: BasicFeatureConfig = field(default_factory=BasicFeatureConfig)
    temporal: TemporalFeatureConfig = field(default_factory=TemporalFeatureConfig)
    flare: FlareFeatureConfig = field(default_factory=FlareFeatureConfig)
    spectral: SpectralFeatureConfig = field(default_factory=SpectralFeatureConfig)

    enable_basic: bool = True
    enable_temporal: bool = True
    enable_flare: bool = True
    enable_spectral: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        """Build a full pipeline config from a nested plain dict, e.g.:

            {
              "basic":    {"windows_sec": [60, 300]},
              "temporal": {"lags_sec": [12, 60]},
              "flare":    {"soft_col": "CR", "hard_col": "CR_hard"},
              "spectral": {"soft_col": "CR", "hard_col": "CR_hard"},
              "enable_spectral": False
            }

        Any stage key may be omitted, in which case that stage's
        defaults are used.
        """
        d = dict(d)  # shallow copy, don't mutate caller's dict
        basic = BasicFeatureConfig.from_dict(d.pop("basic", {}))
        temporal = TemporalFeatureConfig.from_dict(d.pop("temporal", {}))
        flare = FlareFeatureConfig.from_dict(d.pop("flare", {}))
        spectral = SpectralFeatureConfig.from_dict(d.pop("spectral", {}))
        return cls(basic=basic, temporal=temporal, flare=flare, spectral=spectral, **d)

    def to_dict(self) -> Dict[str, Any]:
        """Inverse of from_dict; useful for logging/serialising a run."""
        return asdict(self)


class FeaturePipeline:
    """Chains all feature stages over a single input DataFrame.

    Each stage's `transform` receives the *output* of the previous
    stage, so later stages can in principle reference earlier-stage
    feature columns (none of the four stages currently do this -- they
    are designed to be independent and only read the original raw
    columns -- but the chaining is in place to support that in future
    without changing this orchestrator).
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        self._stages: List[Tuple[str, Any]] = []
        if self.config.enable_basic:
            self._stages.append(("basic", BasicFeatures(self.config.basic)))
        if self.config.enable_temporal:
            self._stages.append(("temporal", TemporalFeatures(self.config.temporal)))
        if self.config.enable_flare:
            self._stages.append(("flare", FlareFeatures(self.config.flare)))
        if self.config.enable_spectral:
            self._stages.append(("spectral", SpectralFeatures(self.config.spectral)))

    # ------------------------------------------------------------------
    # Public API
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
            If a stage fails, re-raised with the stage name attached so
            the failure is traceable to a specific feature module
            rather than surfacing as an opaque pandas/numpy error.
        """
        out = df
        for stage_name, stage in self._stages:
            try:
                out = stage.transform(out)
            except Exception as exc:  # noqa: BLE001 - intentionally broad, re-raised
                raise RuntimeError(
                    f"FeaturePipeline stage '{stage_name}' failed: {exc}"
                ) from exc
        return out

    def transform_separately(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run every enabled stage independently on the *raw* input.

        Unlike `transform`, each stage receives the original `df`
        rather than the previous stage's output. Useful for debugging
        a single stage's output in isolation, or for inspecting which
        stage produced a given column without name collisions across
        stages masking the source.
        """
        results = {}
        for stage_name, stage in self._stages:
            try:
                results[stage_name] = stage.transform(df)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"FeaturePipeline stage '{stage_name}' failed: {exc}"
                ) from exc
        return results
