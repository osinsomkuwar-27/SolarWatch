"""
feature_pipeline.py
====================

Orchestrator that chains basic_features, temporal_features,
flare_features, and spectral_features into a single transform, with
one combined configuration object that can be constructed from a
plain dict (e.g. parsed from YAML/JSON config files).

Usage
-----
>>> from solar.ml.features import FeaturePipeline, PipelineConfig
>>> pipe = FeaturePipeline(PipelineConfig())
>>> features_df = pipe.transform(raw_df)
>>> pipe.feature_names()  # full list of generated columns, in order

Extended for HEL1OS / combined instrument mode
----------------------------------------------
>>> from solar.ml.features.feature_pipeline import CombinedFeaturePipeline
>>> combined = CombinedFeaturePipeline(solexs_config=PipelineConfig(),
...                                    helios_config=HEL1OSPipelineConfig())
>>> out = combined.transform(df)   # df must have both SoLEXS and HEL1OS columns
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
        """Build a full pipeline config from a nested plain dict, e.g.::

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


# ══════════════════════════════════════════════════════════════════════════════
# NEW: Combined instrument pipeline — HEL1OS extension
# ══════════════════════════════════════════════════════════════════════════════

class CombinedFeaturePipeline:
    """Run SoLEXS and HEL1OS feature pipelines over a combined DataFrame.

    The input DataFrame must contain columns required by both pipelines.
    Features from both instruments are concatenated in the output.  The
    two pipelines do not share state — each transforms independently.

    This class does NOT modify FeaturePipeline or HEL1OSFeaturePipeline;
    it simply delegates to both and merges their outputs.

    Parameters
    ----------
    solexs_config : PipelineConfig or None
        SoLEXS (soft X-ray) pipeline configuration.
    helios_config : HEL1OSPipelineConfig or None
        HEL1OS (hard X-ray) pipeline configuration.
    instrument : str
        One of 'solexs', 'helios', 'combined'.  Controls which pipelines
        are actually run.
    """

    INSTRUMENT_SOLEXS   = "solexs"
    INSTRUMENT_HELIOS   = "helios"
    INSTRUMENT_COMBINED = "combined"

    def __init__(
        self,
        solexs_config:  Optional[PipelineConfig]      = None,
        helios_config:  Optional["HEL1OSPipelineConfig"] = None,  # noqa: F821
        instrument:     str = "combined",
    ) -> None:
        if instrument not in (
            self.INSTRUMENT_SOLEXS,
            self.INSTRUMENT_HELIOS,
            self.INSTRUMENT_COMBINED,
        ):
            raise ValueError(
                f"instrument must be one of 'solexs', 'helios', 'combined'; "
                f"got '{instrument}'"
            )
        self.instrument = instrument

        self._solexs_pipe: Optional[FeaturePipeline] = None
        self._helios_pipe = None

        if instrument in (self.INSTRUMENT_SOLEXS, self.INSTRUMENT_COMBINED):
            self._solexs_pipe = FeaturePipeline(solexs_config)

        if instrument in (self.INSTRUMENT_HELIOS, self.INSTRUMENT_COMBINED):
            from .helios_features.helios_pipeline import (
                HEL1OSFeaturePipeline,
                HEL1OSPipelineConfig,
            )
            self._helios_pipe = HEL1OSFeaturePipeline(helios_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feature_names(self) -> List[str]:
        """Ordered list of all feature columns both pipelines will add."""
        names: List[str] = []
        if self._solexs_pipe is not None:
            names.extend(self._solexs_pipe.feature_names())
        if self._helios_pipe is not None:
            names.extend(self._helios_pipe.feature_names())
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run both pipelines in order and return the merged output.

        SoLEXS features are computed first; the HEL1OS pipeline then
        receives the SoLEXS-enriched DataFrame (so HEL1OS stages do not
        accidentally rely on SoLEXS-derived columns, but the column names
        are unique across instruments, so no collisions occur).

        Raises
        ------
        RuntimeError  with pipeline name attached on failure.
        """
        out = df
        if self._solexs_pipe is not None:
            try:
                out = self._solexs_pipe.transform(out)
            except RuntimeError as exc:
                raise RuntimeError(f"[SoLEXS] {exc}") from exc

        if self._helios_pipe is not None:
            try:
                out = self._helios_pipe.transform(out)
            except RuntimeError as exc:
                raise RuntimeError(f"[HEL1OS] {exc}") from exc

        return out