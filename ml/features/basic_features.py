"""
basic_features.py
==================

Rolling-window statistics computed directly on the raw count-rate (CR)
time series. These are the foundation features that downstream modules
(temporal, flare, spectral) build on.

Expected input schema
----------------------
A pandas.DataFrame with (at minimum):
    'time' : datetime64[ns]   - monotonically increasing timestamps
    'CR'   : float            - count rate (photons / s, or similar flux unit)

Scientific grounding (Benz 2008, "Flare Observations")
-------------------------------------------------------
- Log counts:        flare energetics follow power-law distributions
                      (Sec. 4, Table 1); log-scale linearises this.
- Rolling mean:       background-level estimate, used in the Neupert
                      effect (Sec. 2.4, Eq. 1-2).
- Rolling std/var:    impulsive-phase onset is marked by a sharp rise
                      in variability (Sec. 1.3).
- Rolling max/min:    peak flux proxy / background floor (Sec. 5.2, 4.5).
- Moving RMS:         total signal power, sensitive to elevated baseline.
- Signal energy:      proxy for radiated energy budget (Sec. 4, Table 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class BasicFeatureConfig:
    """Configuration for BasicFeatures.

    Parameters
    ----------
    windows_sec : list of int
        Rolling window widths, in seconds. Each width is converted to a
        number of samples using the inferred sampling cadence of the
        'time' column. Defaults cover three Benz (2008) timescales:
        1 min (impulsive-phase substructure), 5 min (impulsive phase
        duration), 10 min (flash-phase duration).
    time_col : str
        Name of the datetime column.
    value_col : str
        Name of the count-rate column to operate on.
    min_periods_frac : float
        Fraction of the window that must be non-NaN for a rolling
        statistic to be computed (passed to pandas as min_periods).
    log_offset : float
        Additive offset before taking log10, to avoid log(0).
    clip_negative : bool
        If True, negative CR values (e.g. background-subtracted noise
        dipping below zero) are clipped to 0 before log/energy
        calculations, which are undefined/meaningless for negative flux.
    """

    windows_sec: List[int] = field(default_factory=lambda: [60, 300, 600])
    time_col: str = "time"
    value_col: str = "CR"
    min_periods_frac: float = 0.5
    log_offset: float = 1.0
    clip_negative: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "BasicFeatureConfig":
        """Build a config from a plain dict (e.g. parsed from YAML)."""
        return cls(**d)


class BasicFeatures:
    """Compute rolling-window statistics on a count-rate time series.

    Usage
    -----
    >>> cfg = BasicFeatureConfig(windows_sec=[60, 300])
    >>> bf = BasicFeatures(cfg)
    >>> out = bf.transform(df)
    """

    def __init__(self, config: Optional[BasicFeatureConfig] = None):
        self.config = config or BasicFeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feature_names(self) -> List[str]:
        """Return the list of feature column names this transform adds."""
        names = ["log_counts"]
        for w in self.config.windows_sec:
            names += [
                f"rolling_mean_{w}s",
                f"rolling_std_{w}s",
                f"rolling_max_{w}s",
                f"rolling_min_{w}s",
                f"rolling_var_{w}s",
                f"rms_{w}s",
                f"signal_energy_{w}s",
            ]
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all basic features and return a new DataFrame.

        The input DataFrame is not mutated. The returned DataFrame has
        all original columns plus the new feature columns appended.
        """
        cfg = self.config
        self._validate(df)

        out = df.copy()
        cr = out[cfg.value_col].astype(float)

        if cfg.clip_negative:
            cr = cr.clip(lower=0.0)

        out["log_counts"] = np.log10(cr + cfg.log_offset)

        window_sizes = self._resolve_window_sizes(out[cfg.time_col])

        for w_sec, n_samples in zip(cfg.windows_sec, window_sizes):
            min_periods = max(1, int(round(n_samples * cfg.min_periods_frac)))
            roll = cr.rolling(window=n_samples, min_periods=min_periods)

            out[f"rolling_mean_{w_sec}s"] = roll.mean()
            out[f"rolling_std_{w_sec}s"] = roll.std()
            out[f"rolling_max_{w_sec}s"] = roll.max()
            out[f"rolling_min_{w_sec}s"] = roll.min()
            out[f"rolling_var_{w_sec}s"] = roll.var()

            mean_sq = (cr ** 2).rolling(window=n_samples, min_periods=min_periods).mean()
            out[f"rms_{w_sec}s"] = np.sqrt(mean_sq)

            dt = self._median_dt_seconds(out[cfg.time_col])
            sum_sq = (cr ** 2).rolling(window=n_samples, min_periods=min_periods).sum()
            out[f"signal_energy_{w_sec}s"] = (sum_sq * dt) / n_samples

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:
        cfg = self.config
        if cfg.time_col not in df.columns:
            raise KeyError(f"Missing required time column: '{cfg.time_col}'")
        if cfg.value_col not in df.columns:
            raise KeyError(f"Missing required value column: '{cfg.value_col}'")
        if len(df) == 0:
            return
        if not pd.api.types.is_datetime64_any_dtype(df[cfg.time_col]):
            raise TypeError(f"Column '{cfg.time_col}' must be datetime64 dtype")

    @staticmethod
    def _median_dt_seconds(time_series: pd.Series) -> float:
        """Infer the median sampling cadence in seconds."""
        if len(time_series) < 2:
            return 1.0
        diffs = time_series.diff().dropna().dt.total_seconds()
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return 1.0
        return float(diffs.median())

    def _resolve_window_sizes(self, time_series: pd.Series) -> List[int]:
        """Convert each window width in seconds to a number of samples."""
        dt = self._median_dt_seconds(time_series)
        sizes = []
        for w_sec in self.config.windows_sec:
            n = max(1, int(round(w_sec / dt)))
            sizes.append(n)
        return sizes
