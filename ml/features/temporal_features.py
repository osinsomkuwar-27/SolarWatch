"""
temporal_features.py
=====================

Features describing the *dynamics* (rate of change, memory, smoothed
trend) of the count-rate time series, as opposed to basic_features.py
which describes its instantaneous distributional properties.

Expected input schema
----------------------
A pandas.DataFrame with (at minimum):
    'time' : datetime64[ns]
    'CR'   : float

Scientific grounding (Benz 2008, "Flare Observations")
-------------------------------------------------------
- First derivative (d/dt CR):
      Central to the Neupert effect: d(F_SXR)/dt ~ F_HXR(t)
      (Sec. 2.4, Eq. 2). A rising derivative flags the impulsive phase.
- Second derivative:
      Identifies inflection points -- e.g. the transition from
      impulsive to flash phase (Sec. 1.3).
- Lag features:
      The Neupert effect and electron time-of-flight delays
      (Sec. 2.2; Aschwanden et al. 1995/1996) both involve comparing
      a signal to its own past values at a fixed offset.
- Exponential moving average (EMA):
      A smoothed background tracker, less sensitive to single-sample
      noise spikes than a simple rolling mean, useful for distinguishing
      the "gentle" pre-flare/decay heating (Sec. 2.6, 2.7) from genuine
      impulsive energy release.
- Rolling median:
      Robust background estimator, insensitive to brief outlier counts
      (e.g. cosmic-ray hits on the detector), unlike the rolling mean.
- Time-since-peak / time-since-rise-start:
      Operationalises the phase timeline of Fig. 2 (preflare, impulsive,
      flash, decay durations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class TemporalFeatureConfig:
    """Configuration for TemporalFeatures.

    Parameters
    ----------
    time_col, value_col : str
        Column names for timestamp and count rate.
    lags_sec : list of int
        Lag offsets (seconds) at which to compute CR(t) - CR(t - lag)
        and CR(t) / CR(t - lag).
    ema_spans_sec : list of int
        Span widths (seconds, converted to samples) for exponential
        moving averages.
    median_windows_sec : list of int
        Window widths (seconds) for rolling median.
    derivative_smoothing : int
        Number of samples to smooth CR over (simple rolling mean) before
        differencing, to reduce derivative noise amplification. Use 1
        for no smoothing.
    """

    time_col: str = "time"
    value_col: str = "CR"
    lags_sec: List[int] = field(default_factory=lambda: [12, 60])
    ema_spans_sec: List[int] = field(default_factory=lambda: [60, 300])
    median_windows_sec: List[int] = field(default_factory=lambda: [60, 300])
    derivative_smoothing: int = 3

    @classmethod
    def from_dict(cls, d: dict) -> "TemporalFeatureConfig":
        return cls(**d)


class TemporalFeatures:
    """Compute derivative, lag, EMA, and rolling-median features.

    Usage
    -----
    >>> cfg = TemporalFeatureConfig(lags_sec=[12, 60])
    >>> tf = TemporalFeatures(cfg)
    >>> out = tf.transform(df)
    """

    def __init__(self, config: Optional[TemporalFeatureConfig] = None):
        self.config = config or TemporalFeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feature_names(self) -> List[str]:
        cfg = self.config
        names = [
            "dCR_dt",
            "d2CR_dt2",
            "time_since_start_s",
            "time_since_peak_s",
        ]
        for lag in cfg.lags_sec:
            names += [f"lag_diff_{lag}s", f"lag_ratio_{lag}s"]
        for span in cfg.ema_spans_sec:
            names += [f"ema_{span}s", f"cr_minus_ema_{span}s"]
        for w in cfg.median_windows_sec:
            names += [f"rolling_median_{w}s"]
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all temporal features and return a new DataFrame."""
        cfg = self.config
        self._validate(df)

        out = df.copy()
        if len(out) == 0:
            for name in self.feature_names():
                out[name] = pd.Series(dtype=float)
            return out

        cr = out[cfg.value_col].astype(float)
        t_sec = self._elapsed_seconds(out[cfg.time_col])
        dt = self._median_dt_seconds(out[cfg.time_col])

        # --- Derivatives -------------------------------------------------
        # np.gradient requires >= 2 points; a single-row series has no
        # well-defined rate of change, so derivatives are 0 by convention
        # (matching the natural "no change observed yet" interpretation)
        # rather than raising.
        if len(out) < 2:
            out["dCR_dt"] = 0.0
            out["d2CR_dt2"] = 0.0
        else:
            smooth_n = max(1, cfg.derivative_smoothing)
            cr_smooth = cr.rolling(window=smooth_n, min_periods=1, center=True).mean()

            d1 = np.gradient(cr_smooth.to_numpy(), t_sec.to_numpy())
            out["dCR_dt"] = d1
            d2 = np.gradient(d1, t_sec.to_numpy())
            out["d2CR_dt2"] = d2

        # --- Time-since-start / time-since-peak --------------------------
        out["time_since_start_s"] = t_sec
        peak_idx = int(cr.to_numpy().argmax()) if cr.notna().any() else 0
        peak_t = t_sec.iloc[peak_idx]
        out["time_since_peak_s"] = t_sec - peak_t

        # --- Lag features --------------------------------------------------
        for lag_sec in cfg.lags_sec:
            n_lag = max(1, int(round(lag_sec / dt)))
            shifted = cr.shift(n_lag)
            out[f"lag_diff_{lag_sec}s"] = cr - shifted
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = cr / shifted
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            out[f"lag_ratio_{lag_sec}s"] = ratio

        # --- EMA -------------------------------------------------------------
        for span_sec in cfg.ema_spans_sec:
            n_span = max(1, int(round(span_sec / dt)))
            ema = cr.ewm(span=n_span, adjust=False, min_periods=1).mean()
            out[f"ema_{span_sec}s"] = ema
            out[f"cr_minus_ema_{span_sec}s"] = cr - ema

        # --- Rolling median --------------------------------------------------
        for w_sec in cfg.median_windows_sec:
            n_samples = max(1, int(round(w_sec / dt)))
            min_periods = max(1, n_samples // 2)
            out[f"rolling_median_{w_sec}s"] = cr.rolling(
                window=n_samples, min_periods=min_periods
            ).median()

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
    def _elapsed_seconds(time_series: pd.Series) -> pd.Series:
        t0 = time_series.iloc[0]
        return (time_series - t0).dt.total_seconds()

    @staticmethod
    def _median_dt_seconds(time_series: pd.Series) -> float:
        if len(time_series) < 2:
            return 1.0
        diffs = time_series.diff().dropna().dt.total_seconds()
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return 1.0
        return float(diffs.median())
