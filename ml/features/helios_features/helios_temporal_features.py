"""
helios_features/helios_temporal_features.py
=============================================

Temporal dynamics features for HEL1OS hard X-ray count rates.

Expected input schema
----------------------
    'time'    : datetime64[ns]
    'cdte_CR' : float   — CdTe broadband
    'czt_CR'  : float   — CZT broadband

Scientific grounding (Benz 2008)
---------------------------------
- d/dt CR: HXR derivative directly indicates the rate of particle
  acceleration (§2.2). Peaks sharply in the impulsive phase.
- d²/dt² CR: second derivative identifies the acceleration onset
  inflection — the transition from preflare to impulsive (§1.3).
- Lag differences/ratios: time-of-flight electron delays between HXR
  footpoints and the site of acceleration (§2.2; Aschwanden 1995/1996).
- EMA: provides a slowly-varying background estimate, essential for
  distinguishing the HXR burst above background.
- Cumulative fluence: ∫ CR dt — total HXR radiated energy proxy,
  which (by the Neupert effect) should track soft X-ray peak emission.

Coding style
------------
Identical to TemporalFeatures (SoLEXS) in architecture — same config
dataclass, same feature_names/transform pattern, same helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class HEL1OSTemporalFeatureConfig:
    """Configuration for HEL1OSTemporalFeatures.

    Parameters
    ----------
    time_col : str
    cdte_col, czt_col : str
        Broadband count-rate columns for CdTe and CZT detectors.
    lags_sec : list of int
        Lag offsets (seconds) for diff/ratio features.
    ema_spans_sec : list of int
        EMA span widths in seconds.
    median_windows_sec : list of int
        Rolling median window widths in seconds.
    derivative_smoothing : int
        Pre-smoothing (rolling mean samples) before differencing.
    """

    time_col:             str = "time"
    cdte_col:             str = "cdte_CR"
    czt_col:              str = "czt_CR"
    lags_sec:             List[int] = field(default_factory=lambda: [12, 60])
    ema_spans_sec:        List[int] = field(default_factory=lambda: [60, 300])
    median_windows_sec:   List[int] = field(default_factory=lambda: [60, 300])
    derivative_smoothing: int = 3

    @classmethod
    def from_dict(cls, d: dict) -> "HEL1OSTemporalFeatureConfig":
        return cls(**d)


class HEL1OSTemporalFeatures:
    """Compute derivative, lag, EMA, and cumulative fluence features for HEL1OS.

    Usage
    -----
    >>> cfg = HEL1OSTemporalFeatureConfig(lags_sec=[12, 60])
    >>> htf = HEL1OSTemporalFeatures(cfg)
    >>> out = htf.transform(df)
    """

    def __init__(self, config: Optional[HEL1OSTemporalFeatureConfig] = None):
        self.config = config or HEL1OSTemporalFeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feature_names(self) -> List[str]:
        cfg = self.config
        names: List[str] = []
        for prefix in ("cdte", "czt"):
            names += [
                f"{prefix}_dCR_dt",
                f"{prefix}_d2CR_dt2",
                f"{prefix}_time_since_peak_s",
                f"{prefix}_cumulative_fluence",
            ]
            for lag in cfg.lags_sec:
                names += [f"{prefix}_lag_diff_{lag}s", f"{prefix}_lag_ratio_{lag}s"]
            for span in cfg.ema_spans_sec:
                names += [f"{prefix}_ema_{span}s", f"{prefix}_cr_minus_ema_{span}s"]
            for w in cfg.median_windows_sec:
                names += [f"{prefix}_rolling_median_{w}s"]
        names += ["time_since_start_s"]
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all HEL1OS temporal features and return a new DataFrame."""
        cfg = self.config
        self._validate(df)

        out = df.copy()
        if len(out) == 0:
            for name in self.feature_names():
                out[name] = pd.Series(dtype=float)
            return out

        t_sec = self._elapsed_seconds(out[cfg.time_col])
        dt    = self._median_dt_seconds(out[cfg.time_col])
        out["time_since_start_s"] = t_sec

        for col, prefix in [(cfg.cdte_col, "cdte"), (cfg.czt_col, "czt")]:
            cr = out[col].astype(float)

            # ── Derivatives ──────────────────────────────────
            if len(out) < 2:
                out[f"{prefix}_dCR_dt"]   = 0.0
                out[f"{prefix}_d2CR_dt2"] = 0.0
            else:
                smooth_n  = max(1, cfg.derivative_smoothing)
                cr_smooth = cr.rolling(window=smooth_n, min_periods=1, center=True).mean()
                d1 = np.gradient(cr_smooth.to_numpy(), t_sec.to_numpy())
                out[f"{prefix}_dCR_dt"]   = d1
                d2 = np.gradient(d1, t_sec.to_numpy())
                out[f"{prefix}_d2CR_dt2"] = d2

            # ── Time since peak ───────────────────────────────
            peak_idx = int(cr.to_numpy().argmax()) if cr.notna().any() else 0
            peak_t   = t_sec.iloc[peak_idx]
            out[f"{prefix}_time_since_peak_s"] = t_sec - peak_t

            # ── Cumulative fluence (Neupert effect LHS proxy) ─
            cr_pos = cr.clip(lower=0.0)
            out[f"{prefix}_cumulative_fluence"] = (cr_pos * dt).cumsum()

            # ── Lag features ─────────────────────────────────
            for lag_sec in cfg.lags_sec:
                n_lag   = max(1, int(round(lag_sec / dt)))
                shifted = cr.shift(n_lag)
                out[f"{prefix}_lag_diff_{lag_sec}s"]  = cr - shifted
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = cr / shifted
                ratio = ratio.replace([np.inf, -np.inf], np.nan)
                out[f"{prefix}_lag_ratio_{lag_sec}s"] = ratio

            # ── EMA ───────────────────────────────────────────
            for span_sec in cfg.ema_spans_sec:
                n_span = max(1, int(round(span_sec / dt)))
                ema    = cr.ewm(span=n_span, adjust=False, min_periods=1).mean()
                out[f"{prefix}_ema_{span_sec}s"]          = ema
                out[f"{prefix}_cr_minus_ema_{span_sec}s"] = cr - ema

            # ── Rolling median ────────────────────────────────
            for w_sec in cfg.median_windows_sec:
                n_samples   = max(1, int(round(w_sec / dt)))
                min_periods = max(1, n_samples // 2)
                out[f"{prefix}_rolling_median_{w_sec}s"] = cr.rolling(
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
        for col in (cfg.cdte_col, cfg.czt_col):
            if col not in df.columns:
                raise KeyError(f"Missing required HEL1OS column: '{col}'")
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