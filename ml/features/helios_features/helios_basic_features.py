"""
helios_features/helios_basic_features.py
==========================================

Rolling-window statistics for HEL1OS hard X-ray light curves.

Expected input schema
----------------------
A pandas.DataFrame with (at minimum):
    'time'        : datetime64[ns]   — monotonically increasing timestamps
    'cdte_CR'     : float            — CdTe broadband count rate (20–150 keV)
    'czt_CR'      : float            — CZT broadband count rate (8–60 keV)

Additional optional columns (energy band sub-ranges):
    'cdte_lo_CR'  : float            — CdTe low sub-band (e.g. 20–60 keV)
    'cdte_hi_CR'  : float            — CdTe high sub-band (e.g. 60–150 keV)
    'czt_lo_CR'   : float            — CZT low sub-band (e.g. 8–25 keV)
    'czt_hi_CR'   : float            — CZT high sub-band (e.g. 25–60 keV)

Scientific grounding (Benz 2008, "Flare Observations")
-------------------------------------------------------
- Log counts: HXR flux follows a power-law distribution (§2.2, Fig. 7);
              log-scale linearises the dynamical range.
- Rolling mean: background estimator used in event detection (§5.2).
- Rolling std: impulsive-phase onset signature — variability spikes before peak.
- Rolling max: peak flux proxy (§4.5, §5.2).
- Signal energy: total radiated energy proxy in HXR band (§4, Table 1).
- Energy band features: the ratio of high-sub-band to low-sub-band counts
  within one detector is an instantaneous spectral-slope proxy (§2.2).

Coding style
------------
Identical to BasicFeatures (SoLEXS) in every respect — same dataclass pattern,
same _validate / _resolve_window_sizes / _median_dt_seconds helpers — so the
two modules can be used and tested interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class HEL1OSBasicFeatureConfig:
    """Configuration for HEL1OSBasicFeatures.

    Parameters
    ----------
    windows_sec : list of int
        Rolling window widths in seconds.  Defaults match the SoLEXS
        BasicFeatureConfig defaults for easy cross-instrument comparison.
    time_col : str
        Name of the datetime column.
    cdte_col : str
        Name of the CdTe broadband count-rate column.
    czt_col : str
        Name of the CZT broadband count-rate column.
    cdte_lo_col, cdte_hi_col : str or None
        Optional CdTe sub-band columns.
    czt_lo_col, czt_hi_col : str or None
        Optional CZT sub-band columns.
    min_periods_frac : float
        Fraction of window that must be non-NaN for a rolling stat to compute.
    log_offset : float
        Additive offset before log10 (avoid log(0)).
    clip_negative : bool
        If True, negative CR values are clipped to 0.
    """

    windows_sec:    List[int] = field(default_factory=lambda: [60, 300, 600])
    time_col:       str = "time"
    cdte_col:       str = "cdte_CR"
    czt_col:        str = "czt_CR"
    cdte_lo_col:    Optional[str] = None
    cdte_hi_col:    Optional[str] = None
    czt_lo_col:     Optional[str] = None
    czt_hi_col:     Optional[str] = None
    min_periods_frac: float = 0.5
    log_offset:     float = 1.0
    clip_negative:  bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "HEL1OSBasicFeatureConfig":
        """Build a config from a plain dict (e.g. parsed from YAML)."""
        return cls(**d)


class HEL1OSBasicFeatures:
    """Compute rolling-window statistics on HEL1OS count-rate channels.

    Usage
    -----
    >>> cfg = HEL1OSBasicFeatureConfig(windows_sec=[60, 300])
    >>> hbf = HEL1OSBasicFeatures(cfg)
    >>> out = hbf.transform(df)
    """

    def __init__(self, config: Optional[HEL1OSBasicFeatureConfig] = None):
        self.config = config or HEL1OSBasicFeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feature_names(self) -> List[str]:
        """Return the ordered list of feature column names this transform adds."""
        cfg = self.config
        names: List[str] = ["cdte_log_counts", "czt_log_counts"]

        for w in cfg.windows_sec:
            for prefix in ("cdte", "czt"):
                names += [
                    f"{prefix}_rolling_mean_{w}s",
                    f"{prefix}_rolling_std_{w}s",
                    f"{prefix}_rolling_max_{w}s",
                    f"{prefix}_rolling_min_{w}s",
                    f"{prefix}_rolling_var_{w}s",
                    f"{prefix}_rms_{w}s",
                    f"{prefix}_signal_energy_{w}s",
                ]

        # Energy-band ratio features (only declared if sub-band cols configured)
        if cfg.cdte_lo_col and cfg.cdte_hi_col:
            names += ["cdte_band_ratio"]
            for w in cfg.windows_sec:
                names += [f"cdte_band_ratio_rolling_mean_{w}s"]
        if cfg.czt_lo_col and cfg.czt_hi_col:
            names += ["czt_band_ratio"]
            for w in cfg.windows_sec:
                names += [f"czt_band_ratio_rolling_mean_{w}s"]

        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all HEL1OS basic features and return a new DataFrame.

        The input DataFrame is not mutated. The returned DataFrame has
        all original columns plus the new feature columns appended.
        """
        cfg = self.config
        self._validate(df)

        out = df.copy()

        for col_raw, log_col, prefix in [
            (cfg.cdte_col, "cdte_log_counts", "cdte"),
            (cfg.czt_col,  "czt_log_counts",  "czt"),
        ]:
            cr = out[col_raw].astype(float)
            if cfg.clip_negative:
                cr = cr.clip(lower=0.0)

            out[log_col] = np.log10(cr + cfg.log_offset)
            window_sizes = self._resolve_window_sizes(out[cfg.time_col])
            dt = self._median_dt_seconds(out[cfg.time_col])

            for w_sec, n_samples in zip(cfg.windows_sec, window_sizes):
                min_periods = max(1, int(round(n_samples * cfg.min_periods_frac)))
                roll = cr.rolling(window=n_samples, min_periods=min_periods)

                out[f"{prefix}_rolling_mean_{w_sec}s"] = roll.mean()
                out[f"{prefix}_rolling_std_{w_sec}s"]  = roll.std()
                out[f"{prefix}_rolling_max_{w_sec}s"]  = roll.max()
                out[f"{prefix}_rolling_min_{w_sec}s"]  = roll.min()
                out[f"{prefix}_rolling_var_{w_sec}s"]  = roll.var()

                mean_sq = (cr ** 2).rolling(window=n_samples, min_periods=min_periods).mean()
                out[f"{prefix}_rms_{w_sec}s"] = np.sqrt(mean_sq)

                sum_sq = (cr ** 2).rolling(window=n_samples, min_periods=min_periods).sum()
                out[f"{prefix}_signal_energy_{w_sec}s"] = (sum_sq * dt) / n_samples

        # ── Energy band ratio features ────────────────────────────────────
        out = self._add_band_ratios(out, cfg)

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_band_ratios(self, out: pd.DataFrame, cfg: HEL1OSBasicFeatureConfig) -> pd.DataFrame:
        """Add CdTe and CZT intra-band ratio features if sub-band cols are present."""
        window_sizes = self._resolve_window_sizes(out[cfg.time_col])

        for lo_col, hi_col, ratio_col, prefix in [
            (cfg.cdte_lo_col, cfg.cdte_hi_col, "cdte_band_ratio", "cdte"),
            (cfg.czt_lo_col,  cfg.czt_hi_col,  "czt_band_ratio",  "czt"),
        ]:
            if lo_col is None or hi_col is None:
                continue
            if lo_col not in out.columns or hi_col not in out.columns:
                continue

            lo = out[lo_col].astype(float).clip(lower=0.0)
            hi = out[hi_col].astype(float).clip(lower=0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = hi / (lo + 1e-6)
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            out[ratio_col] = ratio

            for w_sec, n_samples in zip(cfg.windows_sec, window_sizes):
                min_periods = max(1, int(round(n_samples * cfg.min_periods_frac)))
                out[f"{prefix}_band_ratio_rolling_mean_{w_sec}s"] = (
                    ratio.rolling(window=n_samples, min_periods=min_periods).mean()
                )
        return out

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
    def _median_dt_seconds(time_series: pd.Series) -> float:
        if len(time_series) < 2:
            return 1.0
        diffs = time_series.diff().dropna().dt.total_seconds()
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return 1.0
        return float(diffs.median())

    def _resolve_window_sizes(self, time_series: pd.Series) -> List[int]:
        dt = self._median_dt_seconds(time_series)
        return [max(1, int(round(w / dt))) for w in self.config.windows_sec]