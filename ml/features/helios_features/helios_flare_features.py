"""
helios_features/helios_flare_features.py
==========================================

HXR-specific flare-physics features: hardness ratios, cumulative fluence,
and HXR phase classification for HEL1OS CdTe/CZT data.

Expected input schema
----------------------
    'time'    : datetime64[ns]
    'cdte_CR' : float   — CdTe broadband (20–150 keV); acts as "hard" channel
    'czt_CR'  : float   — CZT broadband (8–60 keV);   acts as "soft" channel
                          within the HXR band

Scientific grounding (Benz 2008)
---------------------------------
- Hardness ratio (CdTe / CZT):
    Within the HXR band, the ratio of higher-energy to lower-energy counts
    tracks the instantaneous spectral index.  The spectrum hardens during
    the impulsive phase (Parks & Winckler 1969; Benz §5.2).
- Smoothed hardness:
    Rolling-mean smoothing removes noise without distorting the trend.
- Cumulative fluence:
    ∫ CR dt for each detector.  For CdTe (high-E) this is the hard X-ray
    fluence; by the Neupert effect (§2.4) it should track the SoLEXS
    soft-X-ray peak emission.
- HXR phase labels (preflare / impulsive / flash / decay):
    Rule-based, driven by the CdTe derivative (faster response channel),
    mirroring FlareFeatures (SoLEXS) but adapted for HXR timescales
    (impulsive phase is shorter and the derivative threshold is larger).
- Detector statistics:
    Mean, std, P90 of each detector over the series — quick summary
    features usable directly by downstream classifiers.

Coding style
------------
Identical to FlareFeatures (SoLEXS): same dataclass, same feature_names /
transform public API, same _validate helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

HXR_PHASES = ("preflare", "impulsive", "flash", "decay")


@dataclass
class HEL1OSFlareFeatureConfig:
    """Configuration for HEL1OSFlareFeatures.

    Parameters
    ----------
    time_col : str
    cdte_col : str  — high-energy channel (hardness numerator)
    czt_col  : str  — low-energy channel  (hardness denominator)
    hardness_smooth_sec : int
        Window for smoothing before computing the hardness ratio.
    background_window_sec : int
        Window used to estimate pre-event background.
    impulsive_deriv_threshold : float
        Normalised dCR/dt above which the phase is impulsive.
    flash_amplitude_frac : float
        Fraction of peak-above-background for flash phase.
    decay_deriv_threshold : float
        Normalised dCR/dt below which the phase is decay.
    min_event_amplitude : float
        Minimum (peak - bg) amplitude to consider any flare present.
    """

    time_col:                 str   = "time"
    cdte_col:                 str   = "cdte_CR"
    czt_col:                  str   = "czt_CR"
    hardness_smooth_sec:      int   = 12
    background_window_sec:    int   = 300    # shorter than SXR: HXR decays faster
    impulsive_deriv_threshold: float = 0.08  # larger than SXR: faster rise
    flash_amplitude_frac:     float = 0.5
    decay_deriv_threshold:    float = -0.03
    min_event_amplitude:      float = 1e-6

    @classmethod
    def from_dict(cls, d: dict) -> "HEL1OSFlareFeatureConfig":
        return cls(**d)


class HEL1OSFlareFeatures:
    """Compute HXR hardness, fluence, and phase features for HEL1OS.

    Usage
    -----
    >>> cfg = HEL1OSFlareFeatureConfig()
    >>> hff = HEL1OSFlareFeatures(cfg)
    >>> out = hff.transform(df)
    """

    def __init__(self, config: Optional[HEL1OSFlareFeatureConfig] = None):
        self.config = config or HEL1OSFlareFeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feature_names(self) -> List[str]:
        return [
            # Hardness ratio features
            "hardness_ratio",
            "hardness_smoothed",
            # Cumulative fluence per detector
            "cdte_cumulative_fluence",
            "czt_cumulative_fluence",
            # d(CdTe)/dt for HXR Neupert diagnostic
            "dcdte_dt",
            # HXR phase labels (CdTe-driven, faster than SXR)
            "hxr_phase",
            "hxr_phase_preflare",
            "hxr_phase_impulsive",
            "hxr_phase_flash",
            "hxr_phase_decay",
            # Detector statistics (scalar features broadcast to each row)
            "cdte_stat_mean",
            "cdte_stat_std",
            "cdte_stat_p90",
            "czt_stat_mean",
            "czt_stat_std",
            "czt_stat_p90",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all HEL1OS flare features and return a new DataFrame."""
        cfg = self.config
        self._validate(df)

        out = df.copy()
        if len(out) == 0:
            for name in self.feature_names():
                out[name] = pd.Series(
                    dtype=object if name == "hxr_phase" else float
                )
            return out

        t_sec  = self._elapsed_seconds(out[cfg.time_col])
        dt     = self._median_dt_seconds(out[cfg.time_col])
        cdte   = out[cfg.cdte_col].astype(float).clip(lower=0.0)
        czt    = out[cfg.czt_col].astype(float).clip(lower=0.0)

        # ── Hardness ratio (CdTe / CZT) ──────────────────────────────────
        n_smooth = max(1, int(round(cfg.hardness_smooth_sec / dt)))
        cdte_sm  = cdte.rolling(window=n_smooth, min_periods=1, center=True).mean()
        czt_sm   = czt.rolling(window=n_smooth,  min_periods=1, center=True).mean()

        with np.errstate(divide="ignore", invalid="ignore"):
            hardness_raw = cdte / (czt + 1e-12)
            hardness_sm  = cdte_sm / (czt_sm + 1e-12)

        out["hardness_ratio"]    = hardness_raw.replace([np.inf, -np.inf], np.nan)
        out["hardness_smoothed"] = hardness_sm.replace([np.inf, -np.inf], np.nan)

        # ── Cumulative fluence ────────────────────────────────────────────
        out["cdte_cumulative_fluence"] = (cdte * dt).cumsum()
        out["czt_cumulative_fluence"]  = (czt  * dt).cumsum()

        # ── d(CdTe)/dt ────────────────────────────────────────────────────
        if len(out) < 2:
            out["dcdte_dt"] = 0.0
        else:
            out["dcdte_dt"] = np.gradient(cdte.to_numpy(), t_sec.to_numpy())

        # ── HXR phase classification (CdTe-driven) ────────────────────────
        phase = self._classify_phases(cdte, t_sec, dt)
        out["hxr_phase"] = phase
        for p in HXR_PHASES:
            out[f"hxr_phase_{p}"] = (phase == p).astype(int)

        # ── Detector statistics ───────────────────────────────────────────
        for col, prefix in [(cfg.cdte_col, "cdte"), (cfg.czt_col, "czt")]:
            cr = out[col].astype(float)
            out[f"{prefix}_stat_mean"] = float(cr.mean())
            out[f"{prefix}_stat_std"]  = float(cr.std())
            out[f"{prefix}_stat_p90"]  = float(cr.quantile(0.90))

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_phases(
        self,
        cdte:  pd.Series,
        t_sec: pd.Series,
        dt:    float,
    ) -> pd.Series:
        """Rule-based HXR phase classifier, mirroring FlareFeatures._classify_phases.

        Uses CdTe (the faster, higher-energy channel) as the reference because
        HXR emission peaks earlier than SXR and decays faster (Benz §1.3).
        The derivative threshold is larger than for SXR because HXR has a
        steeper impulsive rise.
        """
        cfg = self.config
        n = len(cdte)
        n_bg     = max(3, int(round(cfg.background_window_sec / dt)))
        n_smooth = max(3, min(n, n_bg // 5))

        cdte_arr = cdte.to_numpy()
        smoothed = cdte.rolling(window=n_smooth, min_periods=1, center=True).mean()
        lead     = smoothed.iloc[: min(n_bg, n)]
        background_level = float(lead.min()) if len(lead) else float(cdte_arr.min())

        amplitude_above_bg = np.clip(smoothed.to_numpy() - background_level, 0.0, None)
        peak_amp = float(amplitude_above_bg.max()) if n else 0.0

        phase = pd.Series(["preflare"] * n, index=cdte.index, dtype=object)
        if peak_amp < cfg.min_event_amplitude:
            return phase
        if n < 2:
            if amplitude_above_bg[0] / peak_amp >= cfg.flash_amplitude_frac:
                phase.iloc[0] = "flash"
            return phase

        deriv    = np.gradient(smoothed.to_numpy(), t_sec.to_numpy())
        bg_safe  = max(background_level, cfg.min_event_amplitude)
        norm_deriv = deriv / bg_safe

        amp_frac  = amplitude_above_bg / peak_amp
        is_impulsive = norm_deriv > cfg.impulsive_deriv_threshold
        is_high_amp  = amp_frac >= cfg.flash_amplitude_frac
        is_receded   = amp_frac < cfg.flash_amplitude_frac

        labels = np.full(n, "preflare", dtype=object)
        labels[is_receded & ~is_impulsive] = "decay"
        labels[is_high_amp & ~is_impulsive] = "flash"
        labels[is_impulsive] = "impulsive"

        peak_idx = int(np.argmax(amplitude_above_bg))
        pre_peak = np.arange(n) < peak_idx
        labels[pre_peak & (labels == "decay")] = "preflare"

        return pd.Series(labels, index=cdte.index)

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