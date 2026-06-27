"""
flare_features.py
==================

Flare-physics-specific features: spectral hardness evolution, the
Neupert effect, and flare-phase classification.

Expected input schema
----------------------
A pandas.DataFrame with:
    'time' : datetime64[ns]
    'CR'   : float     - a "soft" channel count rate (e.g. GOES-like
                         soft X-ray, used as the thermal/SXR proxy)

Optionally, for the soft-hard-soft (SHS) features:
    hard_col : float   - a "hard" channel count rate (e.g. RHESSI-like
                         hard X-ray, used as the non-thermal/HXR proxy)
If hard_col is not supplied, SHS/Neupert features that require two
channels are skipped (filled with NaN) and a warning is not raised
(many real datasets are single-channel; this is treated as the normal
case, not an error).

Scientific grounding (Benz 2008, "Flare Observations")
-------------------------------------------------------
- Neupert effect (Sec. 2.4, Eq. 1-2):
      F_SXR(t) ~ integral_{t0}^{t} F_HXR(t') dt'
      i.e. d/dt F_SXR(t) ~ F_HXR(t).
  We compute:
    * the running time-integral of the hard channel (cumulative HXR
      fluence), the model's LHS-equivalent quantity,
    * the rolling-window correlation between d(SXR)/dt and HXR,
      which is the direct empirical test of the effect
      (cf. Dennis & Zarro 1993, Fig. 12; holds in ~80% of flares but
      is violated in roughly half by relative-timing criteria, Sec 2.7).
- Soft-hard-soft behaviour (Sec. 5.2):
      Spectral hardness rises through the impulsive phase and falls
      back in the decay phase (Parks & Winckler 1969; Kane & Anderson
      1970). We approximate a photon "hardness ratio" HXR/SXR as a
      proxy for spectral index gamma (harder spectrum -> larger
      hard/soft ratio), since we do not have multi-energy-bin spectra
      here, only two band-integrated channels.
- Flare phase classification (Sec. 1.3, Fig. 2):
      preflare   - slow heating, low/flat SXR, near-background HXR
      impulsive  - HXR rises sharply, most particle acceleration occurs
                  here, duration ~3-10 min in a large event
      flash      - SXR/Halpha intensity and line width rise rapidly,
                  largely coincident with the impulsive phase but
                  SXR may peak somewhat later, duration ~5-20 min
      decay      - SXR/HXR decline; corona returns toward pre-flare
                  state, duration of one to several hours
  This is implemented as a simple, inspectable rule-based classifier
  driven by derivative sign/magnitude and amplitude thresholds relative
  to a rolling background -- not a learned model -- so that its
  behaviour is directly traceable to the physical definitions above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class FlareFeatureConfig:
    """Configuration for FlareFeatures.

    Parameters
    ----------
    time_col : str
        Timestamp column.
    soft_col : str
        Column used as the soft/thermal (SXR-like) proxy channel.
    hard_col : str or None
        Column used as the hard/non-thermal (HXR-like) proxy channel.
        If None, SHS/Neupert two-channel features are filled with NaN.
    shs_corr_window_sec : int
        Window (seconds) over which the rolling Neupert correlation
        (between d(soft)/dt and hard channel) is computed.
    hardness_smooth_sec : int
        Smoothing window (seconds) applied to both channels before
        computing the hardness ratio, to reduce ratio noise when either
        channel is near background.
    background_window_sec : int
        Window (seconds) used to estimate a slowly-varying background
        level for the soft channel, used by the phase classifier.
    impulsive_deriv_threshold :
        Threshold, in units of (background-normalised d(soft)/dt per
        second), above which the rise is classified "impulsive" rather
        than "preflare"/"decay". Expressed as a fraction of the
        background level per second; tune to data cadence/units.
    flash_amplitude_frac :
        Fraction of the flare's peak-above-background soft amplitude
        above which a sample is eligible to be classified "flash"
        (the slower, larger-amplitude phase that follows/overlaps the
        impulsive rise, Sec 1.3).
    decay_deriv_threshold :
        Threshold (same units as impulsive_deriv_threshold) below which
        (i.e. sufficiently negative) a falling sample is classified
        "decay".
    min_event_amplitude :
        Minimum (peak - background) amplitude, in raw CR units, for a
        time series to be considered to contain a flare at all. Below
        this, every sample is classified "preflare" (quiescent).
    """

    time_col: str = "time"
    soft_col: str = "CR"
    hard_col: Optional[str] = None
    shs_corr_window_sec: int = 120
    hardness_smooth_sec: int = 12
    background_window_sec: int = 600
    impulsive_deriv_threshold: float = 0.05
    flash_amplitude_frac: float = 0.5
    decay_deriv_threshold: float = -0.02
    min_event_amplitude: float = 1e-6

    @classmethod
    def from_dict(cls, d: dict) -> "FlareFeatureConfig":
        return cls(**d)


PHASES = ("preflare", "impulsive", "flash", "decay")


class FlareFeatures:
    """Compute soft-hard-soft / Neupert-effect features and flare phase.

    Usage
    -----
    >>> cfg = FlareFeatureConfig(soft_col="CR", hard_col="CR_hard")
    >>> ff = FlareFeatures(cfg)
    >>> out = ff.transform(df)
    """

    def __init__(self, config: Optional[FlareFeatureConfig] = None):
        self.config = config or FlareFeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feature_names(self) -> List[str]:
        return [
            "hxr_cumulative_fluence",
            "dSXR_dt",
            "neupert_corr",
            "hardness_ratio",
            "hardness_smoothed",
            "flare_phase",
            "phase_preflare",
            "phase_impulsive",
            "phase_flash",
            "phase_decay",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        self._validate(df)

        out = df.copy()
        if len(out) == 0:
            for name in self.feature_names():
                out[name] = pd.Series(
                    dtype=object if name == "flare_phase" else float
                )
            return out

        t_sec = self._elapsed_seconds(out[cfg.time_col])
        dt = self._median_dt_seconds(out[cfg.time_col])
        soft = out[cfg.soft_col].astype(float).clip(lower=0.0)

        has_hard = cfg.hard_col is not None and cfg.hard_col in out.columns
        hard = out[cfg.hard_col].astype(float).clip(lower=0.0) if has_hard else None

        # --- d(SXR)/dt -----------------------------------------------------
        # np.gradient requires >= 2 points; with a single sample there is
        # no observable rate of change, so we report 0 by convention.
        if len(out) < 2:
            dSXR_dt = np.zeros(len(out))
        else:
            dSXR_dt = np.gradient(soft.to_numpy(), t_sec.to_numpy())
        out["dSXR_dt"] = dSXR_dt

        # --- Neupert effect: cumulative HXR fluence + rolling correlation --
        if has_hard:
            out["hxr_cumulative_fluence"] = (hard * dt).cumsum()
            n_corr = max(2, int(round(cfg.shs_corr_window_sec / dt)))
            out["neupert_corr"] = (
                pd.Series(dSXR_dt, index=out.index)
                .rolling(window=n_corr, min_periods=max(2, n_corr // 2))
                .corr(hard)
            )
        else:
            out["hxr_cumulative_fluence"] = np.nan
            out["neupert_corr"] = np.nan

        # --- Soft-hard-soft hardness ratio ----------------------------------
        if has_hard:
            n_smooth = max(1, int(round(cfg.hardness_smooth_sec / dt)))
            soft_sm = soft.rolling(window=n_smooth, min_periods=1, center=True).mean()
            hard_sm = hard.rolling(window=n_smooth, min_periods=1, center=True).mean()
            with np.errstate(divide="ignore", invalid="ignore"):
                hardness = hard_sm / (soft_sm + 1e-12)
            out["hardness_ratio"] = hard / (soft + 1e-12)
            out["hardness_smoothed"] = hardness
        else:
            out["hardness_ratio"] = np.nan
            out["hardness_smoothed"] = np.nan

        # --- Flare phase classification -------------------------------------
        phase = self._classify_phases(soft, t_sec, dt)
        out["flare_phase"] = phase
        for p in PHASES:
            out[f"phase_{p}"] = (phase == p).astype(int)

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_phases(
        self, soft: pd.Series, t_sec: pd.Series, dt: float
    ) -> pd.Series:
        """Rule-based flare-phase classifier driven by Benz (2008) Fig. 2.

        Logic
        -----
        1. Estimate a fixed pre-event background level as the minimum of
           a lightly-smoothed version of the *first* background_window
           worth of samples. Unlike a sliding-window minimum, this does
           not drift upward once the flare has been elevated for longer
           than one window -- it anchors to the true pre-flare baseline
           for the whole series, which is what "amplitude above
           background" should be measured against throughout a single
           flare event (Sec. 1.3 treats preflare level as a fixed
           reference, not a moving one).
        2. If peak amplitude above this background is negligible, label
           the whole series "preflare" (quiescent / no flare present).
        3. Otherwise, smooth the derivative (same smoothing window as the
           background estimate) before thresholding, since single-sample
           background noise otherwise flips labels at random; normalise
           by the background level for a scale-free rate.
        4. Assign phases by amplitude and (smoothed, normalised) slope:
             - strongly rising                       -> impulsive
             - high amplitude (near/at peak), and not
               strongly rising                        -> flash
             - amplitude has fallen back toward
               background (regardless of small-scale
               wiggles in slope)                       -> decay
             - otherwise (low amplitude, weak slope)   -> preflare
           Decay is keyed off *amplitude having receded*, not off a
           negative-derivative threshold alone, because a real decay
           phase plateaus and wiggles near the local trend long before
           consistently producing a single dominant slope sign at the
           cadence of typical light curve noise.
        """
        cfg = self.config
        n = len(soft)
        n_bg = max(3, int(round(cfg.background_window_sec / dt)))
        n_smooth = max(3, min(n, n_bg // 5))

        soft_arr = soft.to_numpy()

        # Fixed pre-event background: minimum of a smoothed version of the
        # leading background_window of samples (or the whole series if
        # shorter). This stays anchored to the true quiescent level even
        # after the flare has been elevated longer than n_bg samples.
        smoothed = soft.rolling(window=n_smooth, min_periods=1, center=True).mean()
        lead = smoothed.iloc[: min(n_bg, n)]
        background_level = float(lead.min()) if len(lead) else float(soft_arr.min())

        amplitude_above_bg = np.clip(smoothed.to_numpy() - background_level, 0.0, None)
        peak_amp = float(amplitude_above_bg.max()) if n else 0.0

        phase = pd.Series(["preflare"] * n, index=soft.index, dtype=object)
        if peak_amp < cfg.min_event_amplitude:
            return phase

        # A single sample has no well-defined rise/fall to classify by;
        # fall back to amplitude-only (preflare vs flash) since np.gradient
        # requires >= 2 points.
        if n < 2:
            if peak_amp >= cfg.min_event_amplitude and amplitude_above_bg[0] / peak_amp >= cfg.flash_amplitude_frac:
                phase.iloc[0] = "flash"
            return phase

        # Smoothed derivative, normalised by background level (fractional
        # rate of change per second), to suppress single-sample noise
        # flips while staying scale-free across different flare sizes.
        deriv = np.gradient(smoothed.to_numpy(), t_sec.to_numpy())
        bg_safe = max(background_level, cfg.min_event_amplitude)
        norm_deriv = deriv / bg_safe

        amp_frac = amplitude_above_bg / peak_amp

        is_impulsive = norm_deriv > cfg.impulsive_deriv_threshold
        is_high_amp = amp_frac >= cfg.flash_amplitude_frac
        is_receded = amp_frac < cfg.flash_amplitude_frac  # fallen back toward bg

        labels = np.full(n, "preflare", dtype=object)
        # Order matters: later assignments override earlier ones where
        # masks overlap, encoding the priority impulsive > flash > decay.
        labels[is_receded & ~is_impulsive] = "decay"
        labels[is_high_amp & ~is_impulsive] = "flash"
        labels[is_impulsive] = "impulsive"

        # The decay rule above also catches samples *before* the flare
        # has risen at all (amp_frac starts at 0, which is < flash_frac).
        # Restrict "decay" to samples that occur after the series' peak
        # amplitude has first been reached, since decay is defined as
        # the post-peak relaxation phase (Sec. 1.3), not the pre-flare
        # quiescent state, which both happen to have low amplitude.
        peak_idx = int(np.argmax(amplitude_above_bg))
        pre_peak = np.arange(n) < peak_idx
        labels[pre_peak & (labels == "decay")] = "preflare"

        return pd.Series(labels, index=soft.index)

    def _validate(self, df: pd.DataFrame) -> None:
        cfg = self.config
        if cfg.time_col not in df.columns:
            raise KeyError(f"Missing required time column: '{cfg.time_col}'")
        if cfg.soft_col not in df.columns:
            raise KeyError(f"Missing required soft-channel column: '{cfg.soft_col}'")
        if cfg.hard_col is not None and cfg.hard_col not in df.columns:
            raise KeyError(
                f"hard_col='{cfg.hard_col}' was specified but not found in DataFrame"
            )
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
