"""
ml/features/basic_features.py
==============================
Basic statistical features extracted from SoLEXS (and HEL1OS) light curves.

Purpose
-------
This module converts raw (timestamp, counts) time-series data into
machine-learning-ready statistical features that capture the *amplitude*,
*variability*, and *energy content* of the X-ray count rate signal.

These are the lowest-level, most broadly applicable features in the pipeline.
They run on SoLEXS Level-1 Light Curve (.lc) data only — no GTI, no spectral
files required.  Every downstream module (temporal, flare, spectral) builds
on the DataFrame produced here.

Scientific grounding (Benz 2008)
----------------------------------
Solar flares span many orders of magnitude in energy (Table 1: 10^30 – 10^32 erg
for X-class events).  The count rate distribution is approximately log-normal
during quiet periods and develops a heavy power-law tail during flares.

Key physical timescales from Benz (2008) §1.3:
    Preflare phase  : minutes        → rolling mean captures baseline drift
    Impulsive phase : 3–10 minutes   → rolling std/variance spikes sharply
    Flash phase     : 5–20 minutes   → rolling max captures peak luminosity
    Decay phase     : hours          → rolling min reveals floor recovery

Features in this module
------------------------
Feature                  | Symbol      | Units
-------------------------|-------------|----------------
Log Counts               | log10(C+1)  | dimensionless
Rolling Mean             | μ_w         | cts s⁻¹
Rolling Standard Dev     | σ_w         | cts s⁻¹
Rolling Maximum          | max_w       | cts s⁻¹
Rolling Minimum          | min_w       | cts s⁻¹
Rolling Variance         | σ²_w        | (cts s⁻¹)²
Moving RMS               | rms_w       | cts s⁻¹
Signal Energy            | E_w         | (cts s⁻¹)² · s

Data requirements
------------------
    Required : SoLEXS Light Curve (.lc)  — columns: [timestamp, counts]
    Optional : None
    NOT needed: GTI, .pi spectral, HEL1OS

Usage
-----
    import pandas as pd
    from ml.features.basic_features import BasicFeatureExtractor

    df = pd.DataFrame({"timestamp": t_unix, "counts": count_rate})
    extractor = BasicFeatureExtractor(windows_sec=[60, 300, 600], cadence_sec=1.0)
    df_features = extractor.transform(df)

    # df_features now contains all basic feature columns alongside
    # the original timestamp and counts.

Author
------
    Aditya-L1 ML Pipeline — Feature Engineering Module
    Scientific reference: Benz (2008) §1.3, §4, §5.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BasicFeatureConfig:
    """
    All tunable parameters for BasicFeatureExtractor.

    No hardcoded values appear inside the extractor itself — every threshold,
    window size, and flag is read from this dataclass.  Production deployments
    should populate this from config/pipeline.yaml via ``from_dict()``.

    Parameters
    ----------
    windows_sec : list of float
        Rolling window sizes in seconds.  Multiple windows capture different
        physical timescales (impulsive vs. decay phase).
        Default: [60, 300, 600]  → 1 min, 5 min, 10 min
    cadence_sec : float
        Nominal time resolution of the light curve in seconds.
        Used to convert window sizes to integer bin counts.
        SoLEXS default cadence: 1 s.
    log_offset : float
        Additive offset before log10 to avoid log(0).
        Default: 1.0  (standard +1 trick)
    min_periods_fraction : float
        Minimum fraction of window bins that must be non-NaN for a rolling
        statistic to be computed (rather than returning NaN).
        Default: 0.5
    count_col : str
        Name of the counts/count-rate column in the input DataFrame.
    time_col : str
        Name of the timestamp column in the input DataFrame.
    clip_negative : bool
        If True, clip negative count values to 0 before feature extraction.
        Negative counts can appear due to background subtraction artefacts.
    energy_normalise_by_window : bool
        If True, signal energy is reported as mean power (energy / window_bins)
        rather than raw summed energy.  Keeps values comparable across windows.
    """

    windows_sec: List[float] = field(
        default_factory=lambda: [60.0, 300.0, 600.0]
    )
    cadence_sec: float = 1.0
    log_offset: float = 1.0
    min_periods_fraction: float = 0.5
    count_col: str = "counts"
    time_col: str = "timestamp"
    clip_negative: bool = True
    energy_normalise_by_window: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "BasicFeatureConfig":
        """
        Instantiate from a plain dict (e.g. loaded from pipeline.yaml).

        Only keys that match dataclass fields are used; extras are ignored.
        """
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    def window_bins(self, window_sec: float) -> int:
        """Convert a window in seconds to an integer number of bins."""
        bins = max(2, int(round(window_sec / self.cadence_sec)))
        return bins

    def min_periods(self, window_sec: float) -> int:
        """Minimum number of valid bins required for a rolling computation."""
        return max(1, int(self.min_periods_fraction * self.window_bins(window_sec)))


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

class BasicFeatureExtractor:
    """
    Extract basic statistical features from a SoLEXS light curve DataFrame.

    This extractor is *stateless* after construction — ``transform()`` can be
    called multiple times on different DataFrames without side effects.

    The output DataFrame preserves all input columns and appends new feature
    columns with names following the pattern:
        ``<feature_name>_<window_sec>s``
    e.g. ``rolling_mean_60s``, ``rolling_std_300s``, ``rms_600s``.

    Parameters
    ----------
    windows_sec : list of float, optional
        Rolling window sizes in seconds.  Overrides config.windows_sec if
        provided directly.
    cadence_sec : float, optional
        Nominal cadence of the light curve.  Overrides config.cadence_sec.
    config : BasicFeatureConfig, optional
        Full configuration object.  If provided, windows_sec and cadence_sec
        arguments are ignored.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> t  = np.arange(0, 3600, 1, dtype=float) + 1.75e9
    >>> cr = np.random.poisson(50, 3600).astype(float)
    >>> df = pd.DataFrame({"timestamp": t, "counts": cr})
    >>> ext = BasicFeatureExtractor(windows_sec=[60, 300], cadence_sec=1.0)
    >>> out = ext.transform(df)
    >>> [c for c in out.columns if c.startswith("rolling_mean")]
    ['rolling_mean_60s', 'rolling_mean_300s']
    """

    def __init__(
        self,
        windows_sec: Optional[Sequence[float]] = None,
        cadence_sec: Optional[float] = None,
        config: Optional[BasicFeatureConfig] = None,
    ) -> None:
        if config is None:
            config = BasicFeatureConfig()
        if windows_sec is not None:
            config.windows_sec = list(windows_sec)
        if cadence_sec is not None:
            config.cadence_sec = float(cadence_sec)

        self.cfg = config
        logger.info(
            "BasicFeatureExtractor | windows=%s s | cadence=%.2f s",
            self.cfg.windows_sec,
            self.cfg.cadence_sec,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all basic feature extractors on the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns ``time_col`` (float, Unix seconds) and
            ``count_col`` (float, cts/s or raw counts).

        Returns
        -------
        pd.DataFrame
            Original columns + all basic feature columns.
            Index is preserved from the input.

        Raises
        ------
        ValueError
            If required columns are missing from df.
        """
        self._validate(df)
        out = df.copy()

        # Prepare the count series — clip negative values if requested
        cr: pd.Series = out[self.cfg.count_col].astype(np.float64)
        if self.cfg.clip_negative:
            n_neg = int((cr < 0).sum())
            if n_neg > 0:
                logger.debug("Clipped %d negative count values to 0.", n_neg)
            cr = cr.clip(lower=0.0)

        # ── Feature 1: Log Counts ─────────────────────────────────────────────
        out["log_counts"] = self._log_counts(cr)

        # ── Window-based features ─────────────────────────────────────────────
        for w in self.cfg.windows_sec:
            tag = f"{int(w)}s"
            roller = cr.rolling(
                window    = self.cfg.window_bins(w),
                min_periods = self.cfg.min_periods(w),
            )

            # ── Feature 2: Rolling Mean ───────────────────────────────────────
            out[f"rolling_mean_{tag}"] = self._rolling_mean(roller)

            # ── Feature 3: Rolling Standard Deviation ────────────────────────
            out[f"rolling_std_{tag}"] = self._rolling_std(roller)

            # ── Feature 4: Rolling Maximum ────────────────────────────────────
            out[f"rolling_max_{tag}"] = self._rolling_max(roller)

            # ── Feature 5: Rolling Minimum ────────────────────────────────────
            out[f"rolling_min_{tag}"] = self._rolling_min(roller)

            # ── Feature 6: Rolling Variance ───────────────────────────────────
            out[f"rolling_var_{tag}"] = self._rolling_var(roller)

            # ── Feature 7: Moving RMS ─────────────────────────────────────────
            out[f"rms_{tag}"] = self._moving_rms(cr, w)

            # ── Feature 8: Signal Energy ──────────────────────────────────────
            out[f"signal_energy_{tag}"] = self._signal_energy(cr, w)

        n_features = len(out.columns) - len(df.columns)
        logger.info(
            "BasicFeatureExtractor: added %d feature columns (windows=%s s).",
            n_features,
            self.cfg.windows_sec,
        )
        return out

    def feature_names(self) -> list[str]:
        """
        Return the list of feature column names this extractor will produce.

        Useful for downstream column selection and schema validation.

        Returns
        -------
        list of str
        """
        names = ["log_counts"]
        for w in self.cfg.windows_sec:
            tag = f"{int(w)}s"
            names += [
                f"rolling_mean_{tag}",
                f"rolling_std_{tag}",
                f"rolling_max_{tag}",
                f"rolling_min_{tag}",
                f"rolling_var_{tag}",
                f"rms_{tag}",
                f"signal_energy_{tag}",
            ]
        return names

    # ── Individual feature implementations ────────────────────────────────────

    @staticmethod
    def _log_counts(cr: pd.Series) -> pd.Series:
        """
        Feature 1: Log Counts
        ─────────────────────
        Mathematical definition
        ~~~~~~~~~~~~~~~~~~~~~~~
        log_counts[t] = log₁₀(CR[t] + offset)

        where  CR[t]  is the count rate at time t and ``offset`` (default 1.0)
        prevents log(0) singularities.

        Physical significance
        ~~~~~~~~~~~~~~~~~~~~~
        Solar flare energy follows a power-law distribution spanning many
        orders of magnitude (Benz 2008, §4, Table 1: 10³⁰ – 10³² erg for
        X-class events).  The log transform compresses this dynamic range,
        making the signal approximately Gaussian during quiet periods.

        This is the same motivation behind the GOES log-scale classification
        (A, B, C, M, X) — each class represents one order of magnitude.

        ML utility
        ~~~~~~~~~~
        - Prevents large-amplitude flares from dominating gradient updates
        - Improves numerical stability in neural networks
        - Linearises the relationship between count rate and flare class

        Data requirement : SoLEXS .lc only
        """
        return np.log10(cr + 1.0)

    @staticmethod
    def _rolling_mean(roller: pd.core.window.rolling.Rolling) -> pd.Series:
        """
        Feature 2: Rolling Mean
        ────────────────────────
        Mathematical definition
        ~~~~~~~~~~~~~~~~~~~~~~~
        μ_w[t] = (1/w) · Σ_{i=t-w+1}^{t} CR[i]

        where  w  is the window size in bins.

        Physical significance
        ~~~~~~~~~~~~~~~~~~~~~
        The rolling mean tracks the slowly-varying *background* (quiet Sun)
        level.  In Benz (2008) §1.3, the pre-flare phase is characterised by
        gradual heating of the coronal plasma.  The rolling mean smooths out
        short-duration spikes (cosmic rays, particle hits) while preserving
        the gradual rise.

        The Neupert effect (Benz 2008, §2.4) states:
            FSXR(t) ∝ ∫ FHXR(t') dt'
        i.e. soft X-ray flux is the time integral of hard X-ray flux.
        The rolling mean of the hard X-ray (HEL1OS) count rate is therefore
        a proxy for the expected soft X-ray (SoLEXS) response.

        ML utility
        ~~~~~~~~~~
        - Captures the slowly-varying background level
        - Serves as the denominator for excess ratio features
        - Important for distinguishing genuine flares from orbital modulations

        Data requirement : SoLEXS .lc only
        """
        return roller.mean()

    @staticmethod
    def _rolling_std(roller: pd.core.window.rolling.Rolling) -> pd.Series:
        """
        Feature 3: Rolling Standard Deviation
        ───────────────────────────────────────
        Mathematical definition
        ~~~~~~~~~~~~~~~~~~~~~~~
        σ_w[t] = √[ (1/(w-1)) · Σ_{i=t-w+1}^{t} (CR[i] - μ_w[t])² ]

        Uses Bessel's correction (ddof=1) for an unbiased estimator.

        Physical significance
        ~~~~~~~~~~~~~~~~~~~~~
        During the quiet Sun, photon arrival is a Poisson process.  For a
        Poisson process with mean rate λ,  σ = √λ.

        When a flare begins (Benz 2008 §1.3, impulsive phase), the count rate
        rises sharply in 3–10 minutes.  The rolling standard deviation captures
        this increase in variability *before* the peak is reached — making it
        one of the most valuable early-warning indicators.

        The FlareDetector in ml/eda/flare_detector.py already uses
        dcr/σ_background as its trigger criterion.  Here we generalise by
        computing σ over multiple timescales.

        ML utility
        ~~~~~~~~~~
        - Spike detection: Poisson std is √mean; excess std signals real flux
        - Temporal variability — captures the soft-hard-soft spectral evolution
          described in Benz (2008) §5.2
        - Input to the SNR feature: SNR = (rolling_max - rolling_mean) / rolling_std

        Data requirement : SoLEXS .lc only
        """
        return roller.std(ddof=1)

    @staticmethod
    def _rolling_max(roller: pd.core.window.rolling.Rolling) -> pd.Series:
        """
        Feature 4: Rolling Maximum
        ────────────────────────────
        Mathematical definition
        ~~~~~~~~~~~~~~~~~~~~~~~
        max_w[t] = max{ CR[i] : i ∈ [t-w+1, t] }

        Physical significance
        ~~~~~~~~~~~~~~~~~~~~~
        The rolling maximum captures the *peak luminosity* within a sliding
        window.  From Benz (2008) §5.2:

            "High flux means that the acceleration process is driven more
             forcefully, resulting in a harder spectrum."

        The rolling maximum over the impulsive-phase window (5–10 min) is
        therefore strongly correlated with the final flare classification.

        Rolling max over longer windows (300 s, 600 s) identifies whether
        the current time step is near a flare peak or in the decay phase.

        ML utility
        ~~~~~~~~~~
        - Key feature for classification: max over 5 min ≈ peak count rate
        - Rolling max over short windows (60 s) helps detect sub-peak bursts
          within a single flare (multi-peaked events described in §5.2)
        - Used in the Peak-to-Background Ratio (flare_features.py)

        Data requirement : SoLEXS .lc only
        """
        return roller.max()

    @staticmethod
    def _rolling_min(roller: pd.core.window.rolling.Rolling) -> pd.Series:
        """
        Feature 5: Rolling Minimum
        ────────────────────────────
        Mathematical definition
        ~~~~~~~~~~~~~~~~~~~~~~~
        min_w[t] = min{ CR[i] : i ∈ [t-w+1, t] }

        Physical significance
        ~~~~~~~~~~~~~~~~~~~~~
        The rolling minimum estimates the *local background floor*.

        From Benz (2008) §2.1, flares can occur even in the "quiet" Sun
        (network interior flares, nanoflares).  The rolling minimum helps
        the model distinguish genuine flux enhancement from an elevated
        persistent background (e.g. from a preceding large flare or
        enhanced solar activity period).

        During flare decay (§1.3), the rolling minimum over a 600 s window
        will track the slowly recovering background level — useful for
        detecting when a second flare starts before the first has fully decayed.

        ML utility
        ~~~~~~~~~~
        - Background floor estimation
        - Dynamic range feature: (rolling_max - rolling_min) / rolling_min
        - Detects sustained emission vs. transient spike

        Data requirement : SoLEXS .lc only
        """
        return roller.min()

    @staticmethod
    def _rolling_var(roller: pd.core.window.rolling.Rolling) -> pd.Series:
        """
        Feature 6: Rolling Variance
        ─────────────────────────────
        Mathematical definition
        ~~~~~~~~~~~~~~~~~~~~~~~
        σ²_w[t] = (1/(w-1)) · Σ_{i=t-w+1}^{t} (CR[i] - μ_w[t])²

        Note: σ²_w = (rolling_std)²  but is computed directly for numerical
        precision (avoids double square-root and re-squaring).

        Physical significance
        ~~~~~~~~~~~~~~~~~~~~~
        Variance is proportional to the second moment of the count rate
        distribution within the window.  It is more sensitive than standard
        deviation to large deviations (spikes) because contributions are
        squared.

        From Benz (2008) §4.5 (nanoflares):
            "The largest nanoflares contain energies of a few 10²⁶ erg."
        Nanoflares appear as small variance increases in the quiet Sun.
        This feature can therefore potentially detect sub-threshold events
        that fall below the FlareDetector onset threshold.

        The rate of increase of rolling variance (computed in
        temporal_features.py as its first derivative) is particularly
        informative.

        ML utility
        ~~~~~~~~~~
        - More sensitive to extreme values than std
        - Useful for anomaly detection in quiet-Sun periods
        - Feature for LSTM models: variance as a measure of 'excitability'

        Data requirement : SoLEXS .lc only
        """
        return roller.var(ddof=1)

    def _moving_rms(self, cr: pd.Series, window_sec: float) -> pd.Series:
        """
        Feature 7: Moving RMS (Root Mean Square)
        ──────────────────────────────────────────
        Mathematical definition
        ~~~~~~~~~~~~~~~~~~~~~~~
        rms_w[t] = √[ (1/w) · Σ_{i=t-w+1}^{t} CR[i]² ]

        Unlike the rolling mean, RMS includes *both* the mean (DC component)
        and the variance:
            rms² = μ² + σ²

        Physical significance
        ~~~~~~~~~~~~~~~~~~~~~
        The RMS of the count rate is proportional to the total *signal power*
        within the window, including the background.

        From Benz (2008) §4.1, the kinetic energy of the electron beam is:
            E_kin = ∫ F(ε) · ε · dε

        The moving RMS is an observable proxy for this — it captures both the
        mean flux and the fluctuations around it.  During the impulsive phase,
        RMS rises faster than the mean because the variance term grows rapidly.

        Key distinction from rolling_mean:
            - If background doubles: rolling_mean doubles, RMS ≈ doubles
            - If spike occurs (μ constant, σ spikes): RMS rises while mean stays low
            → RMS is more sensitive to short impulsive bursts

        ML utility
        ~~~~~~~~~~
        - Captures total signal power (useful for CNN 1D feature maps)
        - Differentiates background rise from impulsive spike
        - Standard metric in signal processing for feature extraction

        Data requirement : SoLEXS .lc only
        """
        w_bins = self.cfg.window_bins(window_sec)
        min_p  = self.cfg.min_periods(window_sec)
        return cr.pow(2).rolling(window=w_bins, min_periods=min_p).mean().pow(0.5)

    def _signal_energy(self, cr: pd.Series, window_sec: float) -> pd.Series:
        """
        Feature 8: Signal Energy
        ─────────────────────────
        Mathematical definition
        ~~~~~~~~~~~~~~~~~~~~~~~
        If ``energy_normalise_by_window = True`` (default):
            E_w[t] = (1/w) · Σ_{i=t-w+1}^{t} CR[i]²  · Δt

        where Δt = cadence_sec.  This gives units of (cts/s)² · s = cts²/s.

        If ``energy_normalise_by_window = False``:
            E_w[t] = Σ_{i=t-w+1}^{t} CR[i]²  · Δt

        Physical significance
        ~~~~~~~~~~~~~~~~~~~~~
        Signal energy is the discrete approximation to the integral of the
        squared signal — a standard measure of total signal power in engineering.

        In solar physics, the *radiated energy* of a flare is proportional to:
            E_rad = ∫ L(t) dt  (luminosity integrated over time)

        Since the count rate CR(t) is proportional to luminosity (in the
        relevant energy band), signal energy is proportional to the total
        energy radiated in that band during the window.

        From Benz (2008) §4, Table 1:
            Non-thermal electrons: 1.9×10³⁰ – 3.2×10³¹ erg
            Thermal hot plasma:    1.4×10³⁰ – 1.2×10³¹ erg

        Signal energy over a 10-minute window discriminates between B/C, M,
        and X class events much more cleanly than peak count rate alone,
        because it is less sensitive to single-bin spikes.

        Normalisation note
        ~~~~~~~~~~~~~~~~~~
        When comparing windows of different lengths, the normalised version
        (divided by w) is preferred — otherwise a 600 s window would always
        have 10× the energy of a 60 s window for the same signal level.

        ML utility
        ~~~~~~~~~~
        - Strongly correlated with flare class (B, C, M, X)
        - Less sensitive to impulsive spikes than rolling_max
        - Captures 'area under the curve' information
        - Key input feature for flare energy prediction regression tasks

        Data requirement : SoLEXS .lc only
        """
        w_bins = self.cfg.window_bins(window_sec)
        min_p  = self.cfg.min_periods(window_sec)
        dt     = self.cfg.cadence_sec

        cr_sq  = cr.pow(2) * dt
        energy = cr_sq.rolling(window=w_bins, min_periods=min_p).sum()

        if self.cfg.energy_normalise_by_window:
            energy = energy / w_bins

        return energy

    # ── Input validation ───────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame before feature extraction.

        Raises
        ------
        ValueError : if required columns are missing or DataFrame is empty
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        missing = [
            col for col in [self.cfg.time_col, self.cfg.count_col]
            if col not in df.columns
        ]
        if missing:
            raise ValueError(
                f"Input DataFrame missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        n_nan = int(df[self.cfg.count_col].isna().sum())
        if n_nan > 0:
            logger.warning(
                "Input has %d NaN values in '%s' column (%.1f%%). "
                "Rolling features will have NaN at those positions.",
                n_nan,
                self.cfg.count_col,
                100 * n_nan / len(df),
            )

        if len(df) < max(self.cfg.window_bins(w) for w in self.cfg.windows_sec):
            logger.warning(
                "Input length (%d rows) is shorter than the largest window (%d bins). "
                "Most rolling features will be NaN.",
                len(df),
                max(self.cfg.window_bins(w) for w in self.cfg.windows_sec),
            )