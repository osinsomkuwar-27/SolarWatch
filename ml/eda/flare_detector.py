"""
ml/eda/flare_detector.py
=========================
Rule-based flare detector for EDA and auto-labelling.

Purpose
-------
Before training any model, we need to *find* flares in the raw light curves
to (a) understand the data, and (b) generate class labels for the ML pipeline.

This module implements the **derivative-threshold** method described in
Benz (2008), §1.1 and §5.2:
  "A flare is defined observationally as a brightening of any emission
   across the electromagnetic spectrum occurring at a time scale of minutes."

Algorithm
---------
1. Smooth the light curve with a rolling median to remove cosmic-ray spikes.
2. Compute the normalised derivative  dCR/dt / σ_background.
3. Flag onset when the derivative exceeds ``onset_sigma`` for at least
   ``min_rise_bins`` consecutive bins.
4. Flag peak as the maximum within ``peak_window_sec`` of the onset.
5. Flag end when the count rate drops below  ``decay_fraction`` × peak value.
6. Assign GOES-proxy class (Quiet / B-C / M / X) from the peak count rate
   percentile relative to the full-day distribution.

This is the same approach recommended by the mentor: "Plot the soft X-ray
and hard X-ray light curves. Study their behaviour and relate the observations
to the flare physics described in Benz (2008)."

Output dataclass
-----------------
Each detected flare is a ``FlareEvent`` dataclass.

Usage
-----
    from ml.eda.flare_detector import FlareDetector
    from ml.loaders.solexs_loader import LightCurve

    detector = FlareDetector(onset_sigma=3.0, peak_window_sec=300)
    flares   = detector.detect(lc)
    summary  = detector.summary(flares)
    print(summary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

# ── GOES-proxy class labels ───────────────────────────────────────────────────
# We assign a label based on peak count rate percentile.
# The exact thresholds are derived from EDA on the first downloaded day, then
# stored in config/pipeline.yaml (labelling.* keys).
FLARE_CLASS_QUIET = 0   # background / quiet Sun
FLARE_CLASS_BC    = 1   # B/C-class equivalent
FLARE_CLASS_M     = 2   # M-class equivalent
FLARE_CLASS_X     = 3   # X-class equivalent

FLARE_CLASS_NAMES = {0: "Quiet", 1: "B/C", 2: "M", 3: "X"}


@dataclass
class FlareEvent:
    """
    A single detected flare event.

    Attributes
    ----------
    onset_unix : float
        Unix timestamp of the flare onset (start of rapid rise).
    peak_unix : float
        Unix timestamp of the peak count rate.
    end_unix : float
        Unix timestamp when count rate returns to near-background.
    peak_count_rate : float
        Peak count rate in cts/s.
    background_rate : float
        Median background count rate before onset.
    rise_time_sec : float
        onset → peak duration in seconds.
    decay_time_sec : float
        peak → end duration in seconds.
    flare_class : int
        0=Quiet, 1=B/C, 2=M, 3=X  (GOES-proxy).
    detector : str
        SDD1, SDD2, CdTe1, CZT1, etc.
    date_str : str
        YYYYMMDD.
    """
    onset_unix:       float
    peak_unix:        float
    end_unix:         float
    peak_count_rate:  float
    background_rate:  float
    rise_time_sec:    float
    decay_time_sec:   float
    flare_class:      int
    detector:         str
    date_str:         str

    @property
    def duration_sec(self) -> float:
        return self.end_unix - self.onset_unix

    @property
    def flux_ratio(self) -> float:
        """Peak / background ratio (impulsiveness indicator)."""
        if self.background_rate <= 0:
            return float("nan")
        return self.peak_count_rate / self.background_rate

    @property
    def class_name(self) -> str:
        return FLARE_CLASS_NAMES.get(self.flare_class, "?")

    @property
    def onset_utc(self) -> str:
        return datetime.fromtimestamp(self.onset_unix, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    @property
    def peak_utc(self) -> str:
        return datetime.fromtimestamp(self.peak_unix, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    def to_dict(self) -> dict:
        return {
            "onset_utc":        self.onset_utc,
            "peak_utc":         self.peak_utc,
            "peak_count_rate":  self.peak_count_rate,
            "background_rate":  self.background_rate,
            "rise_time_sec":    self.rise_time_sec,
            "decay_time_sec":   self.decay_time_sec,
            "duration_sec":     self.duration_sec,
            "flux_ratio":       self.flux_ratio,
            "flare_class":      self.flare_class,
            "class_name":       self.class_name,
            "detector":         self.detector,
            "date_str":         self.date_str,
        }


class FlareDetector:
    """
    Rule-based flare detector for Aditya-L1 light curves.

    Works on any instrument that provides (time_unix, count_rate) arrays —
    SoLEXS LightCurve or HEL1OS HEL1OSBand.

    Parameters
    ----------
    onset_sigma : float
        How many σ above background the derivative must rise to trigger onset.
        Default 3.0 (3-sigma, conservative).
    min_rise_bins : int
        Minimum consecutive bins above threshold to confirm an onset.
        Prevents triggering on single cosmic-ray spikes.
    peak_window_sec : float
        Look forward this many seconds from onset to find the peak.
    decay_fraction : float
        Flare ends when count rate drops to this fraction of peak.
        Default 0.5 (half-maximum).
    smooth_window_sec : float
        Width of the rolling-median smoother (seconds).
    bc_percentile : float
        Percentile above which a flare is labelled B/C (not quiet).
    m_percentile : float
        Percentile above which a flare is labelled M (not B/C).
    x_percentile : float
        Percentile above which a flare is labelled X (not M).
    min_refractory_sec : float
        Minimum gap between two consecutive flares to avoid re-triggering
        on the same event.
    """

    def __init__(
        self,
        onset_sigma:         float = 3.0,
        min_rise_bins:       int   = 3,
        peak_window_sec:     float = 300.0,
        decay_fraction:      float = 0.5,
        smooth_window_sec:   float = 60.0,
        bc_percentile:       float = 70.0,
        m_percentile:        float = 85.0,
        x_percentile:        float = 95.0,
        min_refractory_sec:  float = 600.0,
    ) -> None:
        self.onset_sigma        = onset_sigma
        self.min_rise_bins      = min_rise_bins
        self.peak_window_sec    = peak_window_sec
        self.decay_fraction     = decay_fraction
        self.smooth_window_sec  = smooth_window_sec
        self.bc_percentile      = bc_percentile
        self.m_percentile       = m_percentile
        self.x_percentile       = x_percentile
        self.min_refractory_sec = min_refractory_sec

        logger.info(
            "FlareDetector | σ=%.1f | peak_window=%.0fs | decay=%.2f",
            onset_sigma, peak_window_sec, decay_fraction,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        time_unix:  npt.NDArray[np.float64],
        count_rate: npt.NDArray[np.float64],
        detector:   str = "SDD2",
        date_str:   str = "",
    ) -> list[FlareEvent]:
        """
        Detect flares in a light curve.

        Parameters
        ----------
        time_unix : ndarray of float64
            Unix timestamps (already GTI-filtered recommended).
        count_rate : ndarray of float64
            Count rate in cts/s, same length as time_unix.
        detector : str
        date_str : str

        Returns
        -------
        list of FlareEvent
            Sorted by onset time.
        """
        if len(time_unix) < 10:
            logger.warning("Light curve too short (%d points); skipping.", len(time_unix))
            return []

        # 1. Smooth
        cr_smooth = self._rolling_median(count_rate, time_unix)

        # 2. Background stats — use the lower 50th percentile (quiet Sun)
        background = float(np.nanpercentile(cr_smooth, 50))
        sigma_bg   = float(np.nanstd(
            cr_smooth[cr_smooth < np.nanpercentile(cr_smooth, 75)]
        ))
        if sigma_bg < 1e-6:
            sigma_bg = max(background * 0.05, 1.0)
            logger.debug("σ_background near-zero; using 5%% of background")

        # 3. Derivative
        dt = np.diff(time_unix)
        dt = np.where(dt > 0, dt, 1.0)   # guard divide-by-zero
        dcr = np.diff(cr_smooth) / dt    # cts/s²
        dcr_full = np.append(dcr, 0.0)  # pad to same length

        # Normalise derivative by background σ
        deriv_norm = dcr_full / sigma_bg

        # 4. Flare class thresholds — calibrated from full-day distribution
        thresh_bc = float(np.nanpercentile(cr_smooth, self.bc_percentile))
        thresh_m  = float(np.nanpercentile(cr_smooth, self.m_percentile))
        thresh_x  = float(np.nanpercentile(cr_smooth, self.x_percentile))

        # 5. Find onsets
        flares: list[FlareEvent] = []
        last_end_unix = -np.inf

        i = 0
        while i < len(time_unix) - self.min_rise_bins:
            # Check refractory period
            if time_unix[i] < last_end_unix + self.min_refractory_sec:
                i += 1
                continue

            # Trigger: min_rise_bins consecutive bins above threshold
            window = deriv_norm[i: i + self.min_rise_bins]
            if np.all(window >= self.onset_sigma):
                onset_idx  = i
                onset_unix = float(time_unix[onset_idx])

                # Find peak within peak_window_sec
                peak_end_unix = onset_unix + self.peak_window_sec
                peak_mask     = (time_unix >= onset_unix) & (time_unix <= peak_end_unix)
                if not peak_mask.any():
                    i += 1
                    continue

                peak_idx       = int(np.argmax(cr_smooth[peak_mask]))
                # Map local index back to global index
                global_indices = np.where(peak_mask)[0]
                peak_global    = global_indices[peak_idx]
                peak_unix      = float(time_unix[peak_global])
                peak_cr        = float(cr_smooth[peak_global])

                # Find end: count rate drops to decay_fraction × peak
                decay_level = background + self.decay_fraction * (peak_cr - background)
                end_global  = peak_global
                for j in range(peak_global + 1, len(time_unix)):
                    if cr_smooth[j] <= decay_level:
                        end_global = j
                        break
                end_unix = float(time_unix[end_global])

                # Assign GOES-proxy class
                if peak_cr >= thresh_x:
                    flare_class = FLARE_CLASS_X
                elif peak_cr >= thresh_m:
                    flare_class = FLARE_CLASS_M
                elif peak_cr >= thresh_bc:
                    flare_class = FLARE_CLASS_BC
                else:
                    flare_class = FLARE_CLASS_QUIET

                flare = FlareEvent(
                    onset_unix      = onset_unix,
                    peak_unix       = peak_unix,
                    end_unix        = end_unix,
                    peak_count_rate = peak_cr,
                    background_rate = background,
                    rise_time_sec   = peak_unix - onset_unix,
                    decay_time_sec  = end_unix  - peak_unix,
                    flare_class     = flare_class,
                    detector        = detector,
                    date_str        = date_str,
                )
                flares.append(flare)
                logger.info(
                    "Flare detected: class=%s onset=%s peak=%.1f cts/s",
                    flare.class_name, flare.onset_utc, peak_cr,
                )

                last_end_unix = end_unix
                i = end_global + 1
            else:
                i += 1

        logger.info(
            "Detection complete: %d flare(s) found in %s %s",
            len(flares), detector, date_str,
        )
        return flares

    def detect_from_lc(self, lc: "LightCurve") -> list[FlareEvent]:  # noqa: F821
        """Convenience wrapper for a SoLEXS LightCurve object."""
        return self.detect(
            time_unix  = lc.time_unix,
            count_rate = lc.count_rate,
            detector   = lc.detector,
            date_str   = lc.date_str,
        )

    def detect_from_helios_band(
        self,
        band: "HEL1OSBand",  # noqa: F821
        detector: str,
        date_str: str = "",
    ) -> list[FlareEvent]:
        """Convenience wrapper for a HEL1OS band."""
        return self.detect(
            time_unix  = band.time_unix,
            count_rate = band.count_rate,
            detector   = detector,
            date_str   = date_str,
        )

    @staticmethod
    def summary(flares: list[FlareEvent]) -> pd.DataFrame:
        """
        Return a tidy DataFrame summary of detected flares.

        Parameters
        ----------
        flares : list of FlareEvent

        Returns
        -------
        pd.DataFrame
        """
        if not flares:
            return pd.DataFrame()
        rows = [f.to_dict() for f in flares]
        df = pd.DataFrame(rows)
        df = df.sort_values("onset_utc").reset_index(drop=True)
        return df

    def label_timeseries(
        self,
        time_unix:  npt.NDArray[np.float64],
        count_rate: npt.NDArray[np.float64],
        flares:     list[FlareEvent],
    ) -> npt.NDArray[np.int32]:
        """
        Create a per-bin class label array aligned to time_unix.

        Each bin gets the class of the flare it belongs to (onset→end),
        or FLARE_CLASS_QUIET if no flare is active.

        This output is consumed by the Dataset Builder (Module 4).

        Parameters
        ----------
        time_unix : ndarray
        count_rate : ndarray (unused here, kept for API symmetry)
        flares : list of FlareEvent

        Returns
        -------
        ndarray of int32, same length as time_unix
        """
        labels = np.zeros(len(time_unix), dtype=np.int32)
        for flare in flares:
            mask = (time_unix >= flare.onset_unix) & (time_unix <= flare.end_unix)
            labels[mask] = flare.flare_class
        return labels

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rolling_median(
        self,
        count_rate: npt.NDArray[np.float64],
        time_unix:  npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Apply a time-aware rolling median smoother.

        Converts to a pandas Series (index=time) and uses the median smoother
        with a window size derived from smooth_window_sec and the median cadence.
        """
        cadence = float(np.nanmedian(np.diff(time_unix)))
        if cadence <= 0:
            cadence = 1.0
        window_bins = max(3, int(self.smooth_window_sec / cadence))

        series   = pd.Series(count_rate)
        smoothed = series.rolling(window_bins, center=True, min_periods=1).median()
        return smoothed.to_numpy(dtype=np.float64)
