"""
ml/eda/helios_eda/helios_statistics.py
========================================
Statistical summaries for HEL1OS (hard X-ray, 8–150 keV) EDA.

What it computes
----------------
1. Basic descriptive statistics (mean, std, min, max, percentiles) for every
   active detector band (CdTe1, CdTe2, CZT1, CZT2).
2. Count-rate histogram per detector.
3. Flare class distribution (from detected FlareEvents; reuses the shared
   FlareDetector — no HEL1OS-specific detector required).
4. Class-imbalance report for each detector independently.
5. Inter-arrival time, rise-time, and decay-time distributions.
6. Auto-correlation function (ACF) — reveals the shorter HXR timescales
   characteristic of the impulsive phase (Benz 2008 §2.2).
7. Cross-detector comparison: side-by-side statistics for all four
   detectors to expose CdTe vs. CZT calibration offsets.
8. Full output cached as JSON (one file per detector, same layout as the
   SoLEXS EDAReport) so downstream modules can load either format.

Physics motivation
------------------
HEL1OS detects hard X-rays (8–150 keV) produced during the impulsive
phase by non-thermal bremsstrahlung of electron beams (Benz 2008 §2.2).
The ACF timescales are expected to be shorter (~tens of seconds) than
the SoLEXS soft X-ray ACF (~minutes) because HXR emission tracks the
electron acceleration, not the slower thermal response.

The four detectors operate in overlapping energy ranges:
  CdTe1 / CdTe2 : 20–150 keV (cadmium telluride, radiation-hard)
  CZT1  / CZT2  : 8–60 keV   (cadmium zinc telluride, better energy res.)
Comparing CdTe1 vs CdTe2 and CZT1 vs CZT2 gives an in-flight
cross-calibration diagnostic; large offsets indicate gain drift.

Usage
-----
    from ml.eda.helios_eda.helios_statistics import HEL1OSEDAStatistics
    from ml.eda.flare_detector import FlareDetector

    stats = HEL1OSEDAStatistics(output_dir=cfg.paths.cache)
    report = stats.compute(
        time_unix  = band.time_unix,
        count_rate = band.count_rate,
        flares     = flares,
        detector   = "CdTe1",
        date_str   = "20260621",
    )
    stats.save(report, "20260621")
    stats.plot_distributions(band.time_unix, band.count_rate, flares,
                             tag="20260621", detector="CdTe1")
    # Cross-detector comparison (pass multiple reports)
    stats.plot_detector_comparison([report_cdte1, report_cdte2, report_czt1, report_czt2],
                                   tag="20260621")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

_STYLE = {
    "figure.dpi":        150,
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

# Detector display metadata ─────────────────────────────────────────────────
_DETECTOR_META = {
    "CdTe1": {"color": "#E84040", "energy": "20–150 keV"},
    "CdTe2": {"color": "#C03030", "energy": "20–150 keV"},
    "CZT1":  {"color": "#3A7FD5", "energy": "8–60 keV"},
    "CZT2":  {"color": "#2A5FA5", "energy": "8–60 keV"},
}
_ALL_DETECTORS = list(_DETECTOR_META.keys())


@dataclass
class HEL1OSEDAReport:
    """
    Full EDA statistical summary for one HEL1OS detector on one day.

    All numeric fields are plain Python floats/ints so the object is
    JSON-serialisable without a custom encoder.  Layout deliberately
    mirrors EDAReport (SoLEXS) so that shared tooling can handle both.
    """
    # ── Identity ──────────────────────────────────────────────
    detector:  str
    date_str:  str
    n_samples: int

    # ── Basic stats ───────────────────────────────────────────
    mean_cr:   float
    std_cr:    float
    min_cr:    float
    max_cr:    float
    p25_cr:    float
    p50_cr:    float
    p75_cr:    float
    p90_cr:    float
    p95_cr:    float
    p99_cr:    float

    # ── Flare population ──────────────────────────────────────
    n_flares:  int
    n_quiet:   int
    n_bc:      int
    n_m:       int
    n_x:       int

    # ── Imbalance ─────────────────────────────────────────────
    class_weights: dict = field(default_factory=dict)

    # ── Timing stats ──────────────────────────────────────────
    mean_rise_sec:           float = float("nan")
    mean_decay_sec:          float = float("nan")
    mean_duration_sec:       float = float("nan")
    median_interarrival_sec: float = float("nan")

    # ── Cadence ───────────────────────────────────────────────
    median_cadence_sec: float = 1.0

    # ── Derived thresholds ────────────────────────────────────
    thresh_bc: float = float("nan")
    thresh_m:  float = float("nan")
    thresh_x:  float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


class HEL1OSEDAStatistics:
    """
    Computes and saves EDA statistics for HEL1OS light curves.

    Parameters
    ----------
    output_dir : Path
        Directory where reports and plots are written.
    bc_percentile : float
        Percentile defining the B/C-class threshold.
    m_percentile : float
        Percentile defining the M-class threshold.
    x_percentile : float
        Percentile defining the X-class threshold.
    """

    def __init__(
        self,
        output_dir:    Path | str,
        bc_percentile: float = 70.0,
        m_percentile:  float = 85.0,
        x_percentile:  float = 95.0,
    ) -> None:
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self.bc_percentile = bc_percentile
        self.m_percentile  = m_percentile
        self.x_percentile  = x_percentile

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(
        self,
        time_unix:  npt.NDArray[np.float64],
        count_rate: npt.NDArray[np.float64],
        flares:     list,
        detector:   str,
        date_str:   str = "",
    ) -> HEL1OSEDAReport:
        """
        Compute the full EDA report for one HEL1OS detector.

        Parameters
        ----------
        time_unix : ndarray
        count_rate : ndarray
        flares : list of FlareEvent
        detector : str  — one of CdTe1, CdTe2, CZT1, CZT2
        date_str : str  — YYYYMMDD

        Returns
        -------
        HEL1OSEDAReport
        """
        cr = count_rate[np.isfinite(count_rate)]

        # ── Basic stats ───────────────────────────────────────
        mean_cr = float(np.nanmean(cr))
        std_cr  = float(np.nanstd(cr))
        min_cr  = float(np.nanmin(cr))
        max_cr  = float(np.nanmax(cr))
        percs   = np.nanpercentile(cr, [25, 50, 75, 90, 95, 99]).tolist()

        thresh_bc = float(np.nanpercentile(cr, self.bc_percentile))
        thresh_m  = float(np.nanpercentile(cr, self.m_percentile))
        thresh_x  = float(np.nanpercentile(cr, self.x_percentile))

        cadence = float(np.nanmedian(np.diff(time_unix))) if len(time_unix) > 1 else 1.0

        # ── Flare class distribution ──────────────────────────
        from ml.eda.flare_detector import (
            FLARE_CLASS_BC, FLARE_CLASS_M,
            FLARE_CLASS_QUIET, FLARE_CLASS_X,
        )

        n_bc    = sum(1 for f in flares if f.flare_class == FLARE_CLASS_BC)
        n_m     = sum(1 for f in flares if f.flare_class == FLARE_CLASS_M)
        n_x     = sum(1 for f in flares if f.flare_class == FLARE_CLASS_X)
        n_flare = n_bc + n_m + n_x
        n_quiet = len(flares) - n_flare + max(0, (len(time_unix) // 60) - len(flares))

        counts_by_class = np.array(
            [max(n_quiet, 1), max(n_bc, 1), max(n_m, 1), max(n_x, 1)],
            dtype=float,
        )
        inv_freq = 1.0 / counts_by_class
        inv_freq /= inv_freq.min()
        class_weights = {int(i): float(round(w, 4)) for i, w in enumerate(inv_freq)}

        # ── Timing stats ──────────────────────────────────────
        rises  = [f.rise_time_sec  for f in flares] if flares else [float("nan")]
        decays = [f.decay_time_sec for f in flares] if flares else [float("nan")]
        durs   = [f.duration_sec   for f in flares] if flares else [float("nan")]
        onsets = [f.onset_unix     for f in flares]
        iat    = float(np.nanmedian(np.diff(onsets))) if len(onsets) > 1 else float("nan")

        report = HEL1OSEDAReport(
            detector      = detector,
            date_str      = date_str,
            n_samples     = int(len(time_unix)),
            mean_cr       = mean_cr,
            std_cr        = std_cr,
            min_cr        = min_cr,
            max_cr        = max_cr,
            p25_cr        = float(percs[0]),
            p50_cr        = float(percs[1]),
            p75_cr        = float(percs[2]),
            p90_cr        = float(percs[3]),
            p95_cr        = float(percs[4]),
            p99_cr        = float(percs[5]),
            n_flares      = len(flares),
            n_quiet       = n_quiet,
            n_bc          = n_bc,
            n_m           = n_m,
            n_x           = n_x,
            class_weights         = class_weights,
            mean_rise_sec         = float(np.nanmean(rises)),
            mean_decay_sec        = float(np.nanmean(decays)),
            mean_duration_sec     = float(np.nanmean(durs)),
            median_interarrival_sec = iat,
            median_cadence_sec    = cadence,
            thresh_bc = thresh_bc,
            thresh_m  = thresh_m,
            thresh_x  = thresh_x,
        )
        logger.info(
            "HEL1OS EDA report: %s %s | n_flares=%d | weights=%s",
            detector, date_str, len(flares), class_weights,
        )
        return report

    def save(self, report: HEL1OSEDAReport, tag: str = "") -> Path:
        """
        Save the EDA report as a JSON file.

        Parameters
        ----------
        report : HEL1OSEDAReport
        tag : str  — appended to the filename (e.g. date string)

        Returns
        -------
        Path
        """
        fname = (
            self._out
            / f"helios_eda_report_{report.detector}_{tag or report.date_str}.json"
        )
        with open(fname, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
        logger.info("HEL1OS EDA report saved: %s", fname)
        return fname

    def plot_distributions(
        self,
        time_unix:  npt.NDArray[np.float64],
        count_rate: npt.NDArray[np.float64],
        flares:     list,
        tag:        str = "",
        detector:   str = "",
    ) -> list[Path]:
        """
        Generate and save all EDA distribution plots for one detector.

        Plots produced
        --------------
        1. Count-rate histogram (linear + log10)
        2. Flare class distribution bar chart  (if any flares detected)
        3. Flare timing histograms              (if any flares detected)
        4. Auto-correlation function (ACF)
        5. Timing plot (count rate vs. UTC time with flare overlays)

        Returns
        -------
        list of Path — saved PNG files
        """
        saved: list[Path] = []
        saved.append(self._plot_cr_histogram(count_rate, tag, detector))
        if flares:
            saved.append(self._plot_class_distribution(flares, tag, detector))
            saved.append(self._plot_timing_distributions(flares, tag, detector))
        saved.append(self._plot_acf(count_rate, time_unix, tag, detector))
        saved.append(self._plot_timing(time_unix, count_rate, flares, tag, detector))
        return saved

    def plot_detector_comparison(
        self,
        reports: list[HEL1OSEDAReport],
        tag:     str = "",
    ) -> Path:
        """
        Side-by-side bar charts comparing key statistics across all four
        HEL1OS detectors (CdTe1, CdTe2, CZT1, CZT2).

        This reveals CdTe vs. CZT gain offsets and intra-pair reproducibility,
        which are important cross-calibration diagnostics before combining
        detector bands into broadband features.

        Parameters
        ----------
        reports : list of HEL1OSEDAReport — one per detector
        tag     : str — appended to the output filename

        Returns
        -------
        Path
        """
        if not reports:
            logger.warning("plot_detector_comparison: no reports supplied.")
            return self._out / f"helios_detector_comparison_{tag}.png"

        detectors = [r.detector for r in reports]
        means     = [r.mean_cr   for r in reports]
        stds      = [r.std_cr    for r in reports]
        p95s      = [r.p95_cr    for r in reports]
        n_flares  = [r.n_flares  for r in reports]
        colors    = [_DETECTOR_META.get(d, {}).get("color", "#888888") for d in detectors]

        with plt.style.context(_STYLE):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            _bar(axes[0], detectors, means, colors, "Mean CR (cts s⁻¹)", "Mean Count Rate")
            _bar(axes[1], detectors, stds,  colors, "Std CR (cts s⁻¹)",  "Count Rate Std Dev")
            _bar(axes[2], detectors, p95s,  colors, "P95 CR (cts s⁻¹)",  "95th Percentile CR")
            _bar(axes[3], detectors, n_flares, colors, "N Flares",        "Detected Flares")

            fig.suptitle(
                f"HEL1OS Detector Comparison — {tag}\n"
                "CdTe: 20–150 keV  |  CZT: 8–60 keV",
                y=1.02,
            )
            plt.tight_layout()
            out = self._out / f"helios_detector_comparison_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved: %s", out)
        return out

    # ── Internal plot methods ─────────────────────────────────────────────────

    def _plot_cr_histogram(
        self,
        count_rate: npt.NDArray[np.float64],
        tag:        str,
        detector:   str,
    ) -> Path:
        """Count-rate histogram (linear + log10) for one HEL1OS detector."""
        color = _DETECTOR_META.get(detector, {}).get("color", "#3A7FD5")
        with plt.style.context(_STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            cr_clean = count_rate[np.isfinite(count_rate) & (count_rate >= 0)]

            axes[0].hist(cr_clean, bins=80, color=color, alpha=0.75, edgecolor="none")
            axes[0].set_xlabel("Count Rate (cts s⁻¹)")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title("Count Rate Distribution (linear)")

            log_cr = np.log10(cr_clean + 1.0)
            axes[1].hist(log_cr, bins=80, color=color, alpha=0.75, edgecolor="none")
            axes[1].set_xlabel("log₁₀(Count Rate + 1)")
            axes[1].set_ylabel("Frequency")
            axes[1].set_title("Count Rate Distribution (log scale)")

            fig.suptitle(
                f"HEL1OS {detector} Count Rate Distribution — {tag}", y=1.02
            )
            plt.tight_layout()
            out = self._out / f"helios_hist_cr_{detector}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved: %s", out)
        return out

    def _plot_class_distribution(
        self,
        flares:   list,
        tag:      str,
        detector: str,
    ) -> Path:
        """Bar chart of GOES-proxy class counts for HEL1OS flares."""
        from ml.eda.flare_detector import (
            FLARE_CLASS_BC, FLARE_CLASS_M,
            FLARE_CLASS_QUIET, FLARE_CLASS_X,
        )
        with plt.style.context(_STYLE):
            class_counts = {
                "Quiet": sum(1 for f in flares if f.flare_class == FLARE_CLASS_QUIET),
                "B/C":   sum(1 for f in flares if f.flare_class == FLARE_CLASS_BC),
                "M":     sum(1 for f in flares if f.flare_class == FLARE_CLASS_M),
                "X":     sum(1 for f in flares if f.flare_class == FLARE_CLASS_X),
            }
            class_colors = ["#AAAAAA", "#66BB6A", "#FFA726", "#EF5350"]
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(
                list(class_counts.keys()),
                list(class_counts.values()),
                color=class_colors,
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, count in zip(bars, class_counts.values()):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    str(count),
                    ha="center", va="bottom", fontsize=9,
                )
            ax.set_ylabel("Number of Flares")
            ax.set_title(
                f"HEL1OS {detector} Flare Class Distribution — {tag}\n"
                "(GOES-proxy: count-rate percentile method)"
            )
            ax.set_ylim(0, max(class_counts.values()) * 1.2 + 1)
            total = sum(class_counts.values())
            if total > 0:
                x_frac = class_counts["X"] / total
                ax.text(
                    0.99, 0.95,
                    f"Class imbalance: X={x_frac*100:.1f}% of events.\n"
                    "→ Use class_weights in training.",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color="#555555",
                    bbox=dict(facecolor="lightyellow", edgecolor="none", alpha=0.8),
                )
            plt.tight_layout()
            out = self._out / f"helios_flare_class_dist_{detector}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved: %s", out)
        return out

    def _plot_timing_distributions(
        self,
        flares:   list,
        tag:      str,
        detector: str,
    ) -> Path:
        """Histograms of rise time, decay time, and duration for HEL1OS flares."""
        with plt.style.context(_STYLE):
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            rises  = [f.rise_time_sec  / 60 for f in flares]
            decays = [f.decay_time_sec / 60 for f in flares]
            durs   = [f.duration_sec   / 60 for f in flares]

            data_labels = [
                (rises,  "Rise Time (min)",      axes[0], "#E84040"),
                (decays, "Decay Time (min)",     axes[1], "#3A7FD5"),
                (durs,   "Total Duration (min)", axes[2], "#2ECC71"),
            ]
            for data, xlabel, ax, color in data_labels:
                ax.hist(data, bins=min(20, max(5, len(data) // 2)),
                        color=color, alpha=0.8, edgecolor="none")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Count")
                median_val = float(np.median(data)) if data else 0.0
                ax.axvline(
                    median_val, color="black", lw=1.2, ls="--",
                    label=f"Median: {median_val:.1f}",
                )
                ax.legend(fontsize=7)

            # HXR timescales are shorter than SXR — impulsive phase ~seconds-minutes
            axes[0].set_title("Rise Time\n(HXR: typically <5 min, Benz 2008 §2.2)", fontsize=8)
            axes[1].set_title("Decay Time\n(faster than SXR gradual phase)", fontsize=8)
            axes[2].set_title("Duration", fontsize=8)

            fig.suptitle(
                f"HEL1OS {detector} Flare Timing — {tag}", y=1.02
            )
            plt.tight_layout()
            out = self._out / f"helios_flare_timing_{detector}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved: %s", out)
        return out

    def _plot_acf(
        self,
        count_rate:  npt.NDArray[np.float64],
        time_unix:   npt.NDArray[np.float64],
        tag:         str,
        detector:    str,
        max_lag_sec: float = 1800.0,   # 30 min — HXR timescales are shorter
    ) -> Path:
        """
        Auto-correlation function for one HEL1OS detector.

        HXR ACF peaks are typically at shorter lags than SXR because hard
        X-ray emission tracks the non-thermal electron population directly
        (Benz 2008 §2.2), while soft X-ray is the time-integral (thermal
        response) of the same energy input.
        """
        with plt.style.context(_STYLE):
            color   = _DETECTOR_META.get(detector, {}).get("color", "#3A7FD5")
            cadence = float(np.nanmedian(np.diff(time_unix)))
            cadence = max(cadence, 1.0)
            max_lag = int(max_lag_sec / cadence)

            cr_clean = count_rate.copy()
            cr_clean[~np.isfinite(cr_clean)] = float(np.nanmedian(cr_clean))
            cr_norm = (cr_clean - np.mean(cr_clean)) / (np.std(cr_clean) + 1e-10)

            n     = len(cr_norm)
            fft_  = np.fft.rfft(cr_norm, n=2 * n)
            acf_  = np.fft.irfft(fft_ * np.conj(fft_))[:n]
            acf_  = acf_ / acf_[0]
            lags_s = np.arange(min(max_lag, n)) * cadence / 60.0  # → minutes

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(lags_s, acf_[:len(lags_s)], color=color, lw=0.9)
            ax.axhline(0, color="gray", lw=0.6, ls=":")
            conf = 1.96 / np.sqrt(n)
            ax.axhline( conf, color="orange", lw=0.8, ls="--", label="95% CI")
            ax.axhline(-conf, color="orange", lw=0.8, ls="--")
            ax.set_xlabel("Lag (minutes)")
            ax.set_ylabel("ACF")
            ax.set_title(
                f"HEL1OS {detector} Auto-Correlation — {tag}\n"
                "HXR peaks at shorter lags than SXR (impulsive vs. thermal response)."
            )
            ax.legend()
            plt.tight_layout()
            out = self._out / f"helios_acf_{detector}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved: %s", out)
        return out

    def _plot_timing(
        self,
        time_unix:  npt.NDArray[np.float64],
        count_rate: npt.NDArray[np.float64],
        flares:     list,
        tag:        str,
        detector:   str,
    ) -> Path:
        """
        Count rate vs. time plot with detected flare onset/peak/end overlaid.

        Provides a quick visual sanity check that the FlareDetector found
        real events and not noise spikes.
        """
        from datetime import datetime, timezone
        color = _DETECTOR_META.get(detector, {}).get("color", "#3A7FD5")
        times = [
            datetime.fromtimestamp(t, tz=timezone.utc) for t in time_unix
        ]
        with plt.style.context(_STYLE):
            import matplotlib.dates as mdates

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(times, count_rate, color=color, lw=0.8,
                    label=f"HEL1OS {detector}")
            ax.fill_between(times, 0, count_rate, color=color, alpha=0.1)

            for f in flares:
                t_on   = datetime.fromtimestamp(f.onset_unix,  tz=timezone.utc)
                t_peak = datetime.fromtimestamp(f.peak_unix,   tz=timezone.utc)
                t_end  = datetime.fromtimestamp(f.end_unix,    tz=timezone.utc)
                ax.axvline(t_on,   color="#FF8C00", lw=1.0, ls="--", alpha=0.8)
                ax.axvline(t_peak, color="#E84040", lw=1.2, ls="-",  alpha=0.8)
                ax.axvline(t_end,  color="#3A7FD5", lw=1.0, ls=":",  alpha=0.8)

            if flares:
                ax.axvline(datetime.fromtimestamp(flares[0].onset_unix, tz=timezone.utc),
                           color="#FF8C00", lw=1.0, ls="--", label="Onset", alpha=0.8)
                ax.axvline(datetime.fromtimestamp(flares[0].peak_unix,  tz=timezone.utc),
                           color="#E84040", lw=1.2, ls="-",  label="Peak",  alpha=0.8)
                ax.axvline(datetime.fromtimestamp(flares[0].end_unix,   tz=timezone.utc),
                           color="#3A7FD5", lw=1.0, ls=":",  label="End",   alpha=0.8)

            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Count Rate (cts s⁻¹)")
            ax.set_title(f"HEL1OS {detector} Light Curve — {tag}")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.legend(loc="upper right", fontsize=7)
            fig.autofmt_xdate()
            plt.tight_layout()
            out = self._out / f"helios_timing_{detector}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved: %s", out)
        return out


# ── Module-level helper ───────────────────────────────────────────────────────

def _bar(
    ax:       plt.Axes,
    labels:   list[str],
    values:   list[float],
    colors:   list[str],
    ylabel:   str,
    title:    str,
) -> None:
    """Utility: labelled bar chart used in detector-comparison plot."""
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(max(values, default=0) * 0.01, 0.01),
            f"{v:.1f}",
            ha="center", va="bottom", fontsize=7,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)