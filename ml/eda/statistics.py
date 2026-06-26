"""
ml/eda/statistics.py
=====================
Statistical summaries for Aditya-L1 light curve EDA.

What it computes
----------------
1. Basic descriptive stats (mean, std, min, max, percentiles).
2. Count rate distribution histogram.
3. Flare class distribution (from detected FlareEvents).
4. Class imbalance report — critical before choosing loss functions.
5. Inter-arrival time distribution (time between flares).
6. Flare duration and rise-time distributions.
7. Auto-correlation function (ACF) — reveals characteristic timescales.
8. Export everything to a JSON / CSV summary in cfg.paths.cache.

Physics motivation
------------------
Benz (2008) §1.3 describes flares as "a brightening occurring at a time
scale of minutes."  The ACF and duration distributions let us verify that
our 1-minute resampling (common_cadence_sec=60 in config) captures the
impulsive phase without aliasing.

The class imbalance numbers directly inform:
  - class_weight in CNN/LSTM training (training.use_class_weights)
  - sampling strategy for windowed datasets (Module 4)

Usage
-----
    from ml.eda.statistics import EDAStatistics
    from ml.eda.flare_detector import FlareDetector

    stats = EDAStatistics(output_dir=cfg.paths.cache)
    report = stats.compute(lc.time_unix, lc.count_rate, flares, lc.detector)
    stats.save(report, "20260621")
    stats.plot_distributions(lc.time_unix, lc.count_rate, flares, "20260621")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

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


@dataclass
class EDAReport:
    """
    Full EDA statistical summary for one day's light curve.

    All numeric fields are Python floats/ints so the object is JSON-serialisable.
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
    n_flares:       int
    n_quiet:        int
    n_bc:           int
    n_m:            int
    n_x:            int

    # ── Imbalance ─────────────────────────────────────────────
    class_weights:  dict = field(default_factory=dict)
    # {0: w0, 1: w1, 2: w2, 3: w3} — inverse frequency weights

    # ── Timing stats ──────────────────────────────────────────
    mean_rise_sec:   float = float("nan")
    mean_decay_sec:  float = float("nan")
    mean_duration_sec: float = float("nan")
    median_interarrival_sec: float = float("nan")

    # ── Cadence ───────────────────────────────────────────────
    median_cadence_sec: float = 1.0

    # ── Derived thresholds (for labelling.yaml override) ──────
    thresh_bc: float = float("nan")
    thresh_m:  float = float("nan")
    thresh_x:  float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


class EDAStatistics:
    """
    Computes and saves EDA statistics for Aditya-L1 light curves.

    Parameters
    ----------
    output_dir : Path
        Directory where reports and plots are saved.
    bc_percentile : float
        Percentile defining the B/C threshold.
    m_percentile : float
        Percentile defining the M threshold.
    x_percentile : float
        Percentile defining the X threshold.
    """

    def __init__(
        self,
        output_dir:   Path | str,
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
        flares:     list,                          # list[FlareEvent]
        detector:   str,
        date_str:   str = "",
    ) -> EDAReport:
        """
        Compute the full EDA report.

        Parameters
        ----------
        time_unix : ndarray
        count_rate : ndarray
        flares : list of FlareEvent
        detector : str
        date_str : str

        Returns
        -------
        EDAReport
        """
        cr = count_rate[np.isfinite(count_rate)]

        # ── Basic stats ───────────────────────────────────────
        mean_cr = float(np.nanmean(cr))
        std_cr  = float(np.nanstd(cr))
        min_cr  = float(np.nanmin(cr))
        max_cr  = float(np.nanmax(cr))
        percs   = np.nanpercentile(cr, [25, 50, 75, 90, 95, 99]).tolist()

        # ── Thresholds ────────────────────────────────────────
        thresh_bc = float(np.nanpercentile(cr, self.bc_percentile))
        thresh_m  = float(np.nanpercentile(cr, self.m_percentile))
        thresh_x  = float(np.nanpercentile(cr, self.x_percentile))

        # ── Cadence ───────────────────────────────────────────
        cadence = float(np.nanmedian(np.diff(time_unix))) if len(time_unix) > 1 else 1.0

        # ── Flare class distribution ──────────────────────────
        from ml.eda.flare_detector import (
            FLARE_CLASS_BC, FLARE_CLASS_M, FLARE_CLASS_QUIET, FLARE_CLASS_X,
        )

        n_bc    = sum(1 for f in flares if f.flare_class == FLARE_CLASS_BC)
        n_m     = sum(1 for f in flares if f.flare_class == FLARE_CLASS_M)
        n_x     = sum(1 for f in flares if f.flare_class == FLARE_CLASS_X)
        n_flare = n_bc + n_m + n_x
        n_quiet = len(flares) - n_flare + max(0, (len(time_unix) // 60) - len(flares))

        # Class weights — inverse of frequency, normalised so min weight = 1
        counts_by_class = np.array([
            max(n_quiet, 1), max(n_bc, 1), max(n_m, 1), max(n_x, 1),
        ], dtype=float)
        inv_freq = 1.0 / counts_by_class
        inv_freq /= inv_freq.min()
        class_weights = {int(i): float(round(w, 4)) for i, w in enumerate(inv_freq)}

        # ── Timing stats ──────────────────────────────────────
        rises   = [f.rise_time_sec   for f in flares] if flares else [float("nan")]
        decays  = [f.decay_time_sec  for f in flares] if flares else [float("nan")]
        durs    = [f.duration_sec    for f in flares] if flares else [float("nan")]
        onsets  = [f.onset_unix      for f in flares]
        iat     = float(np.nanmedian(np.diff(onsets))) if len(onsets) > 1 else float("nan")

        report = EDAReport(
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
            class_weights     = class_weights,
            mean_rise_sec     = float(np.nanmean(rises)),
            mean_decay_sec    = float(np.nanmean(decays)),
            mean_duration_sec = float(np.nanmean(durs)),
            median_interarrival_sec = iat,
            median_cadence_sec  = cadence,
            thresh_bc = thresh_bc,
            thresh_m  = thresh_m,
            thresh_x  = thresh_x,
        )
        logger.info(
            "EDA report: %s %s | n_flares=%d | weights=%s",
            detector, date_str, len(flares), class_weights,
        )
        return report

    def save(self, report: EDAReport, tag: str = "") -> Path:
        """
        Save the EDA report as a JSON file.

        Parameters
        ----------
        report : EDAReport
        tag : str
            Optional suffix for the filename (e.g. date string).

        Returns
        -------
        Path
        """
        fname = self._out / f"eda_report_{report.detector}_{tag or report.date_str}.json"
        with open(fname, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
        logger.info("EDA report saved: %s", fname)
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
        Generate and save all EDA distribution plots.

        Plots produced:
          1. Count rate histogram + KDE
          2. Flare class distribution bar chart
          3. Flare duration histogram
          4. Auto-correlation function (ACF)

        Parameters
        ----------
        time_unix : ndarray
        count_rate : ndarray
        flares : list of FlareEvent
        tag : str
        detector : str

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

        return saved

    # ── Internal plot methods ─────────────────────────────────────────────────

    def _plot_cr_histogram(
        self,
        count_rate: npt.NDArray[np.float64],
        tag: str,
        detector: str,
    ) -> Path:
        """Count rate histogram with log10 y-axis to show the flare tail."""
        with plt.style.context(_STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            cr_clean = count_rate[np.isfinite(count_rate) & (count_rate >= 0)]

            # Linear scale
            axes[0].hist(cr_clean, bins=80, color="#E84040", alpha=0.75, edgecolor="none")
            axes[0].set_xlabel("Count Rate (cts s⁻¹)")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title("Count Rate Distribution (linear)")

            # Log10 count rate — reveals the flare power-law tail
            log_cr = np.log10(cr_clean + 1.0)
            axes[1].hist(log_cr, bins=80, color="#3A7FD5", alpha=0.75, edgecolor="none")
            axes[1].set_xlabel("log₁₀(Count Rate + 1)")
            axes[1].set_ylabel("Frequency")
            axes[1].set_title("Count Rate Distribution (log scale)")

            fig.suptitle(f"{detector} Count Rate Distribution — {tag}", y=1.02)
            plt.tight_layout()

            out = self._out / f"hist_cr_{detector}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved: %s", out)
        return out

    def _plot_class_distribution(
        self,
        flares: list,
        tag:    str,
        detector: str,
    ) -> Path:
        """Bar chart showing count per GOES-proxy class."""
        from ml.eda.flare_detector import (
            FLARE_CLASS_BC, FLARE_CLASS_M, FLARE_CLASS_QUIET, FLARE_CLASS_X,
        )

        with plt.style.context(_STYLE):
            class_counts = {
                "Quiet": sum(1 for f in flares if f.flare_class == FLARE_CLASS_QUIET),
                "B/C":   sum(1 for f in flares if f.flare_class == FLARE_CLASS_BC),
                "M":     sum(1 for f in flares if f.flare_class == FLARE_CLASS_M),
                "X":     sum(1 for f in flares if f.flare_class == FLARE_CLASS_X),
            }
            colors = ["#AAAAAA", "#66BB6A", "#FFA726", "#EF5350"]

            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(
                list(class_counts.keys()),
                list(class_counts.values()),
                color=colors,
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
            ax.set_title(f"Flare Class Distribution — {detector} {tag}\n"
                         "(GOES-proxy: count-rate percentile method)")
            ax.set_ylim(0, max(class_counts.values()) * 1.2 + 1)

            # Add imbalance warning
            total = sum(class_counts.values())
            if total > 0:
                x_frac = class_counts["X"] / total
                note = (
                    f"Class imbalance: X={x_frac*100:.1f}% of events.\n"
                    "→ Use class_weights in training (Module 5/6)."
                )
                ax.text(0.99, 0.95, note, transform=ax.transAxes,
                        ha="right", va="top", fontsize=7, color="#555555",
                        bbox=dict(facecolor="lightyellow", edgecolor="none", alpha=0.8))

            plt.tight_layout()
            out = self._out / f"flare_class_dist_{detector}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved: %s", out)
        return out

    def _plot_timing_distributions(
        self,
        flares: list,
        tag:    str,
        detector: str,
    ) -> Path:
        """Histograms for rise time, decay time, and total duration."""
        with plt.style.context(_STYLE):
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            rises  = [f.rise_time_sec  / 60 for f in flares]  # → minutes
            decays = [f.decay_time_sec / 60 for f in flares]
            durs   = [f.duration_sec   / 60 for f in flares]

            data_labels = [
                (rises,  "Rise Time (min)",     axes[0], "#E84040"),
                (decays, "Decay Time (min)",    axes[1], "#3A7FD5"),
                (durs,   "Total Duration (min)", axes[2], "#2ECC71"),
            ]
            for data, xlabel, ax, color in data_labels:
                ax.hist(data, bins=min(20, max(5, len(data) // 2)),
                        color=color, alpha=0.8, edgecolor="none")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Count")
                ax.axvline(float(np.median(data)), color="black",
                           lw=1.2, ls="--", label=f"Median: {np.median(data):.1f}")
                ax.legend(fontsize=7)

            # Physics context from Benz (2008) §1.3
            axes[0].set_title(
                "Rise Time\n(Benz 2008: impulsive phase 3–10 min)", fontsize=8
            )
            axes[1].set_title("Decay Time\n(typically 1 h for large flares)", fontsize=8)
            axes[2].set_title("Duration", fontsize=8)

            fig.suptitle(f"Flare Timing Distributions — {detector} {tag}", y=1.02)
            plt.tight_layout()

            out = self._out / f"flare_timing_{detector}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved: %s", out)
        return out

    def _plot_acf(
        self,
        count_rate: npt.NDArray[np.float64],
        time_unix:  npt.NDArray[np.float64],
        tag:        str,
        detector:   str,
        max_lag_sec: float = 3600.0,
    ) -> Path:
        """
        Auto-correlation function (ACF) of the count rate.

        Reveals periodicity and characteristic decay timescales.
        The ACF of flare light curves typically shows:
          - A sharp peak at lag=0
          - A broad wing at 3–10 min (impulsive phase correlation)
          - Slow decay at 30–60 min (thermal/decay phase)
        """
        with plt.style.context(_STYLE):
            cadence   = float(np.nanmedian(np.diff(time_unix)))
            cadence   = max(cadence, 1.0)
            max_lag   = int(max_lag_sec / cadence)
            cr_clean  = count_rate.copy()
            cr_clean[~np.isfinite(cr_clean)] = float(np.nanmedian(cr_clean))

            # Normalise
            cr_norm = (cr_clean - np.mean(cr_clean)) / (np.std(cr_clean) + 1e-10)

            # Compute ACF via FFT for speed
            n      = len(cr_norm)
            fft_   = np.fft.rfft(cr_norm, n=2 * n)
            acf_   = np.fft.irfft(fft_ * np.conj(fft_))[:n]
            acf_   = acf_ / acf_[0]   # normalise so ACF[0]=1
            lags_s = np.arange(min(max_lag, n)) * cadence / 60.0  # → minutes

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(lags_s, acf_[:len(lags_s)], color="#9B59B6", lw=0.9)
            ax.axhline(0, color="gray", lw=0.6, ls=":")
            # 95% confidence band for white noise
            conf = 1.96 / np.sqrt(n)
            ax.axhline(conf,  color="orange", lw=0.8, ls="--", label="95% CI")
            ax.axhline(-conf, color="orange", lw=0.8, ls="--")
            ax.set_xlabel("Lag (minutes)")
            ax.set_ylabel("ACF")
            ax.set_title(
                f"Auto-Correlation Function — {detector} {tag}\n"
                "Look for peaks at 3–10 min (impulsive) and 30–60 min (decay)."
            )
            ax.legend()
            plt.tight_layout()

            out = self._out / f"acf_{detector}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved: %s", out)
        return out
