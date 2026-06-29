"""
ml/eda/statistics.py
=====================
Statistical summaries for SoLEXS EDA.
Receives loaded data objects — never reads FITS files directly.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_STYLE = {
    "figure.dpi":        150,
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


def _safe_nanmean(values) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.nanmean(finite)) if len(finite) > 0 else float("nan")


def _safe_nanmedian(values) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.nanmedian(finite)) if len(finite) > 0 else float("nan")


def _safe_nanstd(values) -> float:
    """np.nanstd needs >=2 finite values with the default ddof=0; fewer
    triggers numpy's 'Degrees of freedom <= 0 for slice' RuntimeWarning."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.nanstd(finite)) if len(finite) >= 2 else float("nan")


def _safe_std(values) -> float:
    """Compute std only over finite values and return nan when none exist."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.std(finite)) if finite.size >= 1 else float("nan")


@dataclass
class EDAReport:
    detector:  str
    date_str:  str
    n_samples: int

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

    n_flares:  int
    n_quiet:   int
    n_bc:      int
    n_m:       int
    n_x:       int

    class_weights: dict = field(default_factory=dict)

    mean_rise_sec:           float = float("nan")
    mean_decay_sec:          float = float("nan")
    mean_duration_sec:       float = float("nan")
    median_interarrival_sec: float = float("nan")
    median_cadence_sec:      float = 1.0

    thresh_bc: float = float("nan")
    thresh_m:  float = float("nan")
    thresh_x:  float = float("nan")

    n_gti_intervals:   int   = 0
    gti_coverage_pct:  float = float("nan")
    n_missing_samples: int   = 0

    def to_dict(self) -> dict:
        return asdict(self)


class EDAStatistics:
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

    def compute(
        self,
        time_unix:  npt.NDArray[np.float64],
        count_rate: npt.NDArray[np.float64],
        flares:     list,
        detector:   str,
        date_str:   str = "",
        gti=None,
    ) -> EDAReport:
        cr = count_rate[np.isfinite(count_rate)]

        mean_cr   = float(np.nanmean(cr))
        std_cr    = _safe_nanstd(cr)
        min_cr    = float(np.nanmin(cr))
        max_cr    = float(np.nanmax(cr))
        percs     = np.nanpercentile(cr, [25, 50, 75, 90, 95, 99]).tolist()
        thresh_bc = float(np.nanpercentile(cr, self.bc_percentile))
        thresh_m  = float(np.nanpercentile(cr, self.m_percentile))
        thresh_x  = float(np.nanpercentile(cr, self.x_percentile))
        cadence   = float(np.nanmedian(np.diff(time_unix))) if len(time_unix) > 1 else 1.0

        n_gti_intervals  = gti.n_intervals if gti is not None else 0
        gti_coverage_pct = float("nan")
        n_missing        = 0
        if gti is not None and len(time_unix) > 0:
            mask             = gti.mask_for(time_unix)
            gti_coverage_pct = float(100.0 * mask.sum() / len(mask))
            n_missing        = int((~np.isfinite(count_rate)).sum())

        from ml.eda.flare_detector import (
            FLARE_CLASS_BC, FLARE_CLASS_M, FLARE_CLASS_QUIET, FLARE_CLASS_X,
        )
        n_bc    = sum(1 for f in flares if f.flare_class == FLARE_CLASS_BC)
        n_m     = sum(1 for f in flares if f.flare_class == FLARE_CLASS_M)
        n_x     = sum(1 for f in flares if f.flare_class == FLARE_CLASS_X)
        n_quiet = len(flares) - (n_bc + n_m + n_x) + max(
            0, (len(time_unix) // 60) - len(flares)
        )

        counts_arr = np.array(
            [max(n_quiet, 1), max(n_bc, 1), max(n_m, 1), max(n_x, 1)],
            dtype=float,
        )
        inv_freq = 1.0 / counts_arr
        inv_freq /= inv_freq.min()
        class_weights = {int(i): float(round(w, 4)) for i, w in enumerate(inv_freq)}

        rises  = [f.rise_time_sec  for f in flares] if flares else []
        decays = [f.decay_time_sec for f in flares] if flares else []
        durs   = [f.duration_sec   for f in flares] if flares else []
        onsets = [f.onset_unix     for f in flares]
        iat    = _safe_nanmedian(np.diff(onsets)) if len(onsets) > 1 else float("nan")

        return EDAReport(
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
            class_weights           = class_weights,
            mean_rise_sec           = _safe_nanmean(rises),
            mean_decay_sec          = _safe_nanmean(decays),
            mean_duration_sec       = _safe_nanmean(durs),
            median_interarrival_sec = iat,
            median_cadence_sec      = cadence,
            thresh_bc               = thresh_bc,
            thresh_m                = thresh_m,
            thresh_x                = thresh_x,
            n_gti_intervals         = n_gti_intervals,
            gti_coverage_pct        = gti_coverage_pct,
            n_missing_samples       = n_missing,
        )

    def save(self, report: EDAReport, tag: str = "") -> Path:
        fname = self._out / f"eda_report_{report.detector}_{tag or report.date_str}.json"
        with open(fname, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
        logger.info("Saved: %s", fname)
        return fname

    def save_daily_csv(self, reports: list[EDAReport], path: Path) -> Path:
        if not reports:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(reports[0].to_dict().keys())
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in reports:
                writer.writerow(r.to_dict())
        logger.info("Saved daily CSV: %s", path)
        return path

    def save_flare_candidates_csv(
        self,
        all_flares: dict[str, list],
        path: Path,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for date_str, flares in all_flares.items():
            for f in flares:
                rows.append({
                    "date_str":        date_str,
                    "onset_utc":       f.onset_utc,
                    "class_name":      f.class_name,
                    "peak_count_rate": f.peak_count_rate,
                    "rise_time_sec":   f.rise_time_sec,
                    "decay_time_sec":  f.decay_time_sec,
                    "duration_sec":    f.duration_sec,
                })
        if not rows:
            return path
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved flare candidates: %s", path)
        return path

    def save_aggregate_json(
        self,
        reports:      list[EDAReport],
        missing_days: list[str],
        path:         Path,
    ) -> Path:
        if not reports:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        means = [r.mean_cr for r in reports]
        agg = {
            "n_days_analysed":   len(reports),
            "n_days_missing":    len(missing_days),
            "missing_dates":     missing_days,
            "total_samples":     sum(r.n_samples for r in reports),
            "total_flares":      sum(r.n_flares  for r in reports),
            "mean_cr_all_days":  float(np.mean(means)),
            "std_cr_all_days":   _safe_std(means),
            "observation_span":  f"{reports[0].date_str} → {reports[-1].date_str}",
            "detector":          reports[0].detector,
            "top_flare_days":    [
                {"date": r.date_str, "n_flares": r.n_flares}
                for r in sorted(reports, key=lambda x: x.n_flares, reverse=True)[:10]
            ],
        }
        with open(path, "w") as fh:
            json.dump(agg, fh, indent=2)
        logger.info("Saved aggregate JSON: %s", path)
        return path

    def plot_distributions(
        self,
        time_unix:  npt.NDArray[np.float64],
        count_rate: npt.NDArray[np.float64],
        flares:     list,
        tag:        str = "",
        detector:   str = "",
    ) -> list[Path]:
        saved = []
        saved.append(self._plot_cr_histogram(count_rate, tag, detector))
        if flares:
            saved.append(self._plot_class_distribution(flares, tag, detector))
            saved.append(self._plot_timing_distributions(flares, tag, detector))
        saved.append(self._plot_acf(count_rate, time_unix, tag, detector))
        return saved

    def _plot_cr_histogram(self, count_rate, tag, detector) -> Path:
        with plt.style.context(_STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                                     layout="constrained")
            cr_clean  = count_rate[np.isfinite(count_rate) & (count_rate >= 0)]
            axes[0].hist(cr_clean, bins=80, color="#E84040", alpha=0.75, edgecolor="none")
            axes[0].set_xlabel("Count Rate (cts s⁻¹)")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title("Count Rate Distribution (linear)")
            log_cr = np.log10(cr_clean + 1.0)
            axes[1].hist(log_cr, bins=80, color="#3A7FD5", alpha=0.75, edgecolor="none")
            axes[1].set_xlabel("log₁₀(Count Rate + 1)")
            axes[1].set_ylabel("Frequency")
            axes[1].set_title("Count Rate Distribution (log scale)")
            fig.suptitle(f"{detector} Count Rate Distribution — {tag}")
            out = self._out / f"hist_cr_{detector}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
        return out

    def _plot_class_distribution(self, flares, tag, detector) -> Path:
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
            fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
            bars = ax.bar(list(class_counts), list(class_counts.values()),
                          color=colors, edgecolor="white", linewidth=0.5)
            for bar, count in zip(bars, class_counts.values()):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.2, str(count),
                        ha="center", va="bottom", fontsize=9)
            ax.set_ylabel("Number of Flares")
            ax.set_title(f"Flare Class Distribution — {detector} {tag}")
            ax.set_ylim(0, max(class_counts.values()) * 1.2 + 1)
            out = self._out / f"flare_class_dist_{detector}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
        return out

    def _plot_timing_distributions(self, flares, tag, detector) -> Path:
        with plt.style.context(_STYLE):
            fig, axes = plt.subplots(1, 3, figsize=(14, 4),
                                     layout="constrained")
            rises  = [f.rise_time_sec  / 60 for f in flares]
            decays = [f.decay_time_sec / 60 for f in flares]
            durs   = [f.duration_sec   / 60 for f in flares]
            for data, xlabel, ax, color in [
                (rises,  "Rise Time (min)",      axes[0], "#E84040"),
                (decays, "Decay Time (min)",     axes[1], "#3A7FD5"),
                (durs,   "Total Duration (min)", axes[2], "#2ECC71"),
            ]:
                ax.hist(data, bins=min(20, max(5, len(data) // 2)),
                        color=color, alpha=0.8, edgecolor="none")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Count")
                ax.axvline(float(np.median(data)), color="black", lw=1.2, ls="--",
                           label=f"Median: {np.median(data):.1f}")
                ax.legend(fontsize=7)
            fig.suptitle(f"Flare Timing — {detector} {tag}")
            out = self._out / f"flare_timing_{detector}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
        return out

    def _plot_acf(self, count_rate, time_unix, tag, detector,
                  max_lag_sec: float = 3600.0) -> Path:
        with plt.style.context(_STYLE):
            cadence  = float(np.nanmedian(np.diff(time_unix)))
            cadence  = max(cadence, 1.0)
            max_lag  = int(max_lag_sec / cadence)
            cr_clean = count_rate.copy()
            cr_clean[~np.isfinite(cr_clean)] = float(np.nanmedian(cr_clean))
            std = _safe_std(cr_clean)
            cr_norm  = (cr_clean - np.mean(cr_clean)) / (std + 1e-10 if not np.isnan(std) else 1e-10)
            n        = len(cr_norm)
            fft_     = np.fft.rfft(cr_norm, n=2 * n)
            acf_     = np.fft.irfft(fft_ * np.conj(fft_))[:n]
            acf_     = acf_ / acf_[0]
            lags_s   = np.arange(min(max_lag, n)) * cadence / 60.0
            fig, ax  = plt.subplots(figsize=(10, 4), layout="constrained")
            ax.plot(lags_s, acf_[:len(lags_s)], color="#9B59B6", lw=0.9)
            ax.axhline(0, color="gray", lw=0.6, ls=":")
            conf = 1.96 / np.sqrt(n)
            ax.axhline( conf, color="orange", lw=0.8, ls="--", label="95% CI")
            ax.axhline(-conf, color="orange", lw=0.8, ls="--")
            ax.set_xlabel("Lag (minutes)")
            ax.set_ylabel("ACF")
            ax.set_title(f"Auto-Correlation Function — {detector} {tag}")
            ax.legend()
            out = self._out / f"acf_{detector}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
        return out