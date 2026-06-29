# ════════════════════════════════════════════════════════════════════════════
# ml/eda/light_curve_plotter.py
# ════════════════════════════════════════════════════════════════════════════
"""
ml/eda/light_curve_plotter.py
==============================
Publication-quality light curve visualisations for SoLEXS and HEL1OS.
Accepts loaded data objects — never reads files directly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_STYLE = {
    "figure.dpi":        150,
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.fontsize":   8,
    "lines.linewidth":   0.9,
}
_SOLEXS_COLOR = "#E84040"
_HELIOS_COLOR = "#3A7FD5"
_FLARE_COLOR  = "#FF8C00"


def _unix_to_dt(time_unix: npt.NDArray[np.float64]) -> list[datetime]:
    return [datetime.fromtimestamp(t, tz=timezone.utc) for t in time_unix]


class LightCurvePlotter:
    def __init__(self, output_dir: Path | str, show: bool = False) -> None:
        self._out  = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._show = show

    # ── SoLEXS plots ─────────────────────────────────────────────────────────

    def plot_solexs_day(
        self,
        lc,
        gti_mask:         Optional[npt.NDArray] = None,
        flare_times_unix: Optional[list[float]] = None,
        title:            str = "",
    ) -> Path:
        with plt.style.context(_STYLE):
            # constrained_layout is compatible with autofmt_xdate; tight_layout
            # conflicts with it when date tick labels are rotated.
            fig, ax = plt.subplots(figsize=(14, 4), layout="constrained")
            times   = _unix_to_dt(lc.time_unix)
            cr      = lc.count_rate
            if gti_mask is not None:
                self._shade_invalid(ax, lc.time_unix, ~gti_mask)
            ax.plot(times, cr, color=_SOLEXS_COLOR, lw=0.8,
                    label=f"SoLEXS {lc.detector} (2–22 keV)")
            if flare_times_unix:
                for t_unix in flare_times_unix:
                    ax.axvline(datetime.fromtimestamp(t_unix, tz=timezone.utc),
                               color=_FLARE_COLOR, lw=1.2, ls="--", alpha=0.85,
                               label="Flare onset")
            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Count Rate (cts s⁻¹)")
            ax.set_title(title or f"SoLEXS {lc.detector} — {lc.date_str}")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.legend(loc="upper right")
            ax.set_xlim(times[0], times[-1])
            self._add_note(ax, "Soft X-ray (2–22 keV): traces hot thermal plasma.")
            fig.autofmt_xdate()
            out = self._out / f"solexs_{lc.detector}_{lc.date_str}_lc.png"
            fig.savefig(out, bbox_inches="tight")
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    def plot_cr_histogram(self, lc, date_str: str, detector: str) -> Path:
        with plt.style.context(_STYLE):
            # constrained_layout avoids tight_layout conflicts with suptitle
            fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                                     layout="constrained")
            cr_clean  = lc.count_rate[np.isfinite(lc.count_rate) & (lc.count_rate >= 0)]
            axes[0].hist(cr_clean, bins=80, color=_SOLEXS_COLOR,
                         alpha=0.75, edgecolor="none")
            axes[0].set_xlabel("Count Rate (cts s⁻¹)")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title("Linear scale")
            axes[1].hist(np.log10(cr_clean + 1), bins=80, color=_HELIOS_COLOR,
                         alpha=0.75, edgecolor="none")
            axes[1].set_xlabel("log₁₀(Count Rate + 1)")
            axes[1].set_ylabel("Frequency")
            axes[1].set_title("Log scale")
            fig.suptitle(f"{detector} Count Rate — {date_str}")
            out = self._out / f"hist_cr_{detector}_{date_str}.png"
            fig.savefig(out, bbox_inches="tight")
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    def plot_gti_statistics(self, day, detector: str) -> Path:
        """Bar chart of GTI interval durations for one day."""
        with plt.style.context(_STYLE):
            fig, ax = plt.subplots(figsize=(10, 3), layout="constrained")
            gti     = day.gti
            # Compute interval durations in seconds
            durations = []
            for i in range(gti.n_intervals):
                try:
                    start = gti.t_start[i]
                    stop  = gti.t_stop[i]
                    durations.append(float(stop - start))
                except (AttributeError, IndexError):
                    break
            if durations:
                ax.bar(range(len(durations)), durations,
                       color="#2ECC71", alpha=0.8, edgecolor="none")
                ax.axhline(float(np.mean(durations)), color="black",
                           lw=1.2, ls="--", label=f"Mean: {np.mean(durations):.0f}s")
                ax.legend(fontsize=7)
            ax.set_xlabel("GTI Interval Index")
            ax.set_ylabel("Duration (s)")
            ax.set_title(f"GTI Intervals — {detector} {day.date_str}  "
                         f"({gti.n_intervals} intervals)")
            out = self._out / f"gti_stats_{detector}_{day.date_str}.png"
            fig.savefig(out, bbox_inches="tight")
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    def plot_flare_day_ranking(self, reports, detector: str) -> Path:
        """Horizontal bar chart: days ranked by flare count."""
        with plt.style.context(_STYLE):
            ranked = sorted(reports, key=lambda r: r.n_flares, reverse=True)[:20]
            dates  = [r.date_str  for r in ranked]
            counts = [r.n_flares  for r in ranked]
            fig, ax = plt.subplots(figsize=(10, max(4, len(dates) * 0.35)),
                                   layout="constrained")
            ax.barh(dates, counts, color=_SOLEXS_COLOR, alpha=0.8, edgecolor="none")
            ax.set_xlabel("Number of Detected Flares")
            ax.set_title(f"Flare-Day Ranking — {detector} (top {len(dates)} days)")
            ax.invert_yaxis()
            out = self._out / f"flare_day_ranking_{detector}.png"
            fig.savefig(out, bbox_inches="tight")
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    def plot_observation_coverage(self, all_days, detector: str) -> Path:
        """
        Timeline showing which days have data and their GTI sample fraction.
        """
        with plt.style.context(_STYLE):
            dates      = [d.date_str for d in all_days]
            coverages  = []
            for d in all_days:
                lc   = d.lc
                mask = d.gti.mask_for(lc.time_unix)
                coverages.append(100.0 * mask.sum() / max(len(mask), 1))

            fig, ax = plt.subplots(figsize=(max(10, len(dates) * 0.15), 4),
                                   layout="constrained")
            ax.bar(range(len(dates)), coverages,
                   color=_HELIOS_COLOR, alpha=0.8, edgecolor="none")
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels(dates, rotation=90, fontsize=6)
            ax.set_ylim(0, 110)
            ax.axhline(100, color="gray", lw=0.6, ls=":")
            ax.set_ylabel("GTI Coverage (%)")
            ax.set_title(f"Observation Coverage — {detector}")
            out = self._out / f"observation_coverage_{detector}.png"
            fig.savefig(out, bbox_inches="tight")
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    # ── Dual-instrument plots ─────────────────────────────────────────────────

    def plot_helios_bands(self, lc) -> Path:
        n = len(lc.bands)
        with plt.style.context(_STYLE):
            # constrained_layout handles suptitle spacing without conflicting
            # with autofmt_xdate the way tight_layout does.
            fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True,
                                     layout="constrained")
            if n == 1:
                axes = [axes]
            cmap = plt.cm.get_cmap("cool", n)
            for i, (band, ax) in enumerate(zip(lc.bands, axes)):
                times = _unix_to_dt(band.time_unix)
                color = cmap(i)
                ax.plot(times, band.count_rate, color=color, lw=0.8)
                ax.fill_between(times, 0, band.count_rate, color=color, alpha=0.15)
                ax.set_ylabel("cts s⁻¹", fontsize=8)
                ax.text(0.01, 0.88,
                        f"{band.e_low_kev:.0f}–{band.e_high_kev:.0f} keV",
                        transform=ax.transAxes, fontsize=8, color=color,
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
            axes[-1].set_xlabel("Time (UTC)")
            fig.suptitle(f"HEL1OS {lc.detector} Multi-band — {lc.date_str}",
                         fontsize=11)
            fig.autofmt_xdate()
            out = self._out / f"helios_{lc.detector}_{lc.date_str}_bands.png"
            fig.savefig(out, bbox_inches="tight")
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    def plot_dual_panel(self, solexs_lc, helios_lc) -> Path:
        with plt.style.context(_STYLE):
            fig, (ax_soft, ax_hard) = plt.subplots(
                2, 1, figsize=(14, 6), sharex=True,
                gridspec_kw={"hspace": 0.08},
                layout="constrained",
            )
            t_soft = _unix_to_dt(solexs_lc.time_unix)
            ax_soft.plot(t_soft, solexs_lc.count_rate, color=_SOLEXS_COLOR, lw=0.9,
                         label=f"SoLEXS {solexs_lc.detector} (2–22 keV)")
            ax_soft.set_ylabel("Soft X-ray\n(cts s⁻¹)")
            ax_soft.legend(loc="upper right")
            ax_soft.set_title(
                f"Aditya-L1 Dual-Band — {solexs_lc.date_str}"
            )
            full   = helios_lc.full_band
            t_hard = _unix_to_dt(full.time_unix)
            ax_hard.plot(t_hard, full.count_rate, color=_HELIOS_COLOR, lw=0.9,
                         label=f"HEL1OS {helios_lc.detector} "
                               f"({full.e_low_kev:.0f}–{full.e_high_kev:.0f} keV)")
            ax_hard.set_ylabel("Hard X-ray\n(cts s⁻¹)")
            ax_hard.set_xlabel("Time (UTC)")
            ax_hard.legend(loc="upper right")
            ax_hard.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax_hard.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            fig.autofmt_xdate()
            out = self._out / (
                f"dual_panel_{solexs_lc.date_str}_"
                f"{solexs_lc.detector}_{helios_lc.detector}.png"
            )
            fig.savefig(out, bbox_inches="tight")
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    def plot_neupert(self, solexs_lc, helios_lc) -> Path:
        with plt.style.context(_STYLE):
            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                                     gridspec_kw={"hspace": 0.12},
                                     layout="constrained")
            ax_sxr, ax_dsxr, ax_hxr = axes
            t_soft = _unix_to_dt(solexs_lc.time_unix)
            cr_sxr = solexs_lc.count_rate
            ax_sxr.plot(t_soft, cr_sxr, color=_SOLEXS_COLOR, lw=0.9)
            ax_sxr.set_ylabel("SXR\n(cts s⁻¹)")
            ax_sxr.set_title(f"Neupert Effect — {solexs_lc.date_str}")
            dsxr     = np.gradient(cr_sxr, solexs_lc.time_unix)
            dsxr_pos = np.clip(dsxr, 0, None)
            ax_dsxr.plot(t_soft, dsxr_pos, color="#9B59B6", lw=0.9,
                         label="d(SXR)/dt [clipped ≥ 0]")
            ax_dsxr.set_ylabel("d(SXR)/dt")
            ax_dsxr.legend(loc="upper right", fontsize=7)
            ax_dsxr.axhline(0, color="gray", lw=0.5, ls=":")
            full   = helios_lc.full_band
            t_hard = _unix_to_dt(full.time_unix)
            ax_hxr.plot(t_hard, full.count_rate, color=_HELIOS_COLOR, lw=0.9,
                        label=f"HXR {full.e_low_kev:.0f}–{full.e_high_kev:.0f} keV")
            ax_hxr.set_ylabel("HXR\n(cts s⁻¹)")
            ax_hxr.set_xlabel("Time (UTC)")
            ax_hxr.legend(loc="upper right")
            ax_hxr.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax_hxr.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            fig.autofmt_xdate()
            out = self._out / f"neupert_{solexs_lc.date_str}.png"
            fig.savefig(out, bbox_inches="tight")
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _shade_invalid(ax, time_unix, invalid_mask) -> None:
        if not invalid_mask.any():
            return
        indices = np.where(np.diff(invalid_mask.astype(int)))[0]
        starts  = [0] if invalid_mask[0] else []
        stops: list[int] = []
        for idx in indices:
            if invalid_mask[idx]:
                stops.append(idx + 1)
            else:
                starts.append(idx + 1)
        if invalid_mask[-1]:
            stops.append(len(time_unix) - 1)
        for s, e in zip(starts, stops):
            t0 = datetime.fromtimestamp(time_unix[s], tz=timezone.utc)
            t1 = datetime.fromtimestamp(time_unix[e], tz=timezone.utc)
            ax.axvspan(t0, t1, color="#DDDDDD", alpha=0.4, zorder=0)

    @staticmethod
    def _add_note(ax, note: str) -> None:
        ax.text(0.99, 0.04, note, transform=ax.transAxes,
                fontsize=6.5, color="#555555", ha="right", va="bottom",
                bbox=dict(facecolor="white", alpha=0.55, edgecolor="none"))