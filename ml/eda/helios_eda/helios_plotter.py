# ════════════════════════════════════════════════════════════════════════════
# ml/eda/helios_eda/helios_plotter.py
# ════════════════════════════════════════════════════════════════════════════
"""
ml/eda/helios_eda/helios_plotter.py
=====================================
HEL1OS visualisations. Accepts loaded data objects — no file I/O.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib as mpl
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
_DETECTOR_COLORS = {
    "CdTe1": "#E84040",
    "CdTe2": "#C03030",
    "CZT1":  "#3A7FD5",
    "CZT2":  "#2A5FA5",
}
_FLARE_COLOR  = "#FF8C00"
_DS_MAX_PTS   = 8_000   # downsample threshold (points per trace)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _unix_to_dt(time_unix: npt.NDArray[np.float64]) -> list[datetime]:
    return [datetime.fromtimestamp(t, tz=timezone.utc) for t in time_unix]


def _ds_index(arr: npt.NDArray, max_pts: int = _DS_MAX_PTS) -> npt.NDArray[np.intp]:
    """
    Return an index array for max-min decimation of *arr*.

    Groups samples into ``max_pts // 2`` bucket pairs, then picks the index of
    the maximum and minimum within each bucket.  This preserves every spike and
    trough visible at normal screen resolution while reducing the number of
    plotted points to ≤ ``max_pts``.

    The original array is never modified; callers slice with the returned index.
    When len(arr) ≤ max_pts the full index range is returned (no copy made).
    """
    n = len(arr)
    if n <= max_pts:
        return np.arange(n, dtype=np.intp)
    n_pairs = max_pts // 2
    bucket  = n / n_pairs
    idx: list[int] = []
    for i in range(n_pairs):
        lo = int(i * bucket)
        hi = min(int((i + 1) * bucket), n)
        if lo >= hi:
            continue
        chunk = arr[lo:hi]
        idx.append(lo + int(np.argmax(chunk)))
        idx.append(lo + int(np.argmin(chunk)))
    return np.unique(idx)   # sorts and deduplicates; dtype intp on all platforms


def _add_note(ax, note: str) -> None:
    """Stamp a small annotation in the bottom-right corner of *ax*."""
    ax.text(0.99, 0.04, note, transform=ax.transAxes,
            fontsize=6.5, color="#555555", ha="right", va="bottom",
            bbox=dict(facecolor="white", alpha=0.55, edgecolor="none"))


# ── Public class ─────────────────────────────────────────────────────────────

class HEL1OSPlotter:
    def __init__(
        self,
        output_dir: Path | str,
        show:       bool = False,
        overwrite:  bool = False,
    ) -> None:
        self._out       = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._show      = show
        self._overwrite = overwrite

    # ── skip-if-exists guard ─────────────────────────────────────────────────

    def _should_skip(self, path: Path) -> bool:
        """Return True when the file already exists and overwrite is disabled."""
        return (not self._overwrite) and path.exists()

    # ── save + close helper ──────────────────────────────────────────────────

    def _save(self, fig, path: Path) -> Path:
        """Save *fig* to *path*, optionally show it, then release all memory."""
        fig.savefig(path, bbox_inches="tight")
        if self._show:
            plt.show()
        plt.close(fig)
        return path

    # ── plots ────────────────────────────────────────────────────────────────

    def plot_helios_day(
        self,
        lc,
        flare_times_unix: Optional[list[float]] = None,
        title: str = "",
    ) -> Path:
        out = self._out / f"helios_{lc.detector}_{lc.date_str}_lc.png"
        if self._should_skip(out):
            return out

        full  = lc.full_band
        color = _DETECTOR_COLORS.get(lc.detector, "#3A7FD5")

        # downsample for plotting only — originals untouched
        idx   = _ds_index(full.count_rate)
        t_plt = full.time_unix[idx]
        cr    = full.count_rate[idx]
        times = _unix_to_dt(t_plt)

        with plt.style.context(_STYLE):
            fig, ax = plt.subplots(figsize=(14, 4), layout="constrained")
            ax.plot(times, cr, color=color, lw=0.8,
                    label=f"HEL1OS {lc.detector} "
                          f"({full.e_low_kev:.0f}–{full.e_high_kev:.0f} keV)")
            ax.fill_between(times, 0, cr, color=color, alpha=0.1)
            if flare_times_unix:
                first = True
                for t_unix in flare_times_unix:
                    ax.axvline(datetime.fromtimestamp(t_unix, tz=timezone.utc),
                               color=_FLARE_COLOR, lw=1.2, ls="--", alpha=0.85,
                               label="Flare onset" if first else None)
                    first = False
            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Count Rate (cts s⁻¹)")
            ax.set_title(title or f"HEL1OS {lc.detector} — {lc.date_str}")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.legend(loc="upper right")
            ax.set_xlim(times[0], times[-1])
            _add_note(ax, "Hard X-ray (8–150 keV): impulsive phase (Benz 2008 §2.2).")
            fig.autofmt_xdate()
        return self._save(fig, out)

    def plot_multi_band(self, lc, title: str = "") -> Path:
        out = self._out / f"helios_multiband_{lc.detector}_{lc.date_str}.png"
        if self._should_skip(out):
            return out

        n    = len(lc.bands)
        cmap = mpl.colormaps["cool"].resampled(n)

        with plt.style.context(_STYLE):
            fig, axes = plt.subplots(
                n, 1, figsize=(14, 2.5 * n), sharex=True, layout="constrained"
            )
            if n == 1:
                axes = [axes]
            for i, (band, ax) in enumerate(zip(lc.bands, axes)):
                idx   = _ds_index(band.count_rate)
                times = _unix_to_dt(band.time_unix[idx])
                cr    = band.count_rate[idx]
                color = cmap(i)
                ax.plot(times, cr, color=color, lw=0.8)
                ax.fill_between(times, 0, cr, color=color, alpha=0.15)
                ax.set_ylabel("cts s⁻¹", fontsize=8)
                ax.text(0.01, 0.88,
                        f"{band.e_low_kev:.0f}–{band.e_high_kev:.0f} keV",
                        transform=ax.transAxes, fontsize=8, color=color,
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
            axes[-1].set_xlabel("Time (UTC)")
            fig.suptitle(
                title or f"HEL1OS {lc.detector} Multi-band — {lc.date_str}",
                fontsize=11,
            )
            fig.autofmt_xdate()
        return self._save(fig, out)

    def plot_detector_overlay(
        self,
        lcs:      list,
        date_str: str = "",
        title:    str = "",
    ) -> Path:
        tag = date_str or (lcs[0].date_str if lcs else "")
        out = self._out / f"helios_detector_overlay_{tag}.png"
        if self._should_skip(out):
            return out

        with plt.style.context(_STYLE):
            fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
            for lc in lcs:
                full  = lc.full_band
                color = _DETECTOR_COLORS.get(lc.detector, "#888888")
                idx   = _ds_index(full.count_rate)
                times = _unix_to_dt(full.time_unix[idx])
                ax.plot(times, full.count_rate[idx], color=color, lw=0.8,
                        label=f"HEL1OS {lc.detector} "
                              f"({full.e_low_kev:.0f}–{full.e_high_kev:.0f} keV)")
            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Count Rate (cts s⁻¹)")
            ax.set_title(title or f"HEL1OS Detector Overlay — {date_str}")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.legend(loc="upper right")
            _add_note(ax, "CdTe vs CZT cross-calibration check.")
            fig.autofmt_xdate()
        return self._save(fig, out)

    def plot_hardness_ratio(
        self,
        czt_lc:   object,
        cdte_lc:  object,
        date_str: str = "",
        title:    str = "",
    ) -> Path:
        tag = date_str or czt_lc.date_str
        out = self._out / (
            f"helios_hardness_ratio_{cdte_lc.detector}_{czt_lc.detector}_{tag}.png"
        )
        if self._should_skip(out):
            return out

        czt_full  = czt_lc.full_band
        cdte_full = cdte_lc.full_band
        cdte_cr   = np.interp(
            czt_full.time_unix, cdte_full.time_unix, cdte_full.count_rate
        )
        czt_cr = czt_full.count_rate

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = cdte_cr / (czt_cr + 1e-6)
        ratio = np.where(np.isfinite(ratio), ratio, np.nan)

        # Downsample all three derived arrays with the same index so traces
        # stay temporally aligned.  czte_cr is derived from czt_full so its
        # length matches czt_cr; one shared index suffices.
        idx   = _ds_index(czt_cr)
        t_plt = czt_full.time_unix[idx]
        times = _unix_to_dt(t_plt)

        with plt.style.context(_STYLE):
            fig, (ax_top, ax_bot) = plt.subplots(
                2, 1, figsize=(14, 6), sharex=True, layout="constrained",
            )
            ax_top.plot(times, cdte_cr[idx],
                        color=_DETECTOR_COLORS.get(cdte_lc.detector, "#E84040"),
                        lw=0.8, label=f"{cdte_lc.detector} (high-E)")
            ax_top.plot(times, czt_cr[idx],
                        color=_DETECTOR_COLORS.get(czt_lc.detector, "#3A7FD5"),
                        lw=0.8, label=f"{czt_lc.detector} (low-E)")
            ax_top.set_ylabel("Count Rate (cts s⁻¹)")
            ax_top.legend(loc="upper right")
            ax_top.set_title(
                title or f"HEL1OS Hardness Ratio "
                         f"({cdte_lc.detector}/{czt_lc.detector}) — {date_str}"
            )
            ax_bot.plot(times, ratio[idx], color="#9B59B6", lw=0.8)
            ax_bot.axhline(1.0, color="gray", lw=0.6, ls=":")
            ax_bot.set_ylabel("Hardness Ratio\n(CdTe / CZT)")
            ax_bot.set_xlabel("Time (UTC)")
            _add_note(
                ax_bot,
                "Ratio > 1 → harder spectrum.\nRises in impulsive phase (Benz 2008 §5.2).",
            )
            ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax_bot.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            fig.autofmt_xdate()
        return self._save(fig, out)