"""
ml/eda/helios_eda/helios_plotter.py
=====================================
Publication-quality visualisations specific to HEL1OS data.

What it produces
----------------
1. Multi-band stacked light-curve panel (one subplot per energy band).
2. Single-band daily light curve with flare onset overlays.
3. Detector-comparison overlay: CdTe1 vs CdTe2, CZT1 vs CZT2 on one axis.
4. Hardness-ratio plot: CZT (low energy) vs CdTe (high energy) ratio over time.

Physics context (Benz 2008)
---------------------------
HEL1OS covers 8–150 keV in two detector families:
  CdTe1/CdTe2  — 20–150 keV, cadmium telluride (radiation-hard)
  CZT1/CZT2    — 8–60 keV, cadmium zinc telluride (better resolution)

The hardness ratio HXR_high / HXR_low (CdTe / CZT) provides an in-band
spectral index proxy analogous to the soft-hard-soft behaviour described
in Benz (2008) §5.2 but within the hard X-ray regime.  During the impulsive
phase the spectrum hardens (ratio rises); during the decay it softens.

Usage
-----
    from ml.eda.helios_eda.helios_plotter import HEL1OSPlotter

    plotter = HEL1OSPlotter(output_dir=cfg.paths.cache)
    plotter.plot_helios_day(helios_lc, flare_times_unix=[f.onset_unix for f in flares])
    plotter.plot_detector_overlay(helios_lc, detectors=["CdTe1", "CdTe2"])
    plotter.plot_hardness_ratio(czt_lc, cdte_lc, date_str="20260621")
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

_DETECTOR_COLORS = {
    "CdTe1": "#E84040",
    "CdTe2": "#C03030",
    "CZT1":  "#3A7FD5",
    "CZT2":  "#2A5FA5",
}
_FLARE_COLOR = "#FF8C00"   # amber — flare onset marker


def _unix_to_dt(time_unix: npt.NDArray[np.float64]) -> list[datetime]:
    return [datetime.fromtimestamp(t, tz=timezone.utc) for t in time_unix]


class HEL1OSPlotter:
    """
    Generates and saves diagnostic plots for HEL1OS hard X-ray data.

    Parameters
    ----------
    output_dir : Path
        Directory where PNG files are written.  Created automatically.
    show : bool
        If True, call plt.show() after each plot (disable in batch/CI mode).
    """

    def __init__(self, output_dir: Path | str, show: bool = False) -> None:
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._show = show
        logger.info("HEL1OSPlotter → output_dir=%s", self._out)

    # ── Public API ────────────────────────────────────────────────────────────

    def plot_helios_day(
        self,
        lc: "HEL1OSLightCurve",              # noqa: F821
        flare_times_unix: Optional[list[float]] = None,
        title: str = "",
    ) -> Path:
        """
        Plot a full-day HEL1OS light curve for one detector (full band),
        with optional flare onset markers.

        Parameters
        ----------
        lc : HEL1OSLightCurve
            Loaded by HEL1OSLoader.
        flare_times_unix : list of float, optional
            Unix timestamps of flare onsets to mark.
        title : str
            Subplot title override.

        Returns
        -------
        Path — saved PNG file path.
        """
        full  = lc.full_band
        color = _DETECTOR_COLORS.get(lc.detector, "#3A7FD5")
        times = _unix_to_dt(full.time_unix)

        with plt.style.context(_STYLE):
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(times, full.count_rate, color=color, lw=0.8,
                    label=f"HEL1OS {lc.detector} ({full.e_low_kev:.0f}–{full.e_high_kev:.0f} keV)")
            ax.fill_between(times, 0, full.count_rate, color=color, alpha=0.1)

            if flare_times_unix:
                first = True
                for t_unix in flare_times_unix:
                    t_dt = datetime.fromtimestamp(t_unix, tz=timezone.utc)
                    ax.axvline(
                        t_dt, color=_FLARE_COLOR, lw=1.2, ls="--", alpha=0.85,
                        label="Flare onset" if first else None,
                    )
                    first = False

            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Count Rate (cts s⁻¹)")
            ax.set_title(title or f"HEL1OS {lc.detector} — {lc.date_str}")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.legend(loc="upper right")
            ax.set_xlim(times[0], times[-1])
            self._add_physics_note(
                ax,
                "Hard X-ray (8–150 keV): non-thermal bremsstrahlung.\n"
                "Peaks in impulsive phase (Benz 2008, §2.2).",
            )
            fig.autofmt_xdate()
            plt.tight_layout()

            out = self._out / f"helios_{lc.detector}_{lc.date_str}_lc.png"
            fig.savefig(out, bbox_inches="tight")
            logger.info("Saved: %s", out)
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    def plot_multi_band(
        self,
        lc: "HEL1OSLightCurve",   # noqa: F821
        title: str = "",
    ) -> Path:
        """
        Stacked subplot: one panel per energy band, shared time axis.

        Re-uses the LightCurvePlotter.plot_helios_bands() layout for
        consistency, but written here so HEL1OSPlotter is self-contained.

        Parameters
        ----------
        lc : HEL1OSLightCurve
        title : str

        Returns
        -------
        Path
        """
        n = len(lc.bands)
        with plt.style.context(_STYLE):
            fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
            if n == 1:
                axes = [axes]

            cmap = plt.cm.get_cmap("cool", n)
            for i, (band, ax) in enumerate(zip(lc.bands, axes)):
                times = _unix_to_dt(band.time_unix)
                color = cmap(i)
                ax.plot(times, band.count_rate, color=color, lw=0.8)
                ax.fill_between(times, 0, band.count_rate, color=color, alpha=0.15)
                label = f"{band.e_low_kev:.0f}–{band.e_high_kev:.0f} keV"
                ax.set_ylabel("cts s⁻¹", fontsize=8)
                ax.text(0.01, 0.88, label, transform=ax.transAxes,
                        fontsize=8, color=color,
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
            axes[-1].set_xlabel("Time (UTC)")
            fig.suptitle(
                title or f"HEL1OS {lc.detector} Multi-band — {lc.date_str}",
                y=1.01, fontsize=11,
            )
            fig.autofmt_xdate()
            plt.tight_layout()

            out = self._out / f"helios_multiband_{lc.detector}_{lc.date_str}.png"
            fig.savefig(out, bbox_inches="tight")
            logger.info("Saved: %s", out)
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    def plot_detector_overlay(
        self,
        lcs:       list,    # list[HEL1OSLightCurve]
        date_str:  str = "",
        title:     str = "",
    ) -> Path:
        """
        Overlay count-rate light curves for multiple HEL1OS detectors on one
        set of axes — the primary cross-calibration diagnostic.

        Comparing CdTe1 vs CdTe2 (or CZT1 vs CZT2) should give nearly
        identical count rates for the same event; systematic offsets indicate
        gain drift or dead-time correction issues.

        Parameters
        ----------
        lcs : list of HEL1OSLightCurve — one per detector to overlay
        date_str : str
        title : str

        Returns
        -------
        Path
        """
        with plt.style.context(_STYLE):
            fig, ax = plt.subplots(figsize=(14, 5))
            for lc in lcs:
                full  = lc.full_band
                color = _DETECTOR_COLORS.get(lc.detector, "#888888")
                times = _unix_to_dt(full.time_unix)
                label = (
                    f"HEL1OS {lc.detector} "
                    f"({full.e_low_kev:.0f}–{full.e_high_kev:.0f} keV)"
                )
                ax.plot(times, full.count_rate, color=color, lw=0.8, label=label)

            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Count Rate (cts s⁻¹)")
            ax.set_title(
                title or f"HEL1OS Detector Overlay — {date_str}\n"
                "CdTe1 vs CdTe2 and CZT1 vs CZT2 cross-calibration check"
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.legend(loc="upper right")
            self._add_physics_note(
                ax,
                "Systematic offset between paired detectors\n"
                "flags gain drift (Aditya-L1 calibration doc §3).",
            )
            fig.autofmt_xdate()
            plt.tight_layout()

            tag = date_str or (lcs[0].date_str if lcs else "")
            out = self._out / f"helios_detector_overlay_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            logger.info("Saved: %s", out)
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    def plot_hardness_ratio(
        self,
        czt_lc:   "HEL1OSLightCurve",   # noqa: F821  — low-energy reference
        cdte_lc:  "HEL1OSLightCurve",   # noqa: F821  — high-energy channel
        date_str: str = "",
        title:    str = "",
    ) -> Path:
        """
        Hard X-ray hardness ratio: CdTe (high energy) / CZT (low energy)
        as a function of time.

        During the impulsive phase the spectrum hardens (ratio increases),
        mirroring the soft-hard-soft pattern but within the HXR band
        (Benz 2008 §5.2).  This plot is the HXR analogue of the
        spectral-index evolution used in the FlareFeatures module.

        Parameters
        ----------
        czt_lc  : HEL1OSLightCurve — the CZT detector (lower-energy reference)
        cdte_lc : HEL1OSLightCurve — the CdTe detector (higher-energy numerator)
        date_str : str
        title    : str

        Returns
        -------
        Path
        """
        czt_full  = czt_lc.full_band
        cdte_full = cdte_lc.full_band

        # Interpolate CdTe onto CZT time grid if cadences differ
        cdte_cr = np.interp(
            czt_full.time_unix, cdte_full.time_unix, cdte_full.count_rate
        )
        czt_cr  = czt_full.count_rate

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = cdte_cr / (czt_cr + 1e-6)
        ratio = np.where(np.isfinite(ratio), ratio, np.nan)

        times = _unix_to_dt(czt_full.time_unix)

        with plt.style.context(_STYLE):
            fig, (ax_top, ax_bot) = plt.subplots(
                2, 1, figsize=(14, 6), sharex=True,
                gridspec_kw={"hspace": 0.08},
            )

            # Top: CdTe and CZT count rates
            ax_top.plot(times, cdte_cr, color=_DETECTOR_COLORS.get(cdte_lc.detector, "#E84040"),
                        lw=0.8, label=f"{cdte_lc.detector} (high-E)")
            ax_top.plot(times, czt_cr,  color=_DETECTOR_COLORS.get(czt_lc.detector, "#3A7FD5"),
                        lw=0.8, label=f"{czt_lc.detector} (low-E)")
            ax_top.set_ylabel("Count Rate (cts s⁻¹)")
            ax_top.legend(loc="upper right")
            ax_top.set_title(
                title or
                f"HEL1OS Hardness Ratio "
                f"({cdte_lc.detector}/{czt_lc.detector}) — {date_str}"
            )

            # Bottom: hardness ratio
            ax_bot.plot(times, ratio, color="#9B59B6", lw=0.8)
            ax_bot.axhline(1.0, color="gray", lw=0.6, ls=":")
            ax_bot.set_ylabel("Hardness Ratio\n(CdTe / CZT)")
            ax_bot.set_xlabel("Time (UTC)")
            self._add_physics_note(
                ax_bot,
                "Ratio > 1 → harder spectrum (more high-E emission).\n"
                "Rises in impulsive phase; falls in decay (Benz 2008 §5.2).",
            )
            ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax_bot.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            fig.autofmt_xdate()
            plt.tight_layout()

            tag = date_str or czt_lc.date_str
            out = self._out / f"helios_hardness_ratio_{cdte_lc.detector}_{czt_lc.detector}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            logger.info("Saved: %s", out)
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _add_physics_note(ax: plt.Axes, note: str) -> None:
        ax.text(
            0.99, 0.04, note,
            transform=ax.transAxes,
            fontsize=6.5, color="#555555",
            ha="right", va="bottom",
            bbox=dict(facecolor="white", alpha=0.55, edgecolor="none"),
        )