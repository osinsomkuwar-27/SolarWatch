"""
ml/eda/light_curve_plotter.py
==============================
Publication-quality light curve visualisations for SoLEXS and HEL1OS data.

What it produces
----------------
1. Single-instrument daily light curve (count rate vs. UTC time).
2. Dual-panel plot: SoLEXS soft X-ray + HEL1OS hard X-ray on a shared time axis.
   This is the "Neupert effect" diagnostic described in Benz (2008), Section 2.4 —
   the soft X-ray flux should track the time-integral of the hard X-ray flux.
3. Multi-band HEL1OS plot showing each energy band stacked vertically.
4. Spectrogram (time vs. channel vs. counts) from the SoLEXS PI file.

Physics context (Benz 2008)
---------------------------
- Hard X-rays (HEL1OS, 8–150 keV) peak during the *impulsive phase* and are
  produced by non-thermal bremsstrahlung from electron beams hitting the chromosphere.
- Soft X-rays (SoLEXS, 2–22 keV) peak later, during the *gradual/decay phase*,
  because they trace the hot evaporated plasma.
- The derivative of the soft X-ray flux should match the hard X-ray flux
  (Neupert effect). Plotting both together is the first sanity check after
  loading real data.

Usage
-----
    from ml.loaders.solexs_loader import SoLEXSLoader, LightCurve
    from ml.loaders.helios_loader import HEL1OSLoader, HEL1OSLightCurve
    from ml.eda.light_curve_plotter import LightCurvePlotter

    plotter = LightCurvePlotter(output_dir=cfg.paths.cache)
    plotter.plot_solexs_day(lc, gti_mask=gti.mask_for(lc.time_unix))
    plotter.plot_dual_panel(solexs_lc, helios_lc)
    plotter.plot_helios_bands(helios_lc)
    plotter.plot_neupert(solexs_lc, helios_lc)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# ── Style constants ───────────────────────────────────────────────────────────
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

_SOLEXS_COLOR  = "#E84040"   # deep red  — soft X-ray
_HELIOS_COLOR  = "#3A7FD5"   # steel blue — hard X-ray
_GTI_SHADE     = "#C8F0C8"   # pale green — GTI shading
_FLARE_COLOR   = "#FF8C00"   # amber     — flare onset marker


def _unix_to_dt(time_unix: npt.NDArray[np.float64]) -> list[datetime]:
    """Convert Unix timestamps to timezone-aware datetime objects (UTC)."""
    return [datetime.fromtimestamp(t, tz=timezone.utc) for t in time_unix]


class LightCurvePlotter:
    """
    Generates and saves diagnostic plots for Aditya-L1 data.

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
        logger.info("LightCurvePlotter → output_dir=%s", self._out)

    # ── Public API ────────────────────────────────────────────────────────────

    def plot_solexs_day(
        self,
        lc: "LightCurve",  # noqa: F821  (avoid circular import)
        gti_mask: Optional[npt.NDArray[np.bool_]] = None,
        flare_times_unix: Optional[list[float]] = None,
        title: str = "",
    ) -> Path:
        """
        Plot a full-day SoLEXS light curve with optional GTI shading
        and flare onset markers.

        Parameters
        ----------
        lc : LightCurve
            Loaded by SoLEXSLoader.
        gti_mask : ndarray of bool, optional
            Same length as lc.time_unix; True = valid GTI window.
        flare_times_unix : list of float, optional
            Unix timestamps of detected flare onsets to mark.
        title : str
            Subplot title override.

        Returns
        -------
        Path
            Saved PNG file path.
        """
        with plt.style.context(_STYLE):
            fig, ax = plt.subplots(figsize=(14, 4))

            times = _unix_to_dt(lc.time_unix)
            cr    = lc.count_rate

            # GTI shading — shade invalid (non-GTI) periods
            if gti_mask is not None:
                invalid_mask = ~gti_mask
                self._shade_invalid(ax, lc.time_unix, invalid_mask)

            # Main light curve
            ax.plot(times, cr, color=_SOLEXS_COLOR, lw=0.8,
                    label=f"SoLEXS {lc.detector} (2–22 keV)")

            # Flare onset markers
            if flare_times_unix:
                for t_unix in flare_times_unix:
                    t_dt = datetime.fromtimestamp(t_unix, tz=timezone.utc)
                    ax.axvline(t_dt, color=_FLARE_COLOR, lw=1.2,
                               ls="--", alpha=0.85, label="Flare onset")

            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Count Rate (cts s⁻¹)")
            ax.set_title(title or f"SoLEXS {lc.detector} — {lc.date_str}")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.legend(loc="upper right")
            ax.set_xlim(times[0], times[-1])
            self._add_physics_note(ax,
                "Soft X-ray (2–22 keV): traces hot thermal plasma.\n"
                "Peaks in gradual/decay phase (Benz 2008, §1.3).")
            fig.autofmt_xdate()
            plt.tight_layout()

            out = self._out / f"solexs_{lc.detector}_{lc.date_str}_lc.png"
            fig.savefig(out, bbox_inches="tight")
            logger.info("Saved: %s", out)
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    def plot_helios_bands(
        self,
        lc: "HEL1OSLightCurve",  # noqa: F821
        title: str = "",
    ) -> Path:
        """
        Plot all HEL1OS energy bands as stacked subplots.

        Each panel shows one energy band.  The bottom panel is the full-band
        (widest energy range) which correlates best with flare hard X-ray flux.

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
            fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n),
                                     sharex=True)
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

            out = self._out / f"helios_{lc.detector}_{lc.date_str}_bands.png"
            fig.savefig(out, bbox_inches="tight")
            logger.info("Saved: %s", out)
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    def plot_dual_panel(
        self,
        solexs_lc: "LightCurve",
        helios_lc: "HEL1OSLightCurve",
        title: str = "",
    ) -> Path:
        """
        Dual-panel: SoLEXS soft X-ray (top) + HEL1OS hard X-ray (bottom)
        on a shared time axis.

        This is the primary visual diagnostic for the Neupert effect:
        hard X-ray bursts during the impulsive phase vs. soft X-ray
        gradual rise (Benz 2008, §2.4 — Figure 12).

        Parameters
        ----------
        solexs_lc : LightCurve
        helios_lc : HEL1OSLightCurve

        Returns
        -------
        Path
        """
        with plt.style.context(_STYLE):
            fig, (ax_soft, ax_hard) = plt.subplots(
                2, 1, figsize=(14, 6), sharex=True,
                gridspec_kw={"hspace": 0.08},
            )

            # ── Soft X-ray (SoLEXS) ──────────────────────────
            t_soft = _unix_to_dt(solexs_lc.time_unix)
            ax_soft.plot(t_soft, solexs_lc.count_rate,
                         color=_SOLEXS_COLOR, lw=0.9,
                         label=f"SoLEXS {solexs_lc.detector} (2–22 keV)")
            ax_soft.set_ylabel("Soft X-ray\n(cts s⁻¹)")
            ax_soft.legend(loc="upper right")
            ax_soft.set_title(
                title or
                f"Aditya-L1 Dual-Band Light Curve — {solexs_lc.date_str}"
            )
            self._add_physics_note(ax_soft,
                "Soft X-ray: thermal plasma 1–30 MK.\nPeaks in gradual phase.")

            # ── Hard X-ray (HEL1OS full band) ────────────────
            full = helios_lc.full_band
            t_hard = _unix_to_dt(full.time_unix)
            ax_hard.plot(t_hard, full.count_rate,
                         color=_HELIOS_COLOR, lw=0.9,
                         label=(
                             f"HEL1OS {helios_lc.detector} "
                             f"({full.e_low_kev:.0f}–{full.e_high_kev:.0f} keV)"
                         ))
            ax_hard.set_ylabel("Hard X-ray\n(cts s⁻¹)")
            ax_hard.set_xlabel("Time (UTC)")
            ax_hard.legend(loc="upper right")
            self._add_physics_note(ax_hard,
                "Hard X-ray: non-thermal bremsstrahlung.\n"
                "Peaks in impulsive phase (Benz 2008, §2.2).")

            ax_hard.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax_hard.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            fig.autofmt_xdate()
            plt.tight_layout()

            out = self._out / (
                f"dual_panel_{solexs_lc.date_str}_"
                f"{solexs_lc.detector}_{helios_lc.detector}.png"
            )
            fig.savefig(out, bbox_inches="tight")
            logger.info("Saved: %s", out)
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    def plot_neupert(
        self,
        solexs_lc: "LightCurve",
        helios_lc: "HEL1OSLightCurve",
        title: str = "",
    ) -> Path:
        """
        Plot the Neupert effect diagnostic (Benz 2008, §2.4).

        Three panels:
          1. Soft X-ray count rate (SoLEXS)
          2. dSXR/dt — derivative of soft X-ray (should match HXR shape)
          3. Hard X-ray count rate (HEL1OS full band)

        If the standard flare model holds, panels 2 and 3 should be
        nearly identical in shape.

        Parameters
        ----------
        solexs_lc : LightCurve
        helios_lc : HEL1OSLightCurve

        Returns
        -------
        Path
        """
        with plt.style.context(_STYLE):
            fig, axes = plt.subplots(3, 1, figsize=(14, 8),
                                     sharex=True,
                                     gridspec_kw={"hspace": 0.12})
            ax_sxr, ax_dsxr, ax_hxr = axes

            # ── Panel 1: SXR ──────────────────────────────────
            t_soft = _unix_to_dt(solexs_lc.time_unix)
            cr_sxr = solexs_lc.count_rate
            ax_sxr.plot(t_soft, cr_sxr, color=_SOLEXS_COLOR, lw=0.9)
            ax_sxr.set_ylabel("SXR\n(cts s⁻¹)")
            ax_sxr.set_title(title or f"Neupert Effect Diagnostic — {solexs_lc.date_str}")

            # ── Panel 2: dSXR/dt ──────────────────────────────
            dt_sec = np.diff(solexs_lc.time_unix)
            dt_sec = np.where(dt_sec > 0, dt_sec, np.nan)
            dsxr   = np.gradient(cr_sxr, solexs_lc.time_unix)
            t_mid  = t_soft   # same length, use central diff

            # Clip negative derivative (cooling) to zero for comparison
            dsxr_pos = np.clip(dsxr, 0, None)
            ax_dsxr.plot(t_mid, dsxr_pos, color="#9B59B6", lw=0.9,
                         label="d(SXR)/dt  [clipped ≥ 0]")
            ax_dsxr.set_ylabel("d(SXR)/dt")
            ax_dsxr.legend(loc="upper right", fontsize=7)
            ax_dsxr.axhline(0, color="gray", lw=0.5, ls=":")

            # ── Panel 3: HXR ──────────────────────────────────
            full   = helios_lc.full_band
            t_hard = _unix_to_dt(full.time_unix)
            ax_hxr.plot(t_hard, full.count_rate, color=_HELIOS_COLOR, lw=0.9,
                        label=f"HXR {full.e_low_kev:.0f}–{full.e_high_kev:.0f} keV")
            ax_hxr.set_ylabel("HXR\n(cts s⁻¹)")
            ax_hxr.set_xlabel("Time (UTC)")
            ax_hxr.legend(loc="upper right")

            # Physics annotation
            ax_hxr.text(
                0.01, 0.05,
                "Neupert effect (Benz 2008 §2.4):\n"
                "F_SXR(t) ∝ ∫ F_HXR dt  →  dF_SXR/dt ∝ F_HXR",
                transform=ax_hxr.transAxes, fontsize=7,
                color="gray", va="bottom",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
            )

            ax_hxr.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax_hxr.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            fig.autofmt_xdate()
            plt.tight_layout()

            out = self._out / f"neupert_{solexs_lc.date_str}.png"
            fig.savefig(out, bbox_inches="tight")
            logger.info("Saved: %s", out)
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    def plot_spectrogram(
        self,
        pi: "PISpectrum",  # noqa: F821
        time_slice: Optional[tuple[float, float]] = None,
        log_scale: bool = True,
        title: str = "",
    ) -> Path:
        """
        Plot a SoLEXS PI spectrogram: time (x) vs. energy channel (y)
        vs. count rate (colour).

        This is the EDA equivalent of Figure 28 in Benz (2008) for radio,
        adapted to soft X-ray spectral data.  Flares appear as bright
        vertical stripes at all channel energies.

        Parameters
        ----------
        pi : PISpectrum
            From SoLEXSLoader.load_pi().
        time_slice : (t_start_unix, t_stop_unix), optional
            Restrict to a sub-interval.
        log_scale : bool
            If True, colour scale is log10(counts+1).
        title : str

        Returns
        -------
        Path
        """
        with plt.style.context(_STYLE):
            counts = pi.counts.copy().astype(float)
            tstart = pi.tstart.copy()
            energies = pi.channel_energies_kev

            # Time slice
            if time_slice:
                mask = (tstart >= time_slice[0]) & (tstart <= time_slice[1])
                tstart = tstart[mask]
                counts = counts[mask]

            times_dt = _unix_to_dt(tstart)

            if log_scale:
                display = np.log10(counts + 1.0)
                cbar_label = "log₁₀(counts + 1)"
            else:
                display = counts
                cbar_label = "Counts"

            fig, ax = plt.subplots(figsize=(14, 5))
            # pcolormesh expects (n_time, n_energy) → transpose → (n_energy, n_time)
            pcm = ax.pcolormesh(
                mdates.date2num(times_dt),
                energies,
                display.T,
                cmap="inferno",
                shading="auto",
            )
            cbar = fig.colorbar(pcm, ax=ax, pad=0.01)
            cbar.set_label(cbar_label)

            ax.set_ylabel("Energy (keV)")
            ax.set_xlabel("Time (UTC)")
            ax.set_title(title or f"SoLEXS PI Spectrogram — {pi.detector}")
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            fig.autofmt_xdate()
            plt.tight_layout()

            suffix = f"_{int(time_slice[0])}_{int(time_slice[1])}" if time_slice else ""
            out = self._out / f"spectrogram_{pi.detector}{suffix}.png"
            fig.savefig(out, bbox_inches="tight")
            logger.info("Saved: %s", out)
            if self._show:
                plt.show()
            plt.close(fig)
        return out

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _shade_invalid(
        ax: plt.Axes,
        time_unix: npt.NDArray[np.float64],
        invalid_mask: npt.NDArray[np.bool_],
    ) -> None:
        """Shade non-GTI regions with a light grey fill."""
        if not invalid_mask.any():
            return
        # Find contiguous blocks
        indices = np.where(np.diff(invalid_mask.astype(int)))[0]
        starts  = [0] if invalid_mask[0] else []
        stops: list[int] = []
        for idx in indices:
            if invalid_mask[idx]:      # transition F→T means end
                stops.append(idx + 1)
            else:                      # transition T→F means start
                starts.append(idx + 1)
        if invalid_mask[-1]:
            stops.append(len(time_unix) - 1)

        for s, e in zip(starts, stops):
            t0 = datetime.fromtimestamp(time_unix[s], tz=timezone.utc)
            t1 = datetime.fromtimestamp(time_unix[e], tz=timezone.utc)
            ax.axvspan(t0, t1, color="#DDDDDD", alpha=0.4, zorder=0)

    @staticmethod
    def _add_physics_note(ax: plt.Axes, note: str) -> None:
        """Add a small physics annotation in the lower-right corner."""
        ax.text(
            0.99, 0.04, note,
            transform=ax.transAxes,
            fontsize=6.5, color="#555555",
            ha="right", va="bottom",
            bbox=dict(facecolor="white", alpha=0.55, edgecolor="none"),
        )
