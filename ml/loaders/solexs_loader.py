"""
ml/loaders/solexs_loader.py
============================
Loader for SoLEXS Level-1 FITS data products.

Data structures understood by this module
-----------------------------------------
From the SoLEXS Data Analysis Guide (Section 2.1):

  LC File  — RATE extension
      Columns: TIME (Unix seconds), COUNTS (cts/s)
      Cadence: 1 second, covers full day

  PI File  — SPECTRUM extension (Type-II, one row per second)
      Columns: TSTART, TELAPSE, SPEC_NUM, CHANNEL, COUNTS, EXPOSURE
      n_channels = 340 (168 individual + 172 paired above ch 168)

  GTI File — GTI extension
      Columns: START, STOP  (Unix seconds)

All files are gzip-compressed FITS (.gz).

Usage
-----
    from ml.loaders.solexs_loader import SoLEXSLoader
    loader = SoLEXSLoader(cfg)
    lc   = loader.load_lc("AL1_SOLEXS_20260621_SDD2_L1.lc.gz")
    gti  = loader.load_gti("AL1_SOLEXS_20260621_SDD2_L1.gti.gz")
    pi   = loader.load_pi("AL1_SOLEXS_20260621_SDD2_L1.pi.gz")
    day  = loader.load_day("20260621", detector="SDD2")
"""

from __future__ import annotations

import gzip
import io
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from astropy.time import Time

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Result dataclasses — what every loader call returns
# ─────────────────────────────────────────────────────────────

@dataclass
class LightCurve:
    """
    SoLEXS Level-1 light curve for one detector, one day.

    Attributes
    ----------
    time_unix : ndarray of float64
        Unix timestamps (seconds since 1970-01-01 UTC).
    count_rate : ndarray of float64
        Counts per second in the 2–22 keV band.
    detector : str
        'SDD1' or 'SDD2'.
    date_str : str
        Observation date in YYYYMMDD format.
    header : dict
        Primary FITS header keywords (metadata).
    """
    time_unix:  npt.NDArray[np.float64]
    count_rate: npt.NDArray[np.float64]
    detector:   str
    date_str:   str
    header:     dict = field(default_factory=dict, repr=False)

    @property
    def n_samples(self) -> int:
        return len(self.time_unix)

    @property
    def duration_sec(self) -> float:
        if self.n_samples < 2:
            return 0.0
        return float(self.time_unix[-1] - self.time_unix[0])

    @property
    def time_isot(self) -> list[str]:
        """Return ISO-T UTC strings for each timestamp."""
        t = Time(self.time_unix, format="unix", scale="utc")
        return t.isot.tolist()

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"LightCurve({self.detector}, date={self.date_str}, "
            f"n={self.n_samples}, duration={self.duration_sec:.0f}s)"
        )


@dataclass
class GoodTimeIntervals:
    """
    GTI file — valid observation windows.

    Attributes
    ----------
    starts : ndarray of float64
        Start times in Unix seconds.
    stops : ndarray of float64
        Stop times in Unix seconds.
    """
    starts: npt.NDArray[np.float64]
    stops:  npt.NDArray[np.float64]

    @property
    def n_intervals(self) -> int:
        return len(self.starts)

    @property
    def total_duration_sec(self) -> float:
        return float(np.sum(self.stops - self.starts))

    def mask_for(self, time_unix: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """
        Return a boolean mask: True where time_unix falls inside any GTI.

        Parameters
        ----------
        time_unix : ndarray
            Timestamps to test.
        """
        mask = np.zeros(len(time_unix), dtype=bool)
        for start, stop in zip(self.starts, self.stops):
            mask |= (time_unix >= start) & (time_unix <= stop)
        return mask


@dataclass
class PISpectrum:
    """
    SoLEXS Level-1 Type-II PI spectral file.

    Each row is a 1-second spectrum.

    Attributes
    ----------
    tstart : ndarray of float64
        Start time of each spectrum in Unix seconds.
    counts : ndarray of float64, shape (n_spectra, 340)
        Counts in each of the 340 energy channels.
    channel_energies_kev : ndarray of float64, shape (340,)
        Central energy of each channel in keV.
        Channels 1–168: ~47.6 eV bin width.
        Channels 169–340: ~95.2 eV bin width.
    exposure_sec : ndarray of float64
        Effective exposure time per spectrum (deadtime not yet applied).
    detector : str
        'SDD1' or 'SDD2'.
    """
    tstart:               npt.NDArray[np.float64]   # (N,)
    counts:               npt.NDArray[np.float64]   # (N, 340)
    channel_energies_kev: npt.NDArray[np.float64]   # (340,)
    exposure_sec:         npt.NDArray[np.float64]   # (N,)
    detector:             str

    @property
    def n_spectra(self) -> int:
        return self.counts.shape[0]

    @property
    def total_count_rate(self) -> npt.NDArray[np.float64]:
        """Sum over all channels, divided by exposure — total cts/s."""
        return self.counts.sum(axis=1) / np.clip(self.exposure_sec, 1e-6, None)


@dataclass
class SoLEXSDayData:
    """
    All Level-1 products for one detector on one day.

    The GTI mask is already applied to the light curve.
    """
    lc:       LightCurve
    gti:      GoodTimeIntervals
    pi:       Optional[PISpectrum]
    detector: str
    date_str: str

    @property
    def lc_gti_masked(self) -> LightCurve:
        """
        Light curve filtered to valid GTI windows only.
        Returns a new LightCurve with only GTI-valid rows.
        """
        mask = self.gti.mask_for(self.lc.time_unix)
        return LightCurve(
            time_unix  = self.lc.time_unix[mask],
            count_rate = self.lc.count_rate[mask],
            detector   = self.lc.detector,
            date_str   = self.lc.date_str,
            header     = self.lc.header,
        )


# ─────────────────────────────────────────────────────────────
# Channel energy calibration
# (SoLEXS manual Section 4.5.1 — two regimes)
# ─────────────────────────────────────────────────────────────

def _build_channel_energies(n_channels: int = 340) -> npt.NDArray[np.float64]:
    """
    Compute the centre energy (keV) for each SoLEXS channel.

    Channel structure from the manual:
      - Channels 1–168:   bin width = 47.6 eV   (0.0476 keV)
      - Channels 169–340: bin width = 95.2 eV   (0.0952 keV)

    The spectrum starts at ~2.0 keV (lowest detectable energy).
    """
    BIN_NARROW_KEV = 0.0476   # 47.6 eV
    BIN_WIDE_KEV   = 0.0952   # 95.2 eV
    E_START_KEV    = 2.0      # nominal low-energy threshold

    BREAK_CHANNEL  = 168      # 1-indexed

    energies = np.empty(n_channels, dtype=np.float64)
    e = E_START_KEV
    for i in range(n_channels):
        ch = i + 1   # 1-indexed
        bw = BIN_NARROW_KEV if ch <= BREAK_CHANNEL else BIN_WIDE_KEV
        energies[i] = e + bw / 2.0
        e += bw
    return energies


SOLEXS_CHANNEL_ENERGIES_KEV: npt.NDArray[np.float64] = _build_channel_energies(340)


# ─────────────────────────────────────────────────────────────
# Main Loader Class
# ─────────────────────────────────────────────────────────────

class SoLEXSLoader:
    """
    Loads SoLEXS Level-1 FITS data products.

    Parameters
    ----------
    data_dir : Path or str
        Directory that contains the .lc.gz / .pi.gz / .gti.gz files.
        Typically  ``ml/data/raw/solexs/``.
    """

    def __init__(self, data_dir: Path | str) -> None:
        self._data_dir = Path(data_dir)
        logger.info("SoLEXSLoader initialised | data_dir=%s", self._data_dir)

    # ── internal helpers ──────────────────────────────────────

    def _open_fits(self, filepath: Path) -> fits.HDUList:
        """
        Open a FITS file — handles both plain .fits and .fits.gz.
        Astropy handles gzip transparently via the 'mode' arg.
        """
        logger.debug("Opening FITS file: %s", filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"FITS file not found: {filepath}")
        return fits.open(str(filepath), memmap=False)

    def _resolve(self, filename):
        p = Path(filename)

        if p.exists():
            return p

        return self._data_dir / p

    # ── Public API ────────────────────────────────────────────

    def load_lc(
        self,
        filename: str | Path,
        detector: str = "SDD2",
        date_str: str = "",
    ) -> LightCurve:
        """
        Load a SoLEXS Level-1 light curve FITS file.

        The LC file has a RATE extension with columns:
            TIME   — Unix seconds
            COUNTS — counts/second in 2–22 keV

        Parameters
        ----------
        filename : str or Path
            Filename or full path of the .lc or .lc.gz file.
        detector : str
            'SDD1' or 'SDD2'.
        date_str : str
            YYYYMMDD string (auto-extracted from filename if empty).

        Returns
        -------
        LightCurve
        """
        filepath = self._resolve(filename)
        if not date_str:
            date_str = self._extract_date(filepath.name)

        with self._open_fits(filepath) as hdul:
            hdul.info()
            # Try RATE extension first; fall back to first BINTABLE
            ext = self._find_extension(hdul, ["RATE", "LC", "LIGHTCURVE"])
            data = hdul[ext].data
            header = dict(hdul[0].header)

            time_col  = self._find_column(data.names, ["TIME", "MJD", "TSTART"])
            count_col = self._find_column(data.names, ["COUNTS", "CTR", "COUNT_RATE", "RATE"])

            time_raw  = data[time_col].astype(np.float64)
            counts    = data[count_col].astype(np.float64)

        # If time is MJD rather than Unix, convert it
        if time_raw.max() < 1e8:   # MJD values are ~60000, Unix ~1.7e9
            time_unix = Time(time_raw, format="mjd", scale="utc").unix
            logger.debug("Converted MJD → Unix time")
        else:
            time_unix = time_raw

        lc = LightCurve(
            time_unix  = time_unix,
            count_rate = counts,
            detector   = detector,
            date_str   = date_str,
            header     = header,
        )
        logger.info("Loaded LC: %s", lc)
        return lc

    def load_gti(self, filename: str | Path) -> GoodTimeIntervals:
        """
        Load a SoLEXS Level-1 GTI FITS file.

        GTI extension columns: START, STOP in Unix seconds.

        Parameters
        ----------
        filename : str or Path

        Returns
        -------
        GoodTimeIntervals
        """
        filepath = self._resolve(filename)

        with self._open_fits(filepath) as hdul:
            ext  = self._find_extension(hdul, ["GTI", "STDGTI"])
            data = hdul[ext].data

            start_col = self._find_column(data.names, ["START", "TSTART"])
            stop_col  = self._find_column(data.names, ["STOP",  "TSTOP"])

            starts = data[start_col].astype(np.float64)
            stops  = data[stop_col].astype(np.float64)

        # Same MJD → Unix conversion logic
        if starts.max() < 1e8:
            starts = Time(starts, format="mjd", scale="utc").unix
            stops  = Time(stops,  format="mjd", scale="utc").unix

        gti = GoodTimeIntervals(starts=starts, stops=stops)
        logger.info(
            "Loaded GTI: %d intervals, total %.0f s",
            gti.n_intervals, gti.total_duration_sec,
        )
        return gti

    def load_pi(
        self,
        filename: str | Path,
        detector: str = "SDD2",
    ) -> PISpectrum:
        """
        Load a SoLEXS Level-1 Type-II PI spectral file.

        SPECTRUM extension columns (per manual Section 2.1.3):
            TSTART, TELAPSE, SPEC_NUM, CHANNEL, COUNTS, EXPOSURE

        Parameters
        ----------
        filename : str or Path
        detector : str

        Returns
        -------
        PISpectrum
        """
        filepath = self._resolve(filename)

        with self._open_fits(filepath) as hdul:
            ext  = self._find_extension(hdul, ["SPECTRUM", "SPEC"])
            data = hdul[ext].data

            tstart_col   = self._find_column(data.names, ["TSTART", "TIME"])
            counts_col   = self._find_column(data.names, ["COUNTS"])
            exposure_col = self._find_column(data.names, ["EXPOSURE", "TELAPSE"])

            tstart   = data[tstart_col].astype(np.float64)
            counts   = np.array([row for row in data[counts_col]], dtype=np.float64)
            exposure = data[exposure_col].astype(np.float64)

        # MJD → Unix
        if tstart.max() < 1e8:
            tstart = Time(tstart, format="mjd", scale="utc").unix

        pi = PISpectrum(
            tstart               = tstart,
            counts               = counts,
            channel_energies_kev = SOLEXS_CHANNEL_ENERGIES_KEV.copy(),
            exposure_sec         = exposure,
            detector             = detector,
        )
        logger.info(
            "Loaded PI: detector=%s, n_spectra=%d, shape=%s",
            detector, pi.n_spectra, str(pi.counts.shape),
        )
        return pi

    def load_day(
        self,
        date_str: str,
        detector: str = "SDD2",
        load_pi: bool = True,
    ) -> SoLEXSDayData:
        """
        Load all three Level-1 products for one detector on one day.

        Automatically constructs filenames from the PRADAN naming convention:
            AL1_SOLEXS_YYYYMMDD_SDDn_L1.{lc,gti,pi}.gz

        Parameters
        ----------
        date_str : str
            Date in YYYYMMDD format, e.g. '20260621'.
        detector : str
            'SDD1' or 'SDD2'.
        load_pi : bool
            Whether to load the PI spectrogram (large file).

        Returns
        -------
        SoLEXSDayData
        """
        n = detector[-1]  # '1' or '2'
        base = f"AL1_SOLEXS_{date_str}_SDD{n}_L1"

        lc_file  = self._data_dir / f"{base}.lc.gz"
        gti_file = self._data_dir / f"{base}.gti.gz"
        pi_file  = self._data_dir / f"{base}.pi.gz"

        # Try compressed files first, then fall back to uncompressed

        lc_file = self._data_dir / f"{base}.lc.gz"
        if not lc_file.exists():
            lc_file = self._data_dir / f"{base}.lc"

        gti_file = self._data_dir / f"{base}.gti.gz"
        if not gti_file.exists():
            gti_file = self._data_dir / f"{base}.gti"

        pi_file = self._data_dir / f"{base}.pi.gz"
        if not pi_file.exists():
            pi_file = self._data_dir / f"{base}.pi"
        lc  = self.load_lc(lc_file, detector=detector, date_str=date_str)
        gti = self.load_gti(gti_file)
        pi_data: Optional[PISpectrum] = None
        if load_pi and pi_file.exists():
            pi_data = self.load_pi(pi_file, detector=detector)
        elif load_pi:
            logger.warning("PI file not found, skipping: %s", pi_file)

        return SoLEXSDayData(
            lc       = lc,
            gti      = gti,
            pi       = pi_data,
            detector = detector,
            date_str = date_str,
        )

    # ── Static helpers ────────────────────────────────────────

    @staticmethod
    def _extract_date(filename: str) -> str:
        """Extract YYYYMMDD from filename like AL1_SOLEXS_20260621_SDD2_L1.lc.gz"""
        parts = filename.split("_")
        for part in parts:
            if len(part) == 8 and part.isdigit():
                return part
        return "unknown"

    @staticmethod
    def _find_extension(hdul: fits.HDUList, candidates: list[str]) -> int | str:
        """
        Return the index or name of the first extension whose EXTNAME
        matches one of the candidates (case-insensitive).
        Falls back to extension index 1 (first data extension).
        """
        names_upper = [c.upper() for c in candidates]
        for i, hdu in enumerate(hdul):
            extname = hdu.name.upper()
            if extname in names_upper:
                return i
        logger.warning(
            "None of %s found in HDU list %s — falling back to ext 1",
            candidates, [h.name for h in hdul],
        )
        return 1

    @staticmethod
    def _find_column(col_names: list[str], candidates: list[str]) -> str:
        """
        Return the first column name from col_names that matches
        any of the candidate names (case-insensitive).
        """
        upper_map = {c.upper(): c for c in col_names}
        for cand in candidates:
            if cand.upper() in upper_map:
                return upper_map[cand.upper()]
        raise KeyError(
            f"None of {candidates} found in columns {col_names}"
        )
