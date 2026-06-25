"""
ml/loaders/helios_loader.py
============================
Loader for HEL1OS Level-1 FITS data products.

Data structures understood by this module
-----------------------------------------
From the HEL1OS Data Analysis User Manual (Section 2):

  Light Curve files — one per detector (CdTe1, CdTe2, CZT1, CZT2)
      lightcurve_cdte1.fits, lightcurve_czte1.fits, etc.
      Extension: BINTABLE
      Key columns: MJD, ISOT, CTR (count rate cts/sec), STAT_ERR
      Cadence: 1 second
      Multiple energy-band extensions per file (5 bands + full range)

      CdTe bands: 5-20, 20-30, 30-40, 40-60 keV, full (1.8-90 keV)
      CZT  bands: 20-40, 40-60, 60-80, 80-150 keV, full (18-160 keV)

  Spectral files (Type-II PHA) — 20 second cadence
      hel1os_cdte_spectra_cdte1.fits, etc.
      CdTe: 511 channels | CZT: 341 channels

  Event list: evt.fits (4 extensions, one per detector)

  GTI files: gticdte1.fits, gticzte1.fits, etc.

Usage
-----
    from ml.loaders.helios_loader import HEL1OSLoader
    loader = HEL1OSLoader(data_dir)
    lc = loader.load_lc("lightcurve_czt1.fits", detector="CZT1")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from astropy.time import Time

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Constants from the manual
# ─────────────────────────────────────────────────────────────

# Energy bands for light curves (Section 4, item 5 of manual)
CDTE_BANDS_KEV: list[tuple[float, float]] = [
    (5.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 60.0), (1.8, 90.0)
]
CZT_BANDS_KEV: list[tuple[float, float]] = [
    (20.0, 40.0), (40.0, 60.0), (60.0, 80.0), (80.0, 150.0), (18.0, 160.0)
]

CDTE_N_CHANNELS = 511
CZT_N_CHANNELS  = 341

# Minimum energies recommended for analysis (Section 1, Table)
CDTE_ANALYSIS_MIN_KEV = 9.5
CZT_ANALYSIS_MIN_KEV  = 35.0


# ─────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────

@dataclass
class HEL1OSBand:
    """Count rate in a single HEL1OS energy band."""
    e_low_kev:  float
    e_high_kev: float
    time_unix:  npt.NDArray[np.float64]
    count_rate: npt.NDArray[np.float64]
    stat_err:   npt.NDArray[np.float64]

    def __repr__(self) -> str:
        return (
            f"HEL1OSBand({self.e_low_kev:.0f}–{self.e_high_kev:.0f} keV, "
            f"n={len(self.time_unix)})"
        )


@dataclass
class HEL1OSLightCurve:
    """
    HEL1OS light curve — all energy bands for one detector.

    Attributes
    ----------
    bands : list of HEL1OSBand
        One entry per energy band extension in the FITS file.
    detector : str
        'CdTe1', 'CdTe2', 'CZT1', or 'CZT2'.
    date_str : str
        YYYYMMDD observation date.
    header : dict
        Primary header metadata.
    """
    bands:    list[HEL1OSBand]
    detector: str
    date_str: str
    header:   dict = field(default_factory=dict, repr=False)

    @property
    def full_band(self) -> HEL1OSBand:
        """Return the band with the widest energy range (last in list)."""
        return self.bands[-1]

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    @property
    def hardness_ratio(self) -> Optional[npt.NDArray[np.float64]]:
        """
        Compute hardness ratio = hard_band / soft_band.
        Only meaningful for CZT detectors with at least 2 bands.
        For CZT: band[2] (60-80 keV) / band[0] (20-40 keV).
        For CdTe: band[2] (30-40 keV) / band[0] (5-20 keV).
        """
        if self.n_bands < 3:
            return None
        soft = self.bands[0].count_rate
        hard = self.bands[2].count_rate
        denom = np.where(soft > 0, soft, np.nan)
        return hard / denom

    def __repr__(self) -> str:
        return (
            f"HEL1OSLightCurve(detector={self.detector}, "
            f"date={self.date_str}, n_bands={self.n_bands})"
        )


@dataclass
class HEL1OSSpectrum:
    """
    HEL1OS Level-1 Type-II PHA spectral file.

    Attributes
    ----------
    tstart_unix : ndarray, shape (N,)
        Start time of each 20-second spectrum in Unix seconds.
    counts : ndarray, shape (N, n_channels)
    stat_err : ndarray, shape (N, n_channels)
    n_channels : int
        511 for CdTe, 341 for CZT.
    detector : str
    """
    tstart_unix: npt.NDArray[np.float64]
    counts:      npt.NDArray[np.float64]
    stat_err:    npt.NDArray[np.float64]
    n_channels:  int
    detector:    str

    @property
    def n_spectra(self) -> int:
        return self.counts.shape[0]


@dataclass
class HEL1OSGoodTimeInterval:
    """GTI for one HEL1OS detector."""
    starts:   npt.NDArray[np.float64]
    stops:    npt.NDArray[np.float64]
    detector: str

    def mask_for(self, time_unix: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        mask = np.zeros(len(time_unix), dtype=bool)
        for s, e in zip(self.starts, self.stops):
            mask |= (time_unix >= s) & (time_unix <= e)
        return mask


# ─────────────────────────────────────────────────────────────
# Main Loader
# ─────────────────────────────────────────────────────────────

class HEL1OSLoader:
    """
    Loads HEL1OS Level-1 FITS products.

    Parameters
    ----------
    data_dir : Path or str
        Directory containing the extracted HEL1OS data files.
        After extracting the Level-1 zip, point this at:
        HLS_YYYYMMDD_.../opl1/prod/
        or the specific cdte/ or czt/ subdirectory.
    """

    # Detector family classification
    _CDTE_DETECTORS = {"CdTe1", "CdTe2"}
    _CZT_DETECTORS  = {"CZT1",  "CZT2"}

    def __init__(self, data_dir: Path | str) -> None:
        self._data_dir = Path(data_dir)
        logger.info("HEL1OSLoader initialised | data_dir=%s", self._data_dir)

    # ── internal helpers ──────────────────────────────────────

    def _open_fits(self, filepath: Path) -> fits.HDUList:
        if not filepath.exists():
            raise FileNotFoundError(f"HEL1OS FITS not found: {filepath}")
        logger.debug("Opening: %s", filepath)
        return fits.open(str(filepath), memmap=False)

    def _resolve(self, filename: str | Path) -> Path:
        p = Path(filename)
        return p if p.is_absolute() else self._data_dir / p

    @staticmethod
    def _mjd_to_unix(mjd: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return Time(mjd, format="mjd", scale="utc").unix

    @staticmethod
    def _is_mjd(values: npt.NDArray[np.float64]) -> bool:
        """Heuristic: MJD ~60000, Unix ~1.7e9."""
        return float(np.nanmax(values)) < 1e8

    # ── Public API ────────────────────────────────────────────

    def load_lc(
        self,
        filename: str | Path,
        detector: str,
        date_str: str = "",
    ) -> HEL1OSLightCurve:
        """
        Load a HEL1OS light curve FITS file.

        The file contains multiple BINTABLE extensions, one per energy band.
        Each extension has EXTNAME like:
          CDTE1_LC_BAND_5.00KEV_TO_20.00KEV
          CZT1_LC_BAND_20.00KEV_TO_40.00KEV

        Columns: MJD, ISOT, CTR (cts/sec), STAT_ERR

        Parameters
        ----------
        filename : str or Path
        detector : str
            e.g. 'CdTe1', 'CZT1'
        date_str : str
            YYYYMMDD

        Returns
        -------
        HEL1OSLightCurve
        """
        filepath = self._resolve(filename)
        bands: list[HEL1OSBand] = []

        with self._open_fits(filepath) as hdul:
            header = dict(hdul[0].header)

            for hdu in hdul[1:]:   # skip primary HDU
                if not isinstance(hdu, fits.BinTableHDU):
                    continue

                extname = hdu.name.upper()

                # Parse energy limits from extension name
                e_low, e_high = self._parse_band_from_extname(extname)

                data = hdu.data
                col_names = [c.upper() for c in data.names]

                # Time column
                if "MJD" in col_names:
                    mjd = data["MJD"].astype(np.float64)
                    time_unix = self._mjd_to_unix(mjd)
                elif "TIME" in col_names:
                    t = data["TIME"].astype(np.float64)
                    time_unix = self._mjd_to_unix(t) if self._is_mjd(t) else t
                else:
                    logger.warning("No time column in ext %s, skipping", extname)
                    continue

                # Count rate
                ctr_col  = self._pick_col(data.names, ["CTR", "COUNT_RATE", "RATE", "COUNTS"])
                err_col  = self._pick_col(data.names, ["STAT_ERR", "ERROR", "ERR"], required=False)

                count_rate = data[ctr_col].astype(np.float64)
                stat_err   = (
                    data[err_col].astype(np.float64)
                    if err_col else np.sqrt(np.abs(count_rate))
                )

                bands.append(HEL1OSBand(
                    e_low_kev  = e_low,
                    e_high_kev = e_high,
                    time_unix  = time_unix,
                    count_rate = count_rate,
                    stat_err   = stat_err,
                ))

        if not bands:
            raise ValueError(f"No valid band extensions found in {filepath}")

        lc = HEL1OSLightCurve(
            bands    = bands,
            detector = detector,
            date_str = date_str,
            header   = header,
        )
        logger.info("Loaded HEL1OS LC: %s", lc)
        return lc

    def load_spectra(
        self,
        filename: str | Path,
        detector: str,
    ) -> HEL1OSSpectrum:
        """
        Load a HEL1OS Type-II PHA spectral file.

        FITS structure (Appendix B.2 of manual):
          SPECTRUM extension — BINTABLE
          Columns: SPEC_NUM, CHANNEL, COUNTS, STAT_ERR, ROWID,
                   TSTART (relative secs), TSTOP, EXPOSURE

        Parameters
        ----------
        filename : str or Path
        detector : str

        Returns
        -------
        HEL1OSSpectrum
        """
        filepath = self._resolve(filename)
        n_ch = CDTE_N_CHANNELS if detector in self._CDTE_DETECTORS else CZT_N_CHANNELS

        with self._open_fits(filepath) as hdul:
            ext  = self._find_ext(hdul, ["SPECTRUM"])
            data = hdul[ext].data
            pri_header = dict(hdul[0].header)

            # Absolute start time from primary header
            tstart_keyword = pri_header.get("TSTART", 0.0)

            tstart_rel = data["TSTART"].astype(np.float64)   # relative secs
            counts     = np.array(list(data["COUNTS"]), dtype=np.float64)
            stat_err   = np.array(list(data["STAT_ERR"]), dtype=np.float64)

            # Convert relative → absolute Unix time
            # TSTART in header is MJD
            if self._is_mjd(np.array([tstart_keyword])):
                t0_unix = Time(tstart_keyword, format="mjd", scale="utc").unix
            else:
                t0_unix = float(tstart_keyword)

            tstart_unix = t0_unix + tstart_rel

        spec = HEL1OSSpectrum(
            tstart_unix = tstart_unix,
            counts      = counts,
            stat_err    = stat_err,
            n_channels  = n_ch,
            detector    = detector,
        )
        logger.info(
            "Loaded HEL1OS spectrum: %s, n_spectra=%d, shape=%s",
            detector, spec.n_spectra, str(counts.shape),
        )
        return spec

    def load_gti(
        self,
        filename: str | Path,
        detector: str,
    ) -> HEL1OSGoodTimeInterval:
        """
        Load a HEL1OS GTI FITS file (gticdte1.fits, gticzt1.fits, etc.)

        GTI extension columns: START, STOP (MJD or relative secs).
        From the manual: GTI files span the entire data-dump duration
        for Level-1 data.
        """
        filepath = self._resolve(filename)

        with self._open_fits(filepath) as hdul:
            ext  = self._find_ext(hdul, ["GTI", "STDGTI"])
            data = hdul[ext].data

            start_col = self._pick_col(data.names, ["START", "TSTART"])
            stop_col  = self._pick_col(data.names, ["STOP",  "TSTOP"])

            starts = data[start_col].astype(np.float64)
            stops  = data[stop_col].astype(np.float64)

        if self._is_mjd(starts):
            starts = self._mjd_to_unix(starts)
            stops  = self._mjd_to_unix(stops)

        gti = HEL1OSGoodTimeInterval(
            starts   = starts,
            stops    = stops,
            detector = detector,
        )
        logger.info(
            "Loaded HEL1OS GTI: %s, %d intervals",
            detector, gti.starts.shape[0],
        )
        return gti

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _parse_band_from_extname(extname: str) -> tuple[float, float]:
        """
        Parse energy limits from extension names like:
          CDTE1_LC_BAND_5.00KEV_TO_20.00KEV
          CZT1_LC_BAND_18.00KEV_TO_160.00KEV
        Returns (e_low, e_high) in keV, or (0.0, 0.0) if unparseable.
        """
        try:
            # Find numeric tokens after 'BAND'
            tokens = extname.split("_")
            nums = []
            for tok in tokens:
                tok_clean = tok.replace("KEV", "").replace("KV", "")
                try:
                    nums.append(float(tok_clean))
                except ValueError:
                    pass
            if len(nums) >= 2:
                return nums[-2], nums[-1]
        except Exception:
            pass
        return 0.0, 0.0

    @staticmethod
    def _find_ext(hdul: fits.HDUList, candidates: list[str]) -> int:
        for i, hdu in enumerate(hdul):
            if hdu.name.upper() in [c.upper() for c in candidates]:
                return i
        logger.warning("Extensions %s not found; using ext 1", candidates)
        return 1

    @staticmethod
    def _pick_col(
        col_names: list[str],
        candidates: list[str],
        required: bool = True,
    ) -> Optional[str]:
        upper_map = {c.upper(): c for c in col_names}
        for cand in candidates:
            if cand.upper() in upper_map:
                return upper_map[cand.upper()]
        if required:
            raise KeyError(f"None of {candidates} found in {col_names}")
        return None
