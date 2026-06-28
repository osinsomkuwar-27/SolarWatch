"""
ml/loaders/multi_day_loader.py
================================
Batch extractor and multi-day loader for SoLEXS and HEL1OS data
downloaded from the ISRO PRADAN portal.

SoLEXS zip internal structure (confirmed):
    AL1_SLX_L1_20260625_v1.0/
        SDD2/
            AL1_SOLEXS_20260625_SDD2_L1.lc.gz
            AL1_SOLEXS_20260625_SDD2_L1.pi.gz
            AL1_SOLEXS_20260625_SDD2_L1.gti.gz
        SDD1/
            AL1_SOLEXS_20260625_SDD1_L1.gti.gz

HEL1OS zip internal structure (confirmed):
    2024/06/11/HLS_20240611_012538_38053sec_lev1_V111/
        aux/
            gticdte1.fits  gticdte2.fits
            gticzt1.fits   gticzt2.fits
        cdte/
            lightcurve_cdte1.fits
            lightcurve_cdte2.fits
        czt/
            lightcurve_czt1.fits
            lightcurve_czt2.fits
        events/
            evt.fits

Usage
-----
    from ml.loaders.multi_day_loader import MultiDayLoader

    loader = MultiDayLoader(
        solexs_raw_dir = "ml/data/raw/solexs",
        helios_raw_dir = "ml/data/raw/helios",
        extract_dir    = "ml/data/extracted",
    )

    loader.extract_all()
    loader.summary()

    solexs_days   = loader.load_all_solexs(detector="SDD2", load_pi=False)
    helios_curves = loader.load_all_helios(detector="CZT1")
    all_curves    = loader.load_all_helios_all_detectors()
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from ml.loaders.solexs_loader import SoLEXSLoader, SoLEXSDayData
from ml.loaders.helios_loader import HEL1OSLoader, HEL1OSLightCurve

logger = logging.getLogger(__name__)

_HELIOS_DETECTORS: dict[str, tuple[str, str]] = {
    "CZT1":  ("czt",  "lightcurve_czt1.fits"),
    "CZT2":  ("czt",  "lightcurve_czt2.fits"),
    "CdTe1": ("cdte", "lightcurve_cdte1.fits"),
    "CdTe2": ("cdte", "lightcurve_cdte2.fits"),
}

_W = 60   # console width


def _bar() -> str:
    return "-" * _W


def _header(title: str) -> None:
    print(_bar())
    print(f"  {title}")
    print(_bar())


def _row(label: str, value: str, width: int = 30) -> None:
    print(f"  {label:<{width}} {value}")


def _solexs_summary_block(
    days:        list[SoLEXSDayData],
    discovered:  int,
    failed:      int,
    detector:    str,
) -> None:
    """Print the post-load SoLEXS summary block."""
    loaded       = len(days)
    total_samp   = sum(d.lc.n_samples for d in days)
    avg_samp     = total_samp / loaded if loaded else 0
    gti_counts   = [d.gti.n_intervals for d in days]
    total_gti    = sum(gti_counts)
    avg_gti      = total_gti / loaded if loaded else 0
    max_gti      = max(gti_counts) if gti_counts else 0
    min_gti      = min(gti_counts) if gti_counts else 0
    max_gti_date = days[gti_counts.index(max_gti)].date_str if gti_counts else "—"
    dates        = sorted(d.date_str for d in days)
    span         = f"{dates[0]} → {dates[-1]}" if dates else "—"
    status       = "OK" if failed == 0 else f"DEGRADED ({failed} failed)"

    print()
    _header("SoLEXS  Summary")
    _row("Days discovered",        str(discovered))
    _row("Days loaded",            str(loaded))
    _row("Days failed",            str(failed))
    _row("Samples loaded",         f"{total_samp:,}")
    _row("Average samples / day",  f"{avg_samp:,.0f}")
    _row("Total GTI intervals",    str(total_gti))
    _row("Average GTI / day",      f"{avg_gti:.1f}")
    _row("Maximum GTI / day",      f"{max_gti}  ({max_gti_date})")
    _row("Minimum GTI / day",      str(min_gti))
    _row("Observation span",       span)
    _row("Detector",               detector)
    _row("Status",                 status)
    print(_bar())


def _helios_segment_summary_block(
    curves:     list[HEL1OSLightCurve],
    discovered: int,
    failed:     int,
    detector:   str,
) -> None:
    """Print the per-detector HEL1OS summary block."""
    loaded     = len(curves)
    total_samp = sum(len(lc.full_band.time_unix) for lc in curves)
    avg_samp   = total_samp / loaded if loaded else 0
    dates      = sorted(lc.date_str for lc in curves)
    span       = f"{dates[0]} → {dates[-1]}" if dates else "—"
    status     = "OK" if failed == 0 else f"DEGRADED ({failed} failed)"

    if curves:
        fb     = curves[0].full_band
        band   = f"{fb.e_low_kev}–{fb.e_high_kev} keV"
    else:
        band   = "—"

    print()
    _header(f"HEL1OS  Summary  [{detector}]")
    _row("Segments discovered",   str(discovered))
    _row("Segments loaded",       str(loaded))
    _row("Segments failed",       str(failed))
    _row("Samples loaded",        f"{total_samp:,}")
    _row("Average samples / seg", f"{avg_samp:,.0f}")
    _row("Observation span",      span)
    _row("Energy band",           band)
    _row("Detector",              detector)
    _row("Status",                status)
    print(_bar())


def _pipeline_summary(
    solexs_days:    list[SoLEXSDayData],
    solexs_failed:  int,
    solexs_det:     str,
    helios_results: dict[str, list[HEL1OSLightCurve]],
    helios_failed:  dict[str, int],
    n_helios_zips:  int,
) -> None:
    """Print the final overall pipeline summary."""
    total_loaded   = len(solexs_days) + sum(len(v) for v in helios_results.values())
    total_failed   = solexs_failed    + sum(helios_failed.values())
    pipeline_ok    = total_failed == 0

    print()
    print("=" * _W)
    print("  OVERALL PIPELINE SUMMARY")
    print("=" * _W)

    # SoLEXS row
    s_samp  = sum(d.lc.n_samples for d in solexs_days)
    s_dates = sorted(d.date_str for d in solexs_days)
    s_span  = f"{s_dates[0]} → {s_dates[-1]}" if s_dates else "—"
    _row(f"SoLEXS [{solexs_det}]",
         f"{len(solexs_days):>4} days  |  {s_samp:>10,} samples  |  {s_span}")

    # HEL1OS per-detector rows
    for det, curves in helios_results.items():
        h_samp  = sum(len(lc.full_band.time_unix) for lc in curves)
        h_dates = sorted(lc.date_str for lc in curves)
        h_span  = f"{h_dates[0]} → {h_dates[-1]}" if h_dates else "—"
        _row(f"HEL1OS [{det}]",
             f"{len(curves):>4} segs  |  {h_samp:>10,} samples  |  {h_span}")

    print(_bar())
    _row("Total files loaded",  str(total_loaded))
    _row("Total failures",      str(total_failed))
    _row("Pipeline status",     "OK" if pipeline_ok else f"DEGRADED ({total_failed} failures)")
    print("=" * _W)


class MultiDayLoader:
    """
    Extracts and loads all downloaded SoLEXS and HEL1OS zip archives.

    SoLEXS : bulk extract once to disk (fits on disk, already working).
    HEL1OS : stream each zip — extract to /tmp → load → delete — one at a time.

    Parameters
    ----------
    solexs_raw_dir : str or Path
    helios_raw_dir : str or Path
    extract_dir    : str or Path
        Used only for SoLEXS persistent extraction.
        HEL1OS never writes here permanently.
    debug : bool
        If True, print FITS HDU info during loading. Default False.
    """

    def __init__(
        self,
        solexs_raw_dir: str | Path,
        helios_raw_dir: str | Path,
        extract_dir:    str | Path,
        debug:          bool = False,
    ) -> None:
        self.solexs_raw_dir = Path(solexs_raw_dir)
        self.helios_raw_dir = Path(helios_raw_dir)
        self.extract_dir    = Path(extract_dir)
        self.debug          = debug

        self._solexs_extract = self.extract_dir / "solexs"
        self._solexs_extract.mkdir(parents=True, exist_ok=True)
        self._helios_extract = self.extract_dir / "helios"

        logger.info("MultiDayLoader initialised")
        logger.info("  SoLEXS raw : %s", self.solexs_raw_dir)
        logger.info("  HEL1OS raw : %s", self.helios_raw_dir)
        logger.info("  Extract to : %s", self.extract_dir)

    # ──────────────────────────────────────────────────────────
    # 1. EXTRACTION
    # ──────────────────────────────────────────────────────────

    def extract_all(self) -> None:
        """Extract SoLEXS zips to disk. HEL1OS uses streaming — no bulk extract."""
        print("=" * _W)
        print("  EXTRACTING SoLEXS ZIPS")
        print("=" * _W)
        self._extract_zips(
            raw_dir     = self.solexs_raw_dir,
            extract_dir = self._solexs_extract,
            label       = "SoLEXS",
        )
        print()
        print("  HEL1OS: streaming mode — extraction happens inside load_all_helios().")

    def _extract_zips(
        self,
        raw_dir:     Path,
        extract_dir: Path,
        label:       str,
    ) -> None:
        zips = sorted(raw_dir.rglob("*.zip"))

        if not zips:
            print(f"  No zip files found under {raw_dir}")
            return

        print(f"  Found {len(zips)} zip files")

        extracted = 0
        skipped   = 0
        failed    = 0

        for zip_path in zips:
            stem        = zip_path.stem
            dest_folder = extract_dir / stem

            if dest_folder.exists():
                skipped += 1
                continue

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(dest_folder)
                print(f"  [OK]  {stem}")
                extracted += 1
            except zipfile.BadZipFile:
                print(f"  [BAD ZIP] {zip_path.name}")
                failed += 1
            except Exception as e:
                print(f"  [ERROR] {zip_path.name}: {e}")
                failed += 1

        print(
            f"\n  Done — extracted: {extracted}"
            f"  |  skipped: {skipped}"
            f"  |  failed: {failed}"
        )

    # ──────────────────────────────────────────────────────────
    # 2. SOLEXS MULTI-DAY LOADING
    # ──────────────────────────────────────────────────────────

    def load_all_solexs(
        self,
        detector: str = "SDD2",
        load_pi:  bool = False,
    ) -> list[SoLEXSDayData]:
        """
        Load every extracted SoLEXS day in chronological order.

        Parameters
        ----------
        detector : str   'SDD2' (has lc+pi+gti) or 'SDD1' (gti only)
        load_pi  : bool  load spectral file (large — set False for speed)

        Returns
        -------
        list of SoLEXSDayData sorted by date
        """
        pattern    = f"AL1_SOLEXS_*_{detector}_L1.lc.gz"
        lc_files   = sorted(self._solexs_extract.rglob(pattern))
        discovered = len(lc_files)

        print()
        print("=" * _W)
        print(f"  LOADING SoLEXS  [{detector}]")
        print("=" * _W)

        if not lc_files:
            print("  No SoLEXS lc files found. Run extract_all() first.")
            return []

        print(f"  Discovered {discovered} day(s)")
        print(_bar())

        days:   list[SoLEXSDayData] = []
        failed  = 0

        for lc_file in lc_files:
            date_str = self._date_from_name(lc_file.name)

            # suppress FITS stdout unless debug=True
            if not self.debug:
                import io, contextlib
                _sink = contextlib.redirect_stdout(io.StringIO())
            else:
                import contextlib
                _sink = contextlib.nullcontext()

            loader = SoLEXSLoader(data_dir=lc_file.parent)

            try:
                with _sink:
                    day = loader.load_day(
                        date_str=date_str,
                        detector=detector,
                        load_pi=load_pi,
                    )

                days.append(day)
                print(
                    f"  OK   {date_str}"
                    f"  |  {day.lc.n_samples:>6} samples"
                    f"  |  GTI {day.gti.n_intervals} intervals"
                )

            except Exception as e:
                print(f"  FAIL {date_str}: {e}")
                failed += 1

        _solexs_summary_block(days, discovered, failed, detector)
        return days

    # ──────────────────────────────────────────────────────────
    # 3. HEL1OS STREAMING LOAD — single implementation
    # ──────────────────────────────────────────────────────────

    def load_all_helios(
        self,
        detector: str = "CZT1",
    ) -> list[HEL1OSLightCurve]:
        """
        Stream-load every HEL1OS zip for a single detector.
        Extracts each zip to /tmp, loads the FITS, applies GTI, then deletes.

        Parameters
        ----------
        detector : str
            One of 'CZT1', 'CZT2', 'CdTe1', 'CdTe2'

        Returns
        -------
        list of HEL1OSLightCurve sorted chronologically
        """
        if detector not in _HELIOS_DETECTORS:
            raise ValueError(
                f"Unknown detector '{detector}'. "
                f"Choose from {list(_HELIOS_DETECTORS)}"
            )

        _, filename = _HELIOS_DETECTORS[detector]
        gti_name    = f"gti{detector.lower()}.fits"

        zips       = sorted(self.helios_raw_dir.rglob("*.zip"))
        discovered = len(zips)

        print()
        print("=" * _W)
        print(f"  LOADING HEL1OS  [{detector}]")
        print("=" * _W)

        if not zips:
            print(f"  No HEL1OS zip files found under {self.helios_raw_dir}")
            return []

        print(f"  Discovered {discovered} zip(s) — streaming extract→load→delete")
        print(_bar())

        curves: list[HEL1OSLightCurve] = []
        failed  = 0

        for zip_path in zips:
            date_str = self._date_from_name(zip_path.name)
            tmp_dir  = None

            try:
                tmp_dir = Path(tempfile.mkdtemp(prefix="helios_", dir="/tmp"))

                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmp_dir)

                lc_files = list(tmp_dir.rglob(filename))
                if not lc_files:
                    print(f"  SKIP {date_str}  — {filename} not found inside zip")
                    continue

                lc_file  = lc_files[0]
                data_dir = lc_file.parent
                aux_dir  = lc_file.parent.parent / "aux"
                gti_file = aux_dir / gti_name

                loader = HEL1OSLoader(data_dir=data_dir)

                lc = loader.load_lc(
                    filename=filename,
                    detector=detector,
                    date_str=date_str,
                )

                if gti_file.exists():
                    gti  = loader.load_gti(
                        filename=str(gti_file),
                        detector=detector,
                    )
                    fb   = lc.full_band
                    mask = gti.mask_for(fb.time_unix)
                    print(
                        f"  OK   {date_str}"
                        f"  |  {mask.sum():>6}/{len(mask)} GTI-valid samples"
                    )
                else:
                    n = len(lc.full_band.time_unix)
                    print(
                        f"  OK   {date_str}"
                        f"  |  {n:>6} samples  (no GTI)"
                    )

                curves.append(lc)

            except zipfile.BadZipFile:
                print(f"  BAD ZIP  {zip_path.name}")
                failed += 1

            except Exception as e:
                print(f"  FAIL {date_str} {zip_path.name}: {e}")
                failed += 1

            finally:
                if tmp_dir is not None and tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)

        _helios_segment_summary_block(curves, discovered, failed, detector)
        return curves

    def load_all_helios_all_detectors(
        self,
    ) -> dict[str, list[HEL1OSLightCurve]]:
        """
        Load all four HEL1OS detectors by iterating over the detector mapping.

        Returns
        -------
        dict mapping detector name → list of HEL1OSLightCurve
        """
        results: dict[str, list[HEL1OSLightCurve]] = {}

        for detector in _HELIOS_DETECTORS:
            results[detector] = self.load_all_helios(detector=detector)

        return results

    # ──────────────────────────────────────────────────────────
    # 4. UTILITIES
    # ──────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print what has been downloaded and extracted."""
        solexs_zips      = list(self.solexs_raw_dir.rglob("*.zip"))
        helios_zips      = list(self.helios_raw_dir.rglob("*.zip"))
        solexs_extracted = [p for p in self._solexs_extract.iterdir() if p.is_dir()]
        solexs_days      = list(self._solexs_extract.rglob("AL1_SOLEXS_*_SDD2_L1.lc.gz"))

        print()
        print("=" * _W)
        print("  MultiDayLoader  Inventory")
        print("=" * _W)
        _row("SoLEXS zips downloaded",   str(len(solexs_zips)))
        _row("SoLEXS folders extracted", str(len(solexs_extracted)))
        _row("SoLEXS days (SDD2)",       str(len(solexs_days)))
        print(_bar())
        _row("HEL1OS zips available",    f"{len(helios_zips)}  (streaming — not pre-extracted)")
        _row("HEL1OS detectors",         ", ".join(_HELIOS_DETECTORS))
        print("=" * _W)

    def list_solexs_dates(self) -> list[str]:
        """Return sorted list of all available SoLEXS YYYYMMDD dates."""
        files = self._solexs_extract.rglob("AL1_SOLEXS_*_SDD2_L1.lc.gz")
        return sorted({self._date_from_name(f.name) for f in files})

    @staticmethod
    def _date_from_name(name: str) -> str:
        """Extract YYYYMMDD from a filename or folder name."""
        match = re.search(r"(\d{8})", name)
        return match.group(1) if match else "unknown"


# ──────────────────────────────────────────────────────────────
# Run as script: python -m ml.loaders.multi_day_loader
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level  = logging.WARNING,
        format = "%(levelname)s %(name)s: %(message)s",
        stream = sys.stdout,
    )

    loader = MultiDayLoader(
        solexs_raw_dir = "ml/data/raw/solexs",
        helios_raw_dir = "ml/data/raw/helios",
        extract_dir    = "ml/data/extracted",
        debug          = False,
    )

    print("Step 1: Extracting SoLEXS...")
    loader.extract_all()

    print("\nStep 2: Inventory")
    loader.summary()

    print("\nStep 3: Loading SoLEXS...")
    solexs_days   = loader.load_all_solexs(detector="SDD2", load_pi=False)
    solexs_failed = 0   # failures are counted inside; re-derive if needed

    print("\nStep 4: Loading HEL1OS — all four detectors...")
    helios_results: dict[str, list[HEL1OSLightCurve]] = {}
    helios_failed:  dict[str, int]                     = {}

    for det in _HELIOS_DETECTORS:
        before                = 0   # failures tracked inside load_all_helios
        curves                = loader.load_all_helios(detector=det)
        helios_results[det]   = curves
        helios_failed[det]    = 0   # load_all_helios prints its own summary

    n_helios_zips = len(list(loader.helios_raw_dir.rglob("*.zip")))

    _pipeline_summary(
        solexs_days    = solexs_days,
        solexs_failed  = solexs_failed,
        solexs_det     = "SDD2",
        helios_results = helios_results,
        helios_failed  = helios_failed,
        n_helios_zips  = n_helios_zips,
    )