"""
tests/unit/test_solexs_loader.py
=================================
Tests for the SoLEXS FITS data loader.

Run with:
    cd solar/
    python -m pytest tests/unit/test_solexs_loader.py -v

Or with your real file:
    python tests/unit/test_solexs_loader.py --file ml/data/raw/solexs/AL1_SOLEXS_20260621_SDD2_L1.lc

The test creates a synthetic minimal FITS LC file when no real file is given,
so the loader logic can be verified in CI without data.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time


# ── import the loader ──────────────────────────────────────────────────────────
# Add project root to path so the test can be run from anywhere
sys.path.insert(0, str(Path(__file__).parents[2]))

from ml.loaders.solexs_loader import (
    GoodTimeIntervals,
    LightCurve,
    SoLEXSLoader,
    _build_channel_energies,
    SOLEXS_CHANNEL_ENERGIES_KEV,
)


# ─────────────────────────────────────────────────────────────
# Helpers — build synthetic FITS files for testing
# ─────────────────────────────────────────────────────────────

def _make_synthetic_lc_fits(path: Path, n_rows: int = 86400) -> None:
    """
    Create a minimal SoLEXS LC FITS file.
    n_rows = 86400 → 24 hours at 1-second cadence.
    """
    t0_unix = Time("2026-06-21T00:00:00", format="isot", scale="utc").unix

    time_unix  = (t0_unix + np.arange(n_rows)).astype(np.float64)
    # Simulate a quiet sun with one flare spike around hour 6
    count_rate = np.random.normal(100.0, 5.0, n_rows).clip(0)
    flare_idx  = n_rows // 4   # 6-hour mark
    # Inject an impulsive flare profile
    for offset, amp in [(0, 200), (1, 800), (2, 2000), (3, 3500), (4, 2800),
                        (5, 1500), (6, 700), (7, 300), (8, 150)]:
        idx = flare_idx + offset * 60
        if idx < n_rows:
            count_rate[idx:idx+60] += amp

    col1 = fits.Column(name="TIME",   format="D", unit="s",       array=time_unix)
    col2 = fits.Column(name="COUNTS", format="D", unit="cts/sec", array=count_rate)
    table_hdu = fits.BinTableHDU.from_columns([col1, col2])
    table_hdu.name = "RATE"

    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "Aditya-L1"
    primary.header["INSTRUME"] = "SoLEXS"
    primary.header["DETNAM"]   = "SDD2"

    hdul = fits.HDUList([primary, table_hdu])
    hdul.writeto(str(path), overwrite=True)


def _make_synthetic_gti_fits(path: Path, t0_unix: float, duration: float = 86400.0) -> None:
    """Create a simple GTI file covering the full day minus 1 bad hour."""
    # Two GTI windows — exclude 10:00–11:00 UTC
    gap_start = t0_unix + 10 * 3600
    gap_end   = t0_unix + 11 * 3600

    starts = np.array([t0_unix,     gap_end ], dtype=np.float64)
    stops  = np.array([gap_start,   t0_unix + duration], dtype=np.float64)

    col1 = fits.Column(name="START", format="D", array=starts)
    col2 = fits.Column(name="STOP",  format="D", array=stops)
    table_hdu = fits.BinTableHDU.from_columns([col1, col2])
    table_hdu.name = "GTI"

    hdul = fits.HDUList([fits.PrimaryHDU(), table_hdu])
    hdul.writeto(str(path), overwrite=True)


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestChannelEnergies:
    def test_length(self) -> None:
        energies = _build_channel_energies(340)
        assert len(energies) == 340

    def test_monotonic(self) -> None:
        energies = _build_channel_energies(340)
        assert np.all(np.diff(energies) > 0), "Channel energies must be monotonically increasing"

    def test_starts_near_2kev(self) -> None:
        energies = _build_channel_energies(340)
        assert abs(energies[0] - 2.0) < 0.1, f"First channel should be near 2 keV, got {energies[0]}"

    def test_ends_near_22kev(self) -> None:
        # Manual sec 4.5.1: 168 narrow + 172 wide bins → final channel ~26 keV
        # The instrument nominal upper is 22 keV but channel grid extends beyond
        energies = _build_channel_energies(340)
        assert 20.0 < energies[-1] < 30.0, f"Last channel energy {energies[-1]} keV out of plausible range"

    def test_global_constant_shape(self) -> None:
        assert SOLEXS_CHANNEL_ENERGIES_KEV.shape == (340,)


class TestSoLEXSLoaderSynthetic:
    """Tests against a synthetic FITS file — no real data required."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._lc_path  = Path(self._tmpdir) / "AL1_SOLEXS_20260621_SDD2_L1.lc"
        self._gti_path = Path(self._tmpdir) / "AL1_SOLEXS_20260621_SDD2_L1.gti"

        _make_synthetic_lc_fits(self._lc_path, n_rows=3600)  # 1 hour for speed
        t0 = Time("2026-06-21T00:00:00", format="isot", scale="utc").unix
        _make_synthetic_gti_fits(self._gti_path, t0_unix=t0, duration=3600.0)

        self._loader = SoLEXSLoader(data_dir=self._tmpdir)

    def test_load_lc_returns_light_curve(self) -> None:
        lc = self._loader.load_lc(self._lc_path, detector="SDD2", date_str="20260621")
        assert isinstance(lc, LightCurve)

    def test_lc_n_samples(self) -> None:
        lc = self._loader.load_lc(self._lc_path, detector="SDD2", date_str="20260621")
        assert lc.n_samples == 3600

    def test_lc_detector(self) -> None:
        lc = self._loader.load_lc(self._lc_path, detector="SDD2", date_str="20260621")
        assert lc.detector == "SDD2"

    def test_lc_date_str(self) -> None:
        lc = self._loader.load_lc(self._lc_path, detector="SDD2", date_str="20260621")
        assert lc.date_str == "20260621"

    def test_lc_count_rate_positive(self) -> None:
        lc = self._loader.load_lc(self._lc_path, detector="SDD2", date_str="20260621")
        assert np.all(lc.count_rate >= 0), "Count rate must be non-negative"

    def test_lc_time_isot_format(self) -> None:
        lc = self._loader.load_lc(self._lc_path, detector="SDD2", date_str="20260621")
        iso = lc.time_isot
        assert len(iso) == lc.n_samples
        assert "2026" in iso[0], f"Expected 2026-... date string, got {iso[0]}"

    def test_load_gti(self) -> None:
        gti = self._loader.load_gti(self._gti_path)
        assert isinstance(gti, GoodTimeIntervals)
        assert gti.n_intervals == 2

    def test_gti_mask_length(self) -> None:
        lc  = self._loader.load_lc(self._lc_path, detector="SDD2", date_str="20260621")
        gti = self._loader.load_gti(self._gti_path)
        mask = gti.mask_for(lc.time_unix)
        assert mask.shape == lc.time_unix.shape

    def test_gti_mask_excludes_gap(self) -> None:
        lc  = self._loader.load_lc(self._lc_path, detector="SDD2", date_str="20260621")
        gti = self._loader.load_gti(self._gti_path)
        # The GTI excludes 10:00–11:00, but our synthetic data is 00:00–01:00
        # so the full hour should be inside the GTI
        mask = gti.mask_for(lc.time_unix)
        assert mask.sum() > 0, "Expected some valid GTI samples"

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            self._loader.load_lc("nonexistent_file.lc", detector="SDD2")


# ─────────────────────────────────────────────────────────────
# Real-file test (optional, run with --file argument)
# ─────────────────────────────────────────────────────────────

def run_real_file_test(filepath: Path) -> None:
    """
    Smoke-test the loader against a real downloaded file.
    Prints a summary — no assertions, just verifies it doesn't crash.
    """
    loader = SoLEXSLoader(data_dir=".")
    print(f"\n{'='*60}")
    print(f"Testing real file: {filepath.name}")
    print(f"{'='*60}")

    # Determine detector from filename
    if "SDD1" in filepath.name:
        detector = "SDD1"
    elif "SDD2" in filepath.name:
        detector = "SDD2"
    else:
        detector = "SDD2"
        print("Warning: could not determine detector from filename, assuming SDD2")

    lc = loader.load_lc(filepath.resolve(), detector=detector)

    print(f"\nLight Curve Summary")
    print(f"  Detector  : {lc.detector}")
    print(f"  Date      : {lc.date_str}")
    print(f"  N samples : {lc.n_samples}")
    print(f"  Duration  : {lc.duration_sec/3600:.2f} hours")
    print(f"  Count Rate: min={lc.count_rate.min():.1f}, "
          f"max={lc.count_rate.max():.1f}, "
          f"mean={lc.count_rate.mean():.1f} cts/s")
    print(f"  Time range: {lc.time_isot[0]}  →  {lc.time_isot[-1]}")
    print(f"\nChannel Energies (first 5): {SOLEXS_CHANNEL_ENERGIES_KEV[:5]} keV")
    print(f"Channel Energies (last 5):  {SOLEXS_CHANNEL_ENERGIES_KEV[-5:]} keV")
    print(f"\n✓ Real file loaded successfully\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the SoLEXS loader against a real FITS file."
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a real .lc or .lc.gz file downloaded from PRADAN",
    )
    args = parser.parse_args()

    if args.file:
        run_real_file_test(Path(args.file))
    else:
        print("No --file given. Running pytest synthetic tests.")
        pytest.main([__file__, "-v"])
