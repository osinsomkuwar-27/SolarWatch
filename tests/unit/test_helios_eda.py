"""
tests/unit/test_helios_eda.py
==============================
Unit tests for the HEL1OS EDA module — all pass without real Aditya-L1 data.

Run from solar/:
    python -m pytest tests/unit/test_helios_eda.py -v

What is tested
--------------
1.  FlareDetector detects injected synthetic HXR flares
2.  HEL1OSEDAStatistics.compute() returns HEL1OSEDAReport
3.  n_samples matches input length
4.  Basic statistics are sane
5.  class_weights are all positive
6.  save() writes valid JSON with correct structure
7.  plot_distributions() saves PNG files
8.  plot_detector_comparison() saves a PNG file
9.  HEL1OSPlotter.plot_helios_day() creates a PNG
10. HEL1OSPlotter.plot_multi_band() creates a PNG (duck-typed HEL1OSLightCurve)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pytest

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Synthetic HXR data factory ────────────────────────────────────────────────

def _make_synthetic_hxr_lc(
    duration_sec: int = 3600,
    cadence_sec:  int = 1,
    n_flares:     int = 2,
    seed:         int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic HEL1OS-like light curve: Poisson background + impulsive flares.

    HXR rise timescale is shorter (~10 s) than SXR (~20 s), matching
    Benz (2008) §2.2.
    """
    rng = np.random.default_rng(seed)
    n   = duration_sec // cadence_sec
    t   = np.arange(n, dtype=np.float64) * cadence_sec + 1.75e9
    cr  = rng.poisson(30, size=n).astype(np.float64)

    spacing = n // (n_flares + 1)
    for k in range(n_flares):
        onset = spacing * (k + 1)
        amp, rise, decay = 400.0, 10, 300
        for j in range(n):
            dt = j - onset
            if dt >= 0:
                cr[j] += amp * np.exp(-dt / decay) * (1 - np.exp(-dt / max(rise, 1)))
    return t, cr


def _make_detector():
    from ml.eda.flare_detector import FlareDetector
    return FlareDetector(
        onset_sigma        = 3.0,
        min_rise_bins      = 3,
        smooth_window_sec  = 5.0,
        peak_window_sec    = 200.0,
        min_refractory_sec = 300.0,
    )


# ── Duck-typed HEL1OS objects for plotter tests ───────────────────────────────

@dataclass
class _FakeBand:
    time_unix:   np.ndarray
    count_rate:  np.ndarray
    e_low_kev:   float
    e_high_kev:  float


@dataclass
class _FakeHEL1OSLC:
    detector:  str
    date_str:  str
    bands:     List[_FakeBand]

    @property
    def full_band(self) -> _FakeBand:
        return self.bands[-1]


def _make_fake_lc(detector: str = "CdTe1") -> _FakeHEL1OSLC:
    t, cr = _make_synthetic_hxr_lc(duration_sec=3600, n_flares=1)
    bands = [
        _FakeBand(time_unix=t, count_rate=cr * 0.6, e_low_kev=20, e_high_kev=60),
        _FakeBand(time_unix=t, count_rate=cr * 0.4, e_low_kev=60, e_high_kev=150),
        _FakeBand(time_unix=t, count_rate=cr,        e_low_kev=20, e_high_kev=150),
    ]
    return _FakeHEL1OSLC(detector=detector, date_str="20260621", bands=bands)


# ── 1. FlareDetector on HXR data ─────────────────────────────────────────────

class TestFlareDetectorOnHXR:

    def test_detects_at_least_one_flare(self) -> None:
        t, cr  = _make_synthetic_hxr_lc(n_flares=2)
        flares = _make_detector().detect(t, cr, "CdTe1", "20260621")
        assert len(flares) >= 1

    def test_flare_event_attributes_sane(self) -> None:
        t, cr  = _make_synthetic_hxr_lc(n_flares=1)
        flares = _make_detector().detect(t, cr, "CdTe1", "20260621")
        assert len(flares) >= 1
        f = flares[0]
        assert f.onset_unix < f.peak_unix
        assert f.peak_count_rate > 30
        assert f.detector == "CdTe1"

    def test_goes_class_in_valid_set(self) -> None:
        t, cr  = _make_synthetic_hxr_lc(n_flares=2)
        flares = _make_detector().detect(t, cr, "CZT1", "20260621")
        for f in flares:
            assert f.flare_class in (0, 1, 2, 3)


# ── 2–8. HEL1OSEDAStatistics ──────────────────────────────────────────────────

class TestHEL1OSEDAStatistics:

    @staticmethod
    def _setup(tmp_path: Path):
        from ml.eda.helios_eda.helios_statistics import HEL1OSEDAStatistics
        t, cr  = _make_synthetic_hxr_lc(n_flares=2)
        flares = _make_detector().detect(t, cr, "CdTe1", "20260621")
        stats  = HEL1OSEDAStatistics(output_dir=tmp_path)
        return t, cr, flares, stats

    def test_compute_returns_report(self, tmp_path: Path) -> None:
        from ml.eda.helios_eda.helios_statistics import HEL1OSEDAReport
        t, cr, flares, stats = self._setup(tmp_path)
        report = stats.compute(t, cr, flares, "CdTe1", "20260621")
        assert isinstance(report, HEL1OSEDAReport)

    def test_n_samples_correct(self, tmp_path: Path) -> None:
        t, cr, flares, stats = self._setup(tmp_path)
        report = stats.compute(t, cr, flares, "CdTe1", "20260621")
        assert report.n_samples == len(t)

    def test_statistics_sane(self, tmp_path: Path) -> None:
        t, cr, flares, stats = self._setup(tmp_path)
        report = stats.compute(t, cr, flares, "CdTe1", "20260621")
        assert report.min_cr >= 0
        assert report.mean_cr > 0
        assert report.max_cr > report.mean_cr
        assert report.p50_cr <= report.p99_cr

    def test_class_weights_all_positive(self, tmp_path: Path) -> None:
        t, cr, flares, stats = self._setup(tmp_path)
        report = stats.compute(t, cr, flares, "CdTe1", "20260621")
        for cls, w in report.class_weights.items():
            assert w > 0, f"class {cls} weight={w} must be > 0"

    def test_save_creates_valid_json(self, tmp_path: Path) -> None:
        t, cr, flares, stats = self._setup(tmp_path)
        report = stats.compute(t, cr, flares, "CdTe1", "20260621")
        path   = stats.save(report, tag="20260621")
        assert path.exists()
        data   = json.loads(path.read_text())
        assert data["detector"]  == "CdTe1"
        assert data["n_samples"] == len(t)
        assert "class_weights"   in data

    def test_plot_distributions_creates_pngs(self, tmp_path: Path) -> None:
        t, cr, flares, stats = self._setup(tmp_path)
        paths = stats.plot_distributions(t, cr, flares, tag="20260621", detector="CdTe1")
        assert len(paths) >= 2
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 500

    def test_plot_detector_comparison_creates_png(self, tmp_path: Path) -> None:
        from ml.eda.helios_eda.helios_statistics import HEL1OSEDAStatistics
        t, cr  = _make_synthetic_hxr_lc(n_flares=2)
        stats  = HEL1OSEDAStatistics(output_dir=tmp_path)
        # Build two reports for two detectors
        flares = _make_detector().detect(t, cr, "CdTe1", "20260621")
        r1     = stats.compute(t, cr, flares, "CdTe1", "20260621")
        r2     = stats.compute(t, cr, flares, "CZT1",  "20260621")
        path   = stats.plot_detector_comparison([r1, r2], tag="20260621")
        assert path.exists()
        assert path.stat().st_size > 500


# ── 9–10. HEL1OSPlotter ──────────────────────────────────────────────────────

class TestHEL1OSPlotter:

    def test_plot_helios_day_creates_file(self, tmp_path: Path) -> None:
        from ml.eda.helios_eda.helios_plotter import HEL1OSPlotter
        lc  = _make_fake_lc("CdTe1")
        out = HEL1OSPlotter(output_dir=tmp_path, show=False).plot_helios_day(lc)
        assert out.exists()
        assert out.suffix == ".png"
        assert out.stat().st_size > 1000

    def test_plot_multi_band_creates_file(self, tmp_path: Path) -> None:
        from ml.eda.helios_eda.helios_plotter import HEL1OSPlotter
        lc  = _make_fake_lc("CdTe1")
        out = HEL1OSPlotter(output_dir=tmp_path, show=False).plot_multi_band(lc)
        assert out.exists()
        assert out.suffix == ".png"

    def test_output_filename_has_detector_and_date(self, tmp_path: Path) -> None:
        from ml.eda.helios_eda.helios_plotter import HEL1OSPlotter
        lc  = _make_fake_lc("CdTe1")
        out = HEL1OSPlotter(output_dir=tmp_path, show=False).plot_helios_day(lc)
        assert "CdTe1"    in out.name
        assert "20260621" in out.name


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])