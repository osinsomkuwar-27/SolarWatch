"""
tests/unit/test_eda.py
=======================
Unit tests for the EDA module — all pass without real Aditya-L1 data.

Run from solar/:
    python -m pytest tests/unit/test_eda.py -v

What is tested
--------------
1.  FlareDetector detects injected synthetic flares
2.  FlareEvent attributes are sane
3.  GOES-proxy class is one of {0,1,2,3}
4.  summary() returns a DataFrame with required columns
5.  label_timeseries() returns correct shape and value domain
6.  Empty light curve returns empty list (no crash)
7.  Refractory period prevents overlapping events
8.  LightCurvePlotter saves a valid PNG
9.  Output filename encodes detector and date
10. EDAStatistics.compute() returns EDAReport
11. n_samples matches input length
12. Basic statistics are sane
13. class_weights are all positive
14. save() writes valid JSON
15. plot_distributions() saves PNG files
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Synthetic data factory ────────────────────────────────────────────────────

def _make_synthetic_lc(
    duration_sec: int = 86400,
    cadence_sec:  int = 1,
    n_flares:     int = 3,
    seed:         int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic SoLEXS light curve: Poisson background + sharp impulsive flares.

    Flare shape: fast rise (20 s) + exponential decay (600 s), 
    matching Benz (2008) §1.3 impulsive phase description.
    """
    rng = np.random.default_rng(seed)
    n   = duration_sec // cadence_sec
    t   = np.arange(n, dtype=np.float64) * cadence_sec + 1.75e9

    cr = rng.poisson(50, size=n).astype(np.float64)

    # Inject impulsive flares — fast rise so derivative threshold triggers
    spacing = n // (n_flares + 1)
    for k in range(n_flares):
        onset = spacing * (k + 1)
        amp, rise, decay = 600.0, 20, 600
        for j in range(n):
            dt = j - onset
            if dt >= 0:
                cr[j] += amp * np.exp(-dt / decay) * (1 - np.exp(-dt / max(rise, 1)))
    return t, cr


def _make_detector():
    """Return a FlareDetector tuned for the synthetic LC."""
    from ml.eda.flare_detector import FlareDetector
    return FlareDetector(
        onset_sigma       = 3.0,
        min_rise_bins     = 3,
        smooth_window_sec = 10.0,
        peak_window_sec   = 400.0,
        min_refractory_sec= 600.0,
    )


# ── 1–7  FlareDetector ────────────────────────────────────────────────────────

class TestFlareDetector:

    def test_detects_at_least_one_flare(self) -> None:
        """3 injected flares → ≥1 detected."""
        t, cr  = _make_synthetic_lc(n_flares=3)
        flares = _make_detector().detect(t, cr, "SDD2", "20260621")
        assert len(flares) >= 1, f"Got {len(flares)} flares, expected ≥1"

    def test_detects_all_three_flares(self) -> None:
        """Ideally all 3 injected flares are detected."""
        t, cr  = _make_synthetic_lc(n_flares=3)
        flares = _make_detector().detect(t, cr, "SDD2", "20260621")
        assert len(flares) == 3, f"Got {len(flares)}, expected 3"

    def test_flare_event_attributes(self) -> None:
        """FlareEvent must have sensible timing and count rate values."""
        t, cr  = _make_synthetic_lc(n_flares=1)
        flares = _make_detector().detect(t, cr, "SDD2", "20260621")
        assert len(flares) >= 1
        f = flares[0]
        assert f.onset_unix  < f.peak_unix,   "Peak must be after onset"
        assert f.peak_unix   <= f.end_unix + 1, "End must be at or after peak"
        assert f.peak_count_rate > 50,         "Peak should exceed background"
        assert f.rise_time_sec  >= 0
        assert f.decay_time_sec >= 0
        assert f.detector == "SDD2"
        assert f.date_str == "20260621"

    def test_goes_class_assignment(self) -> None:
        """Flare class must be in {0,1,2,3}."""
        t, cr  = _make_synthetic_lc(n_flares=3)
        flares = _make_detector().detect(t, cr, "SDD2", "20260621")
        for f in flares:
            assert f.flare_class in (0, 1, 2, 3)

    def test_summary_returns_dataframe(self) -> None:
        """summary() returns a DataFrame with required columns."""
        import pandas as pd
        from ml.eda.flare_detector import FlareDetector
        t, cr  = _make_synthetic_lc(n_flares=3)
        flares = _make_detector().detect(t, cr, "SDD2", "20260621")
        df     = FlareDetector.summary(flares)
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            required = {"onset_utc", "peak_utc", "peak_count_rate", "flare_class", "class_name"}
            assert required.issubset(set(df.columns))

    def test_label_timeseries_shape(self) -> None:
        """label_timeseries returns same length as input."""
        t, cr  = _make_synthetic_lc(n_flares=2)
        flares = _make_detector().detect(t, cr, "SDD2", "20260621")
        d      = _make_detector()
        labels = d.label_timeseries(t, cr, flares)
        assert labels.shape == t.shape
        assert labels.dtype == np.int32

    def test_label_timeseries_values(self) -> None:
        """All labels must be in {0,1,2,3}."""
        t, cr  = _make_synthetic_lc(n_flares=2)
        flares = _make_detector().detect(t, cr, "SDD2", "20260621")
        labels = _make_detector().label_timeseries(t, cr, flares)
        assert set(labels.tolist()).issubset({0, 1, 2, 3})

    def test_empty_lc_returns_empty_list(self) -> None:
        """<10 points → [] not a crash."""
        from ml.eda.flare_detector import FlareDetector
        flares = FlareDetector().detect(np.arange(5, dtype=float), np.ones(5)*50, "SDD2")
        assert flares == []

    def test_refractory_period_prevents_overlap(self) -> None:
        """No two consecutive flares should overlap in time."""
        t, cr  = _make_synthetic_lc(n_flares=3)
        flares = _make_detector().detect(t, cr, "SDD2", "20260621")
        if len(flares) > 1:
            for f1, f2 in zip(flares[:-1], flares[1:]):
                assert f2.onset_unix >= f1.end_unix - 1, \
                    "Consecutive flares must not overlap"


# ── 8–9  LightCurvePlotter ────────────────────────────────────────────────────

class TestLightCurvePlotter:

    @staticmethod
    def _fake_lc():
        """Minimal duck-typed LightCurve for plotter tests."""
        from dataclasses import dataclass

        @dataclass
        class FakeLC:
            time_unix: np.ndarray
            count_rate: np.ndarray
            detector: str
            date_str: str
            header: dict

        t, cr = _make_synthetic_lc(duration_sec=3600, n_flares=1)
        return FakeLC(time_unix=t, count_rate=cr,
                      detector="SDD2", date_str="20260621", header={})

    def test_plot_solexs_day_creates_file(self, tmp_path: Path) -> None:
        from ml.eda.light_curve_plotter import LightCurvePlotter
        lc  = self._fake_lc()
        out = LightCurvePlotter(output_dir=tmp_path, show=False).plot_solexs_day(lc)
        assert out.exists()
        assert out.suffix == ".png"
        assert out.stat().st_size > 1000

    def test_output_filename_has_detector_and_date(self, tmp_path: Path) -> None:
        from ml.eda.light_curve_plotter import LightCurvePlotter
        lc  = self._fake_lc()
        out = LightCurvePlotter(output_dir=tmp_path, show=False).plot_solexs_day(lc)
        assert "SDD2"     in out.name
        assert "20260621" in out.name


# ── 10–15  EDAStatistics ─────────────────────────────────────────────────────

class TestEDAStatistics:

    @staticmethod
    def _setup(tmp_path: Path):
        from ml.eda.flare_detector import FlareDetector
        from ml.eda.statistics import EDAStatistics
        t, cr  = _make_synthetic_lc(n_flares=3)
        flares = _make_detector().detect(t, cr, "SDD2", "20260621")
        stats  = EDAStatistics(output_dir=tmp_path)
        return t, cr, flares, stats

    def test_compute_returns_report(self, tmp_path: Path) -> None:
        from ml.eda.statistics import EDAReport
        t, cr, flares, stats = self._setup(tmp_path)
        report = stats.compute(t, cr, flares, "SDD2", "20260621")
        assert isinstance(report, EDAReport)

    def test_n_samples_correct(self, tmp_path: Path) -> None:
        t, cr, flares, stats = self._setup(tmp_path)
        report = stats.compute(t, cr, flares, "SDD2", "20260621")
        assert report.n_samples == len(t)

    def test_statistics_sane(self, tmp_path: Path) -> None:
        t, cr, flares, stats = self._setup(tmp_path)
        report = stats.compute(t, cr, flares, "SDD2", "20260621")
        assert report.min_cr >= 0
        assert report.mean_cr > 0
        assert report.max_cr > report.mean_cr
        assert report.p50_cr <= report.p99_cr

    def test_class_weights_all_positive(self, tmp_path: Path) -> None:
        t, cr, flares, stats = self._setup(tmp_path)
        report = stats.compute(t, cr, flares, "SDD2", "20260621")
        for cls, w in report.class_weights.items():
            assert w > 0, f"class {cls} weight={w} must be > 0"

    def test_save_creates_valid_json(self, tmp_path: Path) -> None:
        t, cr, flares, stats = self._setup(tmp_path)
        report = stats.compute(t, cr, flares, "SDD2", "20260621")
        path   = stats.save(report, tag="20260621")
        assert path.exists()
        data   = json.loads(path.read_text())
        assert data["detector"]  == "SDD2"
        assert data["n_samples"] == len(t)
        assert "class_weights"   in data

    def test_plot_distributions_creates_pngs(self, tmp_path: Path) -> None:
        t, cr, flares, stats = self._setup(tmp_path)
        paths = stats.plot_distributions(t, cr, flares, tag="20260621", detector="SDD2")
        assert len(paths) >= 2
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 500


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
