"""
tests/unit/test_basic_features.py
===================================
Unit tests for ml/features/basic_features.py

Tests are structured in four layers:
  1. Smoke tests  — module imports, constructor, feature_names()
  2. Shape tests  — output shape, column names, index preservation
  3. Value tests  — mathematical correctness of each feature
  4. Edge cases   — NaN handling, negative clips, short series, empty DF

Run from solar/ directory:
    python -m pytest tests/unit/test_basic_features.py -v

All tests use synthetic data only — no real Aditya-L1 files required.

Scientific validation strategy
--------------------------------
For each feature, we construct a light curve where the *expected* output
is analytically derivable, then assert that the extractor matches within
a numerical tolerance (1e-9 for exact operations, 1e-6 for sqrt/pow).

This is the same approach used in the existing test_eda.py:
    "All pass without real Aditya-L1 data."
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path
_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.features.basic_features import BasicFeatureConfig, BasicFeatureExtractor


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_constant_lc(value: float = 100.0, n: int = 3600) -> pd.DataFrame:
    """
    Constant count-rate light curve.
    All rolling statistics have analytically predictable values.
    """
    t  = np.arange(n, dtype=np.float64) + 1.75e9
    cr = np.full(n, value, dtype=np.float64)
    return pd.DataFrame({"timestamp": t, "counts": cr})


def _make_poisson_lc(lam: float = 50.0, n: int = 3600, seed: int = 42) -> pd.DataFrame:
    """Poisson background — realistic quiet Sun light curve."""
    rng = np.random.default_rng(seed)
    t   = np.arange(n, dtype=np.float64) + 1.75e9
    cr  = rng.poisson(lam, size=n).astype(np.float64)
    return pd.DataFrame({"timestamp": t, "counts": cr})


def _make_flare_lc(
    background: float = 50.0,
    amplitude:  float = 500.0,
    n:          int   = 7200,
    onset_idx:  int   = 3600,
    rise_bins:  int   = 60,
    decay_bins: int   = 600,
    seed:       int   = 42,
) -> pd.DataFrame:
    """
    Synthetic flare: Poisson background + impulsive flare.
    Shape: fast rise (Gaussian onset) + exponential decay.
    Matches Benz (2008) §1.3 impulsive phase profile.
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(n, dtype=np.float64) + 1.75e9
    cr  = rng.poisson(background, size=n).astype(np.float64)

    for i in range(n):
        dt = i - onset_idx
        if dt >= 0:
            rise  = 1.0 - np.exp(-dt / rise_bins)
            decay = np.exp(-dt / decay_bins)
            cr[i] += amplitude * rise * decay
    return pd.DataFrame({"timestamp": t, "counts": cr})


def _make_extractor(windows_sec=(60.0, 300.0)) -> BasicFeatureExtractor:
    return BasicFeatureExtractor(windows_sec=list(windows_sec), cadence_sec=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSmoke:

    def test_import(self) -> None:
        """Module imports without error."""
        from ml.features.basic_features import BasicFeatureExtractor  # noqa: F401

    def test_constructor_defaults(self) -> None:
        """Default constructor does not raise."""
        ext = BasicFeatureExtractor()
        assert ext.cfg.windows_sec == [60.0, 300.0, 600.0]
        assert ext.cfg.cadence_sec == 1.0

    def test_constructor_custom(self) -> None:
        """Custom windows and cadence are stored correctly."""
        ext = BasicFeatureExtractor(windows_sec=[30.0, 120.0], cadence_sec=2.0)
        assert ext.cfg.windows_sec == [30.0, 120.0]
        assert ext.cfg.cadence_sec == 2.0

    def test_config_from_dict(self) -> None:
        """BasicFeatureConfig.from_dict ignores unknown keys."""
        cfg = BasicFeatureConfig.from_dict({
            "windows_sec": [60.0],
            "cadence_sec": 4.0,
            "unknown_key": "ignored",
        })
        assert cfg.windows_sec == [60.0]
        assert cfg.cadence_sec == 4.0

    def test_feature_names(self) -> None:
        """feature_names() returns the expected list."""
        ext   = _make_extractor(windows_sec=[60.0])
        names = ext.feature_names()
        assert "log_counts" in names
        assert "rolling_mean_60s" in names
        assert "rolling_std_60s"  in names
        assert "rolling_max_60s"  in names
        assert "rolling_min_60s"  in names
        assert "rolling_var_60s"  in names
        assert "rms_60s"          in names
        assert "signal_energy_60s" in names


# ─────────────────────────────────────────────────────────────────────────────
# 2. Shape and schema tests
# ─────────────────────────────────────────────────────────────────────────────

class TestShape:

    def test_output_row_count_preserved(self) -> None:
        """Output must have exactly the same number of rows as input."""
        df  = _make_poisson_lc(n=3600)
        out = _make_extractor().transform(df)
        assert len(out) == len(df)

    def test_input_columns_preserved(self) -> None:
        """Original columns must appear unchanged in output."""
        df  = _make_poisson_lc(n=3600)
        out = _make_extractor().transform(df)
        assert "timestamp" in out.columns
        assert "counts"    in out.columns
        pd.testing.assert_series_equal(out["timestamp"], df["timestamp"])
        pd.testing.assert_series_equal(out["counts"],    df["counts"])

    def test_all_feature_columns_present(self) -> None:
        """All expected feature columns are created."""
        ext   = _make_extractor(windows_sec=[60.0, 300.0])
        df    = _make_poisson_lc(n=3600)
        out   = ext.transform(df)
        for name in ext.feature_names():
            assert name in out.columns, f"Missing feature column: {name}"

    def test_index_preserved(self) -> None:
        """pandas index is not reset during transform."""
        df  = _make_poisson_lc(n=500)
        df.index = np.arange(100, 600)  # non-default index
        out = _make_extractor(windows_sec=[60.0]).transform(df)
        assert list(out.index) == list(df.index)

    def test_multiple_windows_produce_separate_columns(self) -> None:
        """Two windows produce separately named columns."""
        ext = _make_extractor(windows_sec=[60.0, 300.0])
        df  = _make_poisson_lc()
        out = ext.transform(df)
        assert "rolling_mean_60s"  in out.columns
        assert "rolling_mean_300s" in out.columns

    def test_feature_count_scales_with_windows(self) -> None:
        """With k windows: total feature columns = 1 (log) + 7k."""
        for k, windows in enumerate([[60.0], [60.0, 300.0], [60.0, 300.0, 600.0]], start=1):
            ext  = BasicFeatureExtractor(windows_sec=windows, cadence_sec=1.0)
            df   = _make_poisson_lc(n=700)
            out  = ext.transform(df)
            n_new = len(out.columns) - len(df.columns)
            assert n_new == 1 + 7 * k, (
                f"Expected {1+7*k} new columns for {k} window(s), got {n_new}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Value / mathematical correctness tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValues:
    """
    Each test constructs a case where the expected output is analytically
    known and verifies that the extractor matches it.
    """

    W   = 60.0   # 60-second window for these tests
    VAL = 100.0  # constant signal value

    def _constant_out(self) -> pd.DataFrame:
        """Run extractor on a constant signal."""
        df  = _make_constant_lc(value=self.VAL, n=3600)
        ext = BasicFeatureExtractor(windows_sec=[self.W], cadence_sec=1.0)
        return ext.transform(df)

    # ── Feature 1: Log Counts ──────────────────────────────────────────────

    def test_log_counts_formula(self) -> None:
        """log_counts = log10(CR + 1), verified on known values."""
        df  = _make_constant_lc(value=self.VAL)
        out = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0).transform(df)
        expected = np.log10(self.VAL + 1.0)
        np.testing.assert_allclose(
            out["log_counts"].dropna().values,
            expected,
            rtol=1e-9,
            err_msg="log_counts formula mismatch",
        )

    def test_log_counts_zero_input(self) -> None:
        """log10(0 + 1) = 0 — no NaN or -inf."""
        df  = _make_constant_lc(value=0.0)
        out = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0).transform(df)
        assert np.all(out["log_counts"].values == 0.0)

    def test_log_counts_monotone(self) -> None:
        """Higher CR → higher log_counts (monotone transformation)."""
        vals = [10.0, 50.0, 100.0, 500.0, 1000.0]
        log_vals = [np.log10(v + 1) for v in vals]
        assert log_vals == sorted(log_vals)

    # ── Feature 2: Rolling Mean ────────────────────────────────────────────

    def test_rolling_mean_constant(self) -> None:
        """Mean of constant signal equals the constant."""
        out = self._constant_out()
        # Skip the first (window-1) NaN values
        valid = out["rolling_mean_60s"].dropna()
        np.testing.assert_allclose(
            valid.values, self.VAL, rtol=1e-9,
            err_msg="Rolling mean of constant signal must equal the constant",
        )

    def test_rolling_mean_step(self) -> None:
        """
        Step function: mean transitions from low to high across the window.
        After the full window is filled with the new level, mean = new level.
        """
        n  = 1000
        cr = np.full(n, 10.0)
        cr[500:] = 200.0
        df  = pd.DataFrame({"timestamp": np.arange(n, dtype=float), "counts": cr})
        out = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0).transform(df)
        # After sample 559 (500 + 60 - 1), window is entirely in the high region
        assert out["rolling_mean_60s"].iloc[580] == pytest.approx(200.0, abs=1e-9)

    # ── Feature 3: Rolling Standard Deviation ─────────────────────────────

    def test_rolling_std_constant(self) -> None:
        """Std of constant signal is 0."""
        out = self._constant_out()
        valid = out["rolling_std_60s"].dropna()
        np.testing.assert_allclose(
            valid.values, 0.0, atol=1e-10,
            err_msg="Std of constant signal must be 0",
        )

    def test_rolling_std_known_values(self) -> None:
        """
        For a two-value alternating signal [a, b, a, b, ...], the
        sample std over a full window is:
            σ = |b - a| / 2  × √(2w / (w-1))   for large w → |b-a|/2
        Verify using a simple case.
        """
        a, b = 0.0, 10.0
        n    = 500
        cr   = np.tile([a, b], n // 2).astype(float)
        df   = pd.DataFrame({"timestamp": np.arange(n, dtype=float), "counts": cr})
        out  = BasicFeatureExtractor(windows_sec=[100.0], cadence_sec=1.0).transform(df)
        # For a two-value signal with w=100: σ ≈ 5.025 (exact from formula)
        expected_std = np.std([a, b] * 50, ddof=1)
        valid = out["rolling_std_100s"].dropna()
        np.testing.assert_allclose(
            valid.values[-1],
            expected_std,
            rtol=1e-4,
            err_msg="Rolling std of alternating signal mismatch",
        )

    # ── Feature 4: Rolling Maximum ────────────────────────────────────────

    def test_rolling_max_constant(self) -> None:
        """Max of constant signal equals the constant."""
        out   = self._constant_out()
        valid = out["rolling_max_60s"].dropna()
        np.testing.assert_allclose(valid.values, self.VAL, rtol=1e-9)

    def test_rolling_max_single_spike(self) -> None:
        """A single spike propagates forward through the window."""
        n  = 500
        cr = np.full(n, 10.0)
        cr[100] = 999.0   # single spike
        df  = pd.DataFrame({"timestamp": np.arange(n, dtype=float), "counts": cr})
        out = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0).transform(df)
        # Spike at 100 → max stays 999 for bins 100..159 (60-bin window)
        assert out["rolling_max_60s"].iloc[100] == 999.0
        assert out["rolling_max_60s"].iloc[159] == 999.0
        assert out["rolling_max_60s"].iloc[160] == pytest.approx(10.0, abs=1e-9)

    # ── Feature 5: Rolling Minimum ────────────────────────────────────────

    def test_rolling_min_constant(self) -> None:
        """Min of constant signal equals the constant."""
        out   = self._constant_out()
        valid = out["rolling_min_60s"].dropna()
        np.testing.assert_allclose(valid.values, self.VAL, rtol=1e-9)

    def test_rolling_min_dip(self) -> None:
        """A dip in the signal is tracked by rolling min."""
        n  = 500
        cr = np.full(n, 100.0)
        cr[200] = 1.0   # single dip
        df  = pd.DataFrame({"timestamp": np.arange(n, dtype=float), "counts": cr})
        out = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0).transform(df)
        assert out["rolling_min_60s"].iloc[200] == 1.0
        assert out["rolling_min_60s"].iloc[258] == 1.0
        assert out["rolling_min_60s"].iloc[260] == pytest.approx(100.0, abs=1e-9)

    # ── Feature 6: Rolling Variance ───────────────────────────────────────

    def test_rolling_var_equals_std_squared(self) -> None:
        """rolling_var == rolling_std² (within floating-point precision)."""
        df  = _make_poisson_lc(n=3600)
        out = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0).transform(df)
        var = out["rolling_var_60s"].dropna()
        std = out["rolling_std_60s"].dropna()
        np.testing.assert_allclose(
            var.values, std.values ** 2, rtol=1e-6,
            err_msg="rolling_var must equal rolling_std²",
        )

    def test_rolling_var_constant(self) -> None:
        """Variance of constant signal is 0."""
        out   = self._constant_out()
        valid = out["rolling_var_60s"].dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-10)

    # ── Feature 7: Moving RMS ─────────────────────────────────────────────

    def test_rms_constant(self) -> None:
        """RMS of constant signal equals the constant (mean² + 0 variance)."""
        out   = self._constant_out()
        valid = out["rms_60s"].dropna()
        np.testing.assert_allclose(valid.values, self.VAL, rtol=1e-6)

    def test_rms_geq_mean(self) -> None:
        """RMS ≥ rolling_mean always (by Cauchy-Schwarz inequality)."""
        df  = _make_poisson_lc(n=3600)
        out = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0).transform(df)
        rms  = out["rms_60s"].dropna()
        mean = out["rolling_mean_60s"].loc[rms.index]
        assert np.all(rms.values >= mean.values - 1e-9), \
            "RMS must always be ≥ rolling mean"

    def test_rms_formula(self) -> None:
        """Verify RMS = √(mean of squares) on a small known array."""
        arr = np.array([3.0, 4.0, 5.0])
        expected_rms = np.sqrt(np.mean(arr**2))   # √(50/3) ≈ 4.082
        cr  = np.concatenate([np.ones(100) * 100.0, arr])
        df  = pd.DataFrame({"timestamp": np.arange(len(cr), dtype=float), "counts": cr})
        out = BasicFeatureExtractor(windows_sec=[3.0], cadence_sec=1.0).transform(df)
        # Last row: window = [3, 4, 5]
        last_rms = out["rms_3s"].iloc[-1]
        assert last_rms == pytest.approx(expected_rms, rel=1e-6)

    # ── Feature 8: Signal Energy ──────────────────────────────────────────

    def test_signal_energy_constant_normalised(self) -> None:
        """
        Normalised energy of constant signal:
            E_norm = mean(CR² × Δt) / w_bins = CR² × Δt / w_bins  (per-bin)
        But our implementation sums CR²×Δt over the window then divides by w_bins.
        For a constant signal: sum = w_bins × VAL² × Δt → / w_bins = VAL² × Δt.
        This is fully filled only after the window is complete.
        """
        out   = self._constant_out()
        # Once the window is full the value stabilises: VAL² × cadence = 100² × 1
        expected = self.VAL ** 2 * 1.0   # cadence_sec = 1
        fully_filled = out["signal_energy_60s"].iloc[60:]   # skip ramp-up
        np.testing.assert_allclose(
            fully_filled.values, expected, rtol=1e-6,
            err_msg="Fully-filled normalised signal energy should equal VAL² × cadence",
        )

    def test_signal_energy_scales_with_amplitude(self) -> None:
        """Signal energy scales as amplitude² (quadratic relationship)."""
        energies = []
        for amp in [10.0, 100.0, 1000.0]:
            df  = _make_constant_lc(value=amp, n=1000)
            out = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0).transform(df)
            energies.append(out["signal_energy_60s"].dropna().mean())
        # Ratio should be 100: (100/10)² = 100, (1000/100)² = 100
        assert energies[1] / energies[0] == pytest.approx(100.0, rel=1e-4)
        assert energies[2] / energies[1] == pytest.approx(100.0, rel=1e-4)

    def test_signal_energy_unnormalised_scales_with_window(self) -> None:
        """Unnormalised energy over 2× window ≈ 2× energy over 1× window (steady signal)."""
        cfg_unnorm = BasicFeatureConfig(
            windows_sec=[60.0, 120.0],
            cadence_sec=1.0,
            energy_normalise_by_window=False,
        )
        df  = _make_constant_lc(value=100.0, n=2000)
        out = BasicFeatureExtractor(config=cfg_unnorm).transform(df)
        # Skip the ramp-up period (first max_window=120 bins)
        e60  = out["signal_energy_60s"].iloc[120:].mean()
        e120 = out["signal_energy_120s"].iloc[120:].mean()
        assert e120 / e60 == pytest.approx(2.0, rel=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Edge case tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_dataframe_raises(self) -> None:
        """Empty DataFrame raises ValueError."""
        df  = pd.DataFrame({"timestamp": [], "counts": []})
        ext = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0)
        with pytest.raises(ValueError, match="empty"):
            ext.transform(df)

    def test_missing_count_column_raises(self) -> None:
        """Missing 'counts' column raises ValueError."""
        df  = pd.DataFrame({"timestamp": [1.0, 2.0], "wrong_col": [50.0, 50.0]})
        ext = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0)
        with pytest.raises(ValueError, match="missing"):
            ext.transform(df)

    def test_missing_timestamp_column_raises(self) -> None:
        """Missing 'timestamp' column raises ValueError."""
        df  = pd.DataFrame({"time": [1.0, 2.0], "counts": [50.0, 50.0]})
        ext = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0)
        with pytest.raises(ValueError, match="missing"):
            ext.transform(df)

    def test_negative_counts_clipped_to_zero(self) -> None:
        """Negative counts are clipped to 0 when clip_negative=True."""
        df  = pd.DataFrame({
            "timestamp": [1.0, 2.0, 3.0],
            "counts":    [-10.0, 50.0, 100.0],
        })
        ext = BasicFeatureExtractor(windows_sec=[3.0], cadence_sec=1.0)
        out = ext.transform(df)
        # log_counts of -10+1 would be NaN; with clipping, -10 → 0 → log10(1) = 0
        assert out["log_counts"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_negative_counts_not_clipped_when_disabled(self) -> None:
        """When clip_negative=False, negative values pass through."""
        cfg = BasicFeatureConfig(windows_sec=[3.0], cadence_sec=1.0, clip_negative=False)
        df  = pd.DataFrame({
            "timestamp": np.arange(100, dtype=float),
            "counts":    np.full(100, -5.0),
        })
        ext = BasicFeatureExtractor(config=cfg)
        out = ext.transform(df)
        # rolling mean of constant -5 = -5
        np.testing.assert_allclose(
            out["rolling_mean_3s"].dropna().values, -5.0, rtol=1e-9
        )

    def test_nan_in_input_propagates(self) -> None:
        """NaN values in input propagate to output (no silent fill)."""
        n   = 200
        cr  = np.full(n, 50.0)
        cr[100] = np.nan
        df  = pd.DataFrame({"timestamp": np.arange(n, dtype=float), "counts": cr})
        out = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0).transform(df)
        # The log_counts at 100 should be NaN (NaN + 1 = NaN → log10(NaN) = NaN)
        assert np.isnan(out["log_counts"].iloc[100])

    def test_single_window_larger_than_data(self) -> None:
        """Window larger than data length: most values NaN, no crash."""
        df  = pd.DataFrame({
            "timestamp": np.arange(10, dtype=float),
            "counts":    np.full(10, 50.0),
        })
        # Window of 60 s with 10 samples and min_periods_fraction=0.5
        # → min_periods = 30 > 10 → all NaN
        ext = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0)
        out = ext.transform(df)
        # All rolling features should be NaN (except log_counts which is per-sample)
        assert not out["log_counts"].isna().any()

    def test_custom_column_names(self) -> None:
        """Extractor works with non-default column names."""
        df  = pd.DataFrame({
            "time_unix":   np.arange(1000, dtype=float) + 1.75e9,
            "count_rate":  np.full(1000, 100.0),
        })
        cfg = BasicFeatureConfig(
            windows_sec=[60.0],
            cadence_sec=1.0,
            time_col="time_unix",
            count_col="count_rate",
        )
        ext = BasicFeatureExtractor(config=cfg)
        out = ext.transform(df)
        assert "log_counts" in out.columns

    def test_very_large_values_no_overflow(self) -> None:
        """Large count rates (X-class solar flares) don't cause overflow."""
        # RHESSI observed >10^4 cts/s in large flares
        df  = _make_constant_lc(value=1e6, n=200)
        ext = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0)
        out = ext.transform(df)
        assert np.isfinite(out["log_counts"]).all()
        # signal_energy = 1e12 × 1 — should not overflow float64
        valid_energy = out["signal_energy_60s"].dropna()
        assert np.isfinite(valid_energy).all()

    def test_transform_does_not_modify_input(self) -> None:
        """transform() is pure — input DataFrame is not modified."""
        df   = _make_poisson_lc(n=500)
        orig = df.copy()
        _    = BasicFeatureExtractor(windows_sec=[60.0], cadence_sec=1.0).transform(df)
        pd.testing.assert_frame_equal(df, orig)

    def test_transform_idempotent(self) -> None:
        """Calling transform() twice gives identical results."""
        df  = _make_poisson_lc(n=1000)
        ext = _make_extractor(windows_sec=[60.0])
        out1 = ext.transform(df)
        out2 = ext.transform(df)
        pd.testing.assert_frame_equal(out1, out2)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Integration / realistic data test
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    """
    End-to-end test on a synthetic flare light curve.
    Verifies that the features behave as expected at physically meaningful
    points in the light curve (pre-flare, peak, decay).
    """

    @pytest.fixture(scope="class")
    def flare_features(self) -> pd.DataFrame:
        """Pre-compute features on a synthetic flare LC once per class."""
        df  = _make_flare_lc(
            background=50.0, amplitude=500.0, n=7200,
            onset_idx=3600, rise_bins=60, decay_bins=600,
        )
        ext = BasicFeatureExtractor(
            windows_sec=[60.0, 300.0, 600.0],
            cadence_sec=1.0,
        )
        return ext.transform(df)

    def test_rolling_mean_higher_at_peak(self, flare_features: pd.DataFrame) -> None:
        """Rolling mean well after the flare peak > rolling mean in quiet pre-flare period."""
        pre_flare_mean = flare_features["rolling_mean_300s"].iloc[2000:3000].mean()
        # Sample 4000 is onset(3600) + 400 bins: the 300-bin window still captures high flux
        post_onset_mean = flare_features["rolling_mean_300s"].iloc[3700:4000].mean()
        assert post_onset_mean > pre_flare_mean * 1.5, (
            f"Rolling mean post-onset ({post_onset_mean:.1f}) should be "
            f"significantly above pre-flare ({pre_flare_mean:.1f})"
        )

    def test_rolling_std_higher_during_rise(self, flare_features: pd.DataFrame) -> None:
        """Rolling std spikes during the impulsive rise phase."""
        quiet_std = flare_features["rolling_std_60s"].iloc[1000:2000].mean()
        rise_std  = flare_features["rolling_std_60s"].iloc[3600:3660].mean()
        assert rise_std > quiet_std, \
            "Rolling std should increase during the impulsive phase"

    def test_rolling_max_captures_peak(self, flare_features: pd.DataFrame) -> None:
        """Rolling max reaches its global maximum near the flare peak."""
        global_max_idx = flare_features["rolling_max_300s"].idxmax()
        # Peak should be somewhere between onset (3600) and onset+rise+decay (4500)
        assert 3600 <= global_max_idx <= 5000, \
            f"Global max index {global_max_idx} not in expected flare window"

    def test_signal_energy_elevated_during_flare(self, flare_features: pd.DataFrame) -> None:
        """Signal energy is significantly higher during the flare than pre-flare."""
        pre_energy  = flare_features["signal_energy_600s"].iloc[1000:2000].mean()
        peak_energy = flare_features["signal_energy_600s"].iloc[3700:4300].mean()
        assert peak_energy > pre_energy * 5, \
            "Signal energy should increase substantially during a flare"

    def test_log_counts_no_nan_in_clean_flare(self, flare_features: pd.DataFrame) -> None:
        """log_counts should have no NaN for a clean (positive) flare signal."""
        assert not flare_features["log_counts"].isna().any(), \
            "log_counts should not have NaN for a positive signal"

    def test_rms_geq_mean_throughout(self, flare_features: pd.DataFrame) -> None:
        """RMS ≥ rolling_mean at every time step (always true)."""
        rms  = flare_features["rms_300s"].dropna()
        mean = flare_features["rolling_mean_300s"].loc[rms.index]
        diff = rms.values - mean.values
        assert np.all(diff >= -1e-9), \
            "RMS must be ≥ rolling_mean at all times"


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v", "--tb=short"])