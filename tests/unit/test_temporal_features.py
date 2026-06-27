"""Tests for temporal_features.py"""

import unittest

import numpy as np
import pandas as pd

from ml.features.temporal_features import TemporalFeatures, TemporalFeatureConfig
from .fixtures import make_empty_df, make_constant_df, make_ramp_df, make_synthetic_flare


class TestTemporalFeaturesSmoke(unittest.TestCase):
    def test_default_construction(self):
        tf = TemporalFeatures()
        self.assertIsInstance(tf.config, TemporalFeatureConfig)

    def test_config_from_dict(self):
        cfg = TemporalFeatureConfig.from_dict({"lags_sec": [10, 20]})
        self.assertEqual(cfg.lags_sec, [10, 20])

    def test_feature_names_nonempty(self):
        tf = TemporalFeatures(TemporalFeatureConfig(
            lags_sec=[12], ema_spans_sec=[60], median_windows_sec=[60]
        ))
        names = tf.feature_names()
        self.assertIn("dCR_dt", names)
        self.assertIn("lag_diff_12s", names)
        self.assertIn("ema_60s", names)
        self.assertIn("rolling_median_60s", names)


class TestTemporalFeaturesShape(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_flare()
        self.tf = TemporalFeatures(TemporalFeatureConfig(
            lags_sec=[12, 60], ema_spans_sec=[60, 300], median_windows_sec=[60]
        ))

    def test_output_row_count_preserved(self):
        out = self.tf.transform(self.df)
        self.assertEqual(len(out), len(self.df))

    def test_feature_names_match_actual_new_columns(self):
        out = self.tf.transform(self.df)
        actual_new = set(out.columns) - set(self.df.columns)
        declared = set(self.tf.feature_names())
        self.assertEqual(actual_new, declared)

    def test_input_dataframe_not_mutated(self):
        df_copy = self.df.copy(deep=True)
        self.tf.transform(self.df)
        self.assertTrue(self.df.equals(df_copy))


class TestTemporalFeaturesValues(unittest.TestCase):
    def test_linear_ramp_first_derivative_is_slope(self):
        # CR = t (seconds), cadence 1s -> dCR/dt should be exactly 1 everywhere
        df = make_ramp_df(n=20, cadence_s=1)
        tf = TemporalFeatures(TemporalFeatureConfig(
            lags_sec=[5], ema_spans_sec=[5], median_windows_sec=[5],
            derivative_smoothing=1,
        ))
        out = tf.transform(df)
        self.assertTrue(np.allclose(out["dCR_dt"], 1.0, atol=1e-6))

    def test_linear_ramp_second_derivative_is_zero(self):
        df = make_ramp_df(n=20, cadence_s=1)
        tf = TemporalFeatures(TemporalFeatureConfig(derivative_smoothing=1))
        out = tf.transform(df)
        self.assertTrue(np.allclose(out["d2CR_dt2"], 0.0, atol=1e-6))

    def test_lag_diff_known_value_on_ramp(self):
        # slope=1 ramp: CR[i] - CR[i-5] should be exactly 5 (once warmed up)
        df = make_ramp_df(n=20, cadence_s=1)
        tf = TemporalFeatures(TemporalFeatureConfig(lags_sec=[5]))
        out = tf.transform(df)
        self.assertAlmostEqual(out["lag_diff_5s"].iloc[10], 5.0)

    def test_lag_ratio_known_value(self):
        t = pd.date_range("2020-01-01", periods=10, freq="1s")
        df = pd.DataFrame({"time": t, "CR": [2.0] * 5 + [8.0] * 5})
        tf = TemporalFeatures(TemporalFeatureConfig(lags_sec=[1]))
        out = tf.transform(df)
        # idx5: CR=8, CR(t-1)=2 -> ratio=4.0
        self.assertAlmostEqual(out["lag_ratio_1s"].iloc[5], 4.0)

    def test_time_since_start_matches_elapsed_seconds(self):
        df = make_ramp_df(n=20, cadence_s=1)
        tf = TemporalFeatures()
        out = tf.transform(df)
        self.assertTrue(np.allclose(out["time_since_start_s"], np.arange(20)))

    def test_time_since_peak_zero_at_peak(self):
        df = make_ramp_df(n=20, cadence_s=1)  # max CR is at the last index
        tf = TemporalFeatures()
        out = tf.transform(df)
        peak_idx = int(np.argmax(df["CR"].to_numpy()))
        self.assertAlmostEqual(out["time_since_peak_s"].iloc[peak_idx], 0.0)

    def test_ema_of_constant_series_equals_constant(self):
        df = make_constant_df(n=20, value=15.0, cadence_s=1)
        tf = TemporalFeatures(TemporalFeatureConfig(ema_spans_sec=[5]))
        out = tf.transform(df)
        self.assertTrue(np.allclose(out["ema_5s"], 15.0))
        self.assertTrue(np.allclose(out["cr_minus_ema_5s"], 0.0, atol=1e-9))

    def test_rolling_median_known_sequence(self):
        t = pd.date_range("2020-01-01", periods=7, freq="1s")
        df = pd.DataFrame({"time": t, "CR": [1, 5, 2, 100, 3, 6, 4]})
        tf = TemporalFeatures(TemporalFeatureConfig(median_windows_sec=[3]))
        out = tf.transform(df)
        # idx3 window=[5,2,100] -> median=5 (robust to the 100 outlier)
        self.assertAlmostEqual(out["rolling_median_3s"].iloc[3], 5.0)


class TestTemporalFeaturesEdgeCases(unittest.TestCase):
    def test_empty_dataframe(self):
        df = make_empty_df()
        tf = TemporalFeatures()
        out = tf.transform(df)
        self.assertEqual(len(out), 0)
        for name in tf.feature_names():
            self.assertIn(name, out.columns)

    def test_missing_time_column_raises(self):
        df = pd.DataFrame({"CR": [1.0, 2.0]})
        tf = TemporalFeatures()
        with self.assertRaises(KeyError):
            tf.transform(df)

    def test_single_row_input_no_crash(self):
        t = pd.date_range("2020-01-01", periods=1, freq="4s")
        df = pd.DataFrame({"time": t, "CR": [42.0]})
        tf = TemporalFeatures()
        out = tf.transform(df)
        self.assertEqual(len(out), 1)
        # derivatives undefined for a single point -> defined as 0 by convention
        self.assertEqual(out["dCR_dt"].iloc[0], 0.0)
        self.assertEqual(out["d2CR_dt2"].iloc[0], 0.0)

    def test_two_row_input_no_crash(self):
        t = pd.date_range("2020-01-01", periods=2, freq="4s")
        df = pd.DataFrame({"time": t, "CR": [10.0, 20.0]})
        tf = TemporalFeatures()
        out = tf.transform(df)
        self.assertEqual(len(out), 2)
        self.assertTrue(np.isfinite(out["dCR_dt"]).all())

    def test_nan_in_series_does_not_crash(self):
        t = pd.date_range("2020-01-01", periods=20, freq="4s")
        cr = np.full(20, 10.0)
        cr[8] = np.nan
        df = pd.DataFrame({"time": t, "CR": cr})
        tf = TemporalFeatures(TemporalFeatureConfig(lags_sec=[8]))
        out = tf.transform(df)
        self.assertEqual(len(out), 20)

    def test_zero_lag_safety(self):
        # lags_sec smaller than cadence should still resolve to >= 1 sample
        t = pd.date_range("2020-01-01", periods=10, freq="4s")
        df = pd.DataFrame({"time": t, "CR": list(range(10))})
        tf = TemporalFeatures(TemporalFeatureConfig(lags_sec=[1]))  # < 4s cadence
        out = tf.transform(df)
        self.assertIn("lag_diff_1s", out.columns)
        self.assertFalse(out["lag_diff_1s"].iloc[1:].isna().all())

    def test_idempotent_on_repeated_calls(self):
        df = make_synthetic_flare()
        tf = TemporalFeatures(TemporalFeatureConfig(lags_sec=[12]))
        out1 = tf.transform(df)
        out2 = tf.transform(df)
        pd.testing.assert_frame_equal(out1, out2)


if __name__ == "__main__":
    unittest.main()
