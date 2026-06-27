"""Tests for basic_features.py"""

import unittest

import numpy as np
import pandas as pd

from ml.features.basic_features import BasicFeatures, BasicFeatureConfig
from .fixtures import make_empty_df, make_constant_df, make_synthetic_flare


class TestBasicFeaturesSmoke(unittest.TestCase):
    def test_default_construction(self):
        bf = BasicFeatures()
        self.assertIsInstance(bf.config, BasicFeatureConfig)

    def test_config_from_dict(self):
        cfg = BasicFeatureConfig.from_dict({"windows_sec": [30, 90]})
        self.assertEqual(cfg.windows_sec, [30, 90])

    def test_feature_names_nonempty(self):
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[60]))
        names = bf.feature_names()
        self.assertIn("log_counts", names)
        self.assertIn("rolling_mean_60s", names)
        self.assertEqual(len(names), 1 + 7 * 1)  # 1 log + 7 stats per window

    def test_feature_names_scale_with_windows(self):
        bf1 = BasicFeatures(BasicFeatureConfig(windows_sec=[60]))
        bf2 = BasicFeatures(BasicFeatureConfig(windows_sec=[60, 300, 600]))
        self.assertEqual(len(bf2.feature_names()), 1 + 7 * 3)
        self.assertGreater(len(bf2.feature_names()), len(bf1.feature_names()))


class TestBasicFeaturesShape(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_flare()
        self.bf = BasicFeatures(BasicFeatureConfig(windows_sec=[60, 300]))

    def test_output_row_count_preserved(self):
        out = self.bf.transform(self.df)
        self.assertEqual(len(out), len(self.df))

    def test_original_columns_preserved(self):
        out = self.bf.transform(self.df)
        for col in self.df.columns:
            self.assertIn(col, out.columns)

    def test_index_preserved(self):
        out = self.bf.transform(self.df)
        self.assertTrue(out.index.equals(self.df.index))

    def test_feature_names_match_actual_new_columns(self):
        out = self.bf.transform(self.df)
        actual_new = set(out.columns) - set(self.df.columns)
        declared = set(self.bf.feature_names())
        self.assertEqual(actual_new, declared)

    def test_input_dataframe_not_mutated(self):
        df_copy = self.df.copy(deep=True)
        self.bf.transform(self.df)
        self.assertTrue(self.df.equals(df_copy))


class TestBasicFeaturesValues(unittest.TestCase):
    def test_log_counts_formula(self):
        df = make_constant_df(n=5, value=10.0)
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[3], log_offset=1.0))
        out = bf.transform(df)
        self.assertTrue(np.allclose(out["log_counts"], np.log10(11.0)))

    def test_constant_series_rolling_mean_equals_constant(self):
        df = make_constant_df(n=10, value=10.0, cadence_s=1)
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[5]))
        out = bf.transform(df)
        self.assertTrue(np.allclose(out["rolling_mean_5s"].iloc[4:], 10.0))

    def test_constant_series_std_and_var_are_zero(self):
        df = make_constant_df(n=10, value=10.0, cadence_s=1)
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[5]))
        out = bf.transform(df)
        self.assertTrue(np.allclose(out["rolling_std_5s"].iloc[4:], 0.0, atol=1e-9))
        self.assertTrue(np.allclose(out["rolling_var_5s"].iloc[4:], 0.0, atol=1e-9))

    def test_constant_series_rms_equals_constant(self):
        df = make_constant_df(n=10, value=10.0, cadence_s=1)
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[5]))
        out = bf.transform(df)
        self.assertTrue(np.allclose(out["rms_5s"].iloc[4:], 10.0))

    def test_rolling_mean_known_sequence(self):
        # CR = [1..10], window=3 samples -> mean at idx4 (value 5) = mean(3,4,5)=4
        t = pd.date_range("2020-01-01", periods=10, freq="1s")
        df = pd.DataFrame({"time": t, "CR": list(range(1, 11))})
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[3]))
        out = bf.transform(df)
        self.assertAlmostEqual(out["rolling_mean_3s"].iloc[4], 4.0)

    def test_rms_known_sequence(self):
        t = pd.date_range("2020-01-01", periods=10, freq="1s")
        df = pd.DataFrame({"time": t, "CR": list(range(1, 11))})
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[3]))
        out = bf.transform(df)
        expected_rms = np.sqrt((9 + 16 + 25) / 3)  # values 3,4,5 squared
        self.assertAlmostEqual(out["rms_3s"].iloc[4], expected_rms)

    def test_rolling_max_min_known_sequence(self):
        t = pd.date_range("2020-01-01", periods=10, freq="1s")
        df = pd.DataFrame({"time": t, "CR": [5, 1, 9, 2, 8, 3, 7, 4, 6, 0]})
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[3]))
        out = bf.transform(df)
        # idx2 window = [5,1,9] -> max=9, min=1
        self.assertAlmostEqual(out["rolling_max_3s"].iloc[2], 9.0)
        self.assertAlmostEqual(out["rolling_min_3s"].iloc[2], 1.0)

    def test_signal_energy_nonnegative(self):
        df = make_synthetic_flare()
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[60]))
        out = bf.transform(df)
        valid = out["signal_energy_60s"].dropna()
        self.assertTrue((valid >= 0).all())


class TestBasicFeaturesEdgeCases(unittest.TestCase):
    def test_empty_dataframe(self):
        df = make_empty_df()
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[60]))
        out = bf.transform(df)
        self.assertEqual(len(out), 0)
        for name in bf.feature_names():
            self.assertIn(name, out.columns)

    def test_missing_time_column_raises(self):
        df = pd.DataFrame({"CR": [1.0, 2.0, 3.0]})
        bf = BasicFeatures()
        with self.assertRaises(KeyError):
            bf.transform(df)

    def test_missing_value_column_raises(self):
        df = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=3, freq="1s")})
        bf = BasicFeatures()
        with self.assertRaises(KeyError):
            bf.transform(df)

    def test_non_datetime_time_column_raises(self):
        df = pd.DataFrame({"time": [1, 2, 3], "CR": [1.0, 2.0, 3.0]})
        bf = BasicFeatures()
        with self.assertRaises(TypeError):
            bf.transform(df)

    def test_negative_values_clipped_by_default(self):
        t = pd.date_range("2020-01-01", periods=5, freq="1s")
        df = pd.DataFrame({"time": t, "CR": [-5.0, -1.0, 0.0, 1.0, 5.0]})
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[2], clip_negative=True))
        out = bf.transform(df)
        # log_counts should never be NaN/complex since negatives are clipped to 0
        self.assertFalse(out["log_counts"].isna().any())
        self.assertTrue(np.isfinite(out["log_counts"]).all())

    def test_nan_values_propagate_without_crash(self):
        t = pd.date_range("2020-01-01", periods=20, freq="4s")
        cr = np.full(20, 10.0)
        cr[5] = np.nan
        df = pd.DataFrame({"time": t, "CR": cr})
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[20]))
        out = bf.transform(df)
        self.assertTrue(np.isnan(out["log_counts"].iloc[5]))

    def test_single_row_input(self):
        # With only one sample, the sampling cadence cannot be inferred
        # (the code defaults to 1.0s), so a 60s window is interpreted as
        # 60 samples, of which only 1 is available. Per pandas rolling
        # semantics, this correctly yields NaN -- not the lone value --
        # because min_periods cannot be satisfied. This is the honest
        # answer: we cannot claim a "60-second average" was computed
        # when no actual 60-second span of data was observed.
        t = pd.date_range("2020-01-01", periods=1, freq="4s")
        df = pd.DataFrame({"time": t, "CR": [42.0]})
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[60]))
        out = bf.transform(df)
        self.assertEqual(len(out), 1)
        self.assertTrue(np.isnan(out["rolling_mean_60s"].iloc[0]))
        # log_counts has no such ambiguity -- it's a pointwise transform
        self.assertAlmostEqual(out["log_counts"].iloc[0], np.log10(43.0))

    def test_large_values_no_overflow(self):
        t = pd.date_range("2020-01-01", periods=5, freq="1s")
        df = pd.DataFrame({"time": t, "CR": [1e30] * 5})
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[3]))
        out = bf.transform(df)
        self.assertTrue(np.isfinite(out["log_counts"]).all())

    def test_idempotent_on_repeated_calls(self):
        df = make_synthetic_flare()
        bf = BasicFeatures(BasicFeatureConfig(windows_sec=[60]))
        out1 = bf.transform(df)
        out2 = bf.transform(df)
        pd.testing.assert_frame_equal(out1, out2)


if __name__ == "__main__":
    unittest.main()
