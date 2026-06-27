"""Tests for helios_basic_features.py"""

import unittest

import numpy as np
import pandas as pd

from ml.features.helios_features.helios_basic_features import (
    HEL1OSBasicFeatures,
    HEL1OSBasicFeatureConfig,
)
from .helios_fixtures import (
    make_empty_helios_df,
    make_constant_helios_df,
    make_synthetic_helios_flare,
)


class TestHEL1OSBasicFeaturesSmoke(unittest.TestCase):
    def test_default_construction(self):
        hbf = HEL1OSBasicFeatures()
        self.assertIsInstance(hbf.config, HEL1OSBasicFeatureConfig)

    def test_config_from_dict(self):
        cfg = HEL1OSBasicFeatureConfig.from_dict({"windows_sec": [30, 90]})
        self.assertEqual(cfg.windows_sec, [30, 90])

    def test_feature_names_nonempty(self):
        hbf   = HEL1OSBasicFeatures(HEL1OSBasicFeatureConfig(windows_sec=[60]))
        names = hbf.feature_names()
        self.assertIn("cdte_log_counts", names)
        self.assertIn("czt_log_counts",  names)
        self.assertIn("cdte_rolling_mean_60s", names)
        self.assertIn("czt_rolling_mean_60s",  names)

    def test_feature_names_scale_with_windows(self):
        hbf1 = HEL1OSBasicFeatures(HEL1OSBasicFeatureConfig(windows_sec=[60]))
        hbf2 = HEL1OSBasicFeatures(HEL1OSBasicFeatureConfig(windows_sec=[60, 300]))
        self.assertGreater(len(hbf2.feature_names()), len(hbf1.feature_names()))


class TestHEL1OSBasicFeaturesShape(unittest.TestCase):
    def setUp(self):
        self.df  = make_synthetic_helios_flare()
        self.hbf = HEL1OSBasicFeatures(HEL1OSBasicFeatureConfig(windows_sec=[60, 300]))

    def test_output_row_count_preserved(self):
        out = self.hbf.transform(self.df)
        self.assertEqual(len(out), len(self.df))

    def test_original_columns_preserved(self):
        out = self.hbf.transform(self.df)
        for col in self.df.columns:
            self.assertIn(col, out.columns)

    def test_feature_names_match_actual_new_columns(self):
        out        = self.hbf.transform(self.df)
        actual_new = set(out.columns) - set(self.df.columns)
        declared   = set(self.hbf.feature_names())
        self.assertEqual(actual_new, declared)

    def test_input_dataframe_not_mutated(self):
        df_copy = self.df.copy(deep=True)
        self.hbf.transform(self.df)
        self.assertTrue(self.df.equals(df_copy))


class TestHEL1OSBasicFeaturesValues(unittest.TestCase):
    def test_log_counts_formula(self):
        df  = make_constant_helios_df(n=5, cdte_val=10.0, czt_val=10.0)
        hbf = HEL1OSBasicFeatures(HEL1OSBasicFeatureConfig(windows_sec=[3], log_offset=1.0))
        out = hbf.transform(df)
        self.assertTrue(np.allclose(out["cdte_log_counts"], np.log10(11.0)))
        self.assertTrue(np.allclose(out["czt_log_counts"],  np.log10(11.0)))

    def test_constant_rolling_mean_equals_constant(self):
        df  = make_constant_helios_df(n=10, cdte_val=20.0, czt_val=40.0, cadence_s=1)
        hbf = HEL1OSBasicFeatures(HEL1OSBasicFeatureConfig(windows_sec=[5]))
        out = hbf.transform(df)
        self.assertTrue(np.allclose(out["cdte_rolling_mean_5s"].iloc[4:], 20.0))
        self.assertTrue(np.allclose(out["czt_rolling_mean_5s"].iloc[4:],  40.0))

    def test_constant_std_is_zero(self):
        df  = make_constant_helios_df(n=10, cdte_val=20.0, czt_val=40.0, cadence_s=1)
        hbf = HEL1OSBasicFeatures(HEL1OSBasicFeatureConfig(windows_sec=[5]))
        out = hbf.transform(df)
        self.assertTrue(np.allclose(out["cdte_rolling_std_5s"].iloc[4:], 0.0, atol=1e-9))
        self.assertTrue(np.allclose(out["czt_rolling_std_5s"].iloc[4:],  0.0, atol=1e-9))

    def test_signal_energy_nonnegative(self):
        df  = make_synthetic_helios_flare()
        hbf = HEL1OSBasicFeatures(HEL1OSBasicFeatureConfig(windows_sec=[60]))
        out = hbf.transform(df)
        self.assertTrue((out["cdte_signal_energy_60s"].dropna() >= 0).all())
        self.assertTrue((out["czt_signal_energy_60s"].dropna() >= 0).all())


class TestHEL1OSBasicFeaturesEdgeCases(unittest.TestCase):
    def test_empty_dataframe(self):
        df  = make_empty_helios_df()
        hbf = HEL1OSBasicFeatures(HEL1OSBasicFeatureConfig(windows_sec=[60]))
        out = hbf.transform(df)
        self.assertEqual(len(out), 0)
        for name in hbf.feature_names():
            self.assertIn(name, out.columns)

    def test_missing_time_column_raises(self):
        df  = pd.DataFrame({"cdte_CR": [1.0], "czt_CR": [2.0]})
        hbf = HEL1OSBasicFeatures()
        with self.assertRaises(KeyError):
            hbf.transform(df)

    def test_missing_cdte_column_raises(self):
        t  = pd.date_range("2026-01-01", periods=3, freq="1s")
        df = pd.DataFrame({"time": t, "czt_CR": [1.0, 2.0, 3.0]})
        hbf = HEL1OSBasicFeatures()
        with self.assertRaises(KeyError):
            hbf.transform(df)

    def test_negative_values_clipped(self):
        t  = pd.date_range("2026-01-01", periods=5, freq="1s")
        df = pd.DataFrame({"time": t, "cdte_CR": [-5.0, -1.0, 0.0, 1.0, 5.0],
                           "czt_CR": [-3.0, -1.0, 0.0, 2.0, 4.0]})
        hbf = HEL1OSBasicFeatures(HEL1OSBasicFeatureConfig(windows_sec=[2], clip_negative=True))
        out = hbf.transform(df)
        self.assertFalse(out["cdte_log_counts"].isna().any())
        self.assertFalse(out["czt_log_counts"].isna().any())

    def test_idempotent(self):
        df  = make_synthetic_helios_flare()
        hbf = HEL1OSBasicFeatures(HEL1OSBasicFeatureConfig(windows_sec=[60]))
        pd.testing.assert_frame_equal(hbf.transform(df), hbf.transform(df))


if __name__ == "__main__":
    unittest.main()