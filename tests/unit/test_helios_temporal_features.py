"""Tests for helios_temporal_features.py"""

import unittest

import numpy as np
import pandas as pd

from ml.features.helios_features.helios_temporal_features import (
    HEL1OSTemporalFeatures,
    HEL1OSTemporalFeatureConfig,
)
from .helios_fixtures import (
    make_empty_helios_df,
    make_constant_helios_df,
    make_synthetic_helios_flare,
)


class TestHEL1OSTemporalFeaturesSmoke(unittest.TestCase):
    def test_default_construction(self):
        htf = HEL1OSTemporalFeatures()
        self.assertIsInstance(htf.config, HEL1OSTemporalFeatureConfig)

    def test_config_from_dict(self):
        cfg = HEL1OSTemporalFeatureConfig.from_dict({"lags_sec": [24, 120]})
        self.assertEqual(cfg.lags_sec, [24, 120])

    def test_feature_names_include_both_detectors(self):
        htf   = HEL1OSTemporalFeatures()
        names = htf.feature_names()
        self.assertIn("cdte_dCR_dt",  names)
        self.assertIn("czt_dCR_dt",   names)
        self.assertIn("cdte_cumulative_fluence", names)
        self.assertIn("czt_cumulative_fluence",  names)
        self.assertIn("time_since_start_s", names)


class TestHEL1OSTemporalFeaturesShape(unittest.TestCase):
    def setUp(self):
        self.df  = make_synthetic_helios_flare()
        self.htf = HEL1OSTemporalFeatures()

    def test_output_row_count_preserved(self):
        out = self.htf.transform(self.df)
        self.assertEqual(len(out), len(self.df))

    def test_original_columns_preserved(self):
        out = self.htf.transform(self.df)
        for col in self.df.columns:
            self.assertIn(col, out.columns)

    def test_feature_names_match_actual_new_columns(self):
        out        = self.htf.transform(self.df)
        actual_new = set(out.columns) - set(self.df.columns)
        declared   = set(self.htf.feature_names())
        self.assertEqual(actual_new, declared)

    def test_input_not_mutated(self):
        df_copy = self.df.copy(deep=True)
        self.htf.transform(self.df)
        self.assertTrue(self.df.equals(df_copy))


class TestHEL1OSTemporalFeaturesValues(unittest.TestCase):
    def test_cumulative_fluence_monotonically_nondecreasing(self):
        """Fluence = ∫ CR dt; CR ≥ 0 after clipping, so fluence is non-decreasing."""
        df  = make_synthetic_helios_flare()
        htf = HEL1OSTemporalFeatures()
        out = htf.transform(df)
        cdte_flu = out["cdte_cumulative_fluence"].to_numpy()
        self.assertTrue(np.all(np.diff(cdte_flu) >= -1e-6))   # allow tiny rounding

    def test_time_since_start_starts_at_zero(self):
        df  = make_synthetic_helios_flare()
        htf = HEL1OSTemporalFeatures()
        out = htf.transform(df)
        self.assertAlmostEqual(out["time_since_start_s"].iloc[0], 0.0, places=3)

    def test_constant_series_derivative_near_zero(self):
        df  = make_constant_helios_df(n=20, cdte_val=20.0, czt_val=40.0, cadence_s=1)
        htf = HEL1OSTemporalFeatures()
        out = htf.transform(df)
        self.assertTrue(np.allclose(out["cdte_dCR_dt"].to_numpy(), 0.0, atol=1e-6))
        self.assertTrue(np.allclose(out["czt_dCR_dt"].to_numpy(),  0.0, atol=1e-6))

    def test_lag_diff_zero_for_constant(self):
        df  = make_constant_helios_df(n=20, cdte_val=20.0, czt_val=40.0, cadence_s=1)
        htf = HEL1OSTemporalFeatures(HEL1OSTemporalFeatureConfig(lags_sec=[5]))
        out = htf.transform(df)
        valid = out["cdte_lag_diff_5s"].dropna()
        self.assertTrue(np.allclose(valid, 0.0, atol=1e-9))

    def test_empty_dataframe_no_crash(self):
        df  = make_empty_helios_df()
        htf = HEL1OSTemporalFeatures()
        out = htf.transform(df)
        self.assertEqual(len(out), 0)

    def test_idempotent(self):
        df  = make_synthetic_helios_flare()
        htf = HEL1OSTemporalFeatures()
        pd.testing.assert_frame_equal(htf.transform(df), htf.transform(df))


if __name__ == "__main__":
    unittest.main()