"""Tests for helios_flare_features.py"""

import unittest

import numpy as np
import pandas as pd

from ml.features.helios_features.helios_flare_features import (
    HEL1OSFlareFeatures,
    HEL1OSFlareFeatureConfig,
    HXR_PHASES,
)
from .helios_fixtures import (
    make_empty_helios_df,
    make_constant_helios_df,
    make_synthetic_helios_flare,
)


class TestHEL1OSFlareFeaturesSmoke(unittest.TestCase):
    def test_default_construction(self):
        hff = HEL1OSFlareFeatures()
        self.assertIsInstance(hff.config, HEL1OSFlareFeatureConfig)

    def test_feature_names_includes_hardness_and_fluence(self):
        hff   = HEL1OSFlareFeatures()
        names = hff.feature_names()
        self.assertIn("hardness_ratio",        names)
        self.assertIn("hardness_smoothed",     names)
        self.assertIn("cdte_cumulative_fluence", names)
        self.assertIn("czt_cumulative_fluence",  names)
        self.assertIn("hxr_phase",             names)

    def test_feature_names_includes_all_phases(self):
        hff = HEL1OSFlareFeatures()
        for p in HXR_PHASES:
            self.assertIn(f"hxr_phase_{p}", hff.feature_names())

    def test_feature_names_includes_detector_stats(self):
        hff = HEL1OSFlareFeatures()
        for stat in ("mean", "std", "p90"):
            self.assertIn(f"cdte_stat_{stat}", hff.feature_names())
            self.assertIn(f"czt_stat_{stat}",  hff.feature_names())


class TestHEL1OSFlareFeaturesShape(unittest.TestCase):
    def setUp(self):
        self.df  = make_synthetic_helios_flare()
        self.hff = HEL1OSFlareFeatures()

    def test_output_row_count_preserved(self):
        out = self.hff.transform(self.df)
        self.assertEqual(len(out), len(self.df))

    def test_original_columns_preserved(self):
        out = self.hff.transform(self.df)
        for col in self.df.columns:
            self.assertIn(col, out.columns)

    def test_feature_names_match_actual_new_columns(self):
        out        = self.hff.transform(self.df)
        actual_new = set(out.columns) - set(self.df.columns)
        declared   = set(self.hff.feature_names())
        self.assertEqual(actual_new, declared)

    def test_input_not_mutated(self):
        df_copy = self.df.copy(deep=True)
        self.hff.transform(self.df)
        self.assertTrue(self.df.equals(df_copy))


class TestHEL1OSFlareFeaturesValues(unittest.TestCase):
    def test_hardness_ratio_positive_for_positive_inputs(self):
        df  = make_synthetic_helios_flare()
        hff = HEL1OSFlareFeatures()
        out = hff.transform(df)
        valid = out["hardness_ratio"].dropna()
        self.assertTrue((valid >= 0).all())

    def test_hardness_ratio_large_when_cdte_dominates(self):
        """When CdTe >> CZT, hardness ratio should be >> 1."""
        t   = pd.date_range("2026-01-01", periods=10, freq="1s")
        df  = pd.DataFrame({"time": t, "cdte_CR": [100.0] * 10, "czt_CR": [1.0] * 10})
        hff = HEL1OSFlareFeatures()
        out = hff.transform(df)
        self.assertTrue((out["hardness_ratio"].dropna() > 10).all())

    def test_cumulative_fluence_nondecreasing(self):
        df  = make_synthetic_helios_flare()
        hff = HEL1OSFlareFeatures()
        out = hff.transform(df)
        for col in ("cdte_cumulative_fluence", "czt_cumulative_fluence"):
            flu = out[col].to_numpy()
            self.assertTrue(np.all(np.diff(flu) >= -1e-6))

    def test_hxr_phase_values_in_valid_set(self):
        df  = make_synthetic_helios_flare()
        hff = HEL1OSFlareFeatures()
        out = hff.transform(df)
        valid_phases = set(HXR_PHASES)
        unique       = set(out["hxr_phase"].unique())
        self.assertTrue(unique.issubset(valid_phases))

    def test_phase_dummy_columns_sum_to_one(self):
        df  = make_synthetic_helios_flare()
        hff = HEL1OSFlareFeatures()
        out = hff.transform(df)
        dummy_cols = [f"hxr_phase_{p}" for p in HXR_PHASES]
        row_sums   = out[dummy_cols].sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))

    def test_detector_stats_are_scalars_broadcast_across_rows(self):
        df  = make_synthetic_helios_flare()
        hff = HEL1OSFlareFeatures()
        out = hff.transform(df)
        # Scalar stats are broadcast → all rows equal the first row
        for col in ("cdte_stat_mean", "czt_stat_mean"):
            self.assertTrue(out[col].nunique() == 1)

    def test_constant_series_hardness_is_constant(self):
        df  = make_constant_helios_df(n=20, cdte_val=20.0, czt_val=40.0, cadence_s=1)
        hff = HEL1OSFlareFeatures()
        out = hff.transform(df)
        valid = out["hardness_ratio"].dropna()
        self.assertTrue(np.allclose(valid, 20.0 / (40.0 + 1e-12), rtol=1e-3))

    def test_empty_dataframe_no_crash(self):
        df  = make_empty_helios_df()
        hff = HEL1OSFlareFeatures()
        out = hff.transform(df)
        self.assertEqual(len(out), 0)
        self.assertIn("hardness_ratio", out.columns)

    def test_missing_czte_column_raises(self):
        t  = pd.date_range("2026-01-01", periods=5, freq="1s")
        df = pd.DataFrame({"time": t, "cdte_CR": [1.0] * 5})
        hff = HEL1OSFlareFeatures()
        with self.assertRaises(KeyError):
            hff.transform(df)

    def test_idempotent(self):
        df  = make_synthetic_helios_flare()
        hff = HEL1OSFlareFeatures()
        pd.testing.assert_frame_equal(hff.transform(df), hff.transform(df))


if __name__ == "__main__":
    unittest.main()