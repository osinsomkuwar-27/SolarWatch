"""Tests for helios_spectral_features.py"""

import unittest

import numpy as np
import pandas as pd

from ml.features.helios_features.helios_spectral_features import (
    HEL1OSSpectralFeatures,
    HEL1OSSpectralFeatureConfig,
)
from .helios_fixtures import (
    make_empty_helios_df,
    make_constant_helios_df,
    make_synthetic_helios_flare,
)


class TestHEL1OSSpectralFeaturesSmoke(unittest.TestCase):
    def test_default_construction(self):
        hsf = HEL1OSSpectralFeatures()
        self.assertIsInstance(hsf.config, HEL1OSSpectralFeatureConfig)

    def test_feature_names_declared(self):
        hsf   = HEL1OSSpectralFeatures()
        names = hsf.feature_names()
        for expected in [
            "hxr_photon_index_2pt",
            "hxr_hardness_ratio_2ch",
            "hxr_photon_index_fit",
            "hxr_spectral_fit_r2",
            "hxr_spectral_curvature",
            "hxr_spectral_slope",
            "hxr_spectral_mode",
        ]:
            self.assertIn(expected, names)


class TestHEL1OSSpectralFeaturesShape(unittest.TestCase):
    def setUp(self):
        self.df  = make_synthetic_helios_flare()
        self.hsf = HEL1OSSpectralFeatures()

    def test_output_row_count_preserved(self):
        out = self.hsf.transform(self.df)
        self.assertEqual(len(out), len(self.df))

    def test_original_columns_preserved(self):
        out = self.hsf.transform(self.df)
        for col in self.df.columns:
            self.assertIn(col, out.columns)

    def test_feature_names_match_actual_new_columns(self):
        out        = self.hsf.transform(self.df)
        actual_new = set(out.columns) - set(self.df.columns)
        declared   = set(self.hsf.feature_names())
        self.assertEqual(actual_new, declared)

    def test_input_not_mutated(self):
        df_copy = self.df.copy(deep=True)
        self.hsf.transform(self.df)
        self.assertTrue(self.df.equals(df_copy))


class TestHEL1OSSpectralFeaturesValues(unittest.TestCase):
    def test_two_channel_mode_selected_for_default_config(self):
        df  = make_synthetic_helios_flare()
        hsf = HEL1OSSpectralFeatures()
        out = hsf.transform(df)
        self.assertTrue((out["hxr_spectral_mode"] == "two_channel").all())

    def test_photon_index_finite_during_flare_peak(self):
        """During the flare peak both channels are well above min_flux_for_fit."""
        df  = make_synthetic_helios_flare()
        hsf = HEL1OSSpectralFeatures()
        out = hsf.transform(df)
        peak_idx   = df["cdte_CR"].idxmax()
        gamma_peak = out["hxr_photon_index_2pt"].iloc[peak_idx]
        self.assertTrue(np.isfinite(gamma_peak))

    def test_hardness_ratio_2ch_positive(self):
        df  = make_synthetic_helios_flare()
        hsf = HEL1OSSpectralFeatures()
        out = hsf.transform(df)
        valid = out["hxr_hardness_ratio_2ch"].dropna()
        self.assertTrue((valid >= 0).all())

    def test_multibin_mode_with_three_bins(self):
        """When three bin columns are present, multibin mode is selected."""
        t    = pd.date_range("2026-01-01", periods=10, freq="4s")
        df   = pd.DataFrame({
            "time":       t,
            "cdte_CR":    [20.0] * 10,
            "czt_CR":     [40.0] * 10,
            "hxr_08_20":  [60.0] * 10,
            "hxr_20_60":  [30.0] * 10,
            "hxr_60_150": [10.0] * 10,
        })
        cfg = HEL1OSSpectralFeatureConfig(
            energy_bin_cols={"hxr_08_20": 14.0, "hxr_20_60": 40.0, "hxr_60_150": 100.0}
        )
        hsf = HEL1OSSpectralFeatures(cfg)
        out = hsf.transform(df)
        self.assertTrue((out["hxr_spectral_mode"] == "multibin").all())
        self.assertFalse(out["hxr_photon_index_fit"].isna().all())

    def test_no_spectral_columns_gives_mode_none(self):
        """If neither two-channel nor multibin columns are present → mode='none'."""
        t   = pd.date_range("2026-01-01", periods=5, freq="1s")
        df  = pd.DataFrame({"time": t})
        cfg = HEL1OSSpectralFeatureConfig(cdte_col=None, czt_col=None)
        hsf = HEL1OSSpectralFeatures(cfg)
        out = hsf.transform(df)
        self.assertTrue((out["hxr_spectral_mode"] == "none").all())
        self.assertTrue(out["hxr_photon_index_2pt"].isna().all())

    def test_empty_dataframe_no_crash(self):
        df  = make_empty_helios_df()
        hsf = HEL1OSSpectralFeatures()
        out = hsf.transform(df)
        self.assertEqual(len(out), 0)

    def test_idempotent(self):
        df  = make_synthetic_helios_flare()
        hsf = HEL1OSSpectralFeatures()
        pd.testing.assert_frame_equal(hsf.transform(df), hsf.transform(df))


if __name__ == "__main__":
    unittest.main()