"""Tests for spectral_features.py"""

import unittest

import numpy as np
import pandas as pd

from ml.features.spectral_features import SpectralFeatures, SpectralFeatureConfig
from .fixtures import make_empty_df, make_synthetic_flare


class TestSpectralFeaturesSmoke(unittest.TestCase):
    def test_default_construction(self):
        sf = SpectralFeatures()
        self.assertIsInstance(sf.config, SpectralFeatureConfig)

    def test_config_from_dict(self):
        cfg = SpectralFeatureConfig.from_dict(
            {"energy_bin_cols": {"E1": 5.0, "E2": 10.0}}
        )
        self.assertEqual(cfg.energy_bin_cols, {"E1": 5.0, "E2": 10.0})

    def test_feature_names_nonempty(self):
        sf = SpectralFeatures()
        names = sf.feature_names()
        self.assertIn("photon_index_2pt", names)
        self.assertIn("photon_index_fit", names)
        self.assertIn("spectral_mode", names)


class TestSpectralFeaturesShape(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_flare()
        self.sf = SpectralFeatures(
            SpectralFeatureConfig(soft_col="CR", hard_col="CR_hard")
        )

    def test_output_row_count_preserved(self):
        out = self.sf.transform(self.df)
        self.assertEqual(len(out), len(self.df))

    def test_feature_names_match_actual_new_columns(self):
        out = self.sf.transform(self.df)
        actual_new = set(out.columns) - set(self.df.columns)
        declared = set(self.sf.feature_names())
        self.assertEqual(actual_new, declared)

    def test_input_dataframe_not_mutated(self):
        df_copy = self.df.copy(deep=True)
        self.sf.transform(self.df)
        self.assertTrue(self.df.equals(df_copy))

    def test_spectral_mode_is_two_channel_without_energy_bins(self):
        out = self.sf.transform(self.df)
        self.assertTrue((out["spectral_mode"] == "two_channel").all())


class TestSpectralFeaturesValues(unittest.TestCase):
    def test_two_point_index_recovers_exact_power_law(self):
        """F(E) = E^-gamma at two known energies should recover gamma
        exactly via the two-point log-log slope (Sec 2.2)."""
        e_soft, e_hard = 5.0, 35.0
        gamma_true = 3.0
        f_soft = e_soft ** (-gamma_true)
        f_hard = e_hard ** (-gamma_true)
        t = pd.date_range("2020-01-01", periods=3, freq="1s")
        df = pd.DataFrame(
            {"time": t, "CR": [f_soft] * 3, "CR_hard": [f_hard] * 3}
        )
        cfg = SpectralFeatureConfig(
            soft_col="CR", hard_col="CR_hard",
            soft_energy_kev=e_soft, hard_energy_kev=e_hard,
            min_flux_for_fit=1e-12,
        )
        out = SpectralFeatures(cfg).transform(df)
        self.assertTrue(np.allclose(out["photon_index_2pt"], gamma_true, atol=1e-9))

    def test_multibin_fit_recovers_exact_power_law(self):
        energies = {"E1": 5.0, "E2": 10.0, "E3": 20.0, "E4": 40.0, "E5": 80.0}
        gamma_true = 2.5
        flux_vals = {k: v ** (-gamma_true) for k, v in energies.items()}
        t = pd.date_range("2020-01-01", periods=3, freq="1s")
        df = pd.DataFrame({"time": t, **{k: [v] * 3 for k, v in flux_vals.items()}})
        cfg = SpectralFeatureConfig(
            energy_bin_cols=energies, soft_col=None, hard_col=None,
            min_flux_for_fit=1e-15,
        )
        out = SpectralFeatures(cfg).transform(df)
        self.assertTrue(np.allclose(out["photon_index_fit"], gamma_true, atol=1e-9))
        self.assertTrue(np.allclose(out["spectral_fit_r2"], 1.0, atol=1e-9))
        self.assertTrue((out["spectral_mode"] == "multibin").all())

    def test_pure_power_law_has_near_zero_curvature(self):
        energies = {"E1": 5.0, "E2": 10.0, "E3": 20.0}
        gamma = 2.0
        flux = {k: v ** (-gamma) for k, v in energies.items()}
        t = pd.date_range("2020-01-01", periods=1, freq="1s")
        df = pd.DataFrame({"time": t, **{k: [v] for k, v in flux.items()}})
        cfg = SpectralFeatureConfig(
            energy_bin_cols=energies, soft_col=None, hard_col=None,
            min_flux_for_fit=1e-15,
        )
        out = SpectralFeatures(cfg).transform(df)
        self.assertAlmostEqual(out["spectral_curvature"].iloc[0], 0.0, places=9)

    def test_broken_spectrum_has_nonzero_curvature_and_lower_r2(self):
        """A spectrum that is NOT a clean power law (e.g. a thermal-like
        bump dominating the lowest energy bin, cf. Fig. 7's breaks)
        should show measurable curvature and R^2 below 1."""
        energies = {"E1": 5.0, "E2": 10.0, "E3": 20.0}
        flux = {"E1": 1000.0, "E2": 10.0, "E3": 2.0}
        t = pd.date_range("2020-01-01", periods=1, freq="1s")
        df = pd.DataFrame({"time": t, **{k: [v] for k, v in flux.items()}})
        cfg = SpectralFeatureConfig(
            energy_bin_cols=energies, soft_col=None, hard_col=None,
            min_flux_for_fit=1e-15,
        )
        out = SpectralFeatures(cfg).transform(df)
        self.assertGreater(abs(out["spectral_curvature"].iloc[0]), 1.0)
        self.assertLess(out["spectral_fit_r2"].iloc[0], 0.999)

    def test_hardness_ratio_2ch_known_value(self):
        t = pd.date_range("2020-01-01", periods=2, freq="1s")
        df = pd.DataFrame({"time": t, "CR": [10.0, 10.0], "CR_hard": [2.0, 5.0]})
        cfg = SpectralFeatureConfig(soft_col="CR", hard_col="CR_hard")
        out = SpectralFeatures(cfg).transform(df)
        self.assertAlmostEqual(out["hardness_ratio_2ch"].iloc[0], 0.2, places=4)
        self.assertAlmostEqual(out["hardness_ratio_2ch"].iloc[1], 0.5, places=4)

    def test_below_min_flux_yields_nan_index(self):
        t = pd.date_range("2020-01-01", periods=1, freq="1s")
        df = pd.DataFrame({"time": t, "CR": [1e-6], "CR_hard": [1e-6]})
        cfg = SpectralFeatureConfig(
            soft_col="CR", hard_col="CR_hard", min_flux_for_fit=1e-3
        )
        out = SpectralFeatures(cfg).transform(df)
        self.assertTrue(np.isnan(out["photon_index_2pt"].iloc[0]))


class TestSpectralFeaturesEdgeCases(unittest.TestCase):
    def test_empty_dataframe(self):
        df = make_empty_df()
        sf = SpectralFeatures(SpectralFeatureConfig(soft_col="CR", hard_col="CR_hard"))
        out = sf.transform(df)
        self.assertEqual(len(out), 0)
        for name in sf.feature_names():
            self.assertIn(name, out.columns)

    def test_missing_time_column_raises(self):
        df = pd.DataFrame({"CR": [1.0, 2.0]})
        sf = SpectralFeatures()
        with self.assertRaises(KeyError):
            sf.transform(df)

    def test_no_soft_hard_no_energy_bins_yields_nan_mode_none(self):
        t = pd.date_range("2020-01-01", periods=3, freq="1s")
        df = pd.DataFrame({"time": t, "other_col": [1.0, 2.0, 3.0]})
        cfg = SpectralFeatureConfig(soft_col=None, hard_col=None, energy_bin_cols=None)
        out = SpectralFeatures(cfg).transform(df)
        self.assertTrue((out["spectral_mode"] == "none").all())
        self.assertTrue(out["photon_index_2pt"].isna().all())
        self.assertTrue(out["photon_index_fit"].isna().all())

    def test_only_two_energy_bins_falls_back_to_two_channel_mode(self):
        """energy_bin_cols with only 2 valid bins is below the minimum
        of 3 needed for a curvature-capable fit, so multibin mode should
        not activate even though energy_bin_cols was provided."""
        t = pd.date_range("2020-01-01", periods=2, freq="1s")
        df = pd.DataFrame({"time": t, "E1": [10.0, 10.0], "E2": [5.0, 5.0]})
        cfg = SpectralFeatureConfig(
            energy_bin_cols={"E1": 5.0, "E2": 10.0},
            soft_col=None, hard_col=None,
        )
        out = SpectralFeatures(cfg).transform(df)
        self.assertTrue(out["photon_index_fit"].isna().all())
        self.assertTrue((out["spectral_mode"] == "none").all())

    def test_single_row_input_two_channel(self):
        t = pd.date_range("2020-01-01", periods=1, freq="4s")
        df = pd.DataFrame({"time": t, "CR": [42.0], "CR_hard": [3.0]})
        sf = SpectralFeatures(SpectralFeatureConfig(soft_col="CR", hard_col="CR_hard"))
        out = sf.transform(df)
        self.assertEqual(len(out), 1)
        self.assertTrue(np.isfinite(out["photon_index_2pt"].iloc[0]))

    def test_single_row_input_multibin(self):
        t = pd.date_range("2020-01-01", periods=1, freq="4s")
        energies = {"E1": 5.0, "E2": 10.0, "E3": 20.0}
        df = pd.DataFrame({"time": t, "E1": [100.0], "E2": [20.0], "E3": [5.0]})
        cfg = SpectralFeatureConfig(energy_bin_cols=energies, soft_col=None, hard_col=None)
        out = SpectralFeatures(cfg).transform(df)
        self.assertEqual(len(out), 1)
        self.assertTrue(np.isfinite(out["photon_index_fit"].iloc[0]))

    def test_zero_flux_does_not_crash(self):
        t = pd.date_range("2020-01-01", periods=5, freq="1s")
        df = pd.DataFrame({"time": t, "CR": [0.0] * 5, "CR_hard": [0.0] * 5})
        sf = SpectralFeatures(SpectralFeatureConfig(soft_col="CR", hard_col="CR_hard"))
        out = sf.transform(df)
        self.assertTrue(out["photon_index_2pt"].isna().all())

    def test_idempotent_on_repeated_calls(self):
        df = make_synthetic_flare()
        sf = SpectralFeatures(SpectralFeatureConfig(soft_col="CR", hard_col="CR_hard"))
        out1 = sf.transform(df)
        out2 = sf.transform(df)
        pd.testing.assert_frame_equal(out1, out2)


if __name__ == "__main__":
    unittest.main()
