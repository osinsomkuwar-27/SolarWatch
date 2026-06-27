"""Tests for flare_features.py"""

import unittest

import numpy as np
import pandas as pd

from ml.features.flare_features import FlareFeatures, FlareFeatureConfig, PHASES
from .fixtures import make_empty_df, make_constant_df, make_synthetic_flare


class TestFlareFeaturesSmoke(unittest.TestCase):
    def test_default_construction(self):
        ff = FlareFeatures()
        self.assertIsInstance(ff.config, FlareFeatureConfig)

    def test_config_from_dict(self):
        cfg = FlareFeatureConfig.from_dict({"soft_col": "CR", "hard_col": "CR_hard"})
        self.assertEqual(cfg.hard_col, "CR_hard")

    def test_feature_names_nonempty(self):
        ff = FlareFeatures()
        names = ff.feature_names()
        self.assertIn("flare_phase", names)
        self.assertIn("neupert_corr", names)
        self.assertIn("hardness_ratio", names)
        for p in PHASES:
            self.assertIn(f"phase_{p}", names)


class TestFlareFeaturesShape(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_flare()
        self.ff = FlareFeatures(FlareFeatureConfig(soft_col="CR", hard_col="CR_hard"))

    def test_output_row_count_preserved(self):
        out = self.ff.transform(self.df)
        self.assertEqual(len(out), len(self.df))

    def test_feature_names_match_actual_new_columns(self):
        out = self.ff.transform(self.df)
        actual_new = set(out.columns) - set(self.df.columns)
        declared = set(self.ff.feature_names())
        self.assertEqual(actual_new, declared)

    def test_input_dataframe_not_mutated(self):
        df_copy = self.df.copy(deep=True)
        self.ff.transform(self.df)
        self.assertTrue(self.df.equals(df_copy))

    def test_phase_one_hot_columns_sum_to_one(self):
        out = self.ff.transform(self.df)
        total = sum(out[f"phase_{p}"] for p in PHASES)
        self.assertTrue((total == 1).all())

    def test_phase_one_hot_matches_flare_phase_column(self):
        out = self.ff.transform(self.df)
        for p in PHASES:
            mask = out["flare_phase"] == p
            self.assertTrue((out.loc[mask, f"phase_{p}"] == 1).all())
            self.assertTrue((out.loc[~mask, f"phase_{p}"] == 0).all())


class TestFlareFeaturesNeupertAndHardness(unittest.TestCase):
    def test_neupert_corr_high_during_impulsive_rise(self):
        """d(SXR)/dt should correlate strongly with HXR during/just after
        the impulsive phase, since the synthetic soft channel is built
        as the time-integral of the hard channel (Sec 2.4, Eq 1-2)."""
        df = make_synthetic_flare()
        ff = FlareFeatures(FlareFeatureConfig(soft_col="CR", hard_col="CR_hard"))
        out = ff.transform(df)
        impulsive_end_s = df.attrs["impulsive_end_s"]
        window = out[
            (out.index >= 0)
            & (df["time"] - df["time"].iloc[0]).dt.total_seconds().between(
                impulsive_end_s - 120, impulsive_end_s + 60
            )
        ]
        # Should be strongly positive (Neupert effect holds by construction)
        self.assertGreater(window["neupert_corr"].dropna().mean(), 0.8)

    def test_hardness_peaks_during_impulsive_not_decay(self):
        """Soft-hard-soft: hardness should be higher during the impulsive
        rise than during the decay phase (Sec 5.2)."""
        df = make_synthetic_flare()
        ff = FlareFeatures(FlareFeatureConfig(soft_col="CR", hard_col="CR_hard"))
        out = ff.transform(df)
        impulsive_mask = out["flare_phase"] == "impulsive"
        decay_mask = out["flare_phase"] == "decay"
        if impulsive_mask.any() and decay_mask.any():
            impulsive_hardness = out.loc[impulsive_mask, "hardness_smoothed"].mean()
            decay_hardness = out.loc[decay_mask, "hardness_smoothed"].mean()
            self.assertGreater(impulsive_hardness, decay_hardness)

    def test_missing_hard_col_yields_nan_not_crash(self):
        df = make_synthetic_flare()
        ff = FlareFeatures(FlareFeatureConfig(soft_col="CR", hard_col=None))
        out = ff.transform(df)
        self.assertTrue(out["neupert_corr"].isna().all())
        self.assertTrue(out["hardness_ratio"].isna().all())
        self.assertTrue(out["hxr_cumulative_fluence"].isna().all())

    def test_hxr_cumulative_fluence_is_monotonic_nondecreasing(self):
        df = make_synthetic_flare()
        ff = FlareFeatures(FlareFeatureConfig(soft_col="CR", hard_col="CR_hard"))
        out = ff.transform(df)
        diffs = np.diff(out["hxr_cumulative_fluence"].to_numpy())
        self.assertTrue((diffs >= -1e-9).all())  # nondecreasing (integral of >=0 flux)


class TestFlareFeaturesPhaseClassification(unittest.TestCase):
    """These tests encode the ground truth built into make_synthetic_flare:
    preflare [0, 300s), impulsive starting ~300s, flash around the peak,
    decay after the peak, with the hard channel having died down."""

    def setUp(self):
        self.df = make_synthetic_flare()
        self.ff = FlareFeatures(FlareFeatureConfig(soft_col="CR", hard_col="CR_hard"))
        self.out = self.ff.transform(self.df)
        self.t = (self.df["time"] - self.df["time"].iloc[0]).dt.total_seconds()

    def test_early_preflare_samples_classified_preflare(self):
        early = self.out[self.t < 100]
        self.assertTrue((early["flare_phase"] == "preflare").mean() > 0.9)

    def test_all_four_phases_present(self):
        seen = set(self.out["flare_phase"].unique())
        self.assertEqual(seen, set(PHASES))

    def test_phase_transitions_are_monotonic_and_ordered(self):
        """The phase sequence should visit preflare -> impulsive -> flash
        -> decay with no flickering back to an earlier phase once it has
        moved on (a single clean pass through Benz 2008 Fig. 2's phase
        order), for this single, cleanly-separated synthetic flare."""
        order = {"preflare": 0, "impulsive": 1, "flash": 2, "decay": 3}
        phases = self.out["flare_phase"].to_numpy()
        ranks = np.array([order[p] for p in phases])
        # ranks should be non-decreasing throughout
        self.assertTrue((np.diff(ranks) >= 0).all())

    def test_impulsive_onset_near_ground_truth(self):
        impulsive_start_s = self.df.attrs["preflare_end_s"]
        first_impulsive_t = self.t[self.out["flare_phase"] == "impulsive"].iloc[0]
        # within 30s of the true preflare/impulsive boundary
        self.assertLess(abs(first_impulsive_t - impulsive_start_s), 30)

    def test_decay_occurs_after_peak(self):
        peak_t = self.df.attrs["peak_time_s"]
        decay_times = self.t[self.out["flare_phase"] == "decay"]
        if len(decay_times):
            self.assertGreaterEqual(decay_times.min(), peak_t - 30)

    def test_quiescent_series_is_all_preflare(self):
        """A flat, noiseless series with no flare should be classified
        entirely as preflare (quiescent), never impulsive/flash/decay."""
        df = make_constant_df(n=50, value=10.0, cadence_s=4)
        out = self.ff.transform(df)
        self.assertTrue((out["flare_phase"] == "preflare").all())


class TestFlareFeaturesEdgeCases(unittest.TestCase):
    def test_empty_dataframe(self):
        df = make_empty_df()
        ff = FlareFeatures(FlareFeatureConfig(soft_col="CR", hard_col="CR_hard"))
        out = ff.transform(df)
        self.assertEqual(len(out), 0)
        for name in ff.feature_names():
            self.assertIn(name, out.columns)

    def test_missing_soft_col_raises(self):
        df = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=3, freq="1s")})
        ff = FlareFeatures(FlareFeatureConfig(soft_col="CR"))
        with self.assertRaises(KeyError):
            ff.transform(df)

    def test_hard_col_specified_but_absent_raises(self):
        df = pd.DataFrame(
            {"time": pd.date_range("2020-01-01", periods=3, freq="1s"), "CR": [1, 2, 3]}
        )
        ff = FlareFeatures(FlareFeatureConfig(soft_col="CR", hard_col="nonexistent"))
        with self.assertRaises(KeyError):
            ff.transform(df)

    def test_single_row_input_no_crash(self):
        t = pd.date_range("2020-01-01", periods=1, freq="4s")
        df = pd.DataFrame({"time": t, "CR": [42.0], "CR_hard": [3.0]})
        ff = FlareFeatures(FlareFeatureConfig(soft_col="CR", hard_col="CR_hard"))
        out = ff.transform(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out["flare_phase"].iloc[0], "preflare")

    def test_flat_zero_series_no_crash(self):
        t = pd.date_range("2020-01-01", periods=20, freq="4s")
        df = pd.DataFrame({"time": t, "CR": [0.0] * 20, "CR_hard": [0.0] * 20})
        ff = FlareFeatures(FlareFeatureConfig(soft_col="CR", hard_col="CR_hard"))
        out = ff.transform(df)
        self.assertTrue((out["flare_phase"] == "preflare").all())

    def test_idempotent_on_repeated_calls(self):
        df = make_synthetic_flare()
        ff = FlareFeatures(FlareFeatureConfig(soft_col="CR", hard_col="CR_hard"))
        out1 = ff.transform(df)
        out2 = ff.transform(df)
        pd.testing.assert_frame_equal(out1, out2)


if __name__ == "__main__":
    unittest.main()
