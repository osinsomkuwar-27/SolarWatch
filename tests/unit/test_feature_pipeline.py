"""Tests for feature_pipeline.py"""

import unittest

import numpy as np
import pandas as pd

from ml.features import FeaturePipeline, PipelineConfig
from ml.features.basic_features import BasicFeatureConfig
from .fixtures import make_empty_df, make_synthetic_flare


class TestPipelineConfigSmoke(unittest.TestCase):
    def test_default_config(self):
        cfg = PipelineConfig()
        self.assertTrue(cfg.enable_basic)
        self.assertTrue(cfg.enable_temporal)
        self.assertTrue(cfg.enable_flare)
        self.assertTrue(cfg.enable_spectral)

    def test_from_dict_nested(self):
        cfg = PipelineConfig.from_dict(
            {
                "basic": {"windows_sec": [60, 300]},
                "temporal": {"lags_sec": [12, 60]},
                "flare": {"soft_col": "CR", "hard_col": "CR_hard"},
                "spectral": {"soft_col": "CR", "hard_col": "CR_hard"},
                "enable_spectral": False,
            }
        )
        self.assertEqual(cfg.basic.windows_sec, [60, 300])
        self.assertEqual(cfg.temporal.lags_sec, [12, 60])
        self.assertEqual(cfg.flare.hard_col, "CR_hard")
        self.assertFalse(cfg.enable_spectral)

    def test_from_dict_omitted_stage_uses_defaults(self):
        cfg = PipelineConfig.from_dict({"basic": {"windows_sec": [99]}})
        self.assertEqual(cfg.basic.windows_sec, [99])
        # temporal/flare/spectral untouched -> their own defaults
        self.assertEqual(cfg.temporal.lags_sec, [12, 60])

    def test_to_dict_roundtrip_shape(self):
        cfg = PipelineConfig()
        d = cfg.to_dict()
        self.assertIn("basic", d)
        self.assertIn("enable_basic", d)

    def test_direct_construction_with_stage_configs(self):
        cfg = PipelineConfig(basic=BasicFeatureConfig(windows_sec=[15]))
        self.assertEqual(cfg.basic.windows_sec, [15])


class TestFeaturePipelineShape(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_flare()
        self.cfg = PipelineConfig.from_dict(
            {
                "basic": {"windows_sec": [60, 300]},
                "temporal": {"lags_sec": [12, 60]},
                "flare": {"soft_col": "CR", "hard_col": "CR_hard"},
                "spectral": {"soft_col": "CR", "hard_col": "CR_hard"},
            }
        )
        self.pipe = FeaturePipeline(self.cfg)

    def test_output_row_count_preserved(self):
        out = self.pipe.transform(self.df)
        self.assertEqual(len(out), len(self.df))

    def test_all_original_columns_preserved(self):
        out = self.pipe.transform(self.df)
        for col in self.df.columns:
            self.assertIn(col, out.columns)

    def test_feature_names_matches_actual_output(self):
        out = self.pipe.transform(self.df)
        actual_new = set(out.columns) - set(self.df.columns)
        declared = set(self.pipe.feature_names())
        self.assertEqual(actual_new, declared)

    def test_input_dataframe_not_mutated(self):
        df_copy = self.df.copy(deep=True)
        self.pipe.transform(self.df)
        self.assertTrue(self.df.equals(df_copy))

    def test_stage_count_matches_enabled_stages(self):
        self.assertEqual(len(self.pipe._stages), 4)
        cfg2 = PipelineConfig(enable_spectral=False)
        pipe2 = FeaturePipeline(cfg2)
        self.assertEqual(len(pipe2._stages), 3)


class TestFeaturePipelineStageToggling(unittest.TestCase):
    def test_disabling_all_but_basic(self):
        cfg = PipelineConfig(
            enable_basic=True, enable_temporal=False,
            enable_flare=False, enable_spectral=False,
        )
        pipe = FeaturePipeline(cfg)
        df = make_synthetic_flare()
        out = pipe.transform(df)
        self.assertIn("log_counts", out.columns)
        self.assertNotIn("dCR_dt", out.columns)
        self.assertNotIn("flare_phase", out.columns)
        self.assertNotIn("photon_index_2pt", out.columns)

    def test_disabling_all_stages_returns_input_unchanged(self):
        cfg = PipelineConfig(
            enable_basic=False, enable_temporal=False,
            enable_flare=False, enable_spectral=False,
        )
        pipe = FeaturePipeline(cfg)
        df = make_synthetic_flare()
        out = pipe.transform(df)
        self.assertEqual(list(out.columns), list(df.columns))
        self.assertEqual(pipe.feature_names(), [])

    def test_basic_and_flare_only(self):
        cfg = PipelineConfig(
            enable_basic=True, enable_temporal=False,
            enable_flare=True, enable_spectral=False,
            flare=PipelineConfig().flare,
        )
        cfg.flare.hard_col = "CR_hard"
        pipe = FeaturePipeline(cfg)
        df = make_synthetic_flare()
        out = pipe.transform(df)
        self.assertIn("log_counts", out.columns)
        self.assertIn("flare_phase", out.columns)
        self.assertNotIn("dCR_dt", out.columns)
        self.assertNotIn("photon_index_2pt", out.columns)


class TestFeaturePipelineTransformSeparately(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_flare()
        self.cfg = PipelineConfig.from_dict(
            {
                "flare": {"soft_col": "CR", "hard_col": "CR_hard"},
                "spectral": {"soft_col": "CR", "hard_col": "CR_hard"},
            }
        )
        self.pipe = FeaturePipeline(self.cfg)

    def test_returns_dict_with_all_stage_names(self):
        results = self.pipe.transform_separately(self.df)
        self.assertEqual(set(results.keys()), {"basic", "temporal", "flare", "spectral"})

    def test_each_stage_receives_raw_input_not_chained(self):
        results = self.pipe.transform_separately(self.df)
        # flare stage output should NOT contain basic_features' columns,
        # since transform_separately does not chain stages
        self.assertFalse(any("rolling_mean" in c for c in results["flare"].columns))
        self.assertTrue(any("rolling_mean" in c for c in results["basic"].columns))

    def test_each_stage_output_has_consistent_row_count(self):
        results = self.pipe.transform_separately(self.df)
        for name, out in results.items():
            self.assertEqual(len(out), len(self.df), msg=f"stage {name} row count mismatch")


class TestFeaturePipelineErrorHandling(unittest.TestCase):
    def test_missing_required_column_raises_with_stage_name(self):
        df = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=5, freq="1s")})
        pipe = FeaturePipeline(PipelineConfig())
        with self.assertRaises(RuntimeError) as ctx:
            pipe.transform(df)
        self.assertIn("basic", str(ctx.exception))

    def test_transform_separately_also_raises_with_stage_name(self):
        df = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=5, freq="1s")})
        pipe = FeaturePipeline(PipelineConfig())
        with self.assertRaises(RuntimeError) as ctx:
            pipe.transform_separately(df)
        self.assertIn("basic", str(ctx.exception))


class TestFeaturePipelineEdgeCases(unittest.TestCase):
    def test_empty_dataframe_full_pipeline(self):
        df = make_empty_df()
        cfg = PipelineConfig.from_dict(
            {
                "flare": {"soft_col": "CR", "hard_col": "CR_hard"},
                "spectral": {"soft_col": "CR", "hard_col": "CR_hard"},
            }
        )
        pipe = FeaturePipeline(cfg)
        out = pipe.transform(df)
        self.assertEqual(len(out), 0)

    def test_single_row_full_pipeline_no_crash(self):
        t = pd.date_range("2020-01-01", periods=1, freq="4s")
        df = pd.DataFrame({"time": t, "CR": [42.0], "CR_hard": [3.0]})
        cfg = PipelineConfig.from_dict(
            {
                "flare": {"soft_col": "CR", "hard_col": "CR_hard"},
                "spectral": {"soft_col": "CR", "hard_col": "CR_hard"},
            }
        )
        pipe = FeaturePipeline(cfg)
        out = pipe.transform(df)
        self.assertEqual(len(out), 1)
        numeric = out.select_dtypes(include=[float])
        self.assertFalse(np.isinf(numeric.to_numpy()).any())

    def test_idempotent_on_repeated_calls(self):
        df = make_synthetic_flare()
        cfg = PipelineConfig.from_dict(
            {
                "flare": {"soft_col": "CR", "hard_col": "CR_hard"},
                "spectral": {"soft_col": "CR", "hard_col": "CR_hard"},
            }
        )
        pipe = FeaturePipeline(cfg)
        out1 = pipe.transform(df)
        out2 = pipe.transform(df)
        pd.testing.assert_frame_equal(out1, out2)

    def test_full_pipeline_on_synthetic_flare_end_to_end(self):
        """Smoke test that the complete, realistic configuration runs
        end-to-end and produces sane, finite values throughout."""
        df = make_synthetic_flare()
        cfg = PipelineConfig.from_dict(
            {
                "basic": {"windows_sec": [60, 300]},
                "temporal": {"lags_sec": [12, 60], "ema_spans_sec": [60, 300]},
                "flare": {"soft_col": "CR", "hard_col": "CR_hard"},
                "spectral": {"soft_col": "CR", "hard_col": "CR_hard"},
            }
        )
        pipe = FeaturePipeline(cfg)
        out = pipe.transform(df)
        self.assertEqual(len(out), len(df))
        self.assertGreater(len(out.columns), len(df.columns))
        # at least the core physically-meaningful columns must be present
        for col in ["log_counts", "dCR_dt", "flare_phase", "neupert_corr", "photon_index_2pt"]:
            self.assertIn(col, out.columns)


if __name__ == "__main__":
    unittest.main()
