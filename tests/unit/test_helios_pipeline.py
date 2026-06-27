"""Tests for helios_pipeline.py (HEL1OSFeaturePipeline) and
   CombinedFeaturePipeline in feature_pipeline.py."""

import unittest

import numpy as np
import pandas as pd

from ml.features.helios_features import (
    HEL1OSFeaturePipeline,
    HEL1OSPipelineConfig,
    HEL1OSBasicFeatureConfig,
)
from ml.features.feature_pipeline import CombinedFeaturePipeline
from ml.features import PipelineConfig
from .helios_fixtures import make_empty_helios_df, make_synthetic_helios_flare


class TestHEL1OSPipelineConfigSmoke(unittest.TestCase):
    def test_default_config(self):
        cfg = HEL1OSPipelineConfig()
        self.assertTrue(cfg.enable_basic)
        self.assertTrue(cfg.enable_temporal)
        self.assertTrue(cfg.enable_flare)
        self.assertTrue(cfg.enable_spectral)

    def test_from_dict_nested(self):
        cfg = HEL1OSPipelineConfig.from_dict({
            "basic":    {"windows_sec": [60, 300]},
            "temporal": {"lags_sec": [12]},
            "enable_spectral": False,
        })
        self.assertEqual(cfg.basic.windows_sec, [60, 300])
        self.assertEqual(cfg.temporal.lags_sec, [12])
        self.assertFalse(cfg.enable_spectral)

    def test_to_dict_roundtrip(self):
        cfg = HEL1OSPipelineConfig()
        d   = cfg.to_dict()
        self.assertIn("basic",         d)
        self.assertIn("enable_basic",  d)


class TestHEL1OSFeaturePipelineShape(unittest.TestCase):
    def setUp(self):
        self.df   = make_synthetic_helios_flare()
        self.pipe = HEL1OSFeaturePipeline(HEL1OSPipelineConfig())

    def test_output_row_count_preserved(self):
        out = self.pipe.transform(self.df)
        self.assertEqual(len(out), len(self.df))

    def test_all_original_columns_preserved(self):
        out = self.pipe.transform(self.df)
        for col in self.df.columns:
            self.assertIn(col, out.columns)

    def test_feature_names_matches_actual_output(self):
        out        = self.pipe.transform(self.df)
        actual_new = set(out.columns) - set(self.df.columns)
        declared   = set(self.pipe.feature_names())
        self.assertEqual(actual_new, declared)

    def test_input_not_mutated(self):
        df_copy = self.df.copy(deep=True)
        self.pipe.transform(self.df)
        self.assertTrue(self.df.equals(df_copy))

    def test_stage_count_matches_enabled(self):
        self.assertEqual(len(self.pipe._stages), 4)
        pipe2 = HEL1OSFeaturePipeline(HEL1OSPipelineConfig(enable_spectral=False))
        self.assertEqual(len(pipe2._stages), 3)


class TestHEL1OSPipelineStageToggling(unittest.TestCase):
    def test_basic_only(self):
        cfg  = HEL1OSPipelineConfig(
            enable_basic=True, enable_temporal=False,
            enable_flare=False, enable_spectral=False,
        )
        pipe = HEL1OSFeaturePipeline(cfg)
        out  = pipe.transform(make_synthetic_helios_flare())
        self.assertIn("cdte_log_counts", out.columns)
        self.assertNotIn("cdte_dCR_dt",      out.columns)
        self.assertNotIn("hardness_ratio",    out.columns)
        self.assertNotIn("hxr_photon_index_2pt", out.columns)

    def test_disable_all_returns_input_unchanged(self):
        cfg  = HEL1OSPipelineConfig(
            enable_basic=False, enable_temporal=False,
            enable_flare=False, enable_spectral=False,
        )
        pipe = HEL1OSFeaturePipeline(cfg)
        df   = make_synthetic_helios_flare()
        out  = pipe.transform(df)
        self.assertEqual(list(out.columns), list(df.columns))
        self.assertEqual(pipe.feature_names(), [])

    def test_transform_separately_returns_dict(self):
        pipe    = HEL1OSFeaturePipeline(HEL1OSPipelineConfig())
        results = pipe.transform_separately(make_synthetic_helios_flare())
        self.assertEqual(set(results.keys()), {"basic", "temporal", "flare", "spectral"})

    def test_transform_separately_each_stage_independent(self):
        pipe    = HEL1OSFeaturePipeline(HEL1OSPipelineConfig())
        results = pipe.transform_separately(make_synthetic_helios_flare())
        # Flare stage output should NOT contain basic features
        self.assertFalse(
            any("cdte_rolling_mean" in c for c in results["flare"].columns)
        )


class TestHEL1OSPipelineErrorHandling(unittest.TestCase):
    def test_missing_column_raises_with_stage_name(self):
        df   = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=3, freq="1s")})
        pipe = HEL1OSFeaturePipeline(HEL1OSPipelineConfig())
        with self.assertRaises(RuntimeError) as ctx:
            pipe.transform(df)
        self.assertIn("basic", str(ctx.exception))


class TestHEL1OSPipelineEdgeCases(unittest.TestCase):
    def test_empty_dataframe_full_pipeline(self):
        pipe = HEL1OSFeaturePipeline(HEL1OSPipelineConfig())
        out  = pipe.transform(make_empty_helios_df())
        self.assertEqual(len(out), 0)

    def test_idempotent(self):
        pipe = HEL1OSFeaturePipeline(HEL1OSPipelineConfig())
        df   = make_synthetic_helios_flare()
        pd.testing.assert_frame_equal(pipe.transform(df), pipe.transform(df))

    def test_end_to_end_smoke(self):
        pipe = HEL1OSFeaturePipeline(HEL1OSPipelineConfig())
        df   = make_synthetic_helios_flare()
        out  = pipe.transform(df)
        self.assertEqual(len(out), len(df))
        for col in ["cdte_log_counts", "cdte_dCR_dt", "hardness_ratio",
                    "hxr_phase", "hxr_photon_index_2pt"]:
            self.assertIn(col, out.columns)


# ── CombinedFeaturePipeline ───────────────────────────────────────────────────

class TestCombinedFeaturePipelineSmoke(unittest.TestCase):
    def _make_combined_df(self) -> pd.DataFrame:
        """DataFrame with both SoLEXS ('CR') and HEL1OS ('cdte_CR', 'czt_CR') columns."""
        df = make_synthetic_helios_flare()
        df["CR"] = df["cdte_CR"] * 0.1 + df["czt_CR"] * 0.05   # fake SoLEXS proxy
        return df

    def test_solexs_only_instrument(self):
        df   = self._make_combined_df()
        pipe = CombinedFeaturePipeline(
            solexs_config = PipelineConfig(),
            instrument    = "solexs",
        )
        out = pipe.transform(df)
        self.assertIn("log_counts", out.columns)
        self.assertNotIn("cdte_log_counts", out.columns)

    def test_helios_only_instrument(self):
        df   = self._make_combined_df()
        pipe = CombinedFeaturePipeline(
            helios_config = HEL1OSPipelineConfig(),
            instrument    = "helios",
        )
        out = pipe.transform(df)
        self.assertIn("cdte_log_counts", out.columns)
        self.assertNotIn("log_counts", out.columns)

    def test_combined_instrument(self):
        df   = self._make_combined_df()
        pipe = CombinedFeaturePipeline(
            solexs_config = PipelineConfig(),
            helios_config = HEL1OSPipelineConfig(),
            instrument    = "combined",
        )
        out = pipe.transform(df)
        self.assertIn("log_counts",     out.columns)   # SoLEXS
        self.assertIn("cdte_log_counts", out.columns)  # HEL1OS

    def test_invalid_instrument_raises(self):
        with self.assertRaises(ValueError):
            CombinedFeaturePipeline(instrument="unknown")

    def test_feature_names_combined_is_union(self):
        pipe_s = CombinedFeaturePipeline(
            solexs_config=PipelineConfig(), instrument="solexs"
        )
        pipe_h = CombinedFeaturePipeline(
            helios_config=HEL1OSPipelineConfig(), instrument="helios"
        )
        pipe_c = CombinedFeaturePipeline(
            solexs_config=PipelineConfig(),
            helios_config=HEL1OSPipelineConfig(),
            instrument="combined",
        )
        combined_names = set(pipe_c.feature_names())
        self.assertTrue(set(pipe_s.feature_names()).issubset(combined_names))
        self.assertTrue(set(pipe_h.feature_names()).issubset(combined_names))


if __name__ == "__main__":
    unittest.main()