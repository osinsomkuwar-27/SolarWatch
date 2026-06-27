"""
ml/features/run_features.py
============================
CLI runner that orchestrates the existing SoLEXSLoader and
FeaturePipeline to produce a feature-engineered CSV from a raw
Level-1 SoLEXS light curve.

Contains NO feature-extraction logic. It only:
    1. Loads project config (paths + logging) via ml.utils.config.
    2. Loads a day's light curve via SoLEXSLoader.load_day().
    3. Converts the resulting LightCurve into the DataFrame shape
       FeaturePipeline expects ('time', 'CR').
    4. Runs ml.features.feature_pipeline.FeaturePipeline.
    5. Saves output CSV(s) and prints a summary.

Usage
-----
    python -m ml.features.run_features --date 20260621 --detector SDD2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.loaders.solexs_loader import SoLEXSLoader
from ml.features.feature_pipeline import (
    FeaturePipeline,
    PipelineConfig as FeaturePipelineConfig,
)
from ml.utils.config import load_config as load_project_config

logger = logging.getLogger(__name__)


def _lightcurve_to_dataframe(day_data) -> pd.DataFrame:
    """Convert SoLEXSDayData's GTI-masked LightCurve into the
    ('time', 'CR') DataFrame shape FeaturePipeline's stages expect.

    Uses `day_data.lc_gti_masked`, so only samples inside a Good Time
    Interval are passed to feature extraction.
    """
    lc = day_data.lc_gti_masked
    return pd.DataFrame(
        {
            "time": pd.to_datetime(lc.time_unix, unit="s", utc=True),
            "CR": lc.count_rate,
        }
    )


def _load_feature_pipeline_config(
    path: Optional[str],
) -> FeaturePipelineConfig:
    """Build the FEATURE pipeline's own config (basic/temporal/flare/
    spectral stage settings), NOT the project-wide ml.utils.config one.

    There is currently no project utility for loading this specific
    dict shape, so it is parsed directly here. If one is added later
    (e.g. a `features` section in pipeline.yaml), point this at it.
    """
    if path is None:
        return FeaturePipelineConfig()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature pipeline config not found: {p}")

    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        with open(p, "r") as fh:
            raw = yaml.safe_load(fh) or {}
    elif p.suffix.lower() == ".json":
        with open(p, "r") as fh:
            raw = json.load(fh)
    else:
        raise ValueError(f"Unsupported config extension: {p.suffix}")

    return FeaturePipelineConfig.from_dict(raw)


def _save_stagewise(
    pipe: FeaturePipeline, df: pd.DataFrame, out_dir: Path
) -> dict[str, Path]:
    """Run transform_separately() and save one CSV per stage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_results = pipe.transform_separately(df)
    saved: dict[str, Path] = {}
    for stage_name, stage_df in stage_results.items():
        stage_path = out_dir / f"{stage_name}.csv"
        stage_df.to_csv(stage_path, index=False)
        saved[stage_name] = stage_path
        logger.info("Saved stage '%s' -> %s", stage_name, stage_path)
    return saved


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.features.run_features",
        description="Run SoLEXSLoader + FeaturePipeline for one day/detector.",
    )
    parser.add_argument("--date", required=True, help="YYYYMMDD, e.g. 20260621")
    parser.add_argument("--detector", required=True, help="e.g. SDD2")
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path. Default: <processed>/features_<DET>_<DATE>.csv",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to project pipeline.yaml (paths/logging). "
             "Defaults to config/pipeline.yaml per ml.utils.config.",
    )
    parser.add_argument(
        "--feature-config", default=None,
        help="Optional YAML/JSON overriding FeaturePipeline stage settings "
             "(basic/temporal/flare/spectral). Separate from --config.",
    )
    parser.add_argument(
        "--save-stagewise", action="store_true",
        help="Also save basic.csv/temporal.csv/flare.csv/spectral.csv.",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # load_config() sets up logging (file+console) as a side effect and
    # gives us validated paths (paths.raw_solexs, paths.processed).
    project_cfg = load_project_config(args.config)

    try:
        loader = SoLEXSLoader(data_dir=project_cfg.paths.raw_solexs)
        logger.info("Loading day: detector=%s date=%s", args.detector, args.date)
        day_data = loader.load_day(
            date_str=args.date, detector=args.detector, load_pi=False
        )

        df = _lightcurve_to_dataframe(day_data)
        n_rows = len(df)

        feature_cfg = _load_feature_pipeline_config(args.feature_config)
        pipe = FeaturePipeline(feature_cfg)

        logger.info("Running FeaturePipeline.transform()...")
        features_df = pipe.transform(df)
        n_feature_cols = len(features_df.columns) - len(df.columns)

        output_path = (
            Path(args.output) if args.output else
            project_cfg.paths.processed / f"features_{args.detector}_{args.date}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(output_path, index=False)
        logger.info("Saved combined features -> %s", output_path)

        stagewise_paths = None
        if args.save_stagewise:
            stagewise_paths = _save_stagewise(pipe, df, output_path.parent)

        print("\nLoaded:")
        print(f"    detector = {args.detector}")
        print(f"    date = {args.date}")
        print(f"    rows = {n_rows}")
        print("\nGenerated:")
        print(f"    total feature columns = {n_feature_cols}")
        print("\nSaved:")
        print(f"    {output_path}")
        if stagewise_paths:
            for name, path in stagewise_paths.items():
                print(f"    {path}  (stage: {name})")
        print()
        return 0

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except (KeyError, ValueError) as exc:
        logger.error("Invalid input/data: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())