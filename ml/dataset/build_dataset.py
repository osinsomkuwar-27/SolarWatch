"""
Dataset Builder — Command-Line Entry Point
==========================================

Usage
-----
# With defaults (reads ml/data/raw/, writes ml/data/processed/)
python -m ml.dataset.build_dataset

# Override key parameters
python -m ml.dataset.build_dataset \
    --raw-data-dir ml/data/raw \
    --output-dir   ml/data/processed \
    --window-size  120 \
    --horizon      0 \
    --stride       5 \
    --train-frac   0.70 \
    --val-frac     0.15 \
    --scaler       standard

# Load full config from JSON (command-line flags override JSON values)
python -m ml.dataset.build_dataset --config ml/dataset/config.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ml.dataset.builder import DatasetBuilder
from ml.dataset.config import DatasetConfig


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val/test datasets from SoLEXS / HEL1OS feature CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Optional JSON config file (values act as defaults, CLI flags override)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON config file produced by DatasetConfig.to_json().",
    )

    # I/O
    io = parser.add_argument_group("I/O")
    io.add_argument("--raw-data-dir", type=str, default=None)
    io.add_argument("--output-dir",   type=str, default=None)
    io.add_argument(
        "--instrument",
        type=str,
        default=None,
        choices=["solexs", "helios", "combined"],
        help=(
            "Instrument whose feature CSVs to build from. "
            "Determines which filenames are matched and how the scaler is named. "
            "Choices: 'solexs' (features_solexs_<det>_<date>.csv), "
            "'helios' (features_helios_<det>_<date>.csv), "
            "'combined' (features_combined_<date>.csv). "
            "If omitted, the instrument type is auto-inferred from directory contents."
        ),
    )

    # Window
    win = parser.add_argument_group("Window / horizon")
    win.add_argument("--window-size", type=int, default=None)
    win.add_argument("--horizon",     type=int, default=None,
                     dest="prediction_horizon")
    win.add_argument("--stride",      type=int, default=None)

    # Split
    split = parser.add_argument_group("Split")
    split.add_argument("--train-frac", type=float, default=None)
    split.add_argument("--val-frac",   type=float, default=None)

    # Normalisation / imputation
    norm = parser.add_argument_group("Normalisation")
    norm.add_argument(
        "--scaler",
        choices=["standard", "minmax"],
        default=None,
        dest="scaler_type",
    )
    norm.add_argument(
        "--imputation",
        choices=["forward", "median"],
        default=None,
        dest="imputation_strategy",
    )
    norm.add_argument("--min-valid-frac", type=float, default=None,
                      dest="min_valid_fraction")

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    # ── Start from JSON config if provided, else defaults ────────────────
    if args.config is not None:
        cfg = DatasetConfig.from_json(args.config)
    else:
        cfg = DatasetConfig()

    # ── Apply CLI overrides ───────────────────────────────────────────────
    overrides = {
        "raw_data_dir":        args.raw_data_dir,
        "output_dir":          args.output_dir,
        "instrument_tag":      args.instrument,
        "window_size":         args.window_size,
        "prediction_horizon":  args.prediction_horizon,
        "stride":              args.stride,
        "train_frac":          args.train_frac,
        "val_frac":            args.val_frac,
        "scaler_type":         args.scaler_type,
        "imputation_strategy": args.imputation_strategy,
        "min_valid_fraction":  args.min_valid_fraction,
    }
    for attr, value in overrides.items():
        if value is not None:
            setattr(cfg, attr, value)

    builder = DatasetBuilder(cfg)
    builder.run()


if __name__ == "__main__":
    main()