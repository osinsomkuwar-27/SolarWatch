"""
run_features.py
================
Command-line entry point for the feature-engineering pipeline.

Supports three instrument modes:
  --instrument solexs    : SoLEXS-only (existing behaviour, unchanged)
  --instrument helios    : HEL1OS-only
  --instrument combined  : SoLEXS + HEL1OS merged features

Usage
-----
    # SoLEXS only (existing behaviour)
    python -m ml.features.run_features --date 20260621 --instrument solexs

    # HEL1OS only
    python -m ml.features.run_features --date 20260621 --instrument helios \\
        --helios-detector CdTe1

    # Combined
    python -m ml.features.run_features --date 20260621 --instrument combined

Outputs
-------
    features_solexs_<detector>_<date>.csv
    features_helios_<detector>_<date>.csv
    features_combined_<date>.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aditya-L1 Feature Engineering Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--date", required=True,
        help="Observation date YYYYMMDD",
    )
    p.add_argument(
        "--instrument",
        default="solexs",
        choices=["solexs", "helios", "combined"],
        help="Which instrument(s) to process.",
    )
    p.add_argument(
        "--detector", default="SDD2",
        choices=["SDD1", "SDD2"],
        help="SoLEXS detector (solexs / combined modes).",
    )
    p.add_argument(
        "--helios-detector", default="CdTe1",
        choices=["CdTe1", "CdTe2", "CZT1", "CZT2"],
        help="HEL1OS detector (helios / combined modes).",
    )
    p.add_argument(
        "--config", default=None,
        help="Path to pipeline.yaml.",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Override output directory for feature CSVs.",
    )
    return p


def run_features(args: argparse.Namespace) -> None:
    """Dispatch to the appropriate instrument pipeline."""
    from ml.utils.config import load_config
    cfg = load_config(args.config)

    out_dir = Path(args.output_dir) if args.output_dir else cfg.paths.cache / args.date
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.instrument == "solexs":
        _run_solexs(args, cfg, out_dir)
    elif args.instrument == "helios":
        _run_helios(args, cfg, out_dir)
    elif args.instrument == "combined":
        _run_combined(args, cfg, out_dir)
    else:
        logger.error("Unknown instrument: %s", args.instrument)
        sys.exit(1)


# ── SoLEXS (unchanged logic — verbatim from original run_features.py) ────────

def _run_solexs(args, cfg, out_dir: Path) -> Path:
    """Run SoLEXS-only feature pipeline.  Logic is identical to the
    original run_features.py; only wrapped in a function to support the
    combined mode dispatcher above."""
    from ml.loaders.solexs_loader import SoLEXSLoader
    from ml.features import FeaturePipeline, PipelineConfig

    solexs_dir = cfg.paths.raw_solexs
    loader     = SoLEXSLoader(data_dir=solexs_dir)
    n          = args.detector[-1]
    base       = f"AL1_SOLEXS_{args.date}_SDD{n}_L1"
    lc_path    = solexs_dir / f"{base}.lc"
    lc_gz      = solexs_dir / f"{base}.lc.gz"
    lc_file    = lc_path if lc_path.exists() else lc_gz

    if not lc_file.exists():
        logger.error("SoLEXS LC not found: %s", lc_path)
        sys.exit(1)

    lc    = loader.load_lc(lc_file, detector=args.detector, date_str=args.date)
    df    = _solexs_lc_to_df(lc)
    pipe  = FeaturePipeline(PipelineConfig())
    out   = pipe.transform(df)
    csv   = out_dir / f"features_solexs_{args.detector}_{args.date}.csv"
    out.to_csv(csv, index=False)
    logger.info("SoLEXS features saved: %s  (%d rows × %d cols)",
                csv, len(out), len(out.columns))
    return csv


def _solexs_lc_to_df(lc) -> pd.DataFrame:
    """Convert a SoLEXS LightCurve to a features-ready DataFrame."""
    import numpy as np
    times = pd.to_datetime(lc.time_unix, unit="s", utc=True).tz_localize(None)
    return pd.DataFrame({"time": times, "CR": lc.count_rate.astype(float)})


# ── HEL1OS ────────────────────────────────────────────────────────────────────

def _run_helios(args, cfg, out_dir: Path) -> Path:
    """Run HEL1OS-only feature pipeline."""
    from ml.loaders.helios_loader import HEL1OSLoader
    from ml.features.helios_features import (
        HEL1OSFeaturePipeline, HEL1OSPipelineConfig,
    )

    det     = args.helios_detector
    h_dir   = cfg.paths.raw_helios
    subdir = "cdte" if det.lower().startswith("cdte") else "czt"
    h_path = h_dir / subdir / f"lightcurve_{det.lower()}.fits"
    if not h_path.exists():
        logger.error("HEL1OS LC not found: %s", h_path)
        sys.exit(1)

    loader    = HEL1OSLoader(data_dir=h_dir)
    helios_lc = loader.load_lc(h_path, detector=det, date_str=args.date)

    df   = _helios_lc_to_df(helios_lc, cdte_det=det)
    pipe = HEL1OSFeaturePipeline(HEL1OSPipelineConfig())
    out  = pipe.transform(df)
    csv  = out_dir / f"features_helios_{det}_{args.date}.csv"
    out.to_csv(csv, index=False)
    logger.info("HEL1OS features saved: %s  (%d rows × %d cols)",
                csv, len(out), len(out.columns))
    return csv


def _helios_lc_to_df(lc, cdte_det: str = "CdTe1") -> pd.DataFrame:
    """Convert a HEL1OSLightCurve to a features-ready DataFrame.

    Uses full_band count rate for the broadband column.  If the same
    detector is CdTe-family it populates cdte_CR; for CZT-family, czt_CR.
    Both columns are always present (the missing family is filled with NaN).
    """
    full  = lc.full_band
    times = pd.to_datetime(full.time_unix, unit="s", utc=True).tz_localize(None)
    is_cdte = cdte_det.upper().startswith("CDTE")
    df = pd.DataFrame({
        "time":    times,
        "cdte_CR": full.count_rate.astype(float) if is_cdte else float("nan"),
        "czt_CR":  float("nan") if is_cdte else full.count_rate.astype(float),
    })
    # Fill NaN column with zeros to avoid crashing the pipeline on the first row
    df = df.fillna(0.0)
    return df


# ── Combined ──────────────────────────────────────────────────────────────────

def _run_combined(args, cfg, out_dir: Path) -> Path:
    """Run SoLEXS + HEL1OS combined feature pipeline and merge the outputs."""
    from ml.loaders.solexs_loader import SoLEXSLoader
    from ml.loaders.helios_loader import HEL1OSLoader
    from ml.features.feature_pipeline import CombinedFeaturePipeline
    from ml.features import PipelineConfig
    from ml.features.helios_features import HEL1OSPipelineConfig

    # ── Load SoLEXS ───────────────────────────────────────────────────
    solexs_dir = cfg.paths.raw_solexs
    n          = args.detector[-1]
    base       = f"AL1_SOLEXS_{args.date}_SDD{n}_L1"
    lc_path    = solexs_dir / f"{base}.lc"
    lc_gz      = solexs_dir / f"{base}.lc.gz"
    lc_file    = lc_path if lc_path.exists() else lc_gz

    solexs_df = None
    if lc_file.exists():
        from ml.loaders.solexs_loader import SoLEXSLoader
        loader  = SoLEXSLoader(data_dir=solexs_dir)
        lc      = loader.load_lc(lc_file, detector=args.detector, date_str=args.date)
        solexs_df = _solexs_lc_to_df(lc)
    else:
        logger.warning("SoLEXS LC not found at %s; SoLEXS features will be NaN.", lc_file)

    # ── Load HEL1OS ───────────────────────────────────────────────────
    det     = args.helios_detector
    h_dir   = cfg.paths.raw_helios
    subdir = "cdte" if det.lower().startswith("cdte") else "czt"
    h_path = h_dir / subdir / f"lightcurve_{det.lower()}.fits"

    helios_df = None
    if h_path.exists():
        from ml.loaders.helios_loader import HEL1OSLoader
        h_loader  = HEL1OSLoader(data_dir=h_dir)
        helios_lc = h_loader.load_lc(h_path, detector=det, date_str=args.date)
        helios_df = _helios_lc_to_df(helios_lc, cdte_det=det)
    else:
        logger.warning("HEL1OS LC not found at %s; HEL1OS features will be NaN.", h_path)

    if solexs_df is None and helios_df is None:
        logger.error("Neither SoLEXS nor HEL1OS data found. Cannot produce combined features.")
        sys.exit(1)

    # ── Merge on time (outer join, then sort) ─────────────────────────
    df = _merge_solexs_helios(solexs_df, helios_df)

    # ── Run combined pipeline ─────────────────────────────────────────
    pipe = CombinedFeaturePipeline(
        solexs_config = PipelineConfig(),
        helios_config = HEL1OSPipelineConfig(),
        instrument    = "combined",
    )
    out = pipe.transform(df)
    csv = out_dir / f"features_combined_{args.date}.csv"
    out.to_csv(csv, index=False)
    logger.info("Combined features saved: %s  (%d rows × %d cols)",
                csv, len(out), len(out.columns))
    return csv


def _merge_solexs_helios(
    solexs_df: Optional[pd.DataFrame],
    helios_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Outer-merge SoLEXS and HEL1OS DataFrames on the 'time' column.

    Both DataFrames have 'time' as datetime64 column.  After the merge,
    missing values are forward-filled within a 30-second tolerance and
    then filled with 0.0 so neither pipeline crashes on NaN inputs.
    """
    if solexs_df is None and helios_df is not None:
        # Pad with NaN SoLEXS columns
        helios_df = helios_df.copy()
        helios_df["CR"] = 0.0
        return helios_df.sort_values("time").reset_index(drop=True)

    if helios_df is None and solexs_df is not None:
        # Pad with NaN HEL1OS columns
        solexs_df = solexs_df.copy()
        solexs_df["cdte_CR"] = 0.0
        solexs_df["czt_CR"]  = 0.0
        return solexs_df.sort_values("time").reset_index(drop=True)

    # Ensure both time columns use the same datetime precision.
    solexs_df = solexs_df.copy()
    helios_df = helios_df.copy()

    solexs_df["time"] = pd.to_datetime(solexs_df["time"]).astype("datetime64[ns]")
    helios_df["time"] = pd.to_datetime(helios_df["time"]).astype("datetime64[ns]")
    
    merged = pd.merge_asof(
        solexs_df.sort_values("time"),
        helios_df.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("30s"),
    )
    merged = merged.fillna(0.0)
    return merged.sort_values("time").reset_index(drop=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = _build_arg_parser()
    args   = parser.parse_args()
    run_features(args)


if __name__ == "__main__":
    main()