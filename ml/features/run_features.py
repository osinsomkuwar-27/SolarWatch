"""
run_features.py
================
Command-line entry point for the feature-engineering pipeline.

Supports three instrument modes:
  --instrument solexs    : SoLEXS-only processing
  --instrument helios    : HEL1OS-only processing
  --instrument combined  : SoLEXS + HEL1OS merged features

Usage
-----
    python -m ml.features.run_features
    python -m ml.features.run_features --instrument combined
    python -m ml.features.run_features --instrument solexs
    python -m ml.features.run_features --instrument helios

Inputs
------
    ml/data/processed/solexs/<date>.csv
    ml/data/processed/helios/<date>_<detector>.csv

    These are produced by `ml.eda.run_eda` and are the ONLY source this
    module reads from. Raw loaders are never invoked here.

HEL1OS detector handling
-------------------------
    A single observation day can have up to four processed HEL1OS
    datasets (CZT1, CZT2, CdTe1, CdTe2). CdTe and CZT cover distinct,
    non-overlapping energy bands, so they are never merged into one
    dataframe before feature extraction. Every detector found for a
    day is processed independently, both in HEL1OS-only mode and in
    combined mode (one combined feature set per detector, paired with
    that day's SoLEXS data). No detector is silently dropped.

Synchronization
----------------
    The SoLEXS/HEL1OS combined-mode merge_asof tolerance is read from
    pipeline.yaml under `preprocessing.merge_tolerance_sec`. If absent,
    a default of 30 seconds is used. See
    `ml.features.feature_pipeline.get_merge_tolerance_seconds`.

Outputs
-------
    ml/data/features/solexs/YYYYMMDD/features_solexs_YYYYMMDD.csv
    ml/data/features/helios/YYYYMMDD/features_<detector>.csv
    ml/data/features/combined/YYYYMMDD/combined_<detector>.csv
    ml/data/features/solexs_features.csv
    ml/data/features/helios_features.csv
    ml/data/features/combined_features.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_PROCESSED_ROOT = Path("ml/data/processed")
_DEFAULT_OUTPUT_DIR = Path("ml/data/features")
_SUPPORTED_DETECTORS = {"SDD1", "SDD2", "CdTe1", "CdTe2", "CZT1", "CZT2"}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aditya-L1 Feature Engineering Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--instrument",
        default="combined",
        choices=["solexs", "helios", "combined"],
        help="Which instrument(s) to process.",
    )
    p.add_argument(
        "--detector",
        default="SDD2",
        choices=["SDD1", "SDD2"],
        help="SoLEXS detector (solexs / combined modes).",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to pipeline.yaml.",
    )
    p.add_argument(
        "--eda-dir",
        default=None,
        help="Override the processed-dataset input directory "
             "(default: ml/data/processed).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for feature CSVs.",
    )
    return p


def run_features(args: argparse.Namespace) -> None:
    """Discover processed datasets, generate per-instrument features, and save aggregated outputs."""
    from ml.features.feature_pipeline import (
        _clean_and_validate_dataframe,
        discover_processed_datasets,
        get_merge_tolerance_seconds,
    )
    from ml.utils.config import load_config

    cfg = load_config(args.config)
    merge_tolerance_sec = get_merge_tolerance_seconds(cfg)
    logger.info("Using SoLEXS/HEL1OS merge_asof tolerance: %.3fs", merge_tolerance_sec)

    processed_root = Path(args.eda_dir) if args.eda_dir else _DEFAULT_PROCESSED_ROOT
    output_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_processed_datasets(processed_root, args.detector, None)
    discovered = discover_processed_datasets(processed_root, args.detector, None)
    print("DEBUG processed_root =", processed_root.resolve())
    print("DEBUG discovered keys:", list(discovered.keys())[:5])
    for k in list(discovered.keys())[:3]:
        print("DEBUG", k, "solexs:", discovered[k]["solexs"], "helios:", discovered[k]["helios"])
    if not discovered:
        logger.warning("No processed datasets were found under %s", processed_root)
        print(f"No processed datasets were found under {processed_root}. Nothing to process.")
        return

    # ── Discovery-level coverage counts ──────────────────────────────────────
    solexs_discovered = sum(1 for d in discovered.values() if d.get("solexs"))
    helios_discovered = sum(len(d.get("helios", [])) for d in discovered.values())
    combined_candidates = sum(
        1 for d in discovered.values() if d.get("solexs") and d.get("helios")
    )
    solexs_only_dates = sum(
        1 for d in discovered.values() if d.get("solexs") and not d.get("helios")
    )
    helios_only_dates = sum(
        1 for d in discovered.values() if d.get("helios") and not d.get("solexs")
    )

    solexs_frames: List[pd.DataFrame] = []
    helios_frames: List[pd.DataFrame] = []
    combined_frames: List[pd.DataFrame] = []
    solexs_processed = 0
    helios_processed = 0
    combined_processed = 0
    skipped_combined = 0
    detectors_seen_combined: set = set()

    for observation_date in sorted(discovered):
        day_entry = discovered[observation_date]
        helios_paths = day_entry.get("helios", [])
        solexs_paths = day_entry.get("solexs", [])

        if args.instrument in {"combined", "solexs"}:
            for solexs_path in solexs_paths:
                try:
                    frame = _run_solexs(args, observation_date, solexs_path, output_dir)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping SoLEXS for %s due to error: %s", observation_date, exc)
                    print(f"Skipping SoLEXS for {observation_date}: {exc}")
                    frame = None
                if frame is not None:
                    solexs_frames.append(frame)
                    solexs_processed += 1

        if args.instrument in {"combined", "helios"}:
            for helios_path in helios_paths:
                try:
                    frame = _run_helios(observation_date, helios_path, output_dir)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping HEL1OS for %s due to error: %s", observation_date, exc)
                    print(f"Skipping HEL1OS for {observation_date}: {exc}")
                    frame = None
                if frame is not None:
                    helios_frames.append(frame)
                    helios_processed += 1

        if args.instrument == "combined":
            solexs_path = next(iter(solexs_paths), None)
            if solexs_path is None or not helios_paths:
                print(
                    f"DEBUG COMBINED | date={observation_date} | "
                    f"solexs_path={solexs_path} | "
                    f"helios_count={len(helios_paths)}"
                )

                logger.info(
                    "Skipping combined features for %s: missing SoLEXS or HEL1OS data.",
                    observation_date,
                )
                continue

            # ── Option A: every HEL1OS detector gets its own combined dataset ──
            for helios_path in helios_paths:
                detector = _infer_detector(helios_path.name, "helios")
                try:
                    frame = _run_combined(
                        observation_date, solexs_path, helios_path, output_dir, merge_tolerance_sec
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Skipping combined features for %s [%s] due to error: %s",
                        observation_date, detector, exc,
                    )
                    print(f"Skipping combined features for {observation_date} [{detector}]: {exc}")
                    frame = None
                if frame is None:
                    skipped_combined += 1
                else:
                    combined_frames.append(frame)
                    combined_processed += 1
                    if detector:
                        detectors_seen_combined.add(detector)

    if args.instrument in {"combined", "solexs"}:
        _save_aggregated_features(output_dir, solexs_frames, [], [])
    if args.instrument in {"combined", "helios"}:
        _save_aggregated_features(output_dir, [], helios_frames, [])
    if args.instrument == "combined":
        _save_aggregated_features(output_dir, [], [], combined_frames)

    _print_summary(
        solexs_discovered=solexs_discovered,
        solexs_processed=solexs_processed,
        helios_discovered=helios_discovered,
        helios_processed=helios_processed,
        combined_candidates=combined_candidates,
        combined_processed=combined_processed,
        combined_detectors=sorted(detectors_seen_combined),
        solexs_only_dates=solexs_only_dates,
        helios_only_dates=helios_only_dates,
        skipped_combined=skipped_combined,
        solexs_rows=sum(len(f) for f in solexs_frames),
        helios_rows=sum(len(f) for f in helios_frames),
        combined_rows=sum(len(f) for f in combined_frames),
        output_dir=output_dir,
    )


def _run_solexs(
    args: argparse.Namespace,
    observation_date: str,
    data_path: Path,
    output_root: Path,
) -> Optional[pd.DataFrame]:
    """Run SoLEXS-only feature pipeline for a single observation day."""
    from ml.features import FeaturePipeline, PipelineConfig
    from ml.features.feature_pipeline import _clean_and_validate_dataframe

    processed_df = _load_processed_eda_dataframe(data_path)
    if processed_df is None:
        logger.warning("No processed SoLEXS dataset found for %s; skipping.", observation_date)
        return None

    df = _normalise_feature_input(processed_df, instrument="solexs")
    df = _annotate_metadata(df, observation_date, "solexs", args.detector)

    pipe = FeaturePipeline(PipelineConfig())
    out = _clean_and_validate_dataframe(pipe.transform(df))
    csv_path = output_root / "solexs" / observation_date / f"features_solexs_{observation_date}.csv"
    _save_feature_dataframe(
        out,
        csv_path,
        observation_date=observation_date,
        instrument="solexs",
        detector=args.detector,
    )
    return out


def _run_helios(
    observation_date: str,
    data_path: Path,
    output_root: Path,
) -> Optional[pd.DataFrame]:
    """
    Run the HEL1OS feature pipeline for a single (observation_date, detector) pair.

    Every detector found on disk for this day is processed — no detector
    is selected over another, since CdTe and CZT cover distinct energy
    bands and neither is a substitute for the other.
    """
    from ml.features.feature_pipeline import _clean_and_validate_dataframe
    from ml.features.helios_features import HEL1OSFeaturePipeline, HEL1OSPipelineConfig

    processed_df = _load_processed_eda_dataframe(data_path)
    if processed_df is None:
        logger.warning("No processed HEL1OS dataset found for %s (%s); skipping.", observation_date, data_path.name)
        return None

    detector = _infer_detector(data_path.name, "helios")
    if detector is None:
        logger.warning("Could not infer HEL1OS detector from %s; skipping.", data_path.name)
        return None

    df = _normalise_feature_input(processed_df, instrument="helios")
    df = _annotate_metadata(df, observation_date, "helios", detector)

    pipe = HEL1OSFeaturePipeline(HEL1OSPipelineConfig())
    out = _clean_and_validate_dataframe(pipe.transform(df))
    csv_path = output_root / "helios" / observation_date / f"features_{detector}.csv"
    _save_feature_dataframe(
        out,
        csv_path,
        observation_date=observation_date,
        instrument="helios",
        detector=detector,
    )
    return out


def _run_combined(
    observation_date: str,
    solexs_path: Path,
    helios_path: Path,
    output_root: Path,
    merge_tolerance_sec: float,
) -> Optional[pd.DataFrame]:
    """
    Run the SoLEXS + HEL1OS combined feature pipeline for one
    (observation_date, HEL1OS detector) pair.

    Option A: one combined dataset PER detector. The same SoLEXS day is
    paired against every available HEL1OS detector independently, so a
    day with four HEL1OS detectors produces four combined feature
    files — CZT1, CZT2, CdTe1, CdTe2 — rather than one file that
    silently picked a single detector.
    """
    from ml.features import PipelineConfig
    from ml.features.feature_pipeline import CombinedFeaturePipeline, _clean_and_validate_dataframe
    from ml.features.helios_features import HEL1OSPipelineConfig

    solexs_df = _load_processed_eda_dataframe(solexs_path)
    helios_df = _load_processed_eda_dataframe(helios_path)

    if solexs_df is None or helios_df is None:
        logger.warning(
            "No complete processed SoLEXS/HEL1OS pair found for %s (%s); skipping combined features.",
            observation_date, helios_path.name,
        )
        return None

    detector = _infer_detector(helios_path.name, "helios")
    if detector is None:
        logger.warning("Could not infer HEL1OS detector from %s; skipping combined features.", helios_path.name)
        return None

    df = _merge_solexs_helios(
        _normalise_feature_input(solexs_df, instrument="solexs"),
        _normalise_feature_input(helios_df, instrument="helios"),
        tolerance_sec=merge_tolerance_sec,
    )
    df = _annotate_metadata(df, observation_date, "combined", detector)

    pipe = CombinedFeaturePipeline(
        solexs_config=PipelineConfig(),
        helios_config=HEL1OSPipelineConfig(),
        instrument="combined",
    )
    out = _clean_and_validate_dataframe(pipe.transform(df))
    csv_path = output_root / "combined" / observation_date / f"combined_{detector}.csv"
    _save_feature_dataframe(
        out,
        csv_path,
        observation_date=observation_date,
        instrument="combined",
        detector=detector,
    )
    return out


def _save_aggregated_features(
    output_root: Path,
    solexs_frames: List[pd.DataFrame],
    helios_frames: List[pd.DataFrame],
    combined_frames: List[pd.DataFrame],
) -> None:
    from ml.features.feature_pipeline import _clean_and_validate_dataframe

    if solexs_frames:
        solexs_df = pd.concat(solexs_frames, ignore_index=True)
        solexs_df = solexs_df.sort_values(["observation_date", "timestamp"]).reset_index(drop=True)
        _save_feature_dataframe(
            _clean_and_validate_dataframe(solexs_df),
            output_root / "solexs_features.csv",
            observation_date="all",
            instrument="solexs",
            detector="all",
        )

    if helios_frames:
        helios_df = pd.concat(helios_frames, ignore_index=True)
        helios_df = helios_df.sort_values(["observation_date", "detector", "timestamp"]).reset_index(drop=True)
        _save_feature_dataframe(
            _clean_and_validate_dataframe(helios_df),
            output_root / "helios_features.csv",
            observation_date="all",
            instrument="helios",
            detector="all",
        )

    if combined_frames:
        combined_df = pd.concat(combined_frames, ignore_index=True)
        combined_df = combined_df.sort_values(["observation_date", "detector", "timestamp"]).reset_index(drop=True)
        _save_feature_dataframe(
            _clean_and_validate_dataframe(combined_df),
            output_root / "combined_features.csv",
            observation_date="all",
            instrument="combined",
            detector="all",
        )


def _print_summary(
    *,
    solexs_discovered: int,
    solexs_processed: int,
    helios_discovered: int,
    helios_processed: int,
    combined_candidates: int,
    combined_processed: int,
    combined_detectors: List[str],
    solexs_only_dates: int,
    helios_only_dates: int,
    skipped_combined: int,
    solexs_rows: int,
    helios_rows: int,
    combined_rows: int,
    output_dir: Path,
) -> None:
    width = 56
    print("-" * width)
    print("Processed Dataset Summary")
    print("-" * width)
    print(f"SoLEXS observations discovered      : {solexs_discovered}")
    print(f"SoLEXS observations processed       : {solexs_processed}")
    print(f"HEL1OS observations discovered      : {helios_discovered}  (date x detector)")
    print(f"HEL1OS observations processed       : {helios_processed}")
    print(f"Combined candidate observation days : {combined_candidates}")
    print(f"Combined observations generated     : {combined_processed}  (date x detector)")
    print(f"Combined HEL1OS detectors covered   : {', '.join(combined_detectors) if combined_detectors else '-'}")
    print(f"SoLEXS-only observations            : {solexs_only_dates}")
    print(f"HEL1OS-only observations            : {helios_only_dates}")
    print(f"Skipped observations                : {skipped_combined}")
    print(f"Total SoLEXS feature rows           : {solexs_rows}")
    print(f"Total HEL1OS feature rows           : {helios_rows}")
    print(f"Total Combined feature rows         : {combined_rows}")
    print(f"Output directory                    : {output_dir}")
    print("-" * width)


def _load_processed_eda_dataframe(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Read a processed dataframe from disk."""
    if path is None:
        return None
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, low_memory=False)
        else:
            df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to read processed dataset %s: %s", path, exc)
        return None

    return None if df is None or df.empty else df


def _normalise_feature_input(df: Optional[pd.DataFrame], instrument: str) -> pd.DataFrame:
    """
    Normalise a processed dataframe into the feature-pipeline input schema.

    Only renames or adds the columns the downstream feature pipelines
    require (`time`, `timestamp`, `CR` for SoLEXS, `cdte_CR`/`czt_CR`
    for HEL1OS). Every other column produced by the EDA stage is
    preserved unchanged. No column is dropped.
    """
    if df is None:
        raise ValueError("No processed dataframe provided.")

    out = df.copy()
    if "time" not in out.columns:
        for candidate in ["timestamp", "datetime", "date", "obs_time", "t"]:
            if candidate in out.columns:
                out = out.rename(columns={candidate: "time"})
                break
    if "time" not in out.columns:
        raise ValueError("Processed dataset is missing a time column.")

    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    out["timestamp"] = out["time"]

    if instrument == "solexs":
        if "CR" not in out.columns:
            for candidate in ["count_rate", "count_rate_s", "soft_cr", "soft_count_rate"]:
                if candidate in out.columns:
                    out = out.rename(columns={candidate: "CR"})
                    break
        if "CR" not in out.columns:
            numeric_cols = [col for col in out.columns if col != "time" and pd.api.types.is_numeric_dtype(out[col])]
            if not numeric_cols:
                raise ValueError("Processed SoLEXS dataset does not contain a usable count-rate column.")
            out["CR"] = out[numeric_cols[0]].astype(float)
        required = ["time", "CR"]
        remaining = [c for c in out.columns if c not in required]
        return out[list(dict.fromkeys(required + remaining))]

    if instrument == "helios":
        if "cdte_CR" not in out.columns and "czt_CR" not in out.columns:
            for candidate in ["CR", "count_rate", "full_band", "hard_cr"]:
                if candidate in out.columns:
                    out = out.rename(columns={candidate: "cdte_CR"})
                    break
        if "cdte_CR" not in out.columns:
            out["cdte_CR"] = float("nan")
        if "czt_CR" not in out.columns:
            out["czt_CR"] = float("nan")
        required = ["time", "cdte_CR", "czt_CR"]
        remaining = [c for c in out.columns if c not in required]
        return out[list(dict.fromkeys(required + remaining))]

    return out


def _merge_solexs_helios(
    solexs_df: Optional[pd.DataFrame],
    helios_df: Optional[pd.DataFrame],
    tolerance_sec: float,
) -> pd.DataFrame:
    """Outer-merge a SoLEXS DataFrame with ONE HEL1OS detector's DataFrame on 'time'."""
    if solexs_df is None and helios_df is not None:
        helios_df = helios_df.copy()
        helios_df["CR"] = 0.0
        return helios_df.sort_values("time").reset_index(drop=True)

    if helios_df is None and solexs_df is not None:
        solexs_df = solexs_df.copy()
        solexs_df["cdte_CR"] = 0.0
        solexs_df["czt_CR"] = 0.0
        return solexs_df.sort_values("time").reset_index(drop=True)

    solexs_df = solexs_df.copy()
    helios_df = helios_df.copy()

    solexs_df["time"] = pd.to_datetime(solexs_df["time"], utc=True).dt.tz_convert(None).astype("datetime64[ns]")
    helios_df["time"] = pd.to_datetime(helios_df["time"], utc=True).dt.tz_convert(None).astype("datetime64[ns]")

    merged = pd.merge_asof(
        solexs_df.sort_values("time"),
        helios_df.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_sec),
    )
    merged = merged.fillna(0.0)
    return merged.sort_values("time").reset_index(drop=True)


def _annotate_metadata(
    df: pd.DataFrame,
    observation_date: str,
    instrument: str,
    detector: str,
) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns and "time" in out.columns:
        out["timestamp"] = out["time"]
    out["observation_date"] = observation_date
    out["instrument"] = instrument
    out["detector"] = detector
    return out


def _infer_detector(name: str, instrument: str) -> Optional[str]:
    lowered = name.upper()
    for detector in _SUPPORTED_DETECTORS:
        if detector.upper() in lowered:
            return detector
    return None


def _save_feature_dataframe(
    df: pd.DataFrame,
    csv_path: Path,
    observation_date: str,
    instrument: str,
    detector: str,
) -> None:
    """Validate, print a summary, and write a feature DataFrame to disk."""
    try:
        prepared_df = df.copy()
        if prepared_df is None:
            raise ValueError("feature extraction produced no DataFrame")
        if prepared_df.empty:
            raise ValueError("feature extraction produced an empty DataFrame")

        feature_columns = [
            col for col in prepared_df.columns if not _is_metadata_column(col, prepared_df[col])
        ]
        if not feature_columns:
            raise ValueError("feature extraction produced no feature columns")
        if prepared_df[feature_columns].isna().all().all():
            raise ValueError("all extracted feature values are NaN")

        print(
            f"Observation date: {observation_date}\n"
            f"Instrument: {instrument}\n"
            f"Detector(s): {detector}\n"
            f"Rows: {len(prepared_df)}\n"
            f"Feature columns: {len(feature_columns)}\n"
            f"Total extracted features: {len(feature_columns)}\n"
            f"Output CSV: {csv_path}"
        )
        logger.info(
            "Writing feature dataset for date=%s instrument=%s detector=%s rows=%d feature_columns=%d output=%s",
            observation_date,
            instrument,
            detector,
            len(prepared_df),
            len(feature_columns),
            csv_path,
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        prepared_df.to_csv(csv_path, index=False)
        logger.info("Feature dataset saved: %s", csv_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unable to save feature dataset for %s (%s): %s", observation_date, instrument, exc)
        print(f"Unable to save feature dataset for {observation_date}: {exc}")


def _is_metadata_column(column_name: object, series: Optional[pd.Series] = None) -> bool:
    if not isinstance(column_name, str):
        return False
    lower_name = column_name.lower()
    if lower_name in {"time", "timestamp", "date", "observation_date", "instrument", "detector"}:
        return True
    if lower_name.endswith(("_date", "_time", "_dt")):
        return True
    if series is not None and pd.api.types.is_datetime64_any_dtype(series):
        return True
    return False


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_features(args)


if __name__ == "__main__":
    main()