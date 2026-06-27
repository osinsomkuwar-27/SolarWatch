"""
Dataset Builder
================
Converts engineered feature CSVs (``features_*.csv``) produced by the Feature
Pipeline into chronologically-split, normalised NumPy arrays ready for CNN /
LSTM training.

Design principles
-----------------
* **Zero data leakage** – splits are strictly chronological; scaler is fitted
  on training data only and then applied to val/test.
* **Drop-in extensibility** – new observation-day CSVs dropped into
  ``ml/data/processed/`` are picked up automatically without code changes.
* **HEL1OS-ready** – the builder is instrument-agnostic; any CSV that follows
  the same column schema (or a subset) is supported.
* **Production quality** – every step is logged, validated, and recoverable
  via the saved ``metadata.json``.

Outputs (written to ``output_dir``)
-------------------------------------
``X_train.npy``, ``X_val.npy``, ``X_test.npy``
    Shape ``(N, window_size, n_features)`` — float32.
``y_train.npy``, ``y_val.npy``, ``y_test.npy``
    Shape ``(N,)`` — int8 class labels (to be filled by the Label Generator).
    At this stage they contain the raw ``flare_phase`` integer encoding so the
    Label Generator can replace them without re-running the builder.
``scaler_<instrument_tag>.joblib``
    Fitted scikit-learn scaler for inference-time normalisation.
``metadata.json``
    Full provenance record: config, feature list, split sizes, file list, etc.

Usage
-----
>>> from ml.dataset.builder import DatasetBuilder
>>> from ml.dataset.config import DatasetConfig
>>> cfg = DatasetConfig(window_size=60, prediction_horizon=0, stride=5)
>>> builder = DatasetBuilder(cfg)
>>> builder.run()
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ml.dataset.config import (
    CATEGORICAL_COLUMNS,
    NON_FEATURE_COLUMNS,
    DatasetConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal type aliases
# ---------------------------------------------------------------------------

_ScalerType = StandardScaler | MinMaxScaler
_SplitArrays = Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Public builder class
# ---------------------------------------------------------------------------


class DatasetBuilder:
    """
    Orchestrates the full pipeline from raw CSVs to training-ready arrays.

    Parameters
    ----------
    config:
        A :class:`~ml.dataset.config.DatasetConfig` instance.  All knobs live
        there; this class contains no magic numbers.
    """

    # Matches any features_*.csv produced by the Feature Pipeline.
    _CSV_PATTERN = re.compile(r"^features_.+\.csv$", re.IGNORECASE)

    # Per-instrument filename patterns for filtered discovery and auto-inference.
    # Group 1 of each pattern is the date token (used for sorting).
    _INSTRUMENT_PATTERNS: dict = {
        "solexs":   re.compile(r"^features_solexs_[^_]+_(\d{8})\.csv$",  re.IGNORECASE),
        "helios":   re.compile(r"^features_helios_[^_]+_(\d{8})\.csv$",  re.IGNORECASE),
        "combined": re.compile(r"^features_combined_(\d{8})\.csv$",       re.IGNORECASE),
    }

    def __init__(self, config: DatasetConfig) -> None:
        config.validate()
        self.cfg = config
        self._raw_dir = Path(config.raw_data_dir)
        self._out_dir = Path(config.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)

        self._feature_columns: List[str] = []
        self._scaler: Optional[_ScalerType] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full build pipeline end-to-end."""
        t0 = time.perf_counter()
        logger.info("=== DatasetBuilder starting ===")
        logger.info("Config: %s", self.cfg.to_dict())

        # 1. Discover and load all observation-day CSVs
        csv_files = self._discover_csvs()
        df_raw = self._load_and_concatenate(csv_files)

        # 2. Select and clean feature columns
        df_features, df_labels_raw = self._prepare_features(df_raw)

        # 3. Chronological train / val / test split (at row level, before windowing)
        splits_feat, splits_label = self._chronological_split(df_features, df_labels_raw)

        # 4. Fit scaler on training data; transform all splits
        splits_feat_scaled = self._fit_and_scale(splits_feat)

        # 5. Build sliding-window arrays
        split_X, split_y = self._build_windows(splits_feat_scaled, splits_label)

        # 6. Persist arrays, scaler, and metadata
        self._save_arrays(split_X, split_y)
        self._save_scaler()
        self._save_metadata(
            csv_files=csv_files,
            n_raw_rows=len(df_raw),
            split_X=split_X,
            split_y=split_y,
        )

        elapsed = time.perf_counter() - t0
        logger.info("=== DatasetBuilder completed in %.1f s ===", elapsed)

    # ------------------------------------------------------------------
    # Step 1 – Discover CSVs
    # ------------------------------------------------------------------

    def _discover_csvs(self) -> List[Path]:
        """
        Return a chronologically sorted list of feature CSV paths found in
        raw_data_dir, filtered by instrument_tag.

        Supported filename conventions
        --------------------------------
        solexs   : features_solexs_<detector>_<YYYYMMDD>.csv
        helios   : features_helios_<detector>_<YYYYMMDD>.csv
        combined : features_combined_<YYYYMMDD>.csv

        If instrument_tag was not explicitly set by the caller (i.e. it still
        holds the dataclass default of ``"solexs"``), this method first checks
        whether the directory contains *only* files that unambiguously match a
        single instrument type and, if so, auto-infers the tag.  A warning is
        logged so the operator knows inference occurred.
        """
        if not self._raw_dir.exists():
            raise FileNotFoundError(
                f"raw_data_dir does not exist: {self._raw_dir.resolve()}"
            )

        all_csv = [
            p for p in self._raw_dir.iterdir()
            if p.is_file() and self._CSV_PATTERN.match(p.name)
        ]

        # ── Auto-infer instrument tag when not explicitly provided ────────
        # We treat the default value ("solexs") as "not explicitly set" only
        # when the directory contains no solexs files but does contain files
        # of exactly one other instrument type.  Explicit CLI --instrument
        # flags flow through DatasetConfig and are always respected as-is.
        tag = self.cfg.instrument_tag
        matched_for_tag = self._filter_by_instrument(all_csv, tag)

        if not matched_for_tag:
            # Current tag yields nothing — attempt auto-inference.
            inferred: Optional[str] = None
            for candidate in ("solexs", "helios", "combined"):
                if candidate == tag:
                    continue
                if self._filter_by_instrument(all_csv, candidate):
                    if inferred is not None:
                        # Multiple instrument types present: ambiguous.
                        inferred = None
                        break
                    inferred = candidate

            if inferred is not None:
                logger.warning(
                    "instrument_tag=%r yielded no files; auto-inferred %r "
                    "from directory contents.  Pass --instrument explicitly "
                    "to suppress this warning.",
                    tag,
                    inferred,
                )
                self.cfg.instrument_tag = inferred
                tag = inferred
                matched_for_tag = self._filter_by_instrument(all_csv, tag)

        if not matched_for_tag:
            raise FileNotFoundError(
                f"No feature CSVs matching instrument_tag={tag!r} found in "
                f"{self._raw_dir}.  Expected filenames:\n"
                f"  solexs   → features_solexs_<detector>_<YYYYMMDD>.csv\n"
                f"  helios   → features_helios_<detector>_<YYYYMMDD>.csv\n"
                f"  combined → features_combined_<YYYYMMDD>.csv"
            )

        # Chronological sort: extract the date token captured by the pattern.
        pattern = self._INSTRUMENT_PATTERNS[tag]
        files = sorted(
            matched_for_tag,
            key=lambda p: (pattern.match(p.name).group(1), p.name),
        )

        logger.info(
            "Discovered %d CSV file(s) for instrument=%r: %s",
            len(files),
            tag,
            [f.name for f in files],
        )
        return files

    def _filter_by_instrument(self, paths: List[Path], tag: str) -> List[Path]:
        """Return only paths whose filename matches the pattern for *tag*."""
        pattern = self._INSTRUMENT_PATTERNS.get(tag)
        if pattern is None:
            raise ValueError(
                f"Unknown instrument_tag: {tag!r}. "
                f"Valid values: {list(self._INSTRUMENT_PATTERNS)}"
            )
        return [p for p in paths if pattern.match(p.name)]

    # ------------------------------------------------------------------
    # Step 2 – Load and concatenate
    # ------------------------------------------------------------------

    def _load_and_concatenate(self, csv_files: List[Path]) -> pd.DataFrame:
        """
        Load every CSV, parse timestamps, sort chronologically, and concatenate.
        Each file represents one observation day; together they form the full
        multi-day dataset.
        """
        frames: List[pd.DataFrame] = []
        for path in csv_files:
            logger.info("Loading %s …", path.name)
            df = pd.read_csv(path, low_memory=False)

            if "time" not in df.columns:
                raise ValueError(f"Column 'time' missing from {path.name}")

            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            n_bad_ts = df["time"].isna().sum()
            if n_bad_ts:
                logger.warning("  %d rows with un-parseable timestamps dropped.", n_bad_ts)
                df = df.dropna(subset=["time"])

            df = df.sort_values("time").reset_index(drop=True)
            df["_source_file"] = path.name   # provenance column (dropped later)
            frames.append(df)
            logger.info("  → %d rows loaded.", len(df))

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("time").reset_index(drop=True)
        logger.info("Total rows after concatenation: %d", len(combined))
        return combined

    # ------------------------------------------------------------------
    # Step 3 – Feature preparation
    # ------------------------------------------------------------------

    def _prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select numeric feature columns, apply quality gates, and impute NaNs.

        Returns
        -------
        df_features:
            DataFrame containing only the selected, imputed feature columns.
        labels_raw:
            Series with the raw ``flare_phase`` string (for later Label Generator
            consumption) — kept aligned by index.
        """
        # ── Extract raw label series before any column filtering ──────────
        if "flare_phase" not in df.columns:
            raise ValueError("Column 'flare_phase' not found — cannot extract labels.")
        labels_raw: pd.Series = df["flare_phase"].copy()

        # ── Drop all non-feature and categorical columns ──────────────────
        drop_cols = set(NON_FEATURE_COLUMNS) | set(CATEGORICAL_COLUMNS) | {"_source_file"}
        available_drop = [c for c in drop_cols if c in df.columns]
        df_feat = df.drop(columns=available_drop)

        # ── Keep only numeric columns ─────────────────────────────────────
        df_feat = df_feat.select_dtypes(include=[np.number])

        # ── Quality gate: drop columns with too many NaNs ─────────────────
        valid_frac = df_feat.notna().mean()
        keep_mask = valid_frac >= self.cfg.min_valid_fraction
        dropped_cols = valid_frac[~keep_mask].index.tolist()
        if dropped_cols:
            logger.warning(
                "Dropping %d low-coverage feature column(s): %s",
                len(dropped_cols),
                dropped_cols,
            )
        df_feat = df_feat.loc[:, keep_mask]

        logger.info(
            "Feature matrix shape after quality gate: %s", df_feat.shape
        )

        # ── Impute NaNs ───────────────────────────────────────────────────
        df_feat = self._impute(df_feat)

        # Store the final column list for metadata and inference
        self._feature_columns = df_feat.columns.tolist()
        logger.info("Final feature count: %d", len(self._feature_columns))

        return df_feat, labels_raw

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values according to the configured imputation strategy."""
        strategy = self.cfg.imputation_strategy
        if strategy == "forward":
            # Forward-fill preserves causal ordering; back-fill handles leading NaNs
            df = df.ffill().bfill()
        elif strategy == "median":
            medians = df.median()
            df = df.fillna(medians)
        else:
            raise ValueError(f"Unknown imputation_strategy: {strategy!r}")

        remaining = df.isna().sum().sum()
        if remaining:
            logger.warning(
                "%d NaN values remain after imputation — filling with 0.", remaining
            )
            df = df.fillna(0.0)
        return df

    # ------------------------------------------------------------------
    # Step 4 – Chronological split
    # ------------------------------------------------------------------

    def _chronological_split(
        self,
        df_feat: pd.DataFrame,
        labels_raw: pd.Series,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """
        Split rows in strict temporal order.

        Rationale
        ---------
        Solar flare prediction is a forecasting task.  Any random shuffle of
        the data would contaminate the future into training, inflating reported
        metrics and producing a model that cannot generalise to real-time
        operations.  The split is therefore a simple prefix/suffix cut on the
        sorted time axis.
        """
        n = len(df_feat)
        n_train = int(n * self.cfg.train_frac)
        n_val = int(n * self.cfg.val_frac)
        # Test gets everything that remains (avoids rounding errors)

        idxs = {
            "train": slice(0, n_train),
            "val":   slice(n_train, n_train + n_val),
            "test":  slice(n_train + n_val, n),
        }

        splits_feat: Dict[str, pd.DataFrame] = {}
        splits_label: Dict[str, pd.Series] = {}
        for name, sl in idxs.items():
            splits_feat[name] = df_feat.iloc[sl].reset_index(drop=True)
            splits_label[name] = labels_raw.iloc[sl].reset_index(drop=True)
            logger.info("Split %-5s → %d rows", name, len(splits_feat[name]))

        return splits_feat, splits_label

    # ------------------------------------------------------------------
    # Step 5 – Normalisation
    # ------------------------------------------------------------------

    def _fit_and_scale(
        self, splits: Dict[str, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """
        Fit scaler on training data only; transform all splits.

        Returns float32 numpy arrays (cheaper memory-wise for large datasets).
        """
        if self.cfg.scaler_type == "standard":
            self._scaler = StandardScaler()
        else:
            self._scaler = MinMaxScaler()

        logger.info("Fitting %s scaler on training split …", self.cfg.scaler_type)
        train_arr = splits["train"].values.astype(np.float64)
        self._scaler.fit(train_arr)

        scaled: Dict[str, np.ndarray] = {}
        for name, df in splits.items():
            arr = df.values.astype(np.float64)
            scaled[name] = self._scaler.transform(arr).astype(np.float32)
            logger.info("Scaled split %-5s shape: %s", name, scaled[name].shape)

        return scaled

    # ------------------------------------------------------------------
    # Step 6 – Sliding-window construction
    # ------------------------------------------------------------------

    def _build_windows(
        self,
        splits_feat: Dict[str, np.ndarray],
        splits_label: Dict[str, pd.Series],
    ) -> Tuple[_SplitArrays, _SplitArrays]:
        """
        Convert flat (N_rows × F_features) arrays into
        windowed (N_windows × window_size × F_features) tensors.

        The label for each window is drawn from the row at position
        ``window_end + prediction_horizon``.  If the horizon extends beyond the
        split boundary, the trailing windows are discarded.

        Window indexing
        ---------------
        For a window of size W and horizon H:

            input  rows : [i, i+1, ..., i+W-1]
            label  row  : i + W - 1 + H

        This is the *last valid* causal position; no future information leaks
        into the feature matrix.
        """
        W = self.cfg.window_size
        H = self.cfg.prediction_horizon
        S = self.cfg.stride

        # Raw phase → integer encoding (consistent with Label Generator)
        PHASE_MAP = {
            "preflare":  0,
            "impulsive": 1,
            "flash":     2,
            "decay":     3,
        }

        split_X: _SplitArrays = {}
        split_y: _SplitArrays = {}

        for name in ("train", "val", "test"):
            feat_arr = splits_feat[name]        # (N, F)  float32
            label_ser = splits_label[name]      # (N,)    str

            N, F = feat_arr.shape
            # Maximum valid start index so that label index stays in-bounds
            max_start = N - W - H

            if max_start <= 0:
                logger.warning(
                    "Split %s has %d rows — not enough for window_size=%d + "
                    "horizon=%d.  Writing empty arrays.",
                    name, N, W, H,
                )
                split_X[name] = np.empty((0, W, F), dtype=np.float32)
                split_y[name] = np.empty((0,), dtype=np.int8)
                continue

            # ── Pre-allocate arrays ───────────────────────────────────────
            starts = np.arange(0, max_start + 1, S)
            n_windows = len(starts)
            X = np.empty((n_windows, W, F), dtype=np.float32)
            y = np.empty((n_windows,), dtype=np.int8)

            # ── Encode labels ──────────────────────────────────────────────
            label_int = label_ser.map(PHASE_MAP).fillna(-1).astype(np.int8).values

            # ── Fill windows ──────────────────────────────────────────────
            for j, i in enumerate(starts):
                X[j] = feat_arr[i : i + W]
                label_idx = i + W - 1 + H
                y[j] = label_int[label_idx]

            split_X[name] = X
            split_y[name] = y
            logger.info(
                "Windows %-5s → X: %s  y: %s  (label distribution: %s)",
                name,
                X.shape,
                y.shape,
                {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            )

        return split_X, split_y

    # ------------------------------------------------------------------
    # Step 7 – Persist
    # ------------------------------------------------------------------

    def _save_arrays(
        self,
        split_X: _SplitArrays,
        split_y: _SplitArrays,
    ) -> None:
        """Save X and y arrays as compressed .npy files."""
        for split in ("train", "val", "test"):
            x_path = self._out_dir / f"X_{split}.npy"
            y_path = self._out_dir / f"y_{split}.npy"
            np.save(str(x_path), split_X[split])
            np.save(str(y_path), split_y[split])
            logger.info("Saved %s  %s", x_path.name, split_X[split].shape)
            logger.info("Saved %s  %s", y_path.name, split_y[split].shape)

    def _save_scaler(self) -> None:
        """Persist the fitted scaler for use during inference."""
        if self._scaler is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        scaler_path = self._out_dir / f"scaler_{self.cfg.instrument_tag}.joblib"
        joblib.dump(self._scaler, scaler_path)
        logger.info("Scaler saved → %s", scaler_path)

    def _save_metadata(
        self,
        csv_files: List[Path],
        n_raw_rows: int,
        split_X: _SplitArrays,
        split_y: _SplitArrays,
    ) -> None:
        """Write a full provenance JSON so every artefact is reproducible."""
        phase_map_inv = {0: "preflare", 1: "impulsive", 2: "flash", 3: "decay"}

        label_distributions: dict = {}
        for split in ("train", "val", "test"):
            y = split_y[split]
            unique, counts = np.unique(y, return_counts=True)
            label_distributions[split] = {
                phase_map_inv.get(int(k), str(k)): int(v)
                for k, v in zip(unique, counts)
            }

        metadata = {
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "instrument": self.cfg.instrument_tag,
            "config": self.cfg.to_dict(),
            "source_files": [f.name for f in csv_files],
            "n_raw_rows": n_raw_rows,
            "n_features": len(self._feature_columns),
            "feature_columns": self._feature_columns,
            "phase_encoding": phase_map_inv,
            "splits": {
                split: {
                    "X_shape": list(split_X[split].shape),
                    "y_shape": list(split_y[split].shape),
                    "label_distribution": label_distributions[split],
                }
                for split in ("train", "val", "test")
            },
            "output_files": {
                "X_train": "X_train.npy",
                "X_val":   "X_val.npy",
                "X_test":  "X_test.npy",
                "y_train": "y_train.npy",
                "y_val":   "y_val.npy",
                "y_test":  "y_test.npy",
                "scaler":  f"scaler_{self.cfg.instrument_tag}.joblib",
                "metadata": "metadata.json",
            },
        }

        meta_path = self._out_dir / "metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("Metadata saved → %s", meta_path)