"""
Dataset Builder Configuration
==============================
Centralised, type-safe configuration for the Dataset Builder module.
All tuneable hyper-parameters live here; no magic numbers anywhere else.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------

#: Columns produced by the Feature Pipeline that are NOT model inputs.
#: These are either meta-data, labels-in-waiting, or leakage sources.
NON_FEATURE_COLUMNS: List[str] = [
    "time",
    # Phase columns — these are the ground truth; never feed back as input
    "flare_phase",
    "phase_preflare",
    "phase_impulsive",
    "phase_flash",
    "phase_decay",
    # time_since_peak leaks knowledge of the future peak time
    "time_since_peak_s",
]

#: Columns that are categorical / non-numeric even after the feature pipeline.
#: They will be excluded automatically if still present.
CATEGORICAL_COLUMNS: List[str] = [
    "spectral_mode",
]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """
    Full specification for one Dataset Builder run.

    Parameters
    ----------
    raw_data_dir:
        Directory that is scanned for ``features_*.csv`` files.
        Drop any new observation day CSV there — no code changes required.
    output_dir:
        Where ``.npy`` arrays, the scaler, and ``metadata.json`` are written.
    window_size:
        Number of consecutive 1-second time-steps in each input window (T).
    prediction_horizon:
        How many steps *ahead* of the last window sample the label is drawn from.
        0 → label the *last* sample of the window (nowcasting).
        N → label the sample N steps after the window ends (forecasting).
    train_frac:
        Fraction of chronologically ordered samples assigned to the training set.
    val_frac:
        Fraction assigned to validation.  test_frac = 1 - train_frac - val_frac.
    stride:
        Step size between consecutive sliding windows (1 = fully overlapping,
        window_size = non-overlapping).  Large strides reduce memory and training
        time at the cost of coverage.
    scaler_type:
        ``"standard"`` (zero-mean, unit-variance) or ``"minmax"`` ([0, 1]).
        StandardScaler is recommended for LSTM/CNN with tanh/sigmoid.
    imputation_strategy:
        How to fill NaN feature values before windowing.
        ``"forward"`` = forward-fill then back-fill (preserves causal order).
        ``"median"``  = global median imputation (simpler, slightly leaks).
    min_valid_fraction:
        Minimum fraction of non-NaN values a feature column must have to be
        retained.  Columns below this threshold are dropped entirely.
    random_seed:
        Fixed seed for any stochastic steps (currently unused, reserved for
        future data-augmentation).
    instrument_tag:
        Short string identifying the payload (``"solexs"`` or ``"helios"``).
        Written to metadata and used to namespace scaler files.
    """

    # In the DatasetConfig dataclass — replace the existing instrument_tag field:

    instrument_tag: Literal["solexs", "helios", "combined"] = "solexs"

    # I/O
    raw_data_dir: str = "ml/data/processed"
    output_dir: str = "ml/data/processed"

    # Window / horizon
    window_size: int = 60          # 60 s of 1-Hz data
    prediction_horizon: int = 0    # nowcast by default

    # Split fractions (must sum to ≤ 1.0; remainder → test)
    train_frac: float = 0.70
    val_frac: float = 0.15

    # Sliding window stride
    stride: int = 1

    # Normalisation
    scaler_type: Literal["standard", "minmax"] = "standard"

    # Imputation
    imputation_strategy: Literal["forward", "median"] = "forward"

    # Quality gate — drop columns with more NaN than this
    min_valid_fraction: float = 0.50

    # Reproducibility
    random_seed: int = 42

    # Instrument tag written to metadata
    instrument_tag: str = "solexs"

    # -----------------------------------------------------------------------
    # Derived helpers (not serialised)
    # -----------------------------------------------------------------------

    @property
    def test_frac(self) -> float:
        return round(1.0 - self.train_frac - self.val_frac, 10)

    def validate(self) -> None:
        """Raise ValueError for any invalid combination of parameters."""
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError(f"train_frac must be in (0, 1), got {self.train_frac}")
        if not (0.0 < self.val_frac < 1.0):
            raise ValueError(f"val_frac must be in (0, 1), got {self.val_frac}")
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1.0 (leaving room for test)")
        if self.window_size < 1:
            raise ValueError(f"window_size must be ≥ 1, got {self.window_size}")
        if self.prediction_horizon < 0:
            raise ValueError(f"prediction_horizon must be ≥ 0, got {self.prediction_horizon}")
        if self.stride < 1:
            raise ValueError(f"stride must be ≥ 1, got {self.stride}")
        if self.scaler_type not in ("standard", "minmax"):
            raise ValueError(f"Unknown scaler_type: {self.scaler_type!r}")
        if self.imputation_strategy not in ("forward", "median"):
            raise ValueError(f"Unknown imputation_strategy: {self.imputation_strategy!r}")
        if not (0.0 <= self.min_valid_fraction <= 1.0):
            raise ValueError(f"min_valid_fraction must be in [0, 1], got {self.min_valid_fraction}")
        if self.instrument_tag not in ("solexs", "helios", "combined"):
            raise ValueError(f"instrument_tag must be 'solexs', 'helios', or 'combined', got {self.instrument_tag!r}")

    # -----------------------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["test_frac"] = self.test_frac  # add derived field for human readers
        return d

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "DatasetConfig":
        with open(path) as fh:
            data = json.load(fh)
        data.pop("test_frac", None)  # derived; not a constructor arg
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetConfig":
        data = dict(data)
        data.pop("test_frac", None)
        return cls(**data)