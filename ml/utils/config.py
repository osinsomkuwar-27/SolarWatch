"""
ml/utils/config.py
==================
Central configuration loader for the Aditya-L1 pipeline.

Usage
-----
    from ml.utils.config import load_config, PipelineConfig
    cfg = load_config()          # loads config/pipeline.yaml
    cfg = load_config("path/to/other.yaml")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Dataclasses — typed wrappers around the YAML sections
# ─────────────────────────────────────────────────────────────

@dataclass
class PathsConfig:
    data_root:    Path
    raw_solexs:   Path
    raw_helios:   Path
    processed:    Path
    cache:        Path
    saved_models: Path
    logs:         Path
    notebooks:    Path

    def __post_init__(self) -> None:
        """Convert every string value to a Path and create dirs."""
        for fname in self.__dataclass_fields__:
            val = getattr(self, fname)
            if isinstance(val, str):
                setattr(self, fname, Path(val))
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for fname in self.__dataclass_fields__:
            p: Path = getattr(self, fname)
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class SoLEXSConfig:
    detectors:            list[str]
    energy_range_kev:     list[float]
    spectral_fit_min_kev: float
    spectral_fit_max_kev: float
    n_channels:           int
    channel_break:        int
    time_cadence_sec:     int
    lc_pattern:           str
    pi_pattern:           str
    gti_pattern:          str
    recommended_detector: str


@dataclass
class HEL1OSConfig:
    detectors:        list[str]
    energy_range_kev: list[float]
    cdte_fit_min_kev: float
    czt_fit_min_kev:  float
    lc_cadence_sec:   int
    spec_cadence_sec: int
    cdte_n_channels:  int
    czt_n_channels:   int
    cdte_bands_kev:   list[list[float]]
    czt_bands_kev:    list[list[float]]


@dataclass
class PreprocessingConfig:
    common_cadence_sec:     int
    clip_sigma:             float
    min_gti_fraction:       float
    background_window_sec:  int


@dataclass
class LabellingConfig:
    method:             str
    quiet_percentile:   float
    bc_percentile:      float
    m_percentile:       float
    peak_window_sec:    int


@dataclass
class FeaturesConfig:
    window_size_min:      int
    forecast_horizon_min: int
    step_size_min:        int
    include:              list[str]


@dataclass
class CNNConfig:
    input_length:              int
    n_features:                int
    n_classes:                 int
    filters:                   list[int]
    kernel_sizes:              list[int]
    dropout:                   float
    dense_units:               list[int]
    learning_rate:             float
    batch_size:                int
    epochs:                    int
    early_stopping_patience:   int


@dataclass
class LSTMConfig:
    input_length:            int
    n_features:              int
    n_classes:               int
    hidden_units:            list[int]
    dropout:                 float
    recurrent_dropout:       float
    dense_units:             list[int]
    learning_rate:           float
    batch_size:              int
    epochs:                  int
    early_stopping_patience: int


@dataclass
class TrainingConfig:
    val_fraction:     float
    test_fraction:    float
    split_strategy:   str
    use_class_weights: bool
    mixed_precision:  bool


@dataclass
class EvaluationConfig:
    metrics:       list[str]
    cm_normalize:  str


@dataclass
class LoggingConfig:
    level:        str
    format:       str
    file:         str
    max_bytes:    int
    backup_count: int


@dataclass
class ProjectConfig:
    name:    str
    version: str
    seed:    int


@dataclass
class PipelineConfig:
    """
    Top-level configuration object.  Every subsystem receives this object
    so no subsystem has to open a file on its own.
    """
    project:        ProjectConfig
    paths:          PathsConfig
    solexs:         SoLEXSConfig
    helios:         HEL1OSConfig
    preprocessing:  PreprocessingConfig
    labelling:      LabellingConfig
    features:       FeaturesConfig
    cnn:            CNNConfig
    lstm:           LSTMConfig
    training:       TrainingConfig
    evaluation:     EvaluationConfig
    logging:        LoggingConfig

    # Raw dict kept for any keys not yet mapped to a dataclass
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)


# ─────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "config" / "pipeline.yaml"


def _resolve_project_root() -> Path:
    """Return the project root (the directory that contains 'config/')."""
    return _DEFAULT_CONFIG_PATH.parent.parent


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    """
    Load and validate the pipeline configuration.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to a YAML config file.  Defaults to ``config/pipeline.yaml``
        relative to the project root.

    Returns
    -------
    PipelineConfig
        Fully typed configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    KeyError
        If a required top-level section is missing.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    # Allow env-var override
    env_override = os.environ.get("SOLAR_CONFIG_PATH")
    if env_override:
        path = Path(env_override)
        logger.info("Config path overridden by env SOLAR_CONFIG_PATH: %s", path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            f"Expected at: {_DEFAULT_CONFIG_PATH}"
        )

    with open(path, "r") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    logger.info("Loaded configuration from %s", path)

    # Resolve all relative paths against project root
    root = _resolve_project_root()

    def _abs(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else root / p

    raw_paths = raw["paths"]
    paths_cfg = PathsConfig(
        data_root    = _abs(raw_paths["data_root"]),
        raw_solexs   = _abs(raw_paths["raw_solexs"]),
        raw_helios   = _abs(raw_paths["raw_helios"]),
        processed    = _abs(raw_paths["processed"]),
        cache        = _abs(raw_paths["cache"]),
        saved_models = _abs(raw_paths["saved_models"]),
        logs         = _abs(raw_paths["logs"]),
        notebooks    = _abs(raw_paths["notebooks"]),
    )

    cfg = PipelineConfig(
        project       = ProjectConfig(**raw["project"]),
        paths         = paths_cfg,
        solexs        = SoLEXSConfig(**raw["solexs"]),
        helios        = HEL1OSConfig(**raw["helios"]),
        preprocessing = PreprocessingConfig(**raw["preprocessing"]),
        labelling     = LabellingConfig(**raw["labelling"]),
        features      = FeaturesConfig(**raw["features"]),
        cnn           = CNNConfig(**raw["cnn"]),
        lstm          = LSTMConfig(**raw["lstm"]),
        training      = TrainingConfig(**raw["training"]),
        evaluation    = EvaluationConfig(**raw["evaluation"]),
        logging       = LoggingConfig(**raw["logging"]),
        _raw          = raw,
    )

    _configure_logging(cfg.logging, paths_cfg.logs)
    logger.info(
        "Pipeline '%s' v%s | seed=%d",
        cfg.project.name, cfg.project.version, cfg.project.seed,
    )
    return cfg


def _configure_logging(log_cfg: LoggingConfig, log_dir: Path) -> None:
    """Set up root logger with file + stream handlers."""
    import logging.handlers

    root_logger = logging.getLogger()

    # Avoid adding duplicate handlers on re-import
    if root_logger.handlers:
        return

    level = getattr(logging, log_cfg.level.upper(), logging.INFO)
    root_logger.setLevel(level)

    fmt = logging.Formatter(log_cfg.format)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    # Rotating file handler
    log_file = log_dir / Path(log_cfg.file).name
    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=log_cfg.max_bytes,
        backupCount=log_cfg.backup_count,
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)
