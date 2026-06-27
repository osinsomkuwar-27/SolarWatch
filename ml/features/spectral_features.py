"""
spectral_features.py
=====================

Features that summarise the *spectral shape* of emission, either from
true multi-energy-bin spectra (if provided) or, in the common case of
only band-integrated soft/hard channels, from a two-point power-law
approximation between them.

Expected input schema
----------------------
Either:

(a) Multi-bin mode -- a DataFrame with:
        'time' : datetime64[ns]
        one column per energy bin, e.g. 'E_06_12', 'E_12_25', 'E_25_50'
    and a separate `energy_bins_kev` mapping column_name -> representative
    energy (keV), passed via SpectralFeatureConfig.energy_bin_cols.
    This mirrors RHESSI-style imaging spectroscopy (Sec. 2.2, Fig. 7/11).

(b) Two-channel mode -- a DataFrame with:
        'time'   : datetime64[ns]
        soft_col : float  (e.g. 'CR', representing a low-energy channel)
        hard_col : float  (e.g. 'CR_hard', representing a high-energy
                           channel)
    plus the representative energies of each channel
    (soft_energy_kev, hard_energy_kev). A power-law index is estimated
    from the two-point log-log slope between them. This is a coarse
    approximation to the true photon power-law index gamma described
    in Sec. 2.2 ("thick target photon spectrum... flatter... than any
    thin target"), valid only as a relative hardness tracker, not a
    substitute for proper spectral fitting.

If neither multi-bin columns nor both soft/hard columns are present,
spectral features are filled with NaN (graceful degradation, consistent
with flare_features.py's handling of a missing hard channel).

Scientific grounding (Benz 2008, "Flare Observations")
-------------------------------------------------------
- Photon power-law index gamma (Sec. 2.2):
      Thick-target bremsstrahlung spectrum is close to a power law;
      gamma = delta - 1 where delta is the electron spectral index.
      Footpoint (thick-target) spectra are flatter (harder, smaller
      gamma) than coronal (thin-target) spectra by ~2 in the simplest
      picture (observed differences vary, Sec. 2.2-2.3, Fig. 11).
- Spectral break / two-component fit (Sec. 2.2, Fig. 7):
      RHESSI spectra are often fit with a thermal component (~10-20 MK)
      at low energies plus a non-thermal power law at high energies,
      sometimes with *two* power-law breaks (Fig. 7: breaks near 12 and
      50 keV). We approximate the "thermal-dominance" of a sample by
      the curvature of the log-log spectrum at low energy, when >= 3
      bins are available.
- Soft-hard-soft spectral evolution (Sec. 5.2, Eq. 6):
      gamma = A * F(E0)^(-alpha); spectral index correlates with flux.
      We expose the instantaneous gamma estimate so that downstream
      code (or feature_pipeline.py) can correlate it with flux exactly
      as Grigis & Benz (2004) do, without re-deriving gamma itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SpectralFeatureConfig:
    """Configuration for SpectralFeatures.

    Parameters
    ----------
    time_col : str
        Timestamp column.
    energy_bin_cols : dict[str, float] or None
        Multi-bin mode: maps a count-rate column name to its
        representative photon energy in keV, e.g.
        {'E_06_12': 9.0, 'E_12_25': 18.0, 'E_25_50': 37.5}.
        If supplied (and >= 3 valid columns are present in the data),
        multi-bin mode is used in addition to (not instead of) the
        two-channel slope below.
    soft_col, hard_col : str or None
        Two-channel mode column names.
    soft_energy_kev, hard_energy_kev : float
        Representative photon energies (keV) of the two channels, used
        for the two-point power-law slope. Defaults follow a generic
        GOES-like soft channel (~5 keV) and a RHESSI-like hard channel
        (~35 keV, matching the E0 used in Benz 2008 Eq. 6).
    min_flux_for_fit : float
        Minimum count rate (in whichever channel) below which a
        power-law fit is not attempted (returns NaN) -- prevents
        nonsensical slopes when both channels are near background/zero.
    curvature_eps : float
        Small constant guarding against division by zero in degenerate
        (zero-variance or equal-energy-spacing) cases.
    """

    time_col: str = "time"
    energy_bin_cols: Optional[Dict[str, float]] = None
    soft_col: Optional[str] = "CR"
    hard_col: Optional[str] = "CR_hard"
    soft_energy_kev: float = 5.0
    hard_energy_kev: float = 35.0
    min_flux_for_fit: float = 1e-3
    curvature_eps: float = 1e-12

    @classmethod
    def from_dict(cls, d: dict) -> "SpectralFeatureConfig":
        return cls(**d)


class SpectralFeatures:
    """Compute power-law / thermal spectral-shape features.

    Usage
    -----
    >>> cfg = SpectralFeatureConfig(soft_col="CR", hard_col="CR_hard")
    >>> sf = SpectralFeatures(cfg)
    >>> out = sf.transform(df)
    """

    def __init__(self, config: Optional[SpectralFeatureConfig] = None):
        self.config = config or SpectralFeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feature_names(self) -> List[str]:
        return [
            "photon_index_2pt",
            "hardness_ratio_2ch",
            "photon_index_fit",
            "spectral_fit_r2",
            "spectral_curvature",
            "spectral_mode",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        self._validate(df)

        out = df.copy()
        if len(out) == 0:
            for name in self.feature_names():
                out[name] = pd.Series(
                    dtype=object if name == "spectral_mode" else float
                )
            return out

        multibin_cols = self._resolve_multibin_cols(out)
        use_multibin = len(multibin_cols) >= 3  # need >=3 points for curvature/fit

        # --- Two-channel two-point slope (always attempted if available) ---
        two_ch_ok = (
            cfg.soft_col is not None
            and cfg.hard_col is not None
            and cfg.soft_col in out.columns
            and cfg.hard_col in out.columns
        )
        if two_ch_ok:
            soft = out[cfg.soft_col].astype(float).clip(lower=0.0)
            hard = out[cfg.hard_col].astype(float).clip(lower=0.0)
            out["photon_index_2pt"] = self._two_point_index(
                soft, hard, cfg.soft_energy_kev, cfg.hard_energy_kev
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                hr = hard / (soft + 1e-12)
            out["hardness_ratio_2ch"] = hr
        else:
            out["photon_index_2pt"] = np.nan
            out["hardness_ratio_2ch"] = np.nan

        # --- Multi-bin power-law fit + curvature ----------------------------
        if use_multibin:
            energies = np.array([multibin_cols[c] for c in multibin_cols])
            flux_matrix = (
                out[list(multibin_cols.keys())]
                .astype(float)
                .clip(lower=0.0)
                .to_numpy()
            )
            idx_fit, r2_fit, curvature = self._multibin_fit(flux_matrix, energies)
            out["photon_index_fit"] = idx_fit
            out["spectral_fit_r2"] = r2_fit
            out["spectral_curvature"] = curvature
            out["spectral_mode"] = "multibin"
        else:
            out["photon_index_fit"] = np.nan
            out["spectral_fit_r2"] = np.nan
            out["spectral_curvature"] = np.nan
            out["spectral_mode"] = "two_channel" if two_ch_ok else "none"

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_multibin_cols(self, df: pd.DataFrame) -> Dict[str, float]:
        cfg = self.config
        if not cfg.energy_bin_cols:
            return {}
        return {
            col: energy
            for col, energy in cfg.energy_bin_cols.items()
            if col in df.columns
        }

    def _two_point_index(
        self,
        soft: pd.Series,
        hard: pd.Series,
        e_soft: float,
        e_hard: float,
    ) -> pd.Series:
        """Two-point log-log slope: gamma = -d(log F)/d(log E).

        A steeper (more negative) flux ratio in log-log space gives a
        *larger* gamma (softer spectrum); this sign convention matches
        the usual photon power-law index definition F(E) ~ E^{-gamma}.
        """
        cfg = self.config
        log_e_ratio = np.log(e_hard / e_soft)

        valid = (soft > cfg.min_flux_for_fit) & (hard > cfg.min_flux_for_fit)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_flux_ratio = np.log(hard / soft)
        gamma = -log_flux_ratio / log_e_ratio
        gamma = gamma.where(valid, np.nan)
        return gamma

    def _multibin_fit(
        self, flux_matrix: np.ndarray, energies: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-row power-law fit F(E) ~ E^{-gamma} via least squares in
        log-log space, plus a curvature diagnostic.

        Returns
        -------
        gamma     : fitted power-law index per row (NaN if insufficient
                    valid bins)
        r2        : R^2 of the log-log linear fit (goodness of power-law
                    approximation; low R^2 flags a spectral break/thermal
                    contribution, cf. Fig. 7's two breaks)
        curvature : second difference of log(flux) vs log(energy) over
                    the first 3 valid bins -- a cheap, robust break/
                    thermal-bump detector that doesn't require a full
                    fit
        """
        cfg = self.config
        n_rows, n_bins = flux_matrix.shape
        log_e = np.log(energies)

        gamma = np.full(n_rows, np.nan)
        r2 = np.full(n_rows, np.nan)
        curvature = np.full(n_rows, np.nan)

        for i in range(n_rows):
            flux = flux_matrix[i, :]
            valid = flux > cfg.min_flux_for_fit
            if valid.sum() < 3:
                continue

            x = log_e[valid]
            y = np.log(flux[valid])

            # Least-squares linear fit: y = m*x + b ; gamma = -m
            A = np.vstack([x, np.ones_like(x)]).T
            (m, b), _residuals, _rank, _sv = np.linalg.lstsq(A, y, rcond=None)
            gamma[i] = -m

            y_pred = m * x + b
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2[i] = 1.0 - ss_res / ss_tot if ss_tot > cfg.curvature_eps else np.nan

            xs = x[:3]
            ys = y[:3]
            h1 = xs[1] - xs[0]
            h2 = xs[2] - xs[1]
            denom = h1 * h2 * (h1 + h2)
            if abs(denom) > cfg.curvature_eps:
                curvature[i] = (
                    2.0 * (h1 * ys[2] - (h1 + h2) * ys[1] + h2 * ys[0]) / denom
                )

        return gamma, r2, curvature

    def _validate(self, df: pd.DataFrame) -> None:
        cfg = self.config
        if cfg.time_col not in df.columns:
            raise KeyError(f"Missing required time column: '{cfg.time_col}'")
        if len(df) == 0:
            return
        if not pd.api.types.is_datetime64_any_dtype(df[cfg.time_col]):
            raise TypeError(f"Column '{cfg.time_col}' must be datetime64 dtype")
