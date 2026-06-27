"""
helios_features/helios_spectral_features.py
=============================================

Spectral-shape features for HEL1OS hard X-ray data.

Expected input schema
----------------------
    'time'    : datetime64[ns]
    'cdte_CR' : float   — CdTe broadband (20–150 keV)
    'czt_CR'  : float   — CZT broadband  (8–60 keV)

Optionally, for multi-bin mode:
    one column per energy bin (e.g. 'hxr_08_20', 'hxr_20_60', 'hxr_60_150')
    plus a mapping col → representative energy in keV.

Scientific grounding (Benz 2008)
---------------------------------
- Two-point photon index (§2.2, Eq. 2.2):
    F(E) ∝ E^{-γ} in the thick-target bremsstrahlung model.
    A two-point estimate between the CdTe and CZT representative energies
    gives a coarse γ proxy. The sign convention is F_high < F_low ⟹ γ > 0.
- Spectral slope / curvature (§2.2, Fig. 7):
    The HXR spectrum often has a spectral break near 50 keV in large
    events; curvature of the log-log spectrum is a cheap break detector.
- Spectral mode flag:
    Indicates whether multi-bin or two-channel estimation was used,
    so downstream models know the feature's reliability.

Representative energies used by default:
    CZT  centre ≈ 30 keV  (covers 8–60 keV)
    CdTe centre ≈ 70 keV  (covers 20–150 keV)
These are approximate; exact values depend on the sub-band integration
and the loaded detector response.

Coding style
------------
Identical to SpectralFeatures (SoLEXS): same dataclass, same feature_names /
transform public API, same _two_point_index / _multibin_fit helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class HEL1OSSpectralFeatureConfig:
    """Configuration for HEL1OSSpectralFeatures.

    Parameters
    ----------
    time_col : str
    energy_bin_cols : dict[str, float] or None
        Multi-bin mode: column name → representative photon energy (keV).
        E.g. {'hxr_08_20': 14.0, 'hxr_20_60': 40.0, 'hxr_60_150': 100.0}.
    cdte_col, czt_col : str or None
        Two-channel mode columns.
    czte_energy_kev : float
        Representative photon energy of the CZT channel.
    cdte_energy_kev : float
        Representative photon energy of the CdTe channel.
    min_flux_for_fit : float
        Minimum count rate below which the power-law fit returns NaN.
    curvature_eps : float
        Guard against division-by-zero in curvature calculation.
    """

    time_col:         str   = "time"
    energy_bin_cols:  Optional[Dict[str, float]] = None
    cdte_col:         Optional[str] = "cdte_CR"
    czt_col:          Optional[str] = "czt_CR"
    czte_energy_kev:  float = 30.0    # CZT representative energy
    cdte_energy_kev:  float = 70.0    # CdTe representative energy
    min_flux_for_fit: float = 1e-3
    curvature_eps:    float = 1e-12

    @classmethod
    def from_dict(cls, d: dict) -> "HEL1OSSpectralFeatureConfig":
        return cls(**d)


class HEL1OSSpectralFeatures:
    """Compute HXR spectral-shape features: photon index, slope, curvature.

    Usage
    -----
    >>> cfg = HEL1OSSpectralFeatureConfig()
    >>> hsf = HEL1OSSpectralFeatures(cfg)
    >>> out = hsf.transform(df)
    """

    def __init__(self, config: Optional[HEL1OSSpectralFeatureConfig] = None):
        self.config = config or HEL1OSSpectralFeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feature_names(self) -> List[str]:
        return [
            "hxr_photon_index_2pt",      # two-point log-log slope (CZT→CdTe)
            "hxr_hardness_ratio_2ch",    # CdTe / CZT ratio (raw)
            "hxr_photon_index_fit",      # multi-bin least-squares γ (NaN if <3 bins)
            "hxr_spectral_fit_r2",       # R² of multi-bin power-law fit
            "hxr_spectral_curvature",    # second difference of log-log spectrum
            "hxr_spectral_slope",        # linear slope of log(CR) vs log(E) over CZT→CdTe
            "hxr_spectral_mode",         # 'multibin' | 'two_channel' | 'none'
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all HEL1OS spectral features and return a new DataFrame."""
        cfg = self.config
        self._validate(df)

        out = df.copy()
        if len(out) == 0:
            for name in self.feature_names():
                out[name] = pd.Series(
                    dtype=object if name == "hxr_spectral_mode" else float
                )
            return out

        multibin_cols = self._resolve_multibin_cols(out)
        use_multibin  = len(multibin_cols) >= 3

        # ── Two-channel two-point slope ───────────────────────────────────
        two_ch_ok = (
            cfg.cdte_col is not None
            and cfg.czt_col is not None
            and cfg.cdte_col in out.columns
            and cfg.czt_col  in out.columns
        )
        if two_ch_ok:
            czte = out[cfg.czt_col].astype(float).clip(lower=0.0)
            cdte = out[cfg.cdte_col].astype(float).clip(lower=0.0)
            out["hxr_photon_index_2pt"] = self._two_point_index(
                czte, cdte, cfg.czte_energy_kev, cfg.cdte_energy_kev
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                hr = cdte / (czte + 1e-12)
            out["hxr_hardness_ratio_2ch"] = hr.replace([np.inf, -np.inf], np.nan)
            # Spectral slope = log(CdTe/CZT) / log(E_cdte/E_czte)
            log_e_ratio = np.log(cfg.cdte_energy_kev / cfg.czte_energy_kev)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_flux_ratio = np.log((cdte + 1e-12) / (czte + 1e-12))
            out["hxr_spectral_slope"] = (log_flux_ratio / log_e_ratio).replace(
                [np.inf, -np.inf], np.nan
            )
        else:
            out["hxr_photon_index_2pt"]  = np.nan
            out["hxr_hardness_ratio_2ch"] = np.nan
            out["hxr_spectral_slope"]    = np.nan

        # ── Multi-bin power-law fit ────────────────────────────────────────
        if use_multibin:
            energies    = np.array([multibin_cols[c] for c in multibin_cols])
            flux_matrix = (
                out[list(multibin_cols.keys())]
                .astype(float).clip(lower=0.0).to_numpy()
            )
            idx_fit, r2_fit, curvature = self._multibin_fit(flux_matrix, energies)
            out["hxr_photon_index_fit"]   = idx_fit
            out["hxr_spectral_fit_r2"]    = r2_fit
            out["hxr_spectral_curvature"] = curvature
            out["hxr_spectral_mode"]      = "multibin"
        else:
            out["hxr_photon_index_fit"]   = np.nan
            out["hxr_spectral_fit_r2"]    = np.nan
            out["hxr_spectral_curvature"] = np.nan
            out["hxr_spectral_mode"]      = "two_channel" if two_ch_ok else "none"

        return out

    # ------------------------------------------------------------------
    # Internal helpers — identical to SpectralFeatures (SoLEXS)
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
        low:    pd.Series,
        high:   pd.Series,
        e_low:  float,
        e_high: float,
    ) -> pd.Series:
        """Two-point log-log slope: γ = -d(log F)/d(log E)."""
        cfg = self.config
        log_e_ratio = np.log(e_high / e_low)
        valid = (low > cfg.min_flux_for_fit) & (high > cfg.min_flux_for_fit)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_flux_ratio = np.log(high / low)
        gamma = -log_flux_ratio / log_e_ratio
        return gamma.where(valid, np.nan)

    def _multibin_fit(
        self, flux_matrix: np.ndarray, energies: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-row power-law fit and curvature; identical to SpectralFeatures."""
        cfg = self.config
        n_rows, n_bins = flux_matrix.shape
        log_e = np.log(energies)

        gamma     = np.full(n_rows, np.nan)
        r2        = np.full(n_rows, np.nan)
        curvature = np.full(n_rows, np.nan)

        for i in range(n_rows):
            flux  = flux_matrix[i, :]
            valid = flux > cfg.min_flux_for_fit
            if valid.sum() < 3:
                continue
            x = log_e[valid]
            y = np.log(flux[valid])
            A = np.vstack([x, np.ones_like(x)]).T
            (m, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            gamma[i] = -m
            y_pred = m * x + b
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2[i]  = 1.0 - ss_res / ss_tot if ss_tot > cfg.curvature_eps else np.nan
            xs, ys = x[:3], y[:3]
            h1, h2 = xs[1] - xs[0], xs[2] - xs[1]
            denom  = h1 * h2 * (h1 + h2)
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