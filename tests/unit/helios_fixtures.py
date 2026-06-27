"""
Shared synthetic test fixtures for HEL1OS feature module tests.

make_synthetic_helios_flare() builds a two-channel (CdTe + CZT) HXR light
curve whose phase timing is shorter than the SoLEXS fixtures, consistent
with the faster impulsive-phase timescales of hard X-ray emission
(Benz 2008, §2.2):
  preflare   ~2 min
  impulsive  ~3 min (steeper rise than SXR)
  flash      ~5 min
  decay      ~20 min (shorter than SXR 1-hour gradual phase)

The CdTe channel is the high-energy (harder) reference; the CZT channel
is the low-energy reference.  By construction the CdTe/CZT hardness ratio
rises during the impulsive phase and falls during the decay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_helios_flare(cadence_s: int = 4, seed: int = 0) -> pd.DataFrame:
    """Return a DataFrame with 'time', 'cdte_CR', 'czt_CR' columns."""
    rng = np.random.default_rng(seed)

    preflare_dur  = 2 * 60
    impulsive_dur = 3 * 60
    flash_dur     = 5 * 60
    decay_dur     = 20 * 60

    t0 = pd.Timestamp("2026-06-21 06:00:00")

    def seg_times(start_s, dur_s):
        n = int(dur_s // cadence_s)
        return start_s + np.arange(n) * cadence_s

    t_pre   = seg_times(0, preflare_dur)
    t_imp   = seg_times(preflare_dur, impulsive_dur)
    t_flash = seg_times(preflare_dur + impulsive_dur, flash_dur)
    t_decay = seg_times(preflare_dur + impulsive_dur + flash_dur, decay_dur)
    t_all   = np.concatenate([t_pre, t_imp, t_flash, t_decay])
    time    = t0 + pd.to_timedelta(t_all, unit="s")

    bg_cdte = 20.0
    bg_czt  = 40.0   # CZT has higher count rate (lower energy, more photons)

    # CdTe (high-energy, harder): sharper rise, faster decay
    cdte = np.zeros_like(t_all, dtype=float)
    imp_start = preflare_dur
    imp_peak  = preflare_dur + impulsive_dur * 0.5
    for i, t in enumerate(t_all):
        if t < imp_start:
            cdte[i] = bg_cdte
        else:
            tau_rise  = 40.0
            tau_decay = 120.0
            if t <= imp_peak:
                cdte[i] = bg_cdte + 300 * (1 - np.exp(-(t - imp_start) / tau_rise))
            else:
                cdte[i] = bg_cdte + 300 * np.exp(-(t - imp_peak) / tau_decay)

    # CZT (lower-energy, softer): slightly later peak, slower decay
    czt = np.zeros_like(t_all, dtype=float)
    for i, t in enumerate(t_all):
        if t < imp_start:
            czt[i] = bg_czt
        else:
            tau_rise  = 60.0
            tau_decay = 180.0
            if t <= imp_peak + 30:
                czt[i] = bg_czt + 600 * (1 - np.exp(-(t - imp_start) / tau_rise))
            else:
                czt[i] = bg_czt + 600 * np.exp(-(t - (imp_peak + 30)) / tau_decay)

    noise_cdte = rng.normal(0, 1.5, size=len(t_all))
    noise_czt  = rng.normal(0, 2.0, size=len(t_all))

    cdte_noisy = np.clip(cdte + noise_cdte, 0, None)
    czt_noisy  = np.clip(czt  + noise_czt,  0, None)

    df = pd.DataFrame({"time": time, "cdte_CR": cdte_noisy, "czt_CR": czt_noisy})
    df.attrs["preflare_end_s"]  = float(preflare_dur)
    df.attrs["impulsive_end_s"] = float(preflare_dur + impulsive_dur)
    df.attrs["bg_cdte"]         = bg_cdte
    df.attrs["bg_czt"]          = bg_czt
    return df


def make_empty_helios_df() -> pd.DataFrame:
    return pd.DataFrame({
        "time":    pd.Series([], dtype="datetime64[ns]"),
        "cdte_CR": pd.Series([], dtype=float),
        "czt_CR":  pd.Series([], dtype=float),
    })


def make_constant_helios_df(
    n: int = 10,
    cdte_val: float = 20.0,
    czt_val:  float = 40.0,
    cadence_s: int = 1,
) -> pd.DataFrame:
    t = pd.date_range("2026-01-01", periods=n, freq=f"{cadence_s}s")
    return pd.DataFrame({"time": t, "cdte_CR": [cdte_val] * n, "czt_CR": [czt_val] * n})