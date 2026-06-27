"""
Shared synthetic test fixtures.

make_synthetic_flare() builds a two-channel (soft + hard) light curve
whose phase timing loosely follows Benz (2008) Fig. 2:
preflare ~5 min, impulsive ~6 min, flash ~10 min (to peak), decay ~1 hr.

The hard channel is a fast-rise/fast-decay pulse (HXR-like). The soft
channel is built as (an approximation of) the time integral of the
hard channel's above-background flux, by construction satisfying the
Neupert effect (Sec. 2.4), plus an exponential decay phase after the
hard channel has died down, plus independent Gaussian noise on both
channels.

This is deliberately NOT a high-fidelity flare simulator -- it exists
only to give the feature modules a light curve with well-defined,
known phase boundaries and a known underlying power-law-free spectral
relationship, so that feature outputs can be checked against known
ground truth rather than only checked for "did not crash".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_flare(cadence_s: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    preflare_dur = 5 * 60
    impulsive_dur = 6 * 60
    flash_dur = 10 * 60
    decay_dur = 60 * 60

    t0 = pd.Timestamp("2002-07-23 00:18:00")

    def seg_times(start_s, dur_s):
        n = int(dur_s // cadence_s)
        return start_s + np.arange(n) * cadence_s

    t_pre = seg_times(0, preflare_dur)
    t_imp = seg_times(preflare_dur, impulsive_dur)
    t_flash = seg_times(preflare_dur + impulsive_dur, flash_dur)
    t_decay = seg_times(preflare_dur + impulsive_dur + flash_dur, decay_dur)

    t_all = np.concatenate([t_pre, t_imp, t_flash, t_decay])
    time = t0 + pd.to_timedelta(t_all, unit="s")

    bg = 8.0

    # Hard channel (HXR-like): fast rise, fast decay pulse during impulsive phase
    hard = np.zeros_like(t_all, dtype=float)
    imp_start = preflare_dur
    imp_peak = preflare_dur + impulsive_dur * 0.6
    for i, t in enumerate(t_all):
        if t < imp_start:
            hard[i] = bg * 0.3
        else:
            tau_rise = 90.0
            tau_decay = 150.0
            if t <= imp_peak:
                hard[i] = bg * 0.3 + 400 * (1 - np.exp(-(t - imp_start) / tau_rise))
            else:
                hard[i] = bg * 0.3 + 400 * np.exp(-(t - imp_peak) / tau_decay)

    # Soft channel: cumulative integral of (hard - background), i.e. built
    # to satisfy the Neupert effect by construction, then an explicit
    # exponential decay phase once the hard-channel pulse has died down.
    dt = cadence_s
    hard_bg = bg * 0.3
    cumulative = np.cumsum(np.clip(hard - hard_bg, 0, None)) * dt
    soft = bg + 0.015 * cumulative

    # Locate the point where the hard channel has decayed back near its
    # own background (this is NOT the same as argmax(cumulative), which
    # is trivially the last sample since cumulative is monotonic).
    above_hard_bg = hard > (hard_bg + 0.05 * (hard.max() - hard_bg))
    if above_hard_bg.any():
        flash_peak_idx = int(np.nonzero(above_hard_bg)[0][-1])
    else:
        flash_peak_idx = len(soft) // 2

    decay_tau = 700.0
    peak_above_bg = soft[flash_peak_idx] - bg
    for i in range(flash_peak_idx, len(soft)):
        t_since_peak = t_all[i] - t_all[flash_peak_idx]
        soft[i] = bg + peak_above_bg * np.exp(-t_since_peak / decay_tau)

    noise_soft = rng.normal(0, 0.5, size=len(t_all))
    noise_hard = rng.normal(0, 0.3, size=len(t_all))
    soft_noisy = np.clip(soft + noise_soft, 0, None)
    hard_noisy = np.clip(hard + noise_hard, 0, None)

    df = pd.DataFrame({"time": time, "CR": soft_noisy, "CR_hard": hard_noisy})

    # Stash known ground-truth boundaries (seconds since start) as
    # DataFrame attrs, so tests can reference them without recomputing.
    df.attrs["preflare_end_s"] = float(preflare_dur)
    df.attrs["impulsive_end_s"] = float(preflare_dur + impulsive_dur)
    df.attrs["peak_idx"] = flash_peak_idx
    df.attrs["peak_time_s"] = float(t_all[flash_peak_idx])
    return df


def make_empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.Series([], dtype="datetime64[ns]"),
            "CR": pd.Series([], dtype=float),
            "CR_hard": pd.Series([], dtype=float),
        }
    )


def make_constant_df(n: int = 10, value: float = 10.0, cadence_s: int = 1) -> pd.DataFrame:
    t = pd.date_range("2020-01-01", periods=n, freq=f"{cadence_s}s")
    return pd.DataFrame({"time": t, "CR": [value] * n, "CR_hard": [value / 5] * n})


def make_ramp_df(n: int = 20, cadence_s: int = 1) -> pd.DataFrame:
    t = pd.date_range("2020-01-01", periods=n, freq=f"{cadence_s}s")
    return pd.DataFrame({"time": t, "CR": np.arange(n, dtype=float)})
