# ml/eda/run_eda.py
"""
ml/eda/run_eda.py
==================
Unified command-line entry point for the full Aditya-L1 EDA pipeline.

Performs in order:
  1. Load SoLEXS days via MultiDayLoader.
  2. SoLEXS EDA — flare detection, plots, statistics.
  3. Load HEL1OS — all four detectors via MultiDayLoader.
  4. HEL1OS EDA — per-detector flare detection, plots, statistics.
  5. Cross-instrument plots on overlapping dates (always produced).
  6. Aggregate report outputs.
  7. Combined pipeline summary.

Usage
-----
    # SoLEXS + HEL1OS, single day
    python -m ml.eda.run_eda --date 20260621

    # Full pipeline, all days
    python -m ml.eda.run_eda --all-days

    # Specific HEL1OS detector
    python -m ml.eda.run_eda --all-days --helios-detector CZT1

    # All HEL1OS detectors (default)
    python -m ml.eda.run_eda --all-days --helios-detector ALL

    # --with-helios is now a no-op kept for backwards compatibility
    python -m ml.eda.run_eda --all-days --with-helios
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_HELIOS_ALL_DETECTORS = ["CZT1", "CZT2", "CdTe1", "CdTe2"]


# ══════════════════════════════════════════════════════════════════════════════
# Logging filter — suppress expected, repetitive GTI-extension warnings
# ══════════════════════════════════════════════════════════════════════════════

class _SuppressExpectedWarnings(logging.Filter):
    """
    Drop log records that are expected behaviour and would otherwise flood
    the console on every HEL1OS file load.

    Only records matching ALL of:
      • level  == WARNING  (never drops ERROR or above)
      • logger == 'ml.loaders.helios_loader'
      • message contains the known GTI-extension phrase

    are filtered out.  Everything else passes through unchanged.
    """
    _KNOWN_PHRASES = (
        "Extensions ['GTI', 'STDGTI'] not found; using ext 1",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.WARNING:
            return True                          # always keep ERROR+
        if record.name != "ml.loaders.helios_loader":
            return True                          # keep other loggers
        msg = record.getMessage()
        return not any(phrase in msg for phrase in self._KNOWN_PHRASES)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aditya-L1 unified EDA pipeline (SoLEXS + HEL1OS)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--date",
        help="Single observation date YYYYMMDD, e.g. 20260621",
    )
    group.add_argument(
        "--all-days",
        action="store_true",
        help="Load and analyse the complete dataset via MultiDayLoader.",
    )
    p.add_argument(
        "--detector", default="SDD2",
        choices=["SDD1", "SDD2"],
        help="SoLEXS detector.",
    )
    p.add_argument(
        "--helios-detector", default="ALL",
        choices=_HELIOS_ALL_DETECTORS + ["ALL"],
        help="HEL1OS detector(s). ALL runs every detector.",
    )
    p.add_argument(
        "--with-helios", action="store_true",
        help="[Deprecated – HEL1OS now always runs.  Kept for backwards compatibility.]",
    )
    p.add_argument("--config",      default=None)
    p.add_argument("--show",        action="store_true")
    p.add_argument("--onset-sigma", type=float, default=3.0)
    p.add_argument("--debug",       action="store_true")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — SoLEXS
# ══════════════════════════════════════════════════════════════════════════════

def run_solexs(days, detector, fd, solexs_stats, solexs_plotter, cache_root):
    """
    Run flare detection, plotting, and statistics for every SoLEXS day.

    Returns:
        reports      (list)  — one EDAReport per successfully processed day
        all_flares   (dict)  — date_str → list[Flare]
        missing_days (list)  — date strings skipped due to empty GTI
    """
    from ml.eda.flare_detector import FlareDetector

    reports, all_flares, missing_days = [], {}, []

    for day in days:
        lc_masked = day.lc_gti_masked
        t_valid   = lc_masked.time_unix
        cr_valid  = lc_masked.count_rate

        if len(t_valid) == 0:
            logger.warning("Day %s: zero GTI-valid samples — skipping.", day.date_str)
            missing_days.append(day.date_str)
            continue

        flares = fd.detect(t_valid, cr_valid, detector, day.date_str)

        solexs_plotter.plot_solexs_day(
            day.lc,
            gti_mask         = day.gti.mask_for(day.lc.time_unix),
            flare_times_unix = [f.onset_unix for f in flares],
        )
        solexs_plotter.plot_cr_histogram(day.lc, day.date_str, detector)
        solexs_plotter.plot_gti_statistics(day, detector)

        report = solexs_stats.compute(
            time_unix  = t_valid,
            count_rate = cr_valid,
            flares     = flares,
            detector   = detector,
            date_str   = day.date_str,
            gti        = day.gti,
        )
        solexs_stats.save(report, tag=day.date_str)
        solexs_stats.plot_distributions(t_valid, cr_valid, flares,
                                        tag=day.date_str, detector=detector)

        flare_df = FlareDetector.summary(flares)
        if not flare_df.empty:
            cache_dir = cache_root / day.date_str
            cache_dir.mkdir(parents=True, exist_ok=True)
            flare_df.to_csv(
                cache_dir / f"flares_{detector}_{day.date_str}.csv", index=False
            )

        reports.append(report)
        all_flares[day.date_str] = flares

    return reports, all_flares, missing_days


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — HEL1OS
# ══════════════════════════════════════════════════════════════════════════════

def run_helios(loader, detectors_to_run, date_filter,
               fd, helios_stats, helios_plotter, cache_root):
    """
    Load and process all requested HEL1OS detectors.

    Returns:
        helios_curves  (dict)  — det → list[HEL1OSLightCurve]
        helios_reports (dict)  — det → list[HEL1OSEDAReport]
    """
    from ml.eda.flare_detector import FlareDetector

    helios_curves  = {}
    helios_reports = {}

    for det in detectors_to_run:
        print(f"    {det}...", end="  ", flush=True)
        curves = loader.load_all_helios(detector=det)
        if date_filter:
            curves = [lc for lc in curves if lc.date_str == date_filter]

        if not curves:
            print("✗  no data")
            helios_curves[det]  = []
            helios_reports[det] = []
            continue

        print(f"✓  {len(curves)} segment(s)")
        helios_curves[det]  = curves
        helios_reports[det] = []

        for lc in curves:
            full   = lc.full_band
            flares = fd.detect(full.time_unix, full.count_rate, det, lc.date_str)

            helios_plotter.plot_helios_day(lc, flare_times_unix=[f.onset_unix for f in flares])
            helios_plotter.plot_multi_band(lc)

            report = helios_stats.compute(
                time_unix  = full.time_unix,
                count_rate = full.count_rate,
                flares     = flares,
                detector   = det,
                date_str   = lc.date_str,
                lc         = lc,
            )
            helios_stats.save(report, tag=lc.date_str)
            helios_stats.plot_distributions(full.time_unix, full.count_rate, flares,
                                            tag=lc.date_str, detector=det)

            flare_df = FlareDetector.summary(flares)
            if not flare_df.empty:
                h_cache = cache_root / lc.date_str / "helios"
                h_cache.mkdir(parents=True, exist_ok=True)
                flare_df.to_csv(
                    h_cache / f"helios_flares_{det}_{lc.date_str}.csv", index=False
                )

            helios_reports[det].append(report)

        _print_helios_summary(helios_reports[det], det)

    # ── Cross-detector plots (within HEL1OS) ─────────────────────────────────
    loaded_dets = {d: helios_curves[d] for d in detectors_to_run if helios_curves.get(d)}
    if len(loaded_dets) > 1:
        date_set = sorted({lc.date_str for curves in loaded_dets.values() for lc in curves})
        for date_str in date_set:
            day_lcs = {}
            for det, curves in loaded_dets.items():
                match = [lc for lc in curves if lc.date_str == date_str]
                if match:
                    day_lcs[det] = max(match, key=lambda x: len(x.full_band.time_unix))

            if len(day_lcs) <= 1:
                continue

            helios_plotter.plot_detector_overlay(list(day_lcs.values()), date_str=date_str)

            det_reports = [
                r for det in day_lcs
                for r in helios_reports.get(det, [])
                if r.date_str == date_str
            ]
            if det_reports:
                helios_stats.plot_detector_comparison(det_reports, tag=date_str)

            if "CdTe1" in day_lcs and "CZT1" in day_lcs:
                helios_plotter.plot_hardness_ratio(
                    czt_lc=day_lcs["CZT1"], cdte_lc=day_lcs["CdTe1"], date_str=date_str
                )

    return helios_curves, helios_reports


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Cross-instrument (SoLEXS ↔ HEL1OS)
# ══════════════════════════════════════════════════════════════════════════════

def run_cross_instrument(solexs_reports, helios_curves, days,
                         solexs_plotter):
    """
    Generate dual-panel and Neupert plots for dates present in both instruments.

    Returns the sorted list of overlapping date strings.
    """
    loaded_dets = {det: curves for det, curves in helios_curves.items() if curves}
    if not loaded_dets:
        return []

    solexs_dates = {r.date_str for r in solexs_reports}
    helios_dates = {lc.date_str for curves in loaded_dets.values() for lc in curves}
    overlap_dates = sorted(solexs_dates & helios_dates)

    if not overlap_dates:
        return []

    _subsection(f"Cross-Instrument Overlap  ({len(overlap_dates)} date(s))")

    solexs_day_map = {d.date_str: d for d in days}
    helios_lc_map: dict[str, dict] = {}
    for det, curves in loaded_dets.items():
        for lc in curves:
            helios_lc_map.setdefault(lc.date_str, {})[det] = lc

    primary_det = "CZT1" if "CZT1" in loaded_dets else next(iter(loaded_dets))

    for date_str in overlap_dates:
        sol_day = solexs_day_map.get(date_str)
        h_lcs   = helios_lc_map.get(date_str, {})
        if sol_day is None or not h_lcs:
            continue
        h_lc = h_lcs.get(primary_det) or next(iter(h_lcs.values()))
        solexs_plotter.plot_dual_panel(sol_day.lc, h_lc)
        solexs_plotter.plot_neupert(sol_day.lc, h_lc)

    _row("Overlapping dates",   str(len(overlap_dates)))
    _row("HEL1OS ref detector", primary_det)
    _row("Plots produced",      "dual_panel, neupert (per overlapping date)")
    print()

    return overlap_dates


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Aggregate outputs
# ══════════════════════════════════════════════════════════════════════════════

def save_reports(solexs_reports, all_flares, missing_days, all_days, detector,
                 helios_curves, helios_reports,
                 solexs_stats, helios_stats, solexs_plotter,
                 reports_dir):
    """Write all multi-day CSV, JSON, and ranking outputs."""
    # SoLEXS aggregates (only meaningful with more than one day)
    if len(solexs_reports) > 1:
        solexs_stats.save_daily_csv(solexs_reports, reports_dir / "daily_statistics.csv")
        solexs_stats.save_flare_candidates_csv(all_flares, reports_dir / "flare_candidates.csv")
        solexs_stats.save_aggregate_json(solexs_reports, missing_days,
                                         reports_dir / "statistics.json")
        solexs_plotter.plot_flare_day_ranking(solexs_reports, detector)
        solexs_plotter.plot_observation_coverage(all_days, detector)

    # HEL1OS aggregates
    all_helios_reports = [r for reps in helios_reports.values() for r in reps]
    if all_helios_reports:
        helios_stats.save_detector_csv(
            all_helios_reports, reports_dir / "helios_daily_statistics.csv"
        )
        loaded_dets = {d: helios_curves[d] for d in helios_curves if helios_curves[d]}
        obs_lcs = {
            f"{det}_{lc.date_str}": lc
            for det, curves in loaded_dets.items()
            for lc in curves
        }
        helios_stats.save_observation_duration_csv(
            obs_lcs, reports_dir / "observation_duration_statistics.csv"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(args: argparse.Namespace) -> None:
    from ml.utils.config import load_config
    from ml.loaders.multi_day_loader import MultiDayLoader
    from ml.eda.flare_detector import FlareDetector
    from ml.eda.light_curve_plotter import LightCurvePlotter
    from ml.eda.statistics import EDAStatistics
    from ml.eda.helios_eda.helios_plotter import HEL1OSPlotter
    from ml.eda.helios_eda.helios_statistics import HEL1OSEDAStatistics

    cfg = load_config(args.config)

    reports_dir = Path("reports/eda")
    plots_dir   = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Construct expensive objects once ─────────────────────────────────────
    loader = MultiDayLoader(
        solexs_raw_dir = cfg.paths.raw_solexs,
        helios_raw_dir = cfg.paths.raw_helios,
        extract_dir    = cfg.paths.extracted,
        debug          = getattr(args, "debug", False),
    )
    fd = FlareDetector(
        onset_sigma   = args.onset_sigma,
        bc_percentile = cfg.labelling.bc_percentile,
        m_percentile  = cfg.labelling.m_percentile,
        x_percentile  = cfg.labelling.m_percentile + 5.0,
    )
    solexs_stats   = EDAStatistics(
        output_dir    = reports_dir,
        bc_percentile = cfg.labelling.bc_percentile,
        m_percentile  = cfg.labelling.m_percentile,
    )
    helios_stats   = HEL1OSEDAStatistics(
        output_dir    = reports_dir,
        bc_percentile = cfg.labelling.bc_percentile,
        m_percentile  = cfg.labelling.m_percentile,
    )
    solexs_plotter = LightCurvePlotter(output_dir=plots_dir, show=args.show)
    helios_plotter = HEL1OSPlotter(output_dir=plots_dir, show=args.show)

    _section("ADITYA-L1 EDA PIPELINE")

    # ── Load SoLEXS ──────────────────────────────────────────────────────────
    print(f"  Loading SoLEXS [{args.detector}]...", end="  ", flush=True)
    all_days = loader.load_all_solexs(detector=args.detector, load_pi=False)
    if not all_days:
        print("✗  No days loaded — run extract_all() first.")
        sys.exit(1)

    days = (
        [d for d in all_days if d.date_str == args.date]
        if args.date else all_days
    )
    if not days:
        print(f"✗  Date {args.date} not found in extracted data.")
        sys.exit(1)
    print(f"✓  {len(days)} day(s) loaded")

    # ── Run SoLEXS EDA ───────────────────────────────────────────────────────
    solexs_reports, all_flares, missing_days = run_solexs(
        days, args.detector, fd, solexs_stats, solexs_plotter, cfg.paths.cache
    )
    _print_solexs_summary(solexs_reports, missing_days, all_flares, args.detector)

    # ── Load + Run HEL1OS EDA (always) ───────────────────────────────────────
    detectors_to_run = (
        _HELIOS_ALL_DETECTORS if args.helios_detector == "ALL"
        else [args.helios_detector]
    )
    print(f"  Loading HEL1OS [{', '.join(detectors_to_run)}]...", flush=True)
    helios_curves, helios_reports = run_helios(
        loader, detectors_to_run, args.date,
        fd, helios_stats, helios_plotter, cfg.paths.cache,
    )

    # ── Cross-instrument analysis (always) ───────────────────────────────────
    run_cross_instrument(solexs_reports, helios_curves, days, solexs_plotter)

    # ── Save aggregate outputs ────────────────────────────────────────────────
    save_reports(
        solexs_reports, all_flares, missing_days, all_days, args.detector,
        helios_curves, helios_reports,
        solexs_stats, helios_stats, solexs_plotter,
        reports_dir,
    )

    # ── Pipeline summary ──────────────────────────────────────────────────────
    _print_pipeline_summary(
        solexs_reports, missing_days, helios_curves, helios_reports, args.detector
    )


# ══════════════════════════════════════════════════════════════════════════════
# Console output helpers
# ══════════════════════════════════════════════════════════════════════════════

_W = 60


def _section(title: str) -> None:
    print()
    print("═" * _W)
    print(f"  {title}")
    print("═" * _W)


def _subsection(title: str) -> None:
    print()
    print("─" * _W)
    print(f"  {title}")
    print("─" * _W)


def _row(label: str, value: str, width: int = 28) -> None:
    print(f"  {label:<{width}} {value}")


def _print_solexs_summary(reports, missing_days, all_flares, detector):
    if not reports:
        return
    _subsection(f"SoLEXS Summary  [{detector}]")

    gti_counts = [r.n_gti_intervals for r in reports]
    max_gti    = max(gti_counts)
    max_date   = reports[gti_counts.index(max_gti)].date_str
    dates      = sorted(r.date_str for r in reports)
    top_flare  = sorted(reports, key=lambda r: r.n_flares, reverse=True)

    mean_cr_values = [r.mean_cr for r in reports]
    mean_cr_str = (
        f"{float(np.nanmean(mean_cr_values)):.2f} cts/s"
        if mean_cr_values else "—"
    )

    _row("Days loaded",           str(len(reports)))
    _row("Days missing",          str(len(missing_days)))
    _row("Total samples",         f"{sum(r.n_samples for r in reports):,}")
    _row("Total flares",          str(sum(r.n_flares for r in reports)))
    _row("Mean CR (all days)",    mean_cr_str)
    _row("Avg GTI intervals/day", f"{float(np.mean(gti_counts)):.1f}")
    _row("Max GTI intervals",     f"{max_gti}  ({max_date})")
    _row("Observation span",      f"{dates[0]} → {dates[-1]}" if dates else "—")
    _row("Status",
         "OK" if not missing_days else f"DEGRADED ({len(missing_days)} missing)")

    if top_flare:
        print(f"\n  Top 3 flare days:")
        for r in top_flare[:3]:
            print(f"    {r.date_str}  →  {r.n_flares} flare(s)")
    print()

    for r in reports:
        sep = "·" * _W
        print(f"  {sep}")
        print(f"  {r.detector}  {r.date_str}  |  "
              f"{r.n_samples:,} samples  |  "
              f"GTI {r.n_gti_intervals} int  |  "
              f"coverage {r.gti_coverage_pct:.1f}%  |  "
              f"{r.n_flares} flare(s)")
        for f in all_flares.get(r.date_str, []):
            print(f"    {f.onset_utc:<22} {f.class_name:<6} "
                  f"peak {f.peak_count_rate:.1f} cts/s  "
                  f"rise {f.rise_time_sec/60:.1f} min  "
                  f"decay {f.decay_time_sec/60:.1f} min")


def _print_helios_summary(reports, detector):
    if not reports:
        return
    _subsection(f"HEL1OS Summary  [{detector}]")
    dates = sorted(r.date_str for r in reports)

    mean_cr_values = [r.mean_cr for r in reports]
    mean_cr_str = (
        f"{float(np.nanmean(mean_cr_values)):.2f} cts/s"
        if mean_cr_values else "—"
    )

    _row("Segments loaded",  str(len(reports)))
    _row("Total samples",    f"{sum(r.n_samples for r in reports):,}")
    _row("Total flares",     str(sum(r.n_flares for r in reports)))
    _row("Mean CR",          mean_cr_str)
    _row("Energy band",      reports[0].energy_band_kev)
    _row("Observation span", f"{dates[0]} → {dates[-1]}" if dates else "—")
    _row("Status",           "OK")
    print()


def _print_pipeline_summary(solexs_reports, missing_days,
                             helios_curves, helios_reports, solexs_detector):
    _section("OVERALL PIPELINE SUMMARY")

    total_solexs = len(solexs_reports)
    total_helios = sum(len(v) for v in helios_curves.values())
    total_failed = len(missing_days) + sum(
        1 for v in helios_curves.values() if not v
    )

    s_dates = sorted(r.date_str for r in solexs_reports)
    s_span  = f"{s_dates[0]} → {s_dates[-1]}" if s_dates else "—"
    _row(f"SoLEXS [{solexs_detector}]",
         f"{total_solexs:>4} days   "
         f"{sum(r.n_samples for r in solexs_reports):>12,} samples   {s_span}")

    for det, reps in helios_reports.items():
        h_dates = sorted(r.date_str for r in reps)
        h_span  = f"{h_dates[0]} → {h_dates[-1]}" if h_dates else "no data"
        _row(f"HEL1OS [{det}]",
             f"{len(reps):>4} segs   "
             f"{sum(r.n_samples for r in reps):>12,} samples   {h_span}")

    print()
    print("  " + "─" * (_W - 2))
    _row("Total files loaded", str(total_solexs + total_helios))
    _row("Total failures",     str(total_failed))
    _row("Pipeline status",
         "✓  OK" if total_failed == 0 else f"✗  DEGRADED ({total_failed} failure(s))")
    print("═" * _W)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args()
    logging.basicConfig(
        level  = logging.WARNING,
        format = "%(levelname)s %(name)s: %(message)s",
        stream = sys.stdout,
    )
    # Attach the targeted filter to the root handler so it covers all handlers
    # configured by basicConfig.  This never touches ERROR-level records or
    # records from any logger other than ml.loaders.helios_loader.
    _filter = _SuppressExpectedWarnings()
    for handler in logging.root.handlers:
        handler.addFilter(_filter)

    run_eda(args)


if __name__ == "__main__":
    main()