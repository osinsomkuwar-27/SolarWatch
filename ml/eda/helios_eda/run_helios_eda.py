"""
ml/eda/helios_eda/run_helios_eda.py
=====================================
Command-line entry point for the HEL1OS EDA pipeline.

What it does (in order)
------------------------
1. Loads config from config/pipeline.yaml.
2. Loads HEL1OS light curve for one detector (CdTe1/CdTe2/CZT1/CZT2).
3. Optionally loads all four detectors for the comparison plots.
4. Detects flares with the shared FlareDetector.
5. Produces all HEL1OSPlotter plots.
6. Computes and saves the HEL1OS EDA statistical report.
7. Saves flare catalogue as CSV.
8. Prints a console summary.

All output goes to  ml/data/cache/<date>/helios/  to keep it separate
from the SoLEXS EDA cache.

Usage
-----
Run from the  solar/  directory:

    # Single detector
    python -m ml.eda.helios_eda.run_helios_eda --date 20260621

    # All four detectors (enables comparison plots)
    python -m ml.eda.helios_eda.run_helios_eda --date 20260621 --all-detectors

    # Specific detector + custom config
    python -m ml.eda.helios_eda.run_helios_eda --date 20260621 \\
        --detector CZT1 --config config/pipeline.yaml

Outputs (all in ml/data/cache/20260621/helios/)
-------------------------------------------------
    helios_CdTe1_20260621_lc.png
    helios_multiband_CdTe1_20260621.png
    helios_hist_cr_CdTe1_20260621.png
    helios_flare_class_dist_CdTe1_20260621.png
    helios_flare_timing_CdTe1_20260621.png
    helios_acf_CdTe1_20260621.png
    helios_timing_CdTe1_20260621.png
    helios_eda_report_CdTe1_20260621.json
    helios_flares_CdTe1_20260621.csv
    helios_detector_comparison_20260621.png  (if --all-detectors)
    helios_detector_overlay_20260621.png     (if --all-detectors)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ALL_DETECTORS = ["CdTe1", "CdTe2", "CZT1", "CZT2"]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aditya-L1 HEL1OS EDA pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--date", required=True,
        help="Observation date in YYYYMMDD format, e.g. 20260621",
    )
    p.add_argument(
        "--detector", default="CdTe1",
        choices=_ALL_DETECTORS,
        help="Primary HEL1OS detector for the single-detector plots.",
    )
    p.add_argument(
        "--all-detectors", action="store_true",
        help="Load all four detectors and produce comparison plots.",
    )
    p.add_argument(
        "--config", default=None,
        help="Path to pipeline.yaml (default: config/pipeline.yaml relative to solar/).",
    )
    p.add_argument(
        "--show", action="store_true",
        help="Display plots interactively (blocks until window closed).",
    )
    p.add_argument(
        "--onset-sigma", type=float, default=3.0,
        help="Flare onset detection threshold in σ units.",
    )
    return p


def run_helios_eda(args: argparse.Namespace) -> None:
    """Execute the full HEL1OS EDA pipeline."""

    # ── 1. Load configuration ─────────────────────────────────────────────────
    from ml.utils.config import load_config
    cfg = load_config(args.config)

    # ── 2. Per-day HEL1OS cache directory ────────────────────────────────────
    cache_dir = cfg.paths.cache / args.date / "helios"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("HEL1OS EDA cache dir: %s", cache_dir)

    # ── 3. Determine which detectors to load ─────────────────────────────────
    detectors_to_load = _ALL_DETECTORS if args.all_detectors else [args.detector]

    # ── 4. Load HEL1OS data for each detector ────────────────────────────────
    from ml.loaders.helios_loader import HEL1OSLoader

    helios_dir = cfg.paths.raw_helios
    h_loader   = HEL1OSLoader(data_dir=helios_dir)

    loaded_lcs: dict = {}
    for det in detectors_to_load:
        lc_fname = f"lightcurve_{det.lower()}.fits"
        lc_path  = helios_dir / lc_fname
        if not lc_path.exists():
            logger.warning("HEL1OS LC not found for %s at %s — skipping.", det, lc_path)
            continue
        loaded_lcs[det] = h_loader.load_lc(lc_path, detector=det, date_str=args.date)
        logger.info("Loaded HEL1OS %s — %d bands", det, len(loaded_lcs[det].bands))

    if not loaded_lcs:
        logger.error(
            "No HEL1OS LC files found in %s for any of %s.\n"
            "Place downloaded files in: %s",
            helios_dir, detectors_to_load, helios_dir,
        )
        sys.exit(1)

    # ── 5. Detect flares + produce plots per detector ─────────────────────────
    from ml.eda.flare_detector import FlareDetector
    from ml.eda.helios_eda.helios_plotter import HEL1OSPlotter
    from ml.eda.helios_eda.helios_statistics import HEL1OSEDAStatistics

    fd = FlareDetector(
        onset_sigma      = args.onset_sigma,
        bc_percentile    = cfg.labelling.bc_percentile,
        m_percentile     = cfg.labelling.m_percentile,
        x_percentile     = cfg.labelling.m_percentile + 5.0,
    )
    plotter = HEL1OSPlotter(output_dir=cache_dir, show=args.show)
    stats   = HEL1OSEDAStatistics(
        output_dir    = cache_dir,
        bc_percentile = cfg.labelling.bc_percentile,
        m_percentile  = cfg.labelling.m_percentile,
    )

    reports     = []
    all_flares  = {}

    for det, lc in loaded_lcs.items():
        full    = lc.full_band
        flares  = fd.detect(full.time_unix, full.count_rate, det, args.date)
        all_flares[det] = flares

        # Plots
        plotter.plot_helios_day(lc, flare_times_unix=[f.onset_unix for f in flares])
        plotter.plot_multi_band(lc)

        # Statistics
        report = stats.compute(
            time_unix  = full.time_unix,
            count_rate = full.count_rate,
            flares     = flares,
            detector   = det,
            date_str   = args.date,
        )
        stats.save(report, tag=args.date)
        stats.plot_distributions(
            full.time_unix, full.count_rate, flares,
            tag=args.date, detector=det,
        )
        reports.append(report)

        # Flare catalogue CSV
        from ml.eda.flare_detector import FlareDetector as FD
        flare_df = FD.summary(flares)
        if not flare_df.empty:
            csv_path = cache_dir / f"helios_flares_{det}_{args.date}.csv"
            flare_df.to_csv(csv_path, index=False)
            logger.info("Flare catalogue saved: %s", csv_path)

    # ── 6. Cross-detector plots (if multiple detectors loaded) ────────────────
    if len(loaded_lcs) > 1:
        plotter.plot_detector_overlay(list(loaded_lcs.values()), date_str=args.date)
        stats.plot_detector_comparison(reports, tag=args.date)

        # Hardness ratio: CdTe1 / CZT1 (if both available)
        if "CdTe1" in loaded_lcs and "CZT1" in loaded_lcs:
            plotter.plot_hardness_ratio(
                czt_lc  = loaded_lcs["CZT1"],
                cdte_lc = loaded_lcs["CdTe1"],
                date_str = args.date,
            )

    # ── 7. Console summary ────────────────────────────────────────────────────
    for report in reports:
        _print_summary(report, all_flares.get(report.detector, []), cache_dir)


def _print_summary(report, flares, cache_dir: Path) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  HEL1OS EDA Summary  |  {report.detector}  |  {report.date_str}")
    print(sep)
    print(f"  Samples          : {report.n_samples:,}")
    print(f"  Cadence          : {report.median_cadence_sec:.1f} s")
    print(f"  Count Rate Mean  : {report.mean_cr:.2f} cts/s")
    print(f"  Count Rate Std   : {report.std_cr:.2f} cts/s")
    print(f"  Count Rate Range : {report.min_cr:.2f} – {report.max_cr:.2f} cts/s")
    print(f"  P50 / P95 / P99  : {report.p50_cr:.1f} / {report.p95_cr:.1f} / {report.p99_cr:.1f}")
    print(sep)
    print(f"  Detected Flares  : {report.n_flares}")
    print(f"    Quiet          : {report.n_quiet}")
    print(f"    B/C            : {report.n_bc}")
    print(f"    M              : {report.n_m}")
    print(f"    X              : {report.n_x}")
    print(f"  Class Weights    : {report.class_weights}")
    print(sep)
    print(f"  Thresholds       : B/C≥{report.thresh_bc:.1f}  M≥{report.thresh_m:.1f}  X≥{report.thresh_x:.1f}")
    if flares:
        print(f"  Mean Rise Time   : {report.mean_rise_sec/60:.1f} min")
        print(f"  Mean Decay Time  : {report.mean_decay_sec/60:.1f} min")
        print(f"  Mean Duration    : {report.mean_duration_sec/60:.1f} min")
    print(sep)
    print(f"  Output directory : {cache_dir}")
    print(sep)
    print()

    if flares:
        print("  Detected Flare Catalogue:")
        print(f"  {'Onset (UTC)':<22} {'Class':<8} {'Peak CR':>10}  "
              f"{'Rise (min)':>10}  {'Decay (min)':>12}")
        print(f"  {'─'*22} {'─'*8} {'─'*10}  {'─'*10}  {'─'*12}")
        for f in flares:
            print(
                f"  {f.onset_utc:<22} {f.class_name:<8} "
                f"{f.peak_count_rate:>10.1f}  "
                f"{f.rise_time_sec/60:>10.1f}  "
                f"{f.decay_time_sec/60:>12.1f}"
            )
        print()


def main() -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args()
    run_helios_eda(args)


if __name__ == "__main__":
    main()