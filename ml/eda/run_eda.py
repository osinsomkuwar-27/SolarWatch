"""
ml/eda/run_eda.py
==================
Command-line entry point for the full EDA pipeline.

What it does (in order)
------------------------
1. Loads config from config/pipeline.yaml.
2. Loads SoLEXS light curve + GTI for the given date and detector.
3. Optionally loads HEL1OS light curve if files are present.
4. Detects flares with FlareDetector.
5. Produces all plots via LightCurvePlotter.
6. Computes and saves the EDA statistical report.
7. Prints a console summary.

All output goes to  ml/data/cache/<date>/  to keep days separate.

Usage
-----
Run from the  solar/  directory:

    # Single day, SoLEXS only
    python -m ml.eda.run_eda --date 20260621

    # SoLEXS + HEL1OS dual-band
    python -m ml.eda.run_eda --date 20260621 --with-helios

    # Specify detector and config path
    python -m ml.eda.run_eda --date 20260621 --detector SDD2 \\
        --config config/pipeline.yaml

    # Show plots interactively (disable in CI/batch)
    python -m ml.eda.run_eda --date 20260621 --show

Outputs (all in ml/data/cache/20260621/)
-----------------------------------------
    solexs_SDD2_20260621_lc.png
    helios_CdTe1_20260621_bands.png      (if --with-helios)
    dual_panel_20260621_SDD2_CdTe1.png   (if --with-helios)
    neupert_20260621.png                 (if --with-helios)
    hist_cr_SDD2_20260621.png
    flare_class_dist_SDD2_20260621.png
    flare_timing_SDD2_20260621.png
    acf_SDD2_20260621.png
    eda_report_SDD2_20260621.json
    flares_SDD2_20260621.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aditya-L1 EDA pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--date", required=True,
        help="Observation date in YYYYMMDD format, e.g. 20260621",
    )
    p.add_argument(
        "--detector", default="SDD2",
        choices=["SDD1", "SDD2"],
        help="SoLEXS detector (SDD2 recommended — SDD1 saturates at high flux).",
    )
    p.add_argument(
        "--helios-detector", default="CdTe1",
        choices=["CdTe1", "CdTe2", "CZT1", "CZT2"],
        help="HEL1OS detector for dual-panel and Neupert plots.",
    )
    p.add_argument(
        "--with-helios", action="store_true",
        help="Include HEL1OS hard X-ray data in EDA (requires files in ml/data/raw/helios/).",
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


def run_eda(args: argparse.Namespace) -> None:
    """Execute the full EDA pipeline."""

    # ── 1. Load configuration ─────────────────────────────────────────────────
    from ml.utils.config import load_config
    cfg = load_config(args.config)

    # ── 2. Per-day cache directory ────────────────────────────────────────────
    cache_dir = cfg.paths.cache / args.date
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("EDA cache dir: %s", cache_dir)

    # ── 3. Load SoLEXS data ───────────────────────────────────────────────────
    from ml.loaders.solexs_loader import SoLEXSLoader, LightCurve, GoodTimeIntervals

    solexs_dir = cfg.paths.raw_solexs
    loader     = SoLEXSLoader(data_dir=solexs_dir)

    # Build expected filenames
    n       = args.detector[-1]   # '1' or '2'
    base    = f"AL1_SOLEXS_{args.date}_SDD{n}_L1"
    lc_path = solexs_dir / f"{base}.lc"
    lc_gz   = solexs_dir / f"{base}.lc.gz"
    gti_path = solexs_dir / f"{base}.gti"
    gti_gz   = solexs_dir / f"{base}.gti.gz"

    lc_file  = lc_path  if lc_path.exists()  else lc_gz
    gti_file = gti_path if gti_path.exists() else gti_gz

    if not lc_file.exists():
        logger.error(
            "SoLEXS LC file not found.\n"
            "  Expected: %s\n"
            "  Also tried: %s\n"
            "  Place the downloaded file in: %s",
            lc_path, lc_gz, solexs_dir,
        )
        sys.exit(1)

    lc: LightCurve = loader.load_lc(lc_file, detector=args.detector, date_str=args.date)

    gti_mask = None
    if gti_file.exists():
        gti: GoodTimeIntervals = loader.load_gti(gti_file)
        gti_mask = gti.mask_for(lc.time_unix)
        logger.info(
            "GTI loaded: %d intervals, valid fraction = %.1f%%",
            gti.n_intervals,
            100 * gti_mask.mean(),
        )
    else:
        logger.warning("GTI file not found at %s — proceeding without GTI mask.", gti_file)

    # ── 4. Detect flares ──────────────────────────────────────────────────────
    from ml.eda.flare_detector import FlareDetector

    detector = FlareDetector(
        onset_sigma     = args.onset_sigma,
        bc_percentile   = cfg.labelling.bc_percentile,
        m_percentile    = cfg.labelling.m_percentile,
        x_percentile    = cfg.labelling.m_percentile + 5.0,
    )

    # Detect on GTI-valid data for cleaner results
    if gti_mask is not None:
        t_valid  = lc.time_unix[gti_mask]
        cr_valid = lc.count_rate[gti_mask]
    else:
        t_valid  = lc.time_unix
        cr_valid = lc.count_rate

    flares = detector.detect(t_valid, cr_valid, args.detector, args.date)

    # ── 5. Plots — SoLEXS ────────────────────────────────────────────────────
    from ml.eda.light_curve_plotter import LightCurvePlotter

    plotter = LightCurvePlotter(output_dir=cache_dir, show=args.show)
    plotter.plot_solexs_day(
        lc,
        gti_mask         = gti_mask,
        flare_times_unix = [f.onset_unix for f in flares],
    )

    # ── 6. HEL1OS (optional) ──────────────────────────────────────────────────
    helios_lc = None
    if args.with_helios:
        from ml.loaders.helios_loader import HEL1OSLoader

        helios_dir = cfg.paths.raw_helios
        lc_fname   = f"lightcurve_{args.helios_detector.lower()}.fits"
        helios_lc_path = helios_dir / lc_fname

        if not helios_lc_path.exists():
            logger.warning(
                "HEL1OS LC file not found at %s. "
                "Skipping dual-panel and Neupert plots.",
                helios_lc_path,
            )
        else:
            h_loader = HEL1OSLoader(data_dir=helios_dir)
            helios_lc = h_loader.load_lc(
                helios_lc_path,
                detector = args.helios_detector,
                date_str = args.date,
            )
            plotter.plot_helios_bands(helios_lc)
            plotter.plot_dual_panel(lc, helios_lc)
            plotter.plot_neupert(lc, helios_lc)

    # ── 7. EDA statistics ─────────────────────────────────────────────────────
    from ml.eda.statistics import EDAStatistics

    stats  = EDAStatistics(
        output_dir    = cache_dir,
        bc_percentile = cfg.labelling.bc_percentile,
        m_percentile  = cfg.labelling.m_percentile,
    )
    report = stats.compute(
        time_unix  = t_valid,
        count_rate = cr_valid,
        flares     = flares,
        detector   = args.detector,
        date_str   = args.date,
    )
    stats.save(report, tag=args.date)
    stats.plot_distributions(t_valid, cr_valid, flares, tag=args.date, detector=args.detector)

    # ── 8. Save flare catalogue as CSV ────────────────────────────────────────
    flare_df = FlareDetector.summary(flares)
    if not flare_df.empty:
        csv_path = cache_dir / f"flares_{args.detector}_{args.date}.csv"
        flare_df.to_csv(csv_path, index=False)
        logger.info("Flare catalogue saved: %s", csv_path)

    # ── 9. Console summary ────────────────────────────────────────────────────
    _print_summary(report, flares, cache_dir)


def _print_summary(report, flares, cache_dir: Path) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Aditya-L1 EDA Summary  |  {report.detector}  |  {report.date_str}")
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
    run_eda(args)


if __name__ == "__main__":
    main()
