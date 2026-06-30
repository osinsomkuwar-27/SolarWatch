"""
Stage 6 — Model Evaluation (temporal holdout + cross-block validation)
=========================================================================
Trains a Random Forest on the labeled+featured master dataset and
evaluates it on a held-out time period, using a TEMPORAL split (never
random shuffling — that leaks adjacent seconds of the same event across
train/test and produces a meaningless near-100% accuracy).

This script auto-detects how many real flare/pre-flare seconds are in
the held-out period and tells you whether the result is statistically
meaningful or not, rather than assuming either way. Confirmed history:
  - On a 5-day dataset with exactly 1 flare event, holding out the day
    WITHOUT the flare gave a real (if uninteresting) "always predicts
    quiet" result; holding out the day WITH the flare gave a degenerate
    all-zero-importance model, because training then had ZERO positive
    examples. Neither was a meaningful evaluation - purely a data
    quantity problem.
  - On a 38-day dataset (Sept 20 - Oct 28, 2024, solar maximum, 299 real
    GOES flares) holding out 2024-10-25 alone gave a real, informative
    result: decent flare recall (0.61) but poor pre-flare precision
    (0.28) - the model over-predicts "pre-flare", plausibly because
    solar-max data has very little genuinely quiet time (flares stack
    close together), making quiet/pre-flare hard to separate.
  - On the full 3-block dataset (July 2024 + Sept-Oct 2024 + June 2026),
    holding out the entire June 2026 block as test showed clearly worse
    performance (flare precision 0.40-0.47) than within-2024 holdouts
    (flare precision 0.82) - confirmed via a controlled comparison
    (training on Sept-Oct alone vs July+Sept-Oct combined, same June
    2026 test set) that this is NOT explained by July's data quality.
    Also confirmed via smoothed-derivative features that it's NOT
    explained by noisy/weak features either (the smoothed features
    became the top-ranked features but didn't move accuracy). The
    remaining explanation is a genuine train/test distribution gap:
    training data is dominated by solar-maximum conditions and doesn't
    represent June 2026's later-cycle activity level well.

TWO MODES:
  1. Single holdout (--test_date or --test_start/--test_end): the
     original mode, trains once, evaluates once.
  2. Cross-block validation (--cross_block_validate): trains on every
     PAIR of blocks, tests on the remaining one, for all 3 combinations,
     and reports the spread of results. This answers "how much does
     performance vary depending on which block is held out" directly,
     rather than relying on one cherry-picked split — useful for
     understanding whether the July<->Sept-Oct gap or the Sept-Oct<->
     June gap (or both) is driving the generalization problem, which
     in turn tells you what kind of NEW data (if any) would help most.

Use --test_date for a single day, or --test_start/--test_end for a
held-out RANGE (e.g. a full week) - useful for checking whether a
single-day result was representative or a fluke of that particular day.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    "soft_xray", "cdte_broadband", "czt_broadband",
    "hard_soft_ratio", "cdte_czt_ratio",
    "slx_d1", "slx_d2", "cdte_d1",
    "slx_d1_smooth_60s", "slx_d1_smooth_300s",
    "slx_roll_mean_5m", "slx_roll_std_5m",
    "slx_roll_mean_30m", "slx_roll_std_30m",
    "cdte_roll_mean_30m", "cdte_roll_std_30m",
    "slx_zscore", "cdte_zscore", "slx_vs_baseline",
    "data_quality",
]

# Below this many real (label != 0) seconds in the TEST set, treat
# precision/recall on classes 1/2 as not statistically meaningful and
# say so explicitly, rather than silently reporting a number that's
# really just noise from a handful of seconds.
MIN_POSITIVE_SECONDS_FOR_MEANINGFUL_TEST = 500

BLOCK_NAMES = {1: "July 2024", 2: "Sept-Oct 2024", 3: "June 2026"}


def temporal_split(df: pd.DataFrame, test_date: str = None,
                    test_start: str = None, test_end: str = None,
                    train_blocks: list = None):
    """
    Split by calendar date — NEVER shuffle randomly for time series.

    train_blocks: if given, restrict TRAINING data to only these block
    numbers. Test data is NEVER restricted by train_blocks — only by
    test_date/test_start/test_end.
    """
    if test_date is not None:
        train = df[df["date"] != test_date]
        test = df[df["date"] == test_date]
    elif test_start is not None and test_end is not None:
        in_range = (df["date"] >= test_start) & (df["date"] <= test_end)
        train = df[~in_range]
        test = df[in_range]
    else:
        raise ValueError("Provide either test_date, or both test_start and test_end")

    if train_blocks is not None:
        train = train[train["block"].isin(train_blocks)]

    return train, test


def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame, label: str = "", verbose: bool = True):
    """
    Core train/evaluate step, factored out so both the single-holdout
    mode and the cross-block validation mode share identical logic —
    avoids any chance of the two modes silently drifting apart.

    Returns a dict of summary metrics (precision/recall/f1 per class,
    plus the fitted pipeline and feature importances) suitable for
    collecting across multiple runs.
    """
    missing_cols = [c for c in FEATURE_COLS if c not in train.columns]
    if missing_cols:
        raise ValueError(f"Expected feature columns missing from input: {missing_cols}")

    n_test_positive = int((test["label"] != 0).sum())
    n_train_flare = int((train["label"] == 2).sum())
    meaningful = (n_test_positive >= MIN_POSITIVE_SECONDS_FOR_MEANINGFUL_TEST
                  and n_train_flare >= MIN_POSITIVE_SECONDS_FOR_MEANINGFUL_TEST)

    X_train, y_train = train[FEATURE_COLS], train["label"]
    X_test, y_test = test[FEATURE_COLS], test["label"]

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1, 2], zero_division=0
    )
    rf = pipeline.named_steps["rf"]
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    if verbose:
        print(f"\n  Train: {len(train)} rows, {train['date'].nunique()} day(s)")
        print(f"  Test:  {len(test)} rows, {test['date'].nunique()} day(s)")
        if not meaningful:
            print(f"  *** CAUTION: only {n_test_positive} non-quiet second(s) in TEST "
                  f"and/or {n_train_flare} flare second(s) in TRAIN — below the "
                  f"{MIN_POSITIVE_SECONDS_FOR_MEANINGFUL_TEST}-second threshold. "
                  f"Treat metrics as a mechanics check, not a real evaluation.")
        print("\n  Confusion matrix (rows=true, cols=predicted; labels [0,1,2]):")
        print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))
        print("\n  Classification report:")
        print(classification_report(y_test, y_pred, labels=[0, 1, 2],
                                      target_names=["quiet", "pre-flare", "flare"],
                                      zero_division=0))
        print("  Top 5 feature importances:")
        print(importances.head(5).to_string())

    return {
        "label": label,
        "meaningful": meaningful,
        "n_train": len(train), "n_test": len(test),
        "precision": dict(zip(["quiet", "pre-flare", "flare"], precision)),
        "recall": dict(zip(["quiet", "pre-flare", "flare"], recall)),
        "f1": dict(zip(["quiet", "pre-flare", "flare"], f1)),
        "support": dict(zip(["quiet", "pre-flare", "flare"], support)),
        "importances": importances,
        "pipeline": pipeline,
    }


def run_cross_block_validation(features_path: str):
    """
    Leave-one-block-out validation: for each block, train on the OTHER
    two blocks and test on it. Runs all 3 combinations and reports the
    spread of results, so a single cherry-picked split can't hide how
    much performance actually varies depending on which period is held
    out. This directly shows whether the generalization gap is uniform
    across all block pairs or concentrated in one specific pair (e.g.
    "July predicts Sept-Oct fine, but nothing predicts June well" would
    point at June 2026 specifically being out-of-distribution, while
    "every combination is roughly equally bad" would point at a more
    general problem needing more diverse data overall).
    """
    print("=" * 60)
    print("CROSS-BLOCK VALIDATION (leave-one-block-out)")
    print("=" * 60)

    df = pd.read_parquet(features_path)
    df["utc"] = pd.to_datetime(df["utc"], utc=True)

    if "block" not in df.columns:
        raise ValueError("Input is missing a 'block' column — required for cross-block validation.")

    blocks_present = sorted(df["block"].unique())
    print(f"Blocks present: {[(b, BLOCK_NAMES.get(b, '?')) for b in blocks_present]}")
    if len(blocks_present) < 3:
        print(f"WARNING: only {len(blocks_present)} block(s) present — cross-block "
              f"validation is most informative with 3+ blocks. Proceeding anyway.")

    results = []
    for held_out in blocks_present:
        train_blocks = [b for b in blocks_present if b != held_out]
        label = f"test={BLOCK_NAMES.get(held_out, held_out)} / train={[BLOCK_NAMES.get(b,b) for b in train_blocks]}"
        print(f"\n--- Fold: held out block {held_out} ({BLOCK_NAMES.get(held_out, '?')}) ---")

        train = df[df["block"].isin(train_blocks)]
        test = df[df["block"] == held_out]

        result = train_and_evaluate(train, test, label=label, verbose=True)
        results.append(result)

    print("\n" + "=" * 60)
    print("CROSS-BLOCK VALIDATION SUMMARY")
    print("=" * 60)
    summary_rows = []
    for r in results:
        row = {"held_out_block": r["label"]}
        for cls in ["quiet", "pre-flare", "flare"]:
            row[f"{cls}_precision"] = round(r["precision"][cls], 3)
            row[f"{cls}_recall"] = round(r["recall"][cls], 3)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    print("\nSpread across folds (max - min), per metric — large spread means")
    print("performance depends heavily on WHICH period is held out, not just")
    print("random noise:")
    for cls in ["quiet", "pre-flare", "flare"]:
        for metric in ["precision", "recall"]:
            col = f"{cls}_{metric}"
            spread = summary_df[col].max() - summary_df[col].min()
            print(f"  {col}: spread={spread:.3f}  (min={summary_df[col].min():.3f}, "
                  f"max={summary_df[col].max():.3f})")

    print("""
INTERPRETATION GUIDE:
  - If flare/pre-flare metrics are consistently weak across ALL folds
    (not just one) -> a general data-diversity problem, more data from
    ANY new period would likely help.
  - If one specific fold is much worse than the other two -> that
    specific held-out period is most out-of-distribution relative to
    the rest of your data. Look at which block that is — it tells you
    what KIND of new data (e.g. similar solar-activity level, similar
    season) would most directly address the gap.
""")

    return results


def run_mechanics_test(features_path: str, test_date: str = None,
                        test_start: str = None, test_end: str = None,
                        train_blocks: list = None):
    print("=" * 60)
    print("STAGE 6 — MODEL EVALUATION (temporal holdout)")
    print("=" * 60)

    df = pd.read_parquet(features_path)
    df["utc"] = pd.to_datetime(df["utc"], utc=True)
    n_days = df["date"].nunique()
    n_flare_total = (df["label"] == 2).sum()
    n_preflare_total = (df["label"] == 1).sum()
    print(f"Loaded {len(df)} rows, {n_days} day(s): "
          f"{sorted(df['date'].unique())[0]} -> {sorted(df['date'].unique())[-1]}")
    print(f"Total real flare seconds: {n_flare_total}  "
          f"pre-flare seconds: {n_preflare_total}  "
          f"(across the whole dataset, before splitting)")

    if test_date is not None:
        print(f"\nTemporal split: training on all days EXCEPT {test_date}, "
              f"testing on {test_date}")
    else:
        print(f"\nTemporal split: training on all days EXCEPT {test_start} -> {test_end}, "
              f"testing on that range")
    if train_blocks is not None:
        print(f"  RESTRICTING training to block(s) {train_blocks} only "
              f"(isolating a specific block's contribution to training)")

    train, test = temporal_split(df, test_date, test_start, test_end, train_blocks)

    print(f"\n  Label distribution in TRAIN:")
    print(f"  {train['label'].value_counts().sort_index().to_string()}")
    print(f"\n  Label distribution in TEST:")
    print(f"  {test['label'].value_counts().sort_index().to_string()}")

    result = train_and_evaluate(train, test, verbose=True)

    print("\n" + "=" * 60)
    print("WHAT TO ACTUALLY LOOK AT IN THIS OUTPUT")
    print("=" * 60)
    print("""
  1. Did training complete without error? -> pipeline mechanics work.
  2. Do quietness/hardness features (slx_zscore, hard_soft_ratio,
     slx_vs_baseline) rank near the top of feature importances? -> good
     sign the features carry real signal, not noise.
  3. Does 'data_quality' rank suspiciously high? -> would suggest the
     model is partly learning "is the instrument working" rather than
     "is a flare happening" - worth a closer look if so.
  4. If the CAUTION fired above, ignore precision/recall/accuracy on
     classes 1 and 2 - not enough real examples to trust the number.
  5. If no caution fired, the metrics reflect genuine model behavior.
""")

    return result["pipeline"], result["importances"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 6: model evaluation")
    parser.add_argument("--features_path", default="data/master/master_dataset_features.parquet")
    parser.add_argument("--test_date", default=None,
                         help="Single date (YYYY-MM-DD) to hold out as test.")
    parser.add_argument("--test_start", default=None,
                         help="Start date (YYYY-MM-DD) of a held-out RANGE.")
    parser.add_argument("--test_end", default=None,
                         help="End date (YYYY-MM-DD) of a held-out RANGE (inclusive).")
    parser.add_argument("--train_blocks", default=None,
                         help="Comma-separated block numbers to restrict TRAINING to "
                              "(e.g. '2' or '1,2'). Test range is unaffected.")
    parser.add_argument("--cross_block_validate", action="store_true",
                         help="Run leave-one-block-out validation across all blocks "
                              "instead of a single holdout split.")
    args = parser.parse_args()

    if args.cross_block_validate:
        run_cross_block_validation(args.features_path)
    else:
        if args.test_date is None and args.test_start is None:
            df_peek = pd.read_parquet(args.features_path, columns=["date"])
            args.test_date = sorted(df_peek["date"].unique())[-1]

        train_blocks = None
        if args.train_blocks is not None:
            train_blocks = [int(b.strip()) for b in args.train_blocks.split(",")]

        run_mechanics_test(args.features_path, args.test_date, args.test_start, args.test_end,
                            train_blocks)