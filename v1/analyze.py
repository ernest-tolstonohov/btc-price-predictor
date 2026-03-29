"""
Post-training diagnostics for the BTC/USDT prediction pipeline.

Prints:
  1. Target distribution — mean, std, median-abs-deviation for each horizon.
     Shows whether the baseline MAE is even beatable given market noise.
  2. Regressor prediction stats — correlation and mean-signed-error vs actuals.
  3. Classifier calibration — predicted probability distribution by outcome.
  4. Signal agreement — how often reg and cls agree, and accuracy when they do.

Usage:
    python analyze.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from train import (
    HORIZONS,
    SPLIT_RATIO,
    create_targets,
    engineer_features,
    load_and_clean_data,
)

DATA_PATH = "data/btc_usdt_features.csv"


def load_test_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Replicate the exact train/test split used in train.py.

    Returns:
        Tuple of (X_test, y_test DataFrame with all target cols, feature_cols).
    """
    df = load_and_clean_data(DATA_PATH)
    df = engineer_features(df)
    df = create_targets(df, HORIZONS)

    bps_cols = [f"target_bps_{h}" for h in HORIZONS]
    dir_cols = [f"target_dir_{h}" for h in HORIZONS]
    feature_cols = joblib.load("models/features.pkl")

    df.dropna(subset=feature_cols, inplace=True)
    split_idx = int(len(df) * SPLIT_RATIO)
    test = df.iloc[split_idx:]

    X_test = test[feature_cols]
    y_test = test[bps_cols + dir_cols]
    return X_test, y_test, feature_cols


def section(title: str) -> None:
    """Print a section header.

    Args:
        title: Section title text.
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    """Run all diagnostic checks and print results."""
    print("Loading test data and models...")
    X_test, y_test, feature_cols = load_test_data()

    reg_models = {h: joblib.load(f"models/model_reg_{h}m.pkl") for h in HORIZONS}
    cls_models = {h: joblib.load(f"models/model_cls_{h}m.pkl") for h in HORIZONS}

    # 1. Target distribution
    section("1. Target Distribution (test set)")
    print(f"  {'Horizon':<10} {'Mean':>8} {'Std':>8} {'MAD':>8} "
          f"{'% >0':>8} {'Baseline MAE':>14}")
    print(f"  {'-'*62}")
    for h in HORIZONS:
        col = f"target_bps_{h}"
        vals = y_test[col].values
        mad = np.median(np.abs(vals - np.median(vals)))
        pct_up = (vals > 0).mean() * 100
        baseline_mae = np.abs(vals).mean()
        print(f"  +{h}m       {vals.mean():>8.2f} {vals.std():>8.2f} "
              f"{mad:>8.2f} {pct_up:>7.1f}% {baseline_mae:>14.2f}")

    # 2. Regressor predictions vs actuals
    section("2. Regressor: Prediction Quality")
    print(f"  {'Horizon':<10} {'Corr':>8} {'MAE':>8} {'Baseline':>10} "
          f"{'Lift %':>8} {'Bias (mean err)':>16}")
    print(f"  {'-'*64}")
    for h in HORIZONS:
        bps_col = f"target_bps_{h}"
        actuals = y_test[bps_col].values
        preds = reg_models[h].predict(X_test)
        corr = np.corrcoef(actuals, preds)[0, 1]
        mae = mean_absolute_error(actuals, preds)
        baseline = np.abs(actuals).mean()
        lift = (baseline - mae) / baseline * 100
        bias = (preds - actuals).mean()
        print(f"  +{h}m       {corr:>8.4f} {mae:>8.2f} {baseline:>10.2f} "
              f"{lift:>7.2f}% {bias:>16.4f}")

    # 3. Regressor: signed error by direction quartile
    section("3. Regressor: Predictions by Actual-Move Quartile")
    print("  Are large moves predicted as large, or is everything near zero?")
    for h in HORIZONS:
        bps_col = f"target_bps_{h}"
        actuals = y_test[bps_col].values
        preds = reg_models[h].predict(X_test)
        quartiles = np.percentile(actuals, [0, 25, 50, 75, 100])
        print(f"\n  +{h}m horizon")
        print(f"  {'Quartile':<22} {'Actual range':>16} {'Mean pred':>12} "
              f"{'Mean actual':>13}")
        labels = ["Q1 (bottom 25%)", "Q2", "Q3", "Q4 (top 25%)"]
        for i, label in enumerate(labels):
            mask = (actuals >= quartiles[i]) & (actuals < quartiles[i + 1])
            if mask.sum() == 0:
                continue
            print(f"  {label:<22} [{quartiles[i]:>6.1f}, {quartiles[i+1]:>6.1f})"
                  f"  {preds[mask].mean():>12.2f} {actuals[mask].mean():>13.2f}")

    # 4. Classifier calibration
    section("4. Classifier: Probability Distribution by Outcome")
    print(f"  {'Horizon':<10} {'Mean prob | actual=1':>22} "
          f"{'Mean prob | actual=0':>22} {'Separation':>12}")
    print(f"  {'-'*70}")
    for h in HORIZONS:
        dir_col = f"target_dir_{h}"
        actuals = y_test[dir_col].values
        proba = cls_models[h].predict_proba(X_test)[:, 1]
        mean_pos = proba[actuals == 1].mean()
        mean_neg = proba[actuals == 0].mean()
        separation = mean_pos - mean_neg
        print(f"  +{h}m       {mean_pos:>22.4f} {mean_neg:>22.4f} "
              f"{separation:>12.4f}")

    # 5. Combined signal agreement
    section("5. Signal Agreement: When Reg and Cls Agree, Are They Right?")
    print(f"  {'Horizon':<10} {'Agreement %':>13} {'Acc when agree':>16} "
          f"{'Acc when disagree':>19} {'Baseline acc':>14}")
    print(f"  {'-'*76}")
    for h in HORIZONS:
        bps_col = f"target_bps_{h}"
        dir_col = f"target_dir_{h}"
        actuals_dir = y_test[dir_col].values
        preds_bps = reg_models[h].predict(X_test)
        proba = cls_models[h].predict_proba(X_test)[:, 1]

        # Reg says up when bps > 0, cls says up when prob > 0.5
        reg_up = preds_bps > 0
        cls_up = proba > 0.5
        agree = reg_up == cls_up

        acc_agree = (reg_up[agree] == actuals_dir[agree]).mean() if agree.sum() > 0 else float("nan")
        acc_disagree = (reg_up[~agree] == actuals_dir[~agree]).mean() if (~agree).sum() > 0 else float("nan")
        baseline = max(actuals_dir.mean(), 1 - actuals_dir.mean())
        pct_agree = agree.mean() * 100

        print(f"  +{h}m       {pct_agree:>12.1f}% {acc_agree:>16.4f} "
              f"{acc_disagree:>19.4f} {baseline:>14.4f}")


if __name__ == "__main__":
    main()
