"""
BTC/USDT multi-horizon price prediction pipeline using LightGBM.

Trains two models per horizon (ten total):
  - Regressor (model_reg_Nm.pkl): predicts price move in basis points.
  - Classifier (model_cls_Nm.pkl): predicts binary price direction (1=up).

Horizons: 1, 3, 5, 10, 15 minutes ahead.

Usage:
    python train.py

Output:
    model_reg_1m.pkl ... model_reg_15m.pkl
    model_cls_1m.pkl ... model_cls_15m.pkl
    model_meta.pkl
    features.pkl
"""

import logging
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_PATH = "data/btc_usdt_features.csv"
HORIZONS = [1, 3, 5, 10, 15]
SPLIT_RATIO = 0.8
EARLY_STOPPING_ROUNDS = 100

SPARSE_COLS = [
    "open_interest",
    "long_short_ratio",
    "long_account",
    "short_account",
]

# Base hyperparameters shared by all regressor horizons (unchanged from v1)
LGB_PARAMS_BASE: dict[str, Any] = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "num_leaves": 48,
    "max_depth": 5,
    "min_child_samples": 150,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "subsample_freq": 1,
    "reg_alpha": 0.2,
    "reg_lambda": 0.5,
    "verbose": -1,
}

# Per-horizon overrides. Longer horizons have weaker signal-to-noise ratio so
# heavy regularisation hurts more than it helps — fewer required samples per
# leaf and lower lambda let the model find patterns that would otherwise be
# pruned away.
LGB_PARAMS_OVERRIDES: dict[int, dict[str, Any]] = {
    10: {"min_child_samples": 75, "reg_lambda": 0.1, "num_leaves": 48},
    15: {"min_child_samples": 50, "reg_lambda": 0.1, "num_leaves": 31},
}

# Classifier hyperparameters are identical across all horizons. No per-horizon
# overrides are applied because the direction signal degrades more uniformly
# than the magnitude signal, so a single regularisation profile is used.
LGB_CLS_PARAMS: dict[str, Any] = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "num_leaves": 48,
    "max_depth": 5,
    "min_child_samples": 150,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.2,
    "reg_lambda": 0.5,
    "min_gain_to_split": 0.01,
    "verbose": -1,
}


def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load raw CSV and apply basic cleaning steps.

    Drops sparse columns, forward-fills funding_rate and spx_close (both
    operations use only past values so they are safe to apply before the
    train/test split), and removes any rows missing core OHLCV data.

    Args:
        path: Path to the CSV file produced by collect_data.py.

    Returns:
        Cleaned DataFrame with a UTC DatetimeIndex named "timestamp".
    """
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)

    # Drop columns that are ~84% null — confirmed via null-count analysis
    cols_to_drop = [c for c in SPARSE_COLS if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"Dropped sparse columns: {cols_to_drop}")

    # Forward fill uses only past observations, so it is leakage-free
    df["funding_rate"] = df["funding_rate"].ffill()
    df["spx_close"] = df["spx_close"].ffill()

    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Compute exponential moving average with span equal to period.

    Args:
        series: Input price series.
        period: EMA span.

    Returns:
        EMA series of identical length.
    """
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder smoothing (alpha = 1 / period).

    Args:
        series: Closing price series.
        period: Lookback period.

    Returns:
        RSI series bounded [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range using Wilder smoothing.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Smoothing period.

    Returns:
        ATR series.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features from raw columns without look-ahead.

    All rolling and shift operations are strictly backward-looking.
    No absolute price levels are added as features — only scale-invariant
    derivatives. Raw OHLCV columns remain in the DataFrame for target
    creation but are excluded from feature_cols in main().

    Args:
        df: Cleaned DataFrame from load_and_clean_data.

    Returns:
        DataFrame with all engineered feature columns appended.
    """
    feat = df.copy()
    close = feat["close"]
    high = feat["high"]
    low = feat["low"]
    volume = feat["volume"]
    vdelta = feat["volume_delta"]

    # Log returns capture relative price momentum across multiple lookbacks
    for lag in [1, 3, 5, 10, 15, 30, 60]:
        feat[f"log_return_{lag}"] = np.log(close / close.shift(lag))

    # Candle structure: normalized range and close position within the candle
    hl_range = high - low
    feat["hl_range_norm"] = hl_range / close
    feat["close_position"] = np.where(hl_range > 0.0, (close - low) / hl_range, 0.5)

    # Volume relative to recent average removes absolute scale dependency
    vol_mean_20 = volume.rolling(20).mean()
    feat["volume_norm"] = volume / vol_mean_20.replace(0.0, np.nan)

    # Cumulative buy/sell pressure over different windows
    for window in [5, 15, 30]:
        feat[f"vdelta_sum_{window}"] = vdelta.rolling(window).sum()

    # Current delta relative to typical magnitude signals unusual pressure
    abs_vdelta_mean_20 = vdelta.abs().rolling(20).mean()
    feat["vdelta_ratio"] = vdelta / abs_vdelta_mean_20.replace(0.0, np.nan)

    # Momentum oscillator
    feat["rsi_14"] = _rsi(close, period=14)

    # EMA cross signals normalized by close — relative distance, not absolute level
    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    feat["ema9_minus_ema21"] = (ema9 - ema21) / close
    feat["ema21_minus_ema50"] = (ema21 - ema50) / close

    # ATR as a fraction of close gives relative volatility independent of price level
    feat["atr_14"] = _atr(high, low, close, period=14) / close

    bb_mean = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mean + 2.0 * bb_std
    bb_lower = bb_mean - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower).replace(0.0, np.nan)
    feat["bb_position_upper"] = (close - bb_upper) / bb_width
    feat["bb_position_lower"] = (close - bb_lower) / bb_width

    # Rolling std normalized by close — relative dispersion, not absolute dollars
    feat["close_std_10"] = close.rolling(10).std() / close
    feat["close_std_30"] = close.rolling(30).std() / close

    # Cyclical time features capture intraday and intraweek seasonality
    idx = feat.index
    feat["hour"] = idx.hour
    feat["day_of_week"] = idx.dayofweek
    feat["minute_of_day"] = idx.hour * 60 + idx.minute

    # Replace absolute SPX level with its rate of change to remove price scale
    feat["spx_pct_change"] = feat["spx_close"].pct_change()
    feat.drop(columns=["spx_close"], inplace=True)

    return feat


def create_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Create basis-point return and binary direction targets per horizon.

    Replaces the v1 fractional-return single target with two targets:
      target_bps_N  : ((close[+N] - close) / close) * 10000 — bps move
      target_dir_N  : 1 if target_bps_N > 0, else 0 — binary direction

    Using bps instead of fractional returns keeps the regressor target in a
    human-readable unit without affecting model behaviour (it is a linear
    rescaling by 10000). The direction target enables a separate classifier
    whose confidence can be combined with the regressor at inference time.

    Args:
        df: Feature DataFrame that still contains the close column.
        horizons: List of forward steps in minutes.

    Returns:
        DataFrame with target columns appended; tail rows whose bps targets
        are NaN are dropped.
    """
    result = df.copy()
    for h in horizons:
        bps = (
            (result["close"].shift(-h) - result["close"]) / result["close"]
        ) * 10_000
        result[f"target_bps_{h}"] = bps
        # Direction is derived from bps so it is NaN-free after the dropna below
        result[f"target_dir_{h}"] = (bps > 0).astype(int)

    # Only bps columns can contain NaN (tail rows); direction inherits the drop
    bps_cols = [f"target_bps_{h}" for h in horizons]
    before = len(result)
    result.dropna(subset=bps_cols, inplace=True)
    logger.info(f"Dropped {before - len(result)} tail rows due to NaN targets")
    return result


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    split_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train and test sets.

    No shuffling is applied. The split index is computed once and reused
    across all five models to guarantee a shared evaluation boundary.
    LightGBM is tree-based so no feature scaling is needed.

    Args:
        df: Full DataFrame containing feature and target columns.
        feature_cols: Ordered list of feature column names.
        target_cols: List of target column names.
        split_ratio: Fraction of rows assigned to training (0 < ratio < 1).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    split_idx = int(len(df) * split_ratio)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[feature_cols]
    X_test = test[feature_cols]
    y_train = train[target_cols]
    y_test = test[target_cols]

    logger.info(
        f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | "
        f"Features: {len(feature_cols)}"
    )
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    horizon: int,
    params: dict[str, Any],
) -> lgb.LGBMRegressor:
    """Train a LightGBM regressor predicting basis-point price moves.

    The interface and early-stopping logic are identical to v1. The only
    semantic change is that the target is now in basis points rather than
    fractional returns, which is a linear rescaling that does not affect
    tree structure or feature importances.

    Args:
        X_train: Training feature matrix.
        y_train: Training bps target series for this horizon.
        X_test: Test feature matrix (early stopping reference only).
        y_test: Test bps target series (early stopping reference only).
        horizon: Forward horizon in minutes (used for logging).
        params: Full LightGBM hyperparameter dict for this horizon.

    Returns:
        Fitted LGBMRegressor.
    """
    logger.info(f"Training regressor for {horizon}m horizon...")
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    logger.info(f"Regressor {horizon}m: best iteration = {model.best_iteration_}")
    return model


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    horizon: int,
) -> lgb.LGBMClassifier:
    """Train a LightGBM binary classifier predicting price direction.

    Uses a fixed hyperparameter set (LGB_CLS_PARAMS) with no per-horizon
    overrides. The classifier predicts P(close[+N] > close[0]).

    Args:
        X_train: Training feature matrix.
        y_train: Binary training labels (1 = price up, 0 = flat/down).
        X_test: Test feature matrix (early stopping reference only).
        y_test: Binary test labels (early stopping reference only).
        horizon: Forward horizon in minutes (used for logging).

    Returns:
        Fitted LGBMClassifier.
    """
    logger.info(f"Training classifier for {horizon}m horizon...")
    model = lgb.LGBMClassifier(**LGB_CLS_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    logger.info(f"Classifier {horizon}m: best iteration = {model.best_iteration_}")
    return model


def evaluate_model(
    model: lgb.LGBMRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    horizon: int,
) -> dict[str, Any]:
    """Compute MAE and RMSE in basis points for a fitted regressor.

    Also computes a naive baseline that always predicts 0 bps (no move).
    A model is only useful if its MAE meaningfully undercuts the baseline.

    Args:
        model: Fitted LGBMRegressor.
        X_train: Training features.
        y_train: Training bps targets.
        X_test: Test features.
        y_test: Test bps targets.
        horizon: Forward horizon in minutes.

    Returns:
        Dict with keys: horizon, best_iter, mae_train_bps, mae_test_bps,
        rmse_test_bps, baseline_mae_bps.
    """
    train_preds: np.ndarray = np.asarray(model.predict(X_train))
    test_preds: np.ndarray = np.asarray(model.predict(X_test))
    y_test_arr = np.asarray(y_test)

    mae_train_bps = float(mean_absolute_error(y_train, train_preds))
    mae_test_bps = float(mean_absolute_error(y_test_arr, test_preds))
    rmse_test_bps = float(np.sqrt(mean_squared_error(y_test_arr, test_preds)))
    baseline_mae_bps = float(
        mean_absolute_error(y_test_arr, np.zeros_like(y_test_arr))
    )

    return {
        "horizon": horizon,
        "best_iter": model.best_iteration_,
        "mae_train_bps": mae_train_bps,
        "mae_test_bps": mae_test_bps,
        "rmse_test_bps": rmse_test_bps,
        "baseline_mae_bps": baseline_mae_bps,
    }


def evaluate_classifier(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    horizon: int,
) -> dict[str, Any]:
    """Compute accuracy, ROC-AUC, and label balance for a fitted classifier.

    The naive baseline always predicts the majority class. A model is only
    useful if its accuracy and AUC meaningfully exceed that baseline.

    Args:
        model: Fitted LGBMClassifier.
        X_test: Test feature matrix.
        y_test: Binary test labels (1 = price up, 0 = flat/down).
        horizon: Forward horizon in minutes.

    Returns:
        Dict with keys: horizon, best_iter, accuracy_test, roc_auc_test,
        frac_positive, baseline_accuracy.
    """
    y_test_arr = np.asarray(y_test)
    test_preds: np.ndarray = np.asarray(model.predict(X_test))
    test_proba: np.ndarray = np.asarray(model.predict_proba(X_test))[:, 1]

    accuracy_test = float(accuracy_score(y_test_arr, test_preds))
    roc_auc_test = float(roc_auc_score(y_test_arr, test_proba))
    frac_positive = float(y_test_arr.mean())

    # Majority class is 1 when more than half of test labels are positive
    majority_class = int(frac_positive >= 0.5)
    baseline_accuracy = float(
        accuracy_score(y_test_arr, np.full_like(y_test_arr, majority_class))
    )

    return {
        "horizon": horizon,
        "best_iter": model.best_iteration_,
        "accuracy_test": accuracy_test,
        "roc_auc_test": roc_auc_test,
        "frac_positive": frac_positive,
        "baseline_accuracy": baseline_accuracy,
    }


def save_artifacts(
    reg_models: dict[int, lgb.LGBMRegressor],
    cls_models: dict[int, lgb.LGBMClassifier],
    feature_cols: list[str],
) -> None:
    """Persist all ten models, the feature list, and model metadata to disk.

    Args:
        reg_models: Mapping from horizon to fitted LGBMRegressor.
        cls_models: Mapping from horizon to fitted LGBMClassifier.
        feature_cols: Ordered list of feature column names used during training.
    """
    for horizon, model in reg_models.items():
        path = f"models/model_reg_{horizon}m.pkl"
        joblib.dump(model, path)
        logger.info(f"Saved {path}")

    for horizon, model in cls_models.items():
        path = f"models/model_cls_{horizon}m.pkl"
        joblib.dump(model, path)
        logger.info(f"Saved {path}")

    joblib.dump(feature_cols, "models/features.pkl")
    logger.info("Saved models/features.pkl")

    model_meta = {
        "reg": {"horizons": HORIZONS, "target": "bps"},
        "cls": {"horizons": HORIZONS, "target": "direction"},
    }
    joblib.dump(model_meta, "models/model_meta.pkl")
    logger.info("Saved models/model_meta.pkl")


def predict_all(
    row: pd.Series,
    reg_models: dict[int, lgb.LGBMRegressor],
    cls_models: dict[int, lgb.LGBMClassifier],
    feature_cols: list[str],
) -> dict[int, dict[str, Any]]:
    """Run regressor and classifier inference for all horizons on one row.

    The combined signal requires both models to agree: the regressor must
    predict a non-zero direction AND the classifier probability must clear a
    confidence threshold (>0.55 for long, <0.45 for short).

    Args:
        row: Series containing at minimum all feature_cols values.
        reg_models: Mapping from horizon to fitted LGBMRegressor (bps).
        cls_models: Mapping from horizon to fitted LGBMClassifier (direction).
        feature_cols: Ordered list of feature names used during training.

    Returns:
        Dict mapping each horizon to a nested dict with keys:
          bps      — predicted basis-point move from the regressor
          prob_up  — classifier P(price up), i.e. predict_proba class-1
          signal   — "long", "short", or "neutral"
    """
    X = pd.DataFrame([row[feature_cols]])
    result: dict[int, dict[str, Any]] = {}

    for horizon in reg_models:
        bps = float(np.asarray(reg_models[horizon].predict(X))[0])
        prob_up = float(np.asarray(cls_models[horizon].predict_proba(X))[0, 1])

        if bps > 0 and prob_up > 0.55:
            signal = "long"
        elif bps < 0 and prob_up < 0.45:
            signal = "short"
        else:
            signal = "neutral"

        result[horizon] = {"bps": bps, "prob_up": prob_up, "signal": signal}

    return result


def main() -> None:
    """Orchestrate the full training pipeline end to end."""
    # Step 1: Load and clean raw data
    df = load_and_clean_data(DATA_PATH)

    # Step 2: Derive all features without look-ahead
    df = engineer_features(df)

    # Step 3: Create bps + direction targets; drop tail rows with NaN bps
    df = create_targets(df, HORIZONS)

    # Step 4: Exclude raw price/volume columns and all target columns from
    # features. The feature set is identical to v1 — only the targets changed.
    bps_target_cols = [f"target_bps_{h}" for h in HORIZONS]
    dir_target_cols = [f"target_dir_{h}" for h in HORIZONS]
    all_target_cols = set(bps_target_cols + dir_target_cols)
    raw_price_cols = {"open", "high", "low", "close", "volume"}
    feature_cols = [
        c for c in df.columns if c not in all_target_cols | raw_price_cols
    ]

    # Step 5: Remove warm-up rows where rolling windows have not yet filled
    before = len(df)
    df.dropna(subset=feature_cols, inplace=True)
    logger.info(f"Dropped {before - len(df)} warm-up rows with NaN features")

    # Step 6: One chronological split shared by all ten models
    all_target_list = bps_target_cols + dir_target_cols
    X_train, X_test, y_train, y_test = split_data(
        df, feature_cols, all_target_list, SPLIT_RATIO
    )

    # Step 7: Train regressors — one per horizon
    reg_models: dict[int, lgb.LGBMRegressor] = {}
    reg_metrics_rows: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        bps_col = f"target_bps_{horizon}"
        horizon_params = {**LGB_PARAMS_BASE, **LGB_PARAMS_OVERRIDES.get(horizon, {})}
        model = train_model(
            X_train,
            y_train[bps_col],
            X_test,
            y_test[bps_col],
            horizon,
            horizon_params,
        )
        metrics = evaluate_model(
            model,
            X_train,
            y_train[bps_col],
            X_test,
            y_test[bps_col],
            horizon,
        )
        reg_models[horizon] = model
        reg_metrics_rows.append(metrics)

    # Step 8: Train classifiers — one per horizon
    cls_models: dict[int, lgb.LGBMClassifier] = {}
    cls_metrics_rows: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        dir_col = f"target_dir_{horizon}"
        model = train_classifier(
            X_train,
            y_train[dir_col],
            X_test,
            y_test[dir_col],
            horizon,
        )
        metrics = evaluate_classifier(model, X_test, y_test[dir_col], horizon)
        cls_models[horizon] = model
        cls_metrics_rows.append(metrics)

    # Step 9: Print regressor evaluation table
    reg_results = pd.DataFrame(reg_metrics_rows).set_index("horizon")
    reg_results.columns = [
        "Best Iter",
        "MAE Train (bps)",
        "MAE Test (bps)",
        "RMSE Test (bps)",
        "Baseline MAE (bps)",
    ]
    print("\n=== Regressor Evaluation (Basis Points) ===")
    print(reg_results.to_string(float_format=lambda x: f"{x:.4f}"))

    # Step 10: Print classifier evaluation table
    cls_results = pd.DataFrame(cls_metrics_rows).set_index("horizon")
    cls_results.columns = [
        "Best Iter",
        "Accuracy Test",
        "ROC-AUC Test",
        "Frac Positive",
        "Baseline Accuracy",
    ]
    print("\n=== Classifier Evaluation (Direction) ===")
    print(cls_results.to_string(float_format=lambda x: f"{x:.4f}"))

    # Step 11: Feature importances for regressors
    for horizon, model in reg_models.items():
        importance = (
            pd.Series(model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
            .head(20)
        )
        print(f"\n--- Regressor Feature Importance: {horizon}m (top 20) ---")
        print(importance.to_string())

    # Step 12: Feature importances for classifiers
    for horizon, model in cls_models.items():
        importance = (
            pd.Series(model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
            .head(20)
        )
        print(f"\n--- Classifier Feature Importance: {horizon}m (top 20) ---")
        print(importance.to_string())

    # Step 13: Persist all ten models, feature list, and metadata
    save_artifacts(reg_models, cls_models, feature_cols)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
