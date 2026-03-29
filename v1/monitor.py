"""
BTC/USDT prediction monitor — runs every minute and tracks accuracy.

Every minute:
  1. Resolves any pending predictions whose target time has passed.
  2. Makes new predictions for the current candle.
  3. Prints a live accuracy table.

Results are appended to results.csv for offline analysis.

Usage:
    python monitor.py
"""

import csv
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from train import engineer_features


FUTURES_BASE = "https://fapi.binance.com"
FEAR_GREED_URL = "https://api.alternative.me/fng/"
SYMBOL = "BTCUSDT"
CANDLES_NEEDED = 250
HORIZONS = [1, 3, 5, 10, 15]
RESULTS_FILE = "results/results.csv"
RESULTS_COLUMNS = [
    "made_at",
    "horizon",
    "target_time",
    "base_price",
    "predicted",
    "actual",
    "error_usd",
    "error_pct",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get(url: str, params: dict) -> dict | list:
    """GET request with a 30-second timeout.

    Args:
        url: Full endpoint URL.
        params: Query string parameters.

    Returns:
        Parsed JSON response.
    """
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_recent_ohlcv(limit: int = CANDLES_NEEDED) -> pd.DataFrame:
    """Fetch the most recent 1-minute OHLCV candles from Binance Futures.

    Args:
        limit: Number of recent candles to retrieve.

    Returns:
        DataFrame indexed by UTC timestamp with columns: open, high, low,
        close, volume, volume_delta.
    """
    batch = _get(
        f"{FUTURES_BASE}/fapi/v1/klines",
        {"symbol": SYMBOL, "interval": "1m", "limit": limit},
    )
    rows = []
    for c in batch:
        total_vol = float(c[5])
        taker_buy_vol = float(c[9])
        rows.append(
            {
                "timestamp": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": total_vol,
                "volume_delta": 2 * taker_buy_vol - total_vol,
            }
        )
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def fetch_latest_funding_rate() -> float:
    """Fetch the most recent BTCUSDT funding rate.

    Returns:
        Latest funding rate as a float.
    """
    data = _get(f"{FUTURES_BASE}/fapi/v1/fundingRate", {"symbol": SYMBOL, "limit": 1})
    return float(data[-1]["fundingRate"]) if data else 0.0


def fetch_latest_fear_greed() -> int:
    """Fetch the latest Fear & Greed Index value.

    Returns:
        Fear & Greed index (0-100).
    """
    data = _get(FEAR_GREED_URL, {"limit": 1})
    return int(data["data"][0]["value"])


def fetch_latest_spx() -> float:
    """Fetch the most recent SPX close from Yahoo Finance.

    Returns:
        Most recent ^GSPC close price, or NaN on failure.
    """
    raw = yf.Ticker("^GSPC").history(period="5d", interval="1h")
    return float(raw["Close"].iloc[-1]) if not raw.empty else float("nan")


def make_predictions(
    ohlcv: pd.DataFrame,
    funding_rate: float,
    fear_greed: int,
    spx_close: float,
    reg_models: dict,
    cls_models: dict,
    feature_cols: list[str],
) -> tuple[float, pd.Timestamp, dict[int, dict[str, Any]]]:
    """Build features from live data and run all horizon models.

    Args:
        ohlcv: Recent OHLCV candles with volume_delta.
        funding_rate: Latest funding rate.
        fear_greed: Latest Fear & Greed value.
        spx_close: Latest SPX close.
        reg_models: Mapping from horizon to fitted LGBMRegressor.
        cls_models: Mapping from horizon to fitted LGBMClassifier.
        feature_cols: Ordered feature list used during training.

    Returns:
        Tuple of (current_close, candle_timestamp, predictions dict). Each
        predictions entry contains: price, bps, prob_up, signal.
    """
    df = ohlcv.copy()
    df["funding_rate"] = funding_rate
    df["fear_greed"] = fear_greed
    df["spx_close"] = spx_close

    df = engineer_features(df)
    row = df.iloc[-1]
    current_close = float(ohlcv["close"].iloc[-1])
    candle_ts = ohlcv.index[-1]

    X = pd.DataFrame([row[feature_cols]])
    predictions: dict[int, dict[str, Any]] = {}
    for h, reg_model in reg_models.items():
        bps = float(np.asarray(reg_model.predict(X))[0])
        prob_up = float(np.asarray(cls_models[h].predict_proba(X))[0, 1])

        if bps > 0 and prob_up > 0.55:
            signal = "long"
        elif bps < 0 and prob_up < 0.45:
            signal = "short"
        else:
            signal = "neutral"

        predictions[h] = {
            "price": current_close * (1.0 + bps / 10_000),
            "bps": bps,
            "prob_up": prob_up,
            "signal": signal,
        }

    return current_close, candle_ts, predictions


def append_result(row: dict) -> None:
    """Append one resolved prediction row to results.csv.

    Args:
        row: Dict with keys matching RESULTS_COLUMNS.
    """
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_results() -> pd.DataFrame:
    """Load all resolved results from disk.

    Returns:
        DataFrame of resolved predictions, or empty DataFrame if none yet.
    """
    if not os.path.exists(RESULTS_FILE):
        return pd.DataFrame(columns=RESULTS_COLUMNS)
    return pd.read_csv(RESULTS_FILE, parse_dates=["made_at", "target_time"])


def print_accuracy_table(resolved: list[dict]) -> None:
    """Print per-horizon accuracy stats from the resolved list.

    Args:
        resolved: List of result dicts collected in the current session.
    """
    if not resolved:
        print("  No resolved predictions yet.")
        return

    df = pd.DataFrame(resolved)
    print(
        f"\n  {'Horizon':<10} {'Count':>6} {'MAE ($)':>10} {'MAE (%)':>10} {'RMSE ($)':>10}"
    )
    print(f"  {'-' * 50}")
    for h in HORIZONS:
        sub = df[df["horizon"] == h]
        if sub.empty:
            continue
        mae_usd = sub["error_usd"].abs().mean()
        mae_pct = sub["error_pct"].abs().mean()
        rmse_usd = np.sqrt((sub["error_usd"] ** 2).mean())
        print(
            f"  +{h}m        {len(sub):>6} {mae_usd:>10.2f} {mae_pct:>9.4f}% {rmse_usd:>10.2f}"
        )


def seconds_until_next_minute() -> float:
    """Compute seconds remaining until the next whole UTC minute.

    Returns:
        Float seconds to sleep.
    """
    now = datetime.now(timezone.utc)
    return 60.0 - now.second - now.microsecond / 1_000_000


def main() -> None:
    """Run the prediction monitor loop indefinitely."""
    logger.info("Loading models...")
    feature_cols: list[str] = joblib.load("models/features.pkl")
    reg_models = {h: joblib.load(f"models/model_reg_{h}m.pkl") for h in HORIZONS}
    cls_models = {h: joblib.load(f"models/model_cls_{h}m.pkl") for h in HORIZONS}
    logger.info(f"Loaded {len(reg_models) * 2} models | {len(feature_cols)} features")

    pending: dict[tuple[int, pd.Timestamp], dict] = {}
    resolved_session: list[dict] = []

    spx_close = fetch_latest_spx()
    spx_fetched_at = datetime.now(timezone.utc)

    fear_greed = fetch_latest_fear_greed()
    fg_fetched_at = datetime.now(timezone.utc)

    logger.info("Aligning to next minute boundary...")
    sleep_secs = seconds_until_next_minute()
    logger.info(f"Sleeping {sleep_secs:.1f}s to align...")
    time.sleep(sleep_secs)

    while True:
        tick_start = datetime.now(timezone.utc)

        try:
            if (tick_start - spx_fetched_at).total_seconds() >= 3600:
                spx_close = fetch_latest_spx()
                spx_fetched_at = tick_start

            if (tick_start - fg_fetched_at).total_seconds() >= 3600:
                fear_greed = fetch_latest_fear_greed()
                fg_fetched_at = tick_start

            funding_rate = fetch_latest_funding_rate()
            ohlcv = fetch_recent_ohlcv()

            current_close, candle_ts, predictions = make_predictions(
                ohlcv, funding_rate, fear_greed, spx_close,
                reg_models, cls_models, feature_cols,
            )

            resolved_keys = [key for key in pending if key[1] <= candle_ts]
            for key in resolved_keys:
                horizon, target_ts = key
                actual = float(
                    ohlcv["close"].reindex([target_ts], method="nearest").iloc[0]
                )
                pred_info = pending.pop(key)
                error_usd = actual - pred_info["predicted"]
                error_pct = error_usd / actual * 100.0
                row = {
                    "made_at": pred_info["made_at"],
                    "horizon": horizon,
                    "target_time": target_ts,
                    "base_price": pred_info["base_price"],
                    "predicted": pred_info["predicted"],
                    "actual": actual,
                    "error_usd": error_usd,
                    "error_pct": error_pct,
                }
                resolved_session.append(row)
                append_result(row)

            for h, pred in predictions.items():
                target_ts = candle_ts + pd.Timedelta(minutes=h)
                pending[(h, target_ts)] = {
                    "predicted": pred["price"],
                    "base_price": current_close,
                    "made_at": candle_ts,
                }

            now_str = tick_start.strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"\n{'=' * 64}")
            print(f"  {now_str}  |  BTC: ${current_close:,.2f}")
            print(f"  Fear & Greed: {fear_greed}  |  Funding: {funding_rate:.6f}")
            print(f"{'=' * 64}")
            print(
                f"  {'Horizon':<10} {'Predicted':>14} {'Change':>10}"
                f"  {'P(up)':>6}  {'Signal':<8}"
            )
            print(f"  {'-' * 54}")
            for h in HORIZONS:
                pred = predictions[h]
                chg = (pred["price"] - current_close) / current_close * 100.0
                sign = "+" if chg >= 0 else "-"
                print(
                    f"  +{h}m        ${pred['price']:>13,.2f}  {sign}{abs(chg):.3f}%"
                    f"  {pred['prob_up']:>6.3f}  {pred['signal']:<8}"
                )

            if resolved_keys:
                print(f"\n  Resolved {len(resolved_keys)} prediction(s) this tick:")
                for row in resolved_session[-len(resolved_keys):]:
                    sign = "+" if row["error_usd"] >= 0 else ""
                    print(
                        f"  +{row['horizon']}m  pred=${row['predicted']:,.2f}  "
                        f"actual=${row['actual']:,.2f}  "
                        f"err={sign}{row['error_usd']:.2f} ({sign}{row['error_pct']:.3f}%)"
                    )

            total_results = load_results()
            if len(total_results) > 0:
                print(f"\n  Accuracy summary ({len(total_results)} total resolved):")
                print_accuracy_table(total_results.to_dict("records"))

            print(f"  Pending: {len(pending)} predictions in queue")

        except Exception as exc:
            logger.error(f"Tick error: {exc}", exc_info=True)

        elapsed = (datetime.now(timezone.utc) - tick_start).total_seconds()
        sleep_for = max(0.5, 60.0 - elapsed)
        logger.info(f"Next tick in {sleep_for:.1f}s")
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
