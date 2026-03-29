"""
BTC/USDT historical data collector for ML model training.

Collects the following from public APIs (no API key required):
- OHLCV 1m + volume delta from Binance Futures /fapi/v1/klines
- Funding rate from /fapi/v1/fundingRate (every 8h, ffilled to 1m)
- Fear & Greed Index from alternative.me (daily, ffilled to 1m)
- SPX close from yfinance ^GSPC (1h, ffilled to 1m)

Dependencies:
    pip install requests pandas yfinance tqdm

Usage:
    python collect_data.py

Output:
    data/btc_usdt_features.csv
"""

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm


FUTURES_BASE = "https://fapi.binance.com"
FEAR_GREED_URL = "https://api.alternative.me/fng/"
SYMBOL = "BTCUSDT"
KLINES_LIMIT = 1000
REQUEST_SLEEP = 0.2  # Seconds between requests to stay within Binance rate limits
OUTPUT_FILE = "data/btc_usdt_features.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get(url: str, params: dict, retries: int = 3) -> dict | list:
    """Make a GET request with exponential backoff on failure.

    Args:
        url: Full endpoint URL.
        params: Query string parameters.
        retries: Maximum retry attempts on request error.

    Returns:
        Parsed JSON response as dict or list.

    Raises:
        requests.HTTPError: When all retry attempts are exhausted.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(REQUEST_SLEEP)
            return response.json()
        except requests.RequestException as exc:
            wait_seconds = 2**attempt
            logger.warning(
                f"Request failed ({attempt + 1}/{retries}): {exc}. "
                f"Retrying in {wait_seconds}s"
            )
            time.sleep(wait_seconds)
    raise requests.HTTPError(f"All {retries} retries exhausted for {url}")


def fetch_ohlcv(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch 1-minute OHLCV candles from Binance Futures.

    Volume delta is derived from the takerBuyBaseAssetVolume field already
    present in each kline: delta = 2 * taker_buy_vol - total_vol. This is
    mathematically identical to aggregating raw aggTrades per minute, but
    requires ~260 requests instead of hundreds of thousands for 6 months.

    Args:
        start_ms: Range start in Unix milliseconds (UTC).
        end_ms: Range end in Unix milliseconds (UTC).

    Returns:
        DataFrame indexed by UTC timestamp with columns: open, high,
        low, close, volume, volume_delta.
    """
    url = f"{FUTURES_BASE}/fapi/v1/klines"
    rows = []
    current_start = start_ms
    total_minutes = (end_ms - start_ms) // 60_000

    with tqdm(total=total_minutes, desc="OHLCV 1m", unit="candles") as pbar:
        while current_start < end_ms:
            batch = _get(
                url,
                {
                    "symbol": SYMBOL,
                    "interval": "1m",
                    "startTime": current_start,
                    "endTime": end_ms,
                    "limit": KLINES_LIMIT,
                },
            )

            if not batch:
                break

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

            pbar.update(len(batch))

            if len(batch) < KLINES_LIMIT:
                break

            current_start = batch[-1][0] + 60_000

    logger.info(f"OHLCV: collected {len(rows)} candles")
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def fetch_funding_rate(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch funding rate history from Binance Futures.

    Funding is settled every 8 hours (~540 records for 6 months).
    Forward-filled to 1-minute resolution during the merge step.

    Args:
        start_ms: Range start in Unix milliseconds (UTC).
        end_ms: Range end in Unix milliseconds (UTC).

    Returns:
        DataFrame indexed by UTC timestamp with column: funding_rate.
    """
    url = f"{FUTURES_BASE}/fapi/v1/fundingRate"
    rows = []
    current_start = start_ms

    with tqdm(desc="Funding Rate", unit="records") as pbar:
        while current_start < end_ms:
            batch = _get(
                url,
                {
                    "symbol": SYMBOL,
                    "startTime": current_start,
                    "endTime": end_ms,
                    "limit": KLINES_LIMIT,
                },
            )

            if not batch:
                break

            for r in batch:
                rows.append(
                    {
                        "timestamp": r["fundingTime"],
                        "funding_rate": float(r["fundingRate"]),
                    }
                )

            pbar.update(len(batch))

            if len(batch) < KLINES_LIMIT:
                break

            current_start = batch[-1]["fundingTime"] + 1

    logger.info(f"Funding rate: collected {len(rows)} records")
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()



def fetch_fear_greed(days: int = 365) -> pd.DataFrame:
    """Fetch Fear & Greed Index from alternative.me.

    Returns daily values which are forward-filled to 1-minute resolution
    during the merge step.

    Args:
        days: Number of historical days to retrieve (max 365 for free tier).

    Returns:
        DataFrame indexed by UTC timestamp with column: fear_greed.
    """
    logger.info(f"Fear & Greed: fetching {days} days...")

    payload = _get(FEAR_GREED_URL, {"limit": days, "format": "json"})
    rows = [
        {"timestamp": int(r["timestamp"]), "fear_greed": int(r["value"])}
        for r in payload["data"]
    ]

    logger.info(f"Fear & Greed: collected {len(rows)} records")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.set_index("timestamp").sort_index()


def fetch_spx(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch S&P 500 hourly prices from Yahoo Finance via yfinance.

    The index only covers weekday trading hours; overnight and weekend gaps
    are forward-filled to 1-minute resolution during the merge step.

    Args:
        start_date: Start date string in "YYYY-MM-DD" format.
        end_date: End date string in "YYYY-MM-DD" format.

    Returns:
        DataFrame indexed by UTC timestamp with column: spx_close.
    """
    logger.info(f"SPX: fetching ^GSPC from {start_date} to {end_date}...")

    raw = yf.Ticker("^GSPC").history(start=start_date, end=end_date, interval="1h")

    if raw.empty:
        logger.warning("SPX: no data returned from yfinance")
        empty = pd.DataFrame(columns=["spx_close"])
        empty.index.name = "timestamp"
        return empty

    df = raw[["Close"]].rename(columns={"Close": "spx_close"})

    # yfinance may return a tz-naive or tz-aware index depending on the version
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("America/New_York").tz_convert("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "timestamp"
    logger.info(f"SPX: collected {len(df)} hourly records")
    return df.sort_index()


def merge_all(
    ohlcv: pd.DataFrame,
    funding: pd.DataFrame,
    fear_greed: pd.DataFrame,
    spx: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all data sources into a single 1-minute resolution DataFrame.

    The OHLCV index is the canonical time grid. Sparse sources (funding rate,
    Fear & Greed, SPX) are forward-filled into that grid.

    Args:
        ohlcv: 1-minute OHLCV data with volume_delta.
        funding: 8-hour funding rate records.
        fear_greed: Daily Fear & Greed Index values.
        spx: Hourly SPX close prices.

    Returns:
        Merged DataFrame with columns in canonical order:
        open, high, low, close, volume, funding_rate, volume_delta,
        fear_greed, spx_close.
    """
    logger.info("Merging all data sources...")

    merged = ohlcv.copy()
    base_index = merged.index

    def ffill_to_grid(df: pd.DataFrame) -> pd.DataFrame:
        """Reindex a sparse DataFrame onto the 1-minute grid via forward fill."""
        if df.empty:
            return pd.DataFrame(index=base_index, columns=df.columns)
        return (
            df.reindex(base_index.union(df.index))
            .sort_index()
            .ffill()
            .reindex(base_index)
        )

    merged["funding_rate"] = ffill_to_grid(funding)["funding_rate"]
    merged["fear_greed"] = ffill_to_grid(fear_greed)["fear_greed"]
    merged["spx_close"] = ffill_to_grid(spx)["spx_close"]

    column_order = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_rate",
        "volume_delta",
        "fear_greed",
        "spx_close",
    ]
    merged = merged[column_order]

    rows_before = len(merged)
    merged.dropna(subset=["open", "close", "volume"], inplace=True)
    dropped = rows_before - len(merged)

    if dropped:
        logger.info(f"Dropped {dropped} rows with missing core OHLCV data")

    logger.info(f"Final dataset: {merged.shape[0]} rows x {merged.shape[1]} columns")
    return merged


def main() -> None:
    """Orchestrate data collection, merge into a single DataFrame, and save."""
    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(days=180)

    end_ms = int(now.timestamp() * 1000)
    start_ms = int(start_dt.timestamp() * 1000)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    logger.info(f"Period: {start_date} → {end_date} | Symbol: {SYMBOL} | 1-minute")

    ohlcv = fetch_ohlcv(start_ms, end_ms)
    funding = fetch_funding_rate(start_ms, end_ms)
    fear_greed = fetch_fear_greed(days=365)
    spx = fetch_spx(start_date, end_date)

    dataset = merge_all(ohlcv, funding, fear_greed, spx)

    dataset.to_csv(OUTPUT_FILE)
    logger.info(f"Saved → {OUTPUT_FILE}")

    if len(dataset):
        logger.info(f"Date range: {dataset.index[0]} → {dataset.index[-1]}")
        logger.info(f"Preview:\n{dataset.head().to_string()}")


if __name__ == "__main__":
    main()
