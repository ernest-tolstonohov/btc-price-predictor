# BTC/USDT Multi-Horizon Price Prediction

A LightGBM pipeline that trains ten models across five prediction horizons to forecast BTC/USDT price movement and direction probability, with a live monitor that resolves predictions against real prices every minute.

## Results

### Regressor — Basis Point Move Prediction

| Horizon | Best Iter | MAE Train (bps) | MAE Test (bps) | Baseline MAE (bps) |
|---------|-----------|-----------------|----------------|--------------------|
| +1m     | 23        | 4.46            | 4.95           | 4.95               |
| +3m     | 2         | 7.71            | 8.59           | 8.59               |
| +5m     | 13        | 9.97            | 11.03          | 11.03              |
| +10m    | 3         | 14.04           | 15.70          | 15.70              |
| +15m    | 5         | 17.12           | 19.29          | 19.29              |

Baseline is always predicting 0 bps. Early stopping triggered at 2–23 iterations across all horizons: the models collapse to near-constant predictions because OHLCV-derived features carry no recoverable magnitude signal at these horizons on a liquid perpetual futures market.

### Classifier — Direction Probability

| Horizon | Best Iter | Accuracy | ROC-AUC | Baseline | Acc when reg+cls agree |
|---------|-----------|----------|---------|----------|------------------------|
| +1m     | 372       | 51.8%    | 0.528   | 50.1%    | 52.2%                  |
| +3m     | 270       | 52.5%    | 0.537   | 50.2%    | 53.1%                  |
| +5m     | 253       | 52.6%    | 0.536   | 50.3%    | 52.7%                  |
| +10m    | 185       | 52.7%    | 0.539   | 50.4%    | 53.4%                  |
| +15m    | 143       | 53.1%    | 0.542   | 50.6%    | 53.9%                  |

The classifiers trained to 143–372 iterations and hold a consistent 2–3 percentage point edge over the majority-class baseline. When the regressor and classifier agree on direction, accuracy reaches 52–54% across all horizons.

**Key finding:** Public OHLCV data on a high-liquidity perpetual futures market hits a signal ceiling at short horizons. Any exploitable pattern in candlestick data is arbitraged away before it persists long enough to predict magnitude. The classifier captures a weak but real directional signal; the regressor confirms that magnitude is not predictable from this feature set alone. The next version incorporates order book imbalance and liquidation flow, which are private to a given market snapshot and cannot be pre-arbitraged in the same way.

## Architecture

```
Data Sources
  Binance Futures REST  ──► OHLCV 1m, funding rate
  alternative.me REST   ──► Fear & Greed Index (daily)
  Yahoo Finance         ──► SPX close (hourly)
         │
         ▼
collect_data.py ──► data/btc_usdt_features.csv (259,200 rows x 9 cols)
         │
         ▼
train.py: feature engineering (29 features, no look-ahead)
         │
         ├──► 5 x LGBMRegressor  ──► model_reg_{N}m.pkl  (bps target)
         └──► 5 x LGBMClassifier ──► model_cls_{N}m.pkl  (direction target)
                                      features.pkl
                                      model_meta.pkl
         │
         ▼
monitor.py
  make_predictions(row) ──► { horizon: { price, bps, prob_up, signal } }
                              signal in { long, short, neutral }
```

## Project Structure

```
btc-price-predictor/
  v1/
    models/           10 model pkl files, features.pkl, model_meta.pkl
    data/             .gitkeep — CSV excluded from git (259k rows, 6 months)
    results/          results.csv written by monitor.py
    collect_data.py   Fetches 6 months of 1m data from public APIs
    train.py          Feature engineering, training, evaluation, artifact saving
    monitor.py        Runs every minute, resolves predictions, tracks rolling MAE
    analyze.py        Post-training diagnostics: correlation, calibration, signal agreement
  requirements.txt
  .gitignore
  README.md
```

## Quickstart

```bash
git clone https://github.com/ernest-tolstonohov/btc-price-predictor
cd btc-price-predictor

# Create and activate virtual environment
python -m venv env
source env/bin/activate        # Linux / macOS
env\Scripts\activate           # Windows

pip install -r requirements.txt

cd v1

# Collect ~6 months of 1-minute data (takes ~5 minutes)
python collect_data.py

# Train all ten models
python train.py

# Live monitor — runs every minute, resolves predictions, writes results.csv
python monitor.py
```

## What's Next

- **Order book imbalance** — bid/ask depth ratio from Binance WebSocket `@depth` stream. Snapshot data is unavailable in historical REST endpoints, so this requires a collection daemon running forward in time.
- **Liquidation flow** — real-time forced-close volume from the `@forceOrder` WebSocket stream, the primary driver of cascading price moves.
- **Aggressive volume** — buy/sell ratio from `@aggTrade` stream aggregated over 15s, 30s, and 60s windows.
- **v2 model comparison** — retrain with order book features and compare classifier AUC and combined-signal accuracy against the v1 baselines documented here.

## Technical Notes

**Chronological split and leakage prevention.** The 80/20 train/test split is strictly chronological with no shuffling. All rolling features (EMA, RSI, ATR, Bollinger Bands) are computed over the full series before splitting, but use only backward-looking windows so no future information enters any training row. Forward-fill of funding rate and SPX close is applied before the split using only past observations.

**Naive baseline comparison.** Every regressor result is reported alongside the MAE of a model that always predicts 0 bps. This is the correct baseline for a zero-mean return series and directly shows whether the model adds value over predicting no movement.

**Combined signal logic.** `make_predictions()` emits a `signal` field per horizon: `long` when predicted bps > 0 and classifier P(up) > 0.55, `short` when predicted bps < 0 and P(up) < 0.45, and `neutral` otherwise. The asymmetric thresholds require both models to agree with confidence before generating a directional signal. When models disagree, accuracy falls to 47–48%, below random — so neutral is the correct output.

**Live prediction resolution.** `monitor.py` timestamps every prediction at the candle it was made on, stores the target candle timestamp, and resolves the prediction when that candle's close price is observed. MAE is tracked as a rolling mean per horizon and written to `results.csv` for offline analysis.

**Feature engineering.** All 29 features are computed from scratch using pandas and numpy without external TA libraries. Absolute price levels are excluded entirely — only scale-invariant derivatives are used (log returns, normalized ranges, EMA crossovers divided by close). This prevents the model from learning price-level patterns that do not generalize across the dataset's price range.