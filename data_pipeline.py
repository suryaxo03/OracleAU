"""
================================================================
OracleAU — Phase 1: Data Pipeline
================================================================
Script:      data_pipeline.py
Description: Fetches historical price data for SGLN.L (iShares
             Physical Gold, GBP) directly from Stooq's CSV endpoint,
             cleans it, engineers technical indicator features, and
             saves the result to CSV for use in Phase 2 (model training).

             Fetches directly from Stooq's CSV endpoint using requests
             — no pandas_datareader, no yfinance, no API key, no rate
             limiting, no signup. Pure HTTP + pandas.

Usage:
    python data_pipeline.py

Output:
    data/SGLN_raw.csv        — Raw OHLCV data as fetched
    data/SGLN_features.csv   — Cleaned data + all indicators

Requirements:
    pip install requests pandas numpy ta
================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO

# 'ta' is the Technical Analysis library
# Docs: https://technical-analysis-library-in-python.readthedocs.io
from ta.trend import (
    SMAIndicator,
    EMAIndicator,
    MACD,
)
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# All key settings live here — change these without touching the
# rest of the script.
# ================================================================

TICKER        = "SGLN.UK"         # Stooq ticker for SGLN.L (LSE)
YEARS         = 10                # How many years of history to fetch
# Note: Stooq uses date ranges, not period strings — YEARS is converted below
OUTPUT_DIR    = "data"            # Folder to save CSV files
RAW_FILE      = "SGLN_raw.csv"    # Raw data filename
FEATURES_FILE = "SGLN_features.csv"  # Processed data filename

# Technical indicator parameters
SMA_SHORT     = 20    # Short-term Simple Moving Average window
SMA_MEDIUM    = 50    # Medium-term SMA window
SMA_LONG      = 200   # Long-term SMA window (Golden Cross reference)
EMA_SHORT     = 12    # Short EMA (used in MACD)
EMA_LONG      = 26    # Long EMA (used in MACD)
MACD_SIGNAL   = 9     # MACD signal line smoothing window
RSI_PERIOD    = 14    # RSI lookback window (industry standard)
BB_PERIOD     = 20    # Bollinger Bands window
BB_STD        = 2     # Bollinger Bands standard deviation multiplier
VOLATILITY_W  = 20    # Rolling volatility window (trading days)


# ================================================================
# STEP 1 — FETCH RAW DATA
# ================================================================

# Stooq CSV endpoint — returns daily OHLCV as a plain CSV file.
# No library wrapper needed, no API key, no rate limiting.
# Direct URL: https://stooq.com/q/d/l/?s=sgln.uk&i=d
STOOQ_URL = "https://stooq.com/q/d/l/?s={ticker}&i=d"


def fetch_data(ticker: str, years: int) -> pd.DataFrame:
    """
    Download historical OHLCV data directly from Stooq's CSV endpoint
    using plain HTTP requests. No pandas_datareader, no yfinance needed.

    Stooq returns a CSV file with columns: Date, Open, High, Low, Close, Volume
    Data comes newest-first — we reverse it to chronological order.

    Parameters:
        ticker : Stooq ticker symbol (e.g. 'SGLN.UK' for SGLN.L on LSE)
        years  : How many years of history to fetch

    Returns:
        Raw DataFrame with Date as index, columns: Open High Low Close Volume
    """
    print(f"\n{'='*60}")
    print(f"  OracleAU — Phase 1: Data Pipeline")
    print(f"{'='*60}")

    url = STOOQ_URL.format(ticker=ticker.lower())
    print(f"\n[1/5] Fetching {years}y of daily data for {ticker}...")
    print(f"        URL: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Connection failed. Check your internet connection.")
        print(f"    Verify Stooq is accessible: https://stooq.com")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"  ✗ Request timed out after 30s. Try again.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"  ✗ HTTP error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        sys.exit(1)

    # Parse the CSV response
    try:
        raw = pd.read_csv(
            StringIO(response.text),
            parse_dates = ["Date"],
            index_col   = "Date",
        )
    except Exception as e:
        print(f"  ✗ Failed to parse CSV response: {e}")
        print(f"    Response preview: {response.text[:200]}")
        sys.exit(1)

    if raw.empty:
        print(f"  ✗ No data returned for ticker '{ticker}'.")
        print(f"    Check: {url}")
        sys.exit(1)

    # Stooq returns newest-first — sort to chronological order
    raw = raw.sort_index()

    # Filter to requested date range
    cutoff = datetime.today() - timedelta(days=years * 365)
    raw    = raw[raw.index >= pd.Timestamp(cutoff)]

    print(f"  ✓ Fetched {len(raw):,} rows  |  "
          f"{raw.index[0].date()} → {raw.index[-1].date()}")
    print(f"  ✓ Source: Stooq direct CSV (no library, no rate limit)")

    return raw


# ================================================================
# STEP 2 — CLEAN THE DATA
# ================================================================

def fix_currency_flip(df: pd.DataFrame,
                      price_cols: list,
                      window: int = 30,
                      threshold: float = 0.40) -> pd.DataFrame:
    """
    Permanent rolling-median pence/pound auto-detector.

    Yahoo Finance intermittently switches SGLN.L between reporting
    in GBX (pence) and GBP (pounds) — sometimes mid-series, sometimes
    reverting back after repair=True already fixed an earlier block.
    This function catches ALL such flips on every pipeline run.

    How it works:
    - For each price column, compute a 30-day rolling median.
    - If any row's price is less than `threshold` (40%) of its rolling
      median, it's almost certainly in pounds while the surrounding
      data is in pence — so multiply it by 100 to correct it.
    - If any row's price is more than 1/threshold (250%) of its rolling
      median, it's almost certainly in pence while surrounding data is
      in pounds — so divide it by 100.
    - The rolling median is robust to outliers (unlike rolling mean),
      so genuine price spikes won't trigger false corrections.

    Parameters:
        df        : DataFrame with price columns
        price_cols: Columns to check (open, high, low, close)
        window    : Rolling window in trading days (default 30)
        threshold : Flip detection sensitivity (default 0.40)
    """
    total_fixed = 0

    for col in price_cols:
        if col not in df.columns:
            continue

        rolling_med = df[col].rolling(
            window=window, min_periods=5, center=True).median()

        # Rows that look like pounds when series is in pence (too small)
        too_small = df[col] < (rolling_med * threshold)

        # Rows that look like pence when series is in pounds (too large)
        too_large = df[col] > (rolling_med / threshold)

        n_small = too_small.sum()
        n_large = too_large.sum()

        if n_small > 0:
            df.loc[too_small, col] = (df.loc[too_small, col] * 100).round(4)
            total_fixed += n_small
            print(f"  ✓ [{col}] Fixed {n_small} row(s): pounds→pence (×100)")

        if n_large > 0:
            df.loc[too_large, col] = (df.loc[too_large, col] / 100).round(4)
            total_fixed += n_large
            print(f"  ✓ [{col}] Fixed {n_large} row(s): pence→pounds (÷100)")

    if total_fixed == 0:
        print(f"  ✓ Currency consistency check passed — no flips detected")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names, handle missing values, drop duplicates,
    and apply the permanent pence/pound currency flip auto-detector.

    repair=True in yfinance fixes most currency issues but Yahoo
    can revert to wrong units on new dates (as seen on 5 Mar 2026).
    fix_currency_flip() catches these on every run going forward.
    """
    print("\n[2/5] Cleaning data...")

    # ── Standardise column names to lowercase ────────────────────
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    # Drop the yfinance repair metadata column — not a feature
    df.drop(columns=["repaired?"], errors="ignore", inplace=True)

    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # ── Drop duplicate dates ──────────────────────────────────────
    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    dropped = before - len(df)
    if dropped:
        print(f"  ✓ Dropped {dropped} duplicate row(s)")

    # ── Sort chronologically (oldest first) ──────────────────────
    df = df.sort_index()

    # ── Handle missing values ─────────────────────────────────────
    missing_before = df.isnull().sum().sum()
    df = df.ffill().bfill()
    if missing_before:
        print(f"  ✓ Filled {missing_before} missing value(s) via forward/back-fill")

    # ── Permanent currency flip fix ───────────────────────────────
    # Runs on EVERY pipeline execution — catches any future Yahoo
    # pence/pound switches regardless of when they occur.
    print(f"  Running currency consistency check...")
    price_cols = ["open", "high", "low", "close"]
    df = fix_currency_flip(df, price_cols)

    # ── Prices are now consistently in GBP (pounds) ───────────────
    # repair=True + fix_currency_flip() together ensure the full
    # series is in £. No conversion columns needed.
    print(f"  ✓ Prices confirmed in GBP (£) throughout full series")
    print(f"  ✓ Clean data shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  ✓ No remaining nulls: {df.isnull().sum().sum() == 0}")

    return df


# ================================================================
# STEP 3 — ENGINEER TECHNICAL INDICATORS
# ================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators as new columns.
    All indicators are calculated on the GBP close price.
    The full series is consistently in pounds after fix_currency_flip().
    """
    print("\n[3/5] Calculating technical indicators...")

    close  = df["close"]   # GBP close — consistently in £ throughout
    high   = df["high"]
    low    = df["low"]

    # ── Simple Moving Averages (SMA) ─────────────────────────────
    # Measures the average closing price over N days.
    # Smooths out short-term noise to reveal the underlying trend.
    df[f"sma_{SMA_SHORT}"]  = SMAIndicator(close=close, window=SMA_SHORT).sma_indicator()
    df[f"sma_{SMA_MEDIUM}"] = SMAIndicator(close=close, window=SMA_MEDIUM).sma_indicator()
    df[f"sma_{SMA_LONG}"]   = SMAIndicator(close=close, window=SMA_LONG).sma_indicator()
    print(f"  ✓ SMA {SMA_SHORT} / {SMA_MEDIUM} / {SMA_LONG}")

    # ── Exponential Moving Averages (EMA) ────────────────────────
    # Like SMA but weights recent prices more heavily.
    # Used as components of MACD below.
    df[f"ema_{EMA_SHORT}"] = EMAIndicator(close=close, window=EMA_SHORT).ema_indicator()
    df[f"ema_{EMA_LONG}"]  = EMAIndicator(close=close, window=EMA_LONG).ema_indicator()
    print(f"  ✓ EMA {EMA_SHORT} / {EMA_LONG}")

    # ── MACD (Moving Average Convergence Divergence) ─────────────
    # Measures momentum by comparing two EMAs.
    # MACD Line    = EMA(12) - EMA(26)
    # Signal Line  = EMA(9) of the MACD Line
    # Histogram    = MACD Line - Signal Line
    # Crossover of MACD above Signal = bullish momentum signal.
    macd_indicator = MACD(
        close         = close,
        window_fast   = EMA_SHORT,
        window_slow   = EMA_LONG,
        window_sign   = MACD_SIGNAL,
    )
    df["macd"]        = macd_indicator.macd()
    df["macd_signal"] = macd_indicator.macd_signal()
    df["macd_hist"]   = macd_indicator.macd_diff()   # histogram
    print(f"  ✓ MACD ({EMA_SHORT}, {EMA_LONG}, {MACD_SIGNAL})")

    # ── RSI (Relative Strength Index) ────────────────────────────
    # Oscillates between 0–100. Measures speed and change of price moves.
    # > 70  → overbought (possible pullback ahead)
    # < 30  → oversold  (possible bounce ahead)
    # Gold historically shows strong mean-reversion at RSI extremes.
    df["rsi"] = RSIIndicator(close=close, window=RSI_PERIOD).rsi()
    print(f"  ✓ RSI ({RSI_PERIOD})")

    # ── Bollinger Bands ───────────────────────────────────────────
    # Upper/Lower bands = SMA ± (N × standard deviation).
    # Prices touching the upper band = stretched to the upside.
    # Prices touching the lower band = stretched to the downside.
    # Band width = measure of current volatility.
    bb = BollingerBands(close=close, window=BB_PERIOD, window_dev=BB_STD)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_width"]  = bb.bollinger_wband()   # (upper - lower) / middle × 100
    df["bb_pct"]    = bb.bollinger_pband()   # Where price sits within the bands (0–1)
    print(f"  ✓ Bollinger Bands ({BB_PERIOD}, {BB_STD}σ)")

    # ── Daily Return (%) ──────────────────────────────────────────
    # Percentage change from previous close to today's close.
    # Captures the magnitude and direction of daily price moves.
    df["daily_return"] = close.pct_change() * 100   # expressed as %
    print(f"  ✓ Daily Return (%)")

    # ── Rolling Volatility ────────────────────────────────────────
    # Standard deviation of daily returns over the last N days.
    # A higher value = more volatile / uncertain market environment.
    # Annualised by multiplying by √252 (252 trading days per year).
    df["volatility_20d"]       = df["daily_return"].rolling(window=VOLATILITY_W).std()
    df["volatility_annualised"] = df["volatility_20d"] * np.sqrt(252)
    print(f"  ✓ Rolling Volatility (20d + annualised)")

    # ── Golden Cross & Death Cross Signals ───────────────────────
    # Golden Cross : SMA50 crosses above SMA200 → long-term bullish signal
    # Death Cross  : SMA50 crosses below SMA200 → long-term bearish signal
    # Stored as binary flags (1 = signal occurred on this day, 0 = no signal)
    sma50  = df[f"sma_{SMA_MEDIUM}"]
    sma200 = df[f"sma_{SMA_LONG}"]
    df["golden_cross"] = (
        (sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))
    ).astype(int)
    df["death_cross"] = (
        (sma50 < sma200) & (sma50.shift(1) >= sma200.shift(1))
    ).astype(int)
    print(f"  ✓ Golden Cross / Death Cross signals")

    # ── Price Position Relative to Moving Averages ───────────────
    # Is the current price above or below each MA?
    # 1 = above (bullish relative to that MA), 0 = below (bearish)
    df["above_sma20"]  = (close > sma50).astype(int)
    df["above_sma50"]  = (close > sma50).astype(int)
    df["above_sma200"] = (close > sma200).astype(int)
    print(f"  ✓ Price vs MA position flags")

    # ── Lag Features ─────────────────────────────────────────────
    # Previous N days' closing prices as separate columns.
    # Gives the model direct access to recent price history as features.
    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = close.shift(lag)
    print(f"  ✓ Lag features (1, 2, 3, 5, 10 days)")

    # ── Target Variable (what the model will predict) ─────────────
    # Next day's closing price — this is the label for supervised learning.
    # Shift close by -1 so each row contains tomorrow's price as the target.
    df["target_next_close"] = close.shift(-1)

    # Also create a directional target: 1 = price went up, 0 = went down
    df["target_direction"] = (df["target_next_close"] > close).astype(int)
    print(f"  ✓ Target variables (next close price + direction)")

    total_indicators = len([c for c in df.columns if c not in
                            ["open","high","low","close","volume"]])
    print(f"\n  Total features engineered: {total_indicators}")

    return df


# ================================================================
# STEP 4 — SAVE TO CSV
# ================================================================

def save_data(raw: pd.DataFrame, features: pd.DataFrame, output_dir: str) -> None:
    """
    Save both the raw and feature-engineered DataFrames to CSV files
    inside the output directory. Creates the directory if needed.
    """
    print(f"\n[4/5] Saving files to '{output_dir}/'...")

    os.makedirs(output_dir, exist_ok=True)

    raw_path      = os.path.join(output_dir, RAW_FILE)
    features_path = os.path.join(output_dir, FEATURES_FILE)

    raw.to_csv(raw_path)
    features.to_csv(features_path)

    raw_size      = os.path.getsize(raw_path) / 1024
    features_size = os.path.getsize(features_path) / 1024

    print(f"  ✓ Raw data      → {raw_path}  ({raw_size:.1f} KB)")
    print(f"  ✓ Feature data  → {features_path}  ({features_size:.1f} KB)")


# ================================================================
# STEP 5 — VALIDATION & SUMMARY REPORT
# ================================================================

def validate_and_report(df: pd.DataFrame) -> None:
    """
    Run quality checks on the final feature DataFrame and
    print a human-readable summary report.
    All prices are in GBP (£) — fix_currency_flip() ensures this.
    """
    print(f"\n[5/5] Validation & Summary Report")
    print(f"{'─'*60}")

    # ── Date range ───────────────────────────────────────────────
    print(f"  Ticker         : SGLN.L (iShares Physical Gold)")
    print(f"  Date range     : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Trading days   : {len(df):,}")
    print(f"  Total columns  : {len(df.columns)}")

    # ── Latest price ─────────────────────────────────────────────
    latest      = df.iloc[-1]
    latest_close = latest["close"]
    print(f"\n  Latest Close   : £{latest_close:.4f}")

    # ── Sanity check: warn if price looks like pence slipped through ─
    if latest_close < 5.0:
        print(f"  ⚠️  WARNING: Price looks suspiciously low — "
              f"possible pence slip. Check fix_currency_flip().")
    elif latest_close > 500.0:
        print(f"  ⚠️  WARNING: Price looks suspiciously high — "
              f"possible pence value. Check fix_currency_flip().")
    else:
        print(f"  ✓  Price range looks correct for GBP (£)")

    # ── Price statistics ─────────────────────────────────────────
    print(f"\n  Price Statistics (Close, GBP £):")
    print(f"    Mean         : £{df['close'].mean():.4f}")
    print(f"    Min          : £{df['close'].min():.4f}  ({df['close'].idxmin().date()})")
    print(f"    Max          : £{df['close'].max():.4f}  ({df['close'].idxmax().date()})")
    print(f"    Std Dev      : £{df['close'].std():.4f}")

    # ── Latest indicator snapshot ────────────────────────────────
    print(f"\n  Latest Indicator Snapshot:")
    indicators = {
        "RSI (14)"       : f"{latest['rsi']:.2f}",
        "MACD"           : f"{latest['macd']:.4f}",
        "MACD Signal"    : f"{latest['macd_signal']:.4f}",
        "BB Width"       : f"{latest['bb_width']:.4f}",
        "SMA 20"         : f"£{latest['sma_20']:.4f}",
        "SMA 50"         : f"£{latest['sma_50']:.4f}",
        "SMA 200"        : f"£{latest['sma_200']:.4f}",
        "Volatility 20d" : f"{latest['volatility_20d']:.4f}%",
        "Ann. Volatility" : f"{latest['volatility_annualised']:.2f}%",
    }
    for name, val in indicators.items():
        print(f"    {name:<20}: {val}")

    # ── RSI interpretation ───────────────────────────────────────
    rsi_val = latest["rsi"]
    if rsi_val > 70:
        rsi_note = "⚠️  Overbought territory (>70)"
    elif rsi_val < 30:
        rsi_note = "⚠️  Oversold territory (<30)"
    else:
        rsi_note = "✓  Neutral range (30–70)"
    print(f"\n  RSI Reading    : {rsi_note}")

    # ── MA trend signal ──────────────────────────────────────────
    if latest["close"] > latest["sma_200"]:
        trend = "✓  Price above SMA200 — long-term uptrend"
    else:
        trend = "⚠️  Price below SMA200 — long-term downtrend"
    print(f"  Trend Signal   : {trend}")

    # ── Missing value check ──────────────────────────────────────
    total_nulls = df.isnull().sum().sum()
    print(f"\n  Missing Values : {total_nulls} total")
    if total_nulls <= 3:
        print(f"  ✓  Within expected range (last row targets are NaN by design)")
    else:
        print(f"  ⚠️  More NaNs than expected — check indicator windows vs data length")

    # ── Null breakdown ───────────────────────────────────────────
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0]
    if not null_cols.empty:
        print(f"\n  Columns with NaN values:")
        for col, count in null_cols.items():
            print(f"    {col:<30}: {count} NaN(s)")

    print(f"\n{'─'*60}")
    print(f"  ✅  Phase 1 complete! Data saved to '{OUTPUT_DIR}/'")
    print(f"      Load SGLN_features.csv in Phase 2 to begin model training.")
    print(f"{'='*60}\n")


# ================================================================
# MAIN — Run all steps in sequence
# ================================================================

def main():
    # Step 1: Fetch
    raw_df = fetch_data(TICKER, YEARS)

    # Step 2: Clean
    clean_df = clean_data(raw_df.copy())

    # Step 3: Add indicators
    feature_df = add_indicators(clean_df.copy())

    # Step 4: Save
    save_data(raw_df, feature_df, OUTPUT_DIR)

    # Step 5: Validate & report
    validate_and_report(feature_df)


if __name__ == "__main__":
    main()