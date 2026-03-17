"""
================================================================
OracleAU — Phase 3: FastAPI Backend
================================================================
Script:      api.py
Description: REST API that serves live SGLN.L price data,
             technical indicators, and 7-day ensemble forecasts
             (XGBoost v2 + LSTM v2) to the Phase 4 frontend.

Endpoints:
    GET /health              — API + model status
    GET /api/price           — Live SGLN price + OHLCV
    GET /api/forecast        — 7-day ensemble forecast + confidence
    GET /api/history         — Historical OHLCV for charting
    GET /api/indicators      — Latest technical indicator values
    GET /api/accuracy        — Past forecast vs actual accuracy
    GET /api/stats           — Visit counts and usage stats

Usage (local):
    uvicorn api:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs

Requirements:
    pip install fastapi uvicorn[standard] pydantic pandas-datareader
    (pandas, numpy, sklearn, xgboost, tensorflow
     already installed from Phase 2)
================================================================
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional
import requests

import numpy as np
import pandas as pd
import joblib
from io import StringIO

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Technical indicators
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# TensorFlow — suppress verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

# MongoDB
try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    # Note: log not yet defined here — warning emitted at startup instead

# ================================================================
# LOGGING
# ================================================================

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("oracleau")


# ================================================================
# ENVIRONMENT — load .env file when running locally
# ================================================================
# python-dotenv reads a .env file in the project root when present.
# On Render, real environment variables are used instead.
# The .env file is gitignored — never committed to the repo.

try:
    from dotenv import load_dotenv
    load_dotenv()   # Loads .env if present, silently skips if not found
except ImportError:
    pass            # python-dotenv not installed — fine on Render

# ================================================================
# CONFIGURATION
# ================================================================

TICKER_STOOQ = "SGLN.UK"   # Stooq ticker (LSE stocks use .UK suffix)
TICKER       = "SGLN.L"    # Display name (how it appears to users)
STOOQ_URL    = "https://stooq.com/q/d/l/?s={ticker}&i=d"
MODELS_DIR        = "models"
FORECAST_LOG_PATH = os.path.join(MODELS_DIR, "forecast_log.json")

# ── MongoDB ───────────────────────────────────────────────────────
# LOCAL:  Set MONGO_URI in your .env file (see .env.example)
# RENDER: Set MONGO_URI in Render dashboard → Environment Variables
MONGO_URI         = os.environ.get("MONGO_URI", "")
MONGO_DB_NAME     = "oracleau"
SEQUENCE_LEN = 45
FORECAST_DAYS = 7

# Ensemble weights (from model_metadata_v2.json)
ENSEMBLE_WEIGHT_XGB  = 0.40
ENSEMBLE_WEIGHT_LSTM = 0.60
CONFIDENCE_THRESHOLD = 0.50   # % difference → HIGH confidence boundary

# Cache settings — avoids hammering Yahoo Finance
CACHE_TTL_SECONDS = 300   # 5 minutes

# Technical indicator parameters (must match Phase 1)
SMA_SHORT   = 20
SMA_MEDIUM  = 50
SMA_LONG    = 200
EMA_SHORT   = 12
EMA_LONG    = 26
MACD_SIGNAL = 9
RSI_PERIOD  = 14
BB_PERIOD   = 20
BB_STD      = 2

# Lag feature columns (must match Phase 2 — excluded from XGBoost)
LAG_COLS = [
    "close_lag_1", "close_lag_2", "close_lag_3",
    "close_lag_5", "close_lag_10",
]

# Columns that are targets / must not enter X
LEAK_COLS = [
    "target_next_close", "target_direction",
    "return_target", "close_raw", "repaired?",
]


# ================================================================
# GLOBAL STATE — models, scalers, cache
# ================================================================

class AppState:
    """Holds all loaded models, scalers, and the data cache."""

    def __init__(self):
        # Models
        self.xgb_model   = None
        self.lstm_model  = None
        self.scaler_X_xgb  = None
        self.scaler_X_lstm = None
        self.scaler_y      = None
        self.metadata      = {}
        self.models_loaded = False

        # Data cache
        self._cache_df        : Optional[pd.DataFrame] = None
        self._cache_timestamp : Optional[datetime]     = None
        self._cache_stale     : bool                   = False

    def is_cache_valid(self) -> bool:
        if self._cache_df is None or self._cache_timestamp is None:
            return False
        age = (datetime.now() - self._cache_timestamp).total_seconds()
        return age < CACHE_TTL_SECONDS

    def set_cache(self, df: pd.DataFrame) -> None:
        self._cache_df        = df.copy()
        self._cache_timestamp = datetime.now()
        self._cache_stale     = False
        log.info(f"Cache updated — {len(df)} rows, "
                 f"latest: {df.index[-1].date()}")

    def get_cache(self) -> Optional[pd.DataFrame]:
        return self._cache_df.copy() if self._cache_df is not None else None

    def mark_stale(self) -> None:
        self._cache_stale = True

    @property
    def cache_age_seconds(self) -> Optional[float]:
        if self._cache_timestamp is None:
            return None
        return (datetime.now() - self._cache_timestamp).total_seconds()


state = AppState()


# ================================================================
# FASTAPI APP & CORS
# ================================================================

app = FastAPI(
    title       = "OracleAU API",
    description = "SGLN.L gold ETF price forecasting — XGBoost v2 + LSTM v2 ensemble",
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# CORS — allow requests from GitHub Pages frontend and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        "http://localhost:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://*.github.io",      # GitHub Pages (any repo)
        "*",                        # Open during development — tighten for prod
    ],
    allow_credentials = True,
    allow_methods     = ["GET"],
    allow_headers     = ["*"],
)


# ================================================================
# STARTUP — Load models on first request
# ================================================================

@app.on_event("startup")
async def startup_event():
    """Load models and connect to MongoDB when the server starts."""
    log.info("=" * 55)
    log.info("  OracleAU API — Starting up")
    log.info("=" * 55)

    # MongoDB connection
    if not MONGO_AVAILABLE:
        log.warning("pymongo not installed — database features disabled")
    elif not MONGO_URI:
        log.warning("MONGO_URI not set — set it in .env (local) or Render dashboard (deploy)")
    else:
        get_db()   # Eagerly connect so first request isn't slow

    # Model loading — logs working directory to help debug path issues
    log.info(f"Working directory: {os.getcwd()}")
    log.info(f"Models directory:  {os.path.abspath(MODELS_DIR)}")
    load_models()
    log.info("Startup complete. Ready to serve requests.")


def load_models() -> None:
    """
    Load XGBoost, LSTM, scalers, and metadata from the models/ directory.
    Sets state.models_loaded = True on success.
    Logs a warning (does not crash) if files are missing.
    """
    required_files = {
        "xgboost"      : os.path.join(MODELS_DIR, "xgboost_v2.pkl"),
        "lstm"         : os.path.join(MODELS_DIR, "lstm_v2.h5"),
        "scaler_X_xgb" : os.path.join(MODELS_DIR, "scaler_X_xgb_v2.pkl"),
        "scaler_X_lstm": os.path.join(MODELS_DIR, "scaler_X_lstm_v2.pkl"),
        "scaler_y"     : os.path.join(MODELS_DIR, "scaler_y_v2.pkl"),
        "metadata"     : os.path.join(MODELS_DIR, "model_metadata_v2.json"),
    }

    missing = [k for k, p in required_files.items() if not os.path.exists(p)]
    if missing:
        log.warning(f"Missing model files: {missing}. "
                    f"Run model_training_v2.py first.")
        log.warning("API will start but /api/forecast will be unavailable.")
        return

    try:
        state.xgb_model    = joblib.load(required_files["xgboost"])
        state.lstm_model   = load_model(required_files["lstm"], compile=False)
        state.scaler_X_xgb  = joblib.load(required_files["scaler_X_xgb"])
        state.scaler_X_lstm = joblib.load(required_files["scaler_X_lstm"])
        state.scaler_y      = joblib.load(required_files["scaler_y"])

        with open(required_files["metadata"]) as f:
            state.metadata = json.load(f)

        # Read ensemble weights from metadata if present
        global ENSEMBLE_WEIGHT_XGB, ENSEMBLE_WEIGHT_LSTM, CONFIDENCE_THRESHOLD
        ew = state.metadata.get("ensemble_weights", {})
        if ew:
            ENSEMBLE_WEIGHT_XGB  = ew.get("xgboost", ENSEMBLE_WEIGHT_XGB)
            ENSEMBLE_WEIGHT_LSTM = ew.get("lstm",    ENSEMBLE_WEIGHT_LSTM)
        CONFIDENCE_THRESHOLD = state.metadata.get(
            "confidence_threshold", CONFIDENCE_THRESHOLD)

        state.models_loaded = True
        best = state.metadata.get("best_model", "unknown").upper()
        log.info(f"✓ XGBoost v2 loaded")
        log.info(f"✓ LSTM v2 loaded")
        log.info(f"✓ Scalers loaded (X_xgb, X_lstm, y)")
        log.info(f"✓ Metadata loaded — best model: {best}")
        log.info(f"  Ensemble weights: XGB={ENSEMBLE_WEIGHT_XGB}  "
                 f"LSTM={ENSEMBLE_WEIGHT_LSTM}")

    except Exception as e:
        log.error(f"Failed to load models: {e}")
        state.models_loaded = False


# ================================================================
# DATA FETCHING & CACHING
# ================================================================

def fetch_sgln_data(years: int = 2) -> pd.DataFrame:
    """
    Fetch SGLN.L OHLCV data directly from Stooq's CSV endpoint.

    Why Stooq instead of yfinance:
    - No API key required
    - No rate limiting (no more YFRateLimitError)
    - No signup or account needed
    - Reliable for LSE stocks using the .UK ticker suffix

    Also applies:
    - Rolling median currency flip detector (catches pence/pound switches)
    - 5-minute in-memory cache so repeated API calls are instant

    Raises HTTPException(503) if Stooq is unreachable.
    """
    # Return cached data if still fresh
    if state.is_cache_valid():
        log.debug("Serving from cache")
        return state.get_cache()

    url = STOOQ_URL.format(ticker=TICKER_STOOQ.lower())
    log.info(f"Fetching fresh data from Stooq: {url}")

    # ── Step 1: Fetch from Stooq ─────────────────────────────────
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        raw = pd.read_csv(
            StringIO(response.text),
            parse_dates = ["Date"],
            index_col   = "Date",
        )
    except Exception as e:
        stale = state.get_cache()
        if stale is not None:
            log.warning(f"Stooq fetch failed ({e}) — serving stale cache")
            state.mark_stale()
            return stale
        log.error(f"Stooq fetch failed with no cache to fall back on: {e}")
        raise HTTPException(
            status_code = 503,
            detail = f"Unable to fetch data from Stooq: {e}"
        )

    if raw is None or raw.empty:
        stale = state.get_cache()
        if stale is not None:
            log.warning("Empty Stooq response — serving stale cache")
            state.mark_stale()
            return stale
        raise HTTPException(
            status_code = 503,
            detail = f"No data returned for {TICKER_STOOQ} from Stooq.",
        )

    # ── Step 2: Sort and validate ─────────────────────────────────
    raw = raw.sort_index()
    log.info(f"✓ Fetched {len(raw)} rows  "
             f"({raw.index[0].date()} → {raw.index[-1].date()})")

    # ── Step 3: Clean, indicators, targets ───────────────────────
    try:
        log.info("Processing: _clean_raw...")
        df = _clean_raw(raw)
        log.info("Processing: _add_indicators...")
        df = _add_indicators(df)
        log.info("Processing: _add_targets...")
        df = _add_targets(df)
    except Exception as e:
        import traceback
        log.error(f"Data processing failed: {e}")
        log.error(traceback.format_exc())
        stale = state.get_cache()
        if stale is not None:
            log.warning("Processing failed — serving stale cache")
            state.mark_stale()
            return stale
        raise HTTPException(
            status_code = 503,
            detail = f"Data processing error: {e}"
        )

    state.set_cache(df)
    return df


def _clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise, deduplicate, fill gaps, fix currency units.

    Stooq returns SGLN.UK in GBX (pence) — the entire series is in
    pence so a rolling-median detector sees nothing anomalous.
    Instead we use an absolute threshold:
      - SGLN in GBP has traded between £15 and £200 since 2016
      - Any median close above £500 means the series is in pence
      - Divide all OHLC columns by 100 to convert to pounds
    This is unconditional and reliable regardless of data source.
    """

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.drop(columns=["repaired?", "Repaired?"], errors="ignore", inplace=True)

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.ffill().bfill()

    # ── Absolute currency correction ─────────────────────────────
    # If the median close is above 500, the full series is in pence.
    # Divide all OHLC columns by 100 unconditionally in that case.
    # Threshold 500 is safely between:
    #   - Max GBP price ever (~£200)  → well below 500
    #   - Min GBX price ever (~1500p) → well above 500
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    median_close = df["close"].median()

    if median_close > 500:
        for col in price_cols:
            df[col] = (df[col] / 100).round(4)
        log.info(f"Currency: series was in pence (median={median_close:.0f}p) "
                 f"— divided all OHLC by 100 → GBP "
                 f"(new median: £{df['close'].median():.2f})")
    else:
        log.info(f"Currency: series already in GBP "
                 f"(median close: £{median_close:.2f})")

    return df


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators. Must match Phase 1 exactly."""
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    df[f"sma_{SMA_SHORT}"]  = SMAIndicator(close, SMA_SHORT).sma_indicator()
    df[f"sma_{SMA_MEDIUM}"] = SMAIndicator(close, SMA_MEDIUM).sma_indicator()
    df[f"sma_{SMA_LONG}"]   = SMAIndicator(close, SMA_LONG).sma_indicator()
    df[f"ema_{EMA_SHORT}"]  = EMAIndicator(close, EMA_SHORT).ema_indicator()
    df[f"ema_{EMA_LONG}"]   = EMAIndicator(close, EMA_LONG).ema_indicator()

    macd_ind = MACD(close, EMA_SHORT, EMA_LONG, MACD_SIGNAL)
    df["macd"]        = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"]   = macd_ind.macd_diff()

    df["rsi"] = RSIIndicator(close, RSI_PERIOD).rsi()

    bb = BollingerBands(close, BB_PERIOD, BB_STD)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_width"]  = bb.bollinger_wband()
    df["bb_pct"]    = bb.bollinger_pband()

    df["daily_return"]          = close.pct_change() * 100
    df["volatility_20d"]        = df["daily_return"].rolling(20).std()
    df["volatility_annualised"] = df["volatility_20d"] * np.sqrt(252)

    sma50  = df[f"sma_{SMA_MEDIUM}"]
    sma200 = df[f"sma_{SMA_LONG}"]
    df["golden_cross"] = ((sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))).astype(int)
    df["death_cross"]  = ((sma50 < sma200) & (sma50.shift(1) >= sma200.shift(1))).astype(int)
    df["above_sma20"]  = (close > df[f"sma_{SMA_SHORT}"]).astype(int)
    df["above_sma50"]  = (close > sma50).astype(int)
    df["above_sma200"] = (close > sma200).astype(int)

    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = close.shift(lag)

    return df


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add target columns (needed to keep feature column order consistent)."""
    df["target_next_close"] = df["close"].shift(-1)
    df["target_direction"]  = (df["target_next_close"] > df["close"]).astype(int)
    return df


# ================================================================
# FEATURE PREPARATION FOR INFERENCE
# ================================================================

def prepare_inference_sequences(df: pd.DataFrame):
    """
    Extract the last SEQUENCE_LEN rows as scaled input sequences
    for both XGBoost and LSTM.

    Returns:
        seq_xgb  : np.ndarray (1, seq_len × n_features_xgb) — flattened
        seq_lstm : np.ndarray (1, seq_len, n_features_lstm)  — 3D
        last_close : float — most recent actual close price
        last_date  : date  — most recent trading date
    """
    if not state.models_loaded:
        raise HTTPException(503, "Models not loaded. Run model_training_v2.py first.")

    # Drop NaN rows — same as Phase 2 training
    clean = df.dropna(subset=[c for c in df.columns
                               if c not in LEAK_COLS]).copy()

    last_close = float(clean["close"].iloc[-1])
    last_date  = clean.index[-1].date()

    # Feature columns — must exactly match what the scalers were fitted on
    meta_cols_xgb  = state.metadata.get("feature_cols_xgb",  [])
    meta_cols_lstm = state.metadata.get("feature_cols_lstm", [])

    # Fall back to deriving feature cols if metadata is missing
    all_cols = [c for c in clean.columns if c not in LEAK_COLS]
    if not meta_cols_xgb:
        meta_cols_xgb  = [c for c in all_cols if c not in LAG_COLS]
    if not meta_cols_lstm:
        meta_cols_lstm = all_cols

    # Ensure all required columns exist
    missing_xgb  = [c for c in meta_cols_xgb  if c not in clean.columns]
    missing_lstm = [c for c in meta_cols_lstm if c not in clean.columns]
    if missing_xgb or missing_lstm:
        raise HTTPException(500, f"Missing feature columns — "
                                 f"XGB: {missing_xgb}, LSTM: {missing_lstm}")

    # Extract last SEQUENCE_LEN rows and scale
    window_xgb  = clean[meta_cols_xgb].iloc[-SEQUENCE_LEN:].values
    window_lstm = clean[meta_cols_lstm].iloc[-SEQUENCE_LEN:].values

    if len(window_xgb) < SEQUENCE_LEN:
        raise HTTPException(500, f"Not enough data — need {SEQUENCE_LEN} rows, "
                                 f"got {len(window_xgb)}")

    scaled_xgb  = state.scaler_X_xgb.transform(window_xgb)
    scaled_lstm = state.scaler_X_lstm.transform(window_lstm)

    # XGBoost: flatten to (1, seq_len × features)
    seq_xgb  = scaled_xgb.flatten().reshape(1, -1)

    # LSTM: keep as (1, seq_len, features)
    seq_lstm = scaled_lstm.reshape(1, SEQUENCE_LEN, scaled_lstm.shape[1])

    return seq_xgb, seq_lstm, scaled_xgb, scaled_lstm, last_close, last_date


# ================================================================
# FORECASTING ENGINE
# ================================================================

def run_forecast(df: pd.DataFrame) -> dict:
    """
    Run both models for FORECAST_DAYS steps and compute ensemble.
    Uses recursive multi-step forecasting — each prediction feeds
    back as input for the next step.
    """
    (seq_xgb, seq_lstm,
     scaled_xgb, scaled_lstm,
     last_close, last_date) = prepare_inference_sequences(df)

    # Running sequences — updated each step
    cur_xgb  = scaled_xgb.copy()    # (seq_len, n_feat_xgb)
    cur_lstm = scaled_lstm.copy()   # (seq_len, n_feat_lstm)

    prices_xgb  = []
    prices_lstm = []
    current_price = last_close

    for day in range(FORECAST_DAYS):
        # ── XGBoost prediction ───────────────────────────────────
        xgb_inp  = cur_xgb.flatten().reshape(1, -1)
        xgb_pred_s = float(state.xgb_model.predict(xgb_inp)[0])

        # ── LSTM prediction ──────────────────────────────────────
        lstm_inp    = cur_lstm.reshape(1, SEQUENCE_LEN, cur_lstm.shape[1])
        lstm_pred_s = float(state.lstm_model.predict(lstm_inp, verbose=0)[0][0])

        # ── Inverse-transform scaled return → % return → price ──
        xgb_return  = float(state.scaler_y.inverse_transform(
            np.array([[xgb_pred_s]])).flatten()[0])
        lstm_return = float(state.scaler_y.inverse_transform(
            np.array([[lstm_pred_s]])).flatten()[0])

        xgb_price  = round(current_price * (1 + xgb_return  / 100), 4)
        lstm_price = round(current_price * (1 + lstm_return / 100), 4)

        prices_xgb.append(xgb_price)
        prices_lstm.append(lstm_price)

        # ── Roll sequences forward ───────────────────────────────
        new_row_xgb       = cur_xgb[-1].copy()
        new_row_xgb[0]    = xgb_pred_s
        cur_xgb           = np.vstack([cur_xgb[1:],  new_row_xgb])

        new_row_lstm      = cur_lstm[-1].copy()
        new_row_lstm[0]   = lstm_pred_s
        cur_lstm          = np.vstack([cur_lstm[1:], new_row_lstm])

        # Use ensemble price as the base for next day's return calculation
        ens_price     = round(
            ENSEMBLE_WEIGHT_XGB  * xgb_price +
            ENSEMBLE_WEIGHT_LSTM * lstm_price, 4)
        current_price = ens_price

    # ── Build forecast dates (business days only) ────────────────
    from pandas.tseries.offsets import BDay
    future_dates = pd.date_range(
        start=pd.Timestamp(last_date) + BDay(1),
        periods=FORECAST_DAYS, freq="B")

    # ── Compute ensemble + confidence per day ───────────────────
    forecast_days = []
    prev_price    = last_close

    for i, (d, xp, lp) in enumerate(zip(future_dates, prices_xgb, prices_lstm)):
        ens_price = round(
            ENSEMBLE_WEIGHT_XGB  * xp +
            ENSEMBLE_WEIGHT_LSTM * lp, 4)
        pct_diff  = round(abs(xp - lp) / lp * 100, 4)

        if pct_diff   <= CONFIDENCE_THRESHOLD:
            confidence = "HIGH"
        elif pct_diff <= CONFIDENCE_THRESHOLD * 3:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        direction = ("UP"      if ens_price > prev_price
                     else "DOWN" if ens_price < prev_price
                     else "NEUTRAL")
        change_pct = round((ens_price - last_close) / last_close * 100, 4)

        forecast_days.append({
            "day"           : i + 1,
            "date"          : str(d.date()),
            "xgboost"       : xp,
            "lstm"          : lp,
            "ensemble"      : ens_price,
            "agreement_pct" : pct_diff,
            "confidence"    : confidence,
            "direction"     : direction,
            "change_pct"    : change_pct,
        })
        prev_price = ens_price

    # ── Overall signal from first 3 days ────────────────────────
    tier_rank  = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    overall_confidence = max(
        [d["confidence"] for d in forecast_days[:3]],
        key=lambda x: tier_rank[x])

    d1_change = forecast_days[0]["ensemble"] - last_close
    overall_direction = (
        "UP"      if d1_change >  0.05
        else "DOWN" if d1_change < -0.05
        else "NEUTRAL")

    return {
        "last_close"          : round(last_close, 4),
        "last_date"           : str(last_date),
        "overall_direction"   : overall_direction,
        "overall_confidence"  : overall_confidence,
        "ensemble_weights"    : {
            "xgboost": ENSEMBLE_WEIGHT_XGB,
            "lstm"   : ENSEMBLE_WEIGHT_LSTM,
        },
        "forecast"            : forecast_days,
        "generated_at"        : datetime.now().isoformat(),
        "stale_data"          : state._cache_stale,
    }



# ================================================================
# DATABASE — MongoDB Atlas connection + helpers
# ================================================================

_mongo_client = None
_mongo_db     = None


def get_db():
    """
    Return MongoDB database handle.
    Lazily initialises the connection on first call.
    Returns None gracefully if MONGO_URI is not set or connection fails.
    """
    global _mongo_client, _mongo_db

    if _mongo_db is not None:
        return _mongo_db

    if not MONGO_AVAILABLE or not MONGO_URI:
        log.warning("MongoDB not configured — MONGO_URI env var missing")
        return None

    try:
        _mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS = 5000,   # 5s timeout
            connectTimeoutMS         = 5000,
        )
        # Ping to confirm connection is alive
        _mongo_client.admin.command("ping")
        _mongo_db = _mongo_client[MONGO_DB_NAME]

        # ── Ensure indexes exist ──────────────────────────────
        _mongo_db.forecasts.create_index([("week_key",  ASCENDING)], unique=True)
        _mongo_db.forecasts.create_index([("base_date", ASCENDING)])
        _mongo_db.accuracy.create_index( [("date",      ASCENDING)], unique=True)
        _mongo_db.visits.create_index(   [("timestamp", DESCENDING)])

        log.info(f"MongoDB connected — db: {MONGO_DB_NAME}")
        return _mongo_db

    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        log.error(f"MongoDB connection failed: {e}")
        return None
    except Exception as e:
        log.error(f"MongoDB unexpected error: {e}")
        return None


def _get_week_key(date_str: str) -> str:
    """
    Return the ISO week key for a date string (e.g. '2026-W12').
    Used to group forecasts by the week they were made.
    """
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"


def _log_forecast(forecast_result: dict) -> None:
    """
    Persist this week's forecast to MongoDB — one entry per ISO week.

    Weekly cycle:
      - Sunday: models retrain, Render redeploys
      - Monday: first /api/forecast call logs the week's predictions
      - Following Monday: /api/accuracy evaluates last week's predictions
                          against actuals now available from Stooq

    Only writes once per week_key — idempotent, safe to call daily.
    Falls back to local JSON file if MongoDB is unavailable.
    """
    base_date = forecast_result.get("last_date")
    week_key  = _get_week_key(base_date)

    entry = {
        "base_date"   : base_date,
        "week_key"    : week_key,
        "last_close"  : forecast_result.get("last_close"),
        "logged_at"   : datetime.now().isoformat(),
        "predictions" : [
            {
                "date"       : d["date"],
                "day"        : d["day"],
                "ensemble"   : d["ensemble"],
                "xgboost"    : d["xgboost"],
                "lstm"       : d["lstm"],
                "direction"  : d["direction"],
                "confidence" : d["confidence"],
            }
            for d in forecast_result.get("forecast", [])
        ]
    }

    # ── Try MongoDB first ─────────────────────────────────────
    db = get_db()
    if db is not None:
        try:
            db.forecasts.update_one(
                {"week_key": week_key},
                {"$setOnInsert": entry},
                upsert=True,   # Insert only if this week not already logged
            )
            log.info(f"Forecast logged to MongoDB — week {week_key} (base: {base_date})")
            return
        except Exception as e:
            log.warning(f"MongoDB forecast write failed, falling back to file: {e}")

    # ── Fallback: local JSON file ─────────────────────────────
    try:
        entries = []
        if os.path.exists(FORECAST_LOG_PATH):
            with open(FORECAST_LOG_PATH) as f:
                entries = json.load(f)
        existing_weeks = {e.get("week_key") for e in entries}
        if week_key not in existing_weeks:
            entries = (entries + [entry])[-52:]   # Keep ~1 year
            os.makedirs(MODELS_DIR, exist_ok=True)
            with open(FORECAST_LOG_PATH, "w") as f:
                json.dump(entries, f, indent=2)
            log.info(f"Forecast logged to file (fallback) — week {week_key}")
    except Exception as e:
        log.warning(f"Forecast file fallback also failed: {e}")


def _load_forecasts() -> list:
    """
    Load all forecast entries — from MongoDB if available, else local file.
    Returns list of forecast dicts sorted oldest-first.
    """
    db = get_db()
    if db is not None:
        try:
            return list(db.forecasts.find(
                {}, {"_id": 0}
            ).sort("base_date", ASCENDING))
        except Exception as e:
            log.warning(f"MongoDB forecast read failed: {e}")

    # Fallback to local file
    try:
        if os.path.exists(FORECAST_LOG_PATH):
            with open(FORECAST_LOG_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _log_visit(endpoint: str) -> None:
    """
    Record a dashboard visit to MongoDB visits collection.
    Fire-and-forget — never raises, never blocks the response.
    """
    db = get_db()
    if db is None:
        return
    try:
        db.visits.insert_one({
            "timestamp" : datetime.now().isoformat(),
            "endpoint"  : endpoint,
            "date"      : str(datetime.now().date()),
        })
    except Exception as e:
        log.debug(f"Visit log failed (non-critical): {e}")


# ================================================================
# HELPER — RSI / MACD signal interpretation
# ================================================================

def _rsi_signal(rsi: float) -> str:
    if rsi   >= 70: return "OVERBOUGHT"
    if rsi   <= 30: return "OVERSOLD"
    if rsi   >= 60: return "BULLISH"
    if rsi   <= 40: return "BEARISH"
    return "NEUTRAL"

def _macd_signal(macd: float, signal: float, hist: float) -> str:
    if macd > signal and hist > 0: return "BULLISH"
    if macd < signal and hist < 0: return "BEARISH"
    return "NEUTRAL"

def _trend_signal(close: float, sma50: float, sma200: float) -> str:
    if close > sma200 and sma50 > sma200: return "STRONG_UPTREND"
    if close > sma200:                    return "UPTREND"
    if close < sma200 and sma50 < sma200: return "STRONG_DOWNTREND"
    return "DOWNTREND"

def _safe(val) -> Optional[float]:
    """Return None instead of NaN for JSON safety."""
    if val is None: return None
    try:
        return None if np.isnan(float(val)) else round(float(val), 4)
    except Exception:
        return None


# ================================================================
# ENDPOINTS
# ================================================================

# ── / ────────────────────────────────────────────────────────────

@app.get("/", tags=["System"], include_in_schema=False)
async def root():
    """Redirect root URL to interactive API docs."""
    return RedirectResponse(url="/docs")


# ── /health ──────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """
    API health check. Returns model status and cache state.
    The frontend pings this on load to verify the backend is alive.
    """
    cache_age = state.cache_age_seconds
    return {
        "status"         : "ok",
        "models_loaded"  : state.models_loaded,
        "best_model"     : state.metadata.get("best_model", "unknown"),
        "ticker"         : TICKER,
        "cache_age_sec"  : round(cache_age, 1) if cache_age else None,
        "cache_stale"    : state._cache_stale,
        "server_time"    : datetime.now().isoformat(),
        "version"        : "2.0.0",
    }


# ── /api/price ───────────────────────────────────────────────────

@app.get("/api/price", tags=["Market Data"])
async def get_price():
    """
    Latest SGLN.L price with OHLCV, daily change, and 52-week range.
    Data is cached for 5 minutes to avoid Yahoo Finance rate limits.
    """
    _log_visit("/api/price")   # Count as a dashboard load
    _log_visit("/api/price")   # Track dashboard load
    df = fetch_sgln_data(years=2)

    # Latest row
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    close      = _safe(latest["close"])
    prev_close = _safe(prev["close"])
    change     = round(close - prev_close, 4) if close and prev_close else None
    change_pct = round((change / prev_close) * 100, 4) if change and prev_close else None

    # 52-week range
    one_year_ago = df.index[-1] - pd.DateOffset(years=1)
    df_1y        = df[df.index >= one_year_ago]
    week_52_high = _safe(df_1y["high"].max())
    week_52_low  = _safe(df_1y["low"].min())

    return {
        "ticker"       : TICKER,
        "name"         : "iShares Physical Gold ETF",
        "currency"     : "GBP",
        "price"        : close,
        "open"         : _safe(latest.get("open")),
        "high"         : _safe(latest.get("high")),
        "low"          : _safe(latest.get("low")),
        "volume"       : int(latest["volume"]) if "volume" in latest and not pd.isna(latest["volume"]) else None,
        "prev_close"   : prev_close,
        "change"       : change,
        "change_pct"   : change_pct,
        "week_52_high" : week_52_high,
        "week_52_low"  : week_52_low,
        "last_updated" : str(df.index[-1].date()),
        "stale_data"   : state._cache_stale,
    }


# ── /api/forecast ─────────────────────────────────────────────────

@app.get("/api/forecast", tags=["Forecast"])
async def get_forecast():
    """
    7-day ensemble price forecast (XGBoost v2 + LSTM v2).
    Returns per-day prices, direction, and confidence level for each day.
    Requires models to be loaded (run model_training_v2.py first).
    """
    if not state.models_loaded:
        raise HTTPException(
            status_code = 503,
            detail      = "Models not loaded. Run model_training_v2.py first.",
        )

    df     = fetch_sgln_data(years=2)
    result = run_forecast(df)
    _log_forecast(result)      # Persist prediction to MongoDB
    _log_visit("/api/forecast") # Track visit
    return result


# ── /api/history ──────────────────────────────────────────────────

@app.get("/api/history", tags=["Market Data"])
async def get_history(
    period: str = Query(
        default = "3m",
        description = "History window: 1w | 1m | 3m | 6m | 1y | 5y",
    )
):
    """
    Historical OHLCV data for charting.
    Use the `period` query parameter to select the time window.
    Example: /api/history?period=6m
    """
    valid_periods = {"1w", "1m", "3m", "6m", "1y", "5y"}
    if period not in valid_periods:
        raise HTTPException(
            status_code = 400,
            detail      = f"Invalid period '{period}'. "
                          f"Choose from: {sorted(valid_periods)}",
        )

    df = fetch_sgln_data(years=5)   # Always fetch max, then slice

    # Calculate cutoff date based on requested period
    end_date = df.index[-1]
    period_map = {
        "1w" : timedelta(weeks=1),
        "1m" : timedelta(days=30),
        "3m" : timedelta(days=91),
        "6m" : timedelta(days=182),
        "1y" : timedelta(days=365),
        "5y" : timedelta(days=365 * 5),
    }
    start_date = end_date - period_map[period]
    df_slice   = df[df.index >= start_date]

    records = []
    for date, row in df_slice.iterrows():
        records.append({
            "date"   : str(date.date()),
            "open"   : _safe(row.get("open")),
            "high"   : _safe(row.get("high")),
            "low"    : _safe(row.get("low")),
            "close"  : _safe(row["close"]),
            "volume" : int(row["volume"]) if "volume" in row and not pd.isna(row["volume"]) else None,
        })

    return {
        "ticker"     : TICKER,
        "period"     : period,
        "start_date" : str(df_slice.index[0].date()),
        "end_date"   : str(df_slice.index[-1].date()),
        "count"      : len(records),
        "data"       : records,
    }


# ── /api/indicators ───────────────────────────────────────────────

@app.get("/api/indicators", tags=["Technical Analysis"])
async def get_indicators():
    """
    Latest values of all technical indicators calculated in Phase 1.
    Includes RSI, MACD, Bollinger Bands, SMAs, volatility, and
    human-readable signal interpretations for each.
    """
    df     = fetch_sgln_data(years=2)
    latest = df.dropna(subset=["sma_200"]).iloc[-1]   # Ensure MA200 is populated

    close  = _safe(latest["close"])
    sma50  = _safe(latest.get(f"sma_{SMA_MEDIUM}"))
    sma200 = _safe(latest.get(f"sma_{SMA_LONG}"))
    rsi    = _safe(latest.get("rsi"))
    macd   = _safe(latest.get("macd"))
    macd_s = _safe(latest.get("macd_signal"))
    macd_h = _safe(latest.get("macd_hist"))

    return {
        "as_of"    : str(df.index[-1].date()),
        "price"    : close,

        "rsi": {
            "value"  : rsi,
            "signal" : _rsi_signal(rsi) if rsi else None,
            "period" : RSI_PERIOD,
        },

        "macd": {
            "macd"      : macd,
            "signal"    : macd_s,
            "histogram" : macd_h,
            "crossover" : _macd_signal(macd, macd_s, macd_h)
                          if all(v is not None for v in [macd, macd_s, macd_h])
                          else None,
        },

        "bollinger_bands": {
            "upper"     : _safe(latest.get("bb_upper")),
            "middle"    : _safe(latest.get("bb_middle")),
            "lower"     : _safe(latest.get("bb_lower")),
            "width"     : _safe(latest.get("bb_width")),
            "pct_b"     : _safe(latest.get("bb_pct")),
        },

        "moving_averages": {
            "sma_20"    : _safe(latest.get(f"sma_{SMA_SHORT}")),
            "sma_50"    : sma50,
            "sma_200"   : sma200,
            "ema_12"    : _safe(latest.get(f"ema_{EMA_SHORT}")),
            "ema_26"    : _safe(latest.get(f"ema_{EMA_LONG}")),
            "above_sma20"  : bool(latest.get("above_sma20",  0)),
            "above_sma50"  : bool(latest.get("above_sma50",  0)),
            "above_sma200" : bool(latest.get("above_sma200", 0)),
            "trend"     : _trend_signal(close, sma50, sma200)
                          if all(v is not None for v in [close, sma50, sma200])
                          else None,
        },

        "volatility": {
            "daily_return_pct"    : _safe(latest.get("daily_return")),
            "volatility_20d"      : _safe(latest.get("volatility_20d")),
            "volatility_annualised": _safe(latest.get("volatility_annualised")),
        },

        "crossover_signals": {
            "golden_cross" : bool(latest.get("golden_cross", 0)),
            "death_cross"  : bool(latest.get("death_cross",  0)),
        },
    }



# ── /api/accuracy ─────────────────────────────────────────────────

@app.get("/api/accuracy", tags=["Forecast"])
async def get_accuracy():
    """
    Weekly forecast accuracy tracker.

    For each logged week, checks whether all 7 predicted days now have
    actual prices available from Stooq. Only returns COMPLETE weeks
    where every day can be evaluated — no pending rows shown.

    Weekly cycle:
      Monday  → forecast logged for the week
      Following Monday → all 7 actuals available → week evaluated + shown
    """
    entries = _load_forecasts()
    if not entries:
        return {
            "weeks"           : [],
            "results"         : [],
            "directional_acc" : None,
            "mae"             : None,
            "total_evaluated" : 0,
            "message"         : "No forecast history yet — results appear "
                                "one week after the first forecast is generated.",
        }

    # Fetch actual prices
    df = fetch_sgln_data(years=2)
    actual_prices = {str(d.date()): row["close"] for d, row in df.iterrows()}

    all_results  = []   # Flat list of day rows for the table
    weeks_meta   = []   # One summary per completed week
    total_correct = 0
    total_err     = 0.0
    total_days    = 0

    # Process most recent weeks first
    for entry in reversed(entries):
        base_close  = entry.get("last_close")
        week_key    = entry.get("week_key", entry.get("base_date", ""))
        base_date   = entry.get("base_date", "")
        predictions = entry.get("predictions", [])

        # ── Check if ALL 7 days have actuals ─────────────────
        day_results = []
        week_complete = True

        prev_close = base_close
        for pred in predictions:
            actual = actual_prices.get(pred["date"])
            if actual is None:
                week_complete = False
                break

            error      = round(actual - pred["ensemble"], 4)
            act_dir    = "UP" if actual > prev_close else "DOWN"
            is_correct = pred["direction"] == act_dir

            day_results.append({
                "date"          : pred["date"],
                "week_key"      : week_key,
                "predicted"     : pred["ensemble"],
                "actual"        : round(actual, 4),
                "error"         : error,
                "pred_direction": pred["direction"],
                "act_direction" : act_dir,
                "correct"       : is_correct,
                "confidence"    : pred["confidence"],
                "status"        : "evaluated",
            })
            prev_close = actual

        # ── Skip incomplete weeks entirely ────────────────────
        if not week_complete or not day_results:
            continue

        # ── Week is complete — compute summary ────────────────
        week_correct = sum(1 for r in day_results if r["correct"])
        week_mae     = round(sum(abs(r["error"]) for r in day_results) / len(day_results), 4)
        week_da      = round(week_correct / len(day_results) * 100, 1)

        weeks_meta.append({
            "week_key"   : week_key,
            "base_date"  : base_date,
            "days"       : len(day_results),
            "correct"    : week_correct,
            "da_pct"     : week_da,
            "mae"        : week_mae,
        })

        all_results.extend(day_results)
        total_correct += week_correct
        total_err     += sum(abs(r["error"]) for r in day_results)
        total_days    += len(day_results)

        # Keep up to 8 weeks (56 days) of history
        if len(weeks_meta) >= 8:
            break

    if total_days == 0:
        return {
            "weeks"           : [],
            "results"         : [],
            "directional_acc" : None,
            "mae"             : None,
            "total_evaluated" : 0,
            "message"         : "No complete weeks yet — check back next week.",
        }

    overall_da  = round(total_correct / total_days * 100, 1)
    overall_mae = round(total_err / total_days, 4)

    return {
        "weeks"           : weeks_meta,
        "results"         : all_results,
        "directional_acc" : overall_da,
        "mae"             : overall_mae,
        "total_evaluated" : total_days,
        "total_weeks"     : len(weeks_meta),
    }



# ── /api/stats ────────────────────────────────────────────────────

@app.get("/api/stats", tags=["System"])
async def get_stats():
    """
    Usage statistics from MongoDB.
    Returns total dashboard visits, visits today, and forecast count.
    """
    db = get_db()
    if db is None:
        return {
            "total_visits"    : None,
            "visits_today"    : None,
            "total_forecasts" : None,
            "db_connected"    : False,
            "message"         : "Database not connected",
        }

    try:
        today = str(datetime.now().date())

        total_visits    = db.visits.count_documents({})
        visits_today    = db.visits.count_documents({"date": today})
        total_forecasts = db.forecasts.count_documents({})

        # Visits over last 7 days
        from datetime import timedelta
        last_7 = [
            str((datetime.now() - timedelta(days=i)).date())
            for i in range(6, -1, -1)
        ]
        daily_visits = []
        for d in last_7:
            count = db.visits.count_documents({"date": d})
            daily_visits.append({"date": d, "count": count})

        return {
            "total_visits"    : total_visits,
            "visits_today"    : visits_today,
            "total_forecasts" : total_forecasts,
            "daily_visits"    : daily_visits,
            "db_connected"    : True,
        }

    except Exception as e:
        log.error(f"Stats query failed: {e}")
        return {
            "total_visits" : None,
            "visits_today" : None,
            "db_connected" : False,
            "message"      : str(e),
        }


# ================================================================
# MAIN — Run with uvicorn
# ================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        log_level = "info",
    )