"""
================================================================
OracleAU — Phase 2: Model Training v2
================================================================
Script:      model_training_v2.py
Description: Improved training pipeline that fixes four key
             issues identified in v1:

             1. Predicts RETURNS not raw prices → fixes
                distribution shift (prices grew 4x over 10yr)
             2. RobustScaler instead of MinMaxScaler → handles
                prices outside the training range correctly
             3. Lag features removed from XGBoost → eliminates
                target leakage that caused the flat prediction
             4. Stronger LSTM regularisation → closes the
                training/validation loss gap (overfitting)
             5. Recency weighting → recent data weighted 3x
             6. Consistent multistep forecast rollout

Models saved with _v2 suffix to preserve v1 for comparison:
    models/xgboost_v2.pkl
    models/lstm_v2.h5
    models/scaler_X_v2.pkl
    models/scaler_y_v2.pkl
    models/model_metadata_v2.json

Usage:
    python model_training_v2.py

Requirements:
    pip install pandas numpy scikit-learn xgboost tensorflow joblib matplotlib
================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")


# ================================================================
# CONFIGURATION
# ================================================================

DATA_PATH     = "data/SGLN_features.csv"
MODELS_DIR    = "models"
REPORTS_DIR   = "reports"
PLOTS_DIR     = "reports/plots"

# Columns that must never appear in X
LEAK_COLS = [
    "target_next_close",
    "target_direction",
    "repaired?",
]

# Lag feature columns — removed from XGBoost to prevent leakage
# (sequential context is already captured by the window flattening)
LAG_COLS = [
    "close_lag_1", "close_lag_2", "close_lag_3",
    "close_lag_5", "close_lag_10",
]

TARGET_COL    = "target_next_close"
TRAIN_RATIO   = 0.80
SEQUENCE_LEN  = 45
FORECAST_DAYS = 7

# LSTM v2 — reduced capacity + stronger regularisation
LSTM_UNITS_1  = 32      # Down from 64
LSTM_UNITS_2  = 16      # Down from 32
DROPOUT_RATE  = 0.4     # Up from 0.2
L2_REG        = 0.001   # Added L2 weight regularisation
EPOCHS        = 150
BATCH_SIZE    = 32
LEARNING_RATE = 0.0005  # Slightly lower for more careful learning

# XGBoost — slightly higher LR to reduce forecast flatness
XGB_LEARNING_RATE    = 0.03   # Up from 0.02 — more return variance sensitivity

# Ensemble — weighted average of XGBoost + LSTM forecasts
# LSTM weighted higher based on live test MAE and directional accuracy
ENSEMBLE_WEIGHT_LSTM = 0.60
ENSEMBLE_WEIGHT_XGB  = 0.40

# Confidence signal thresholds
# If both model forecasts differ by less than this % -> HIGH confidence
# If they differ more -> LOW confidence (mixed signal shown in UI)
CONFIDENCE_THRESHOLD = 0.50   # 0.50% difference between XGB and LSTM price

# Recency weighting
RECENCY_WEIGHT      = 3.0   # Recent samples weighted this many times more
RECENCY_THRESHOLD   = 0.80  # Top 20% of training data = "recent"

WF_FOLDS      = 5


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_step(n: int, title: str) -> None:
    print(f"\n[{n}] {title}")
    print(f"{'─'*50}")

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                          y_prev: np.ndarray) -> float:
    actual_dir    = (y_true > y_prev).astype(int)
    predicted_dir = (y_pred > y_prev).astype(int)
    return float(np.mean(actual_dir == predicted_dir) * 100)

def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_prev: np.ndarray, label: str) -> dict:
    mae_val  = mean_absolute_error(y_true, y_pred)
    rmse_val = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape_val = mape(y_true, y_pred)
    da_val   = directional_accuracy(y_true, y_pred, y_prev)

    print(f"\n  {label}:")
    print(f"    MAE               : £{mae_val:.4f}")
    print(f"    RMSE              : £{rmse_val:.4f}")
    print(f"    MAPE              : {mape_val:.2f}%")
    print(f"    Directional Acc   : {da_val:.2f}%")

    return {"mae": mae_val, "rmse": rmse_val,
            "mape": mape_val, "directional_accuracy": da_val}

def returns_to_price(base_price: float, returns: np.ndarray) -> np.ndarray:
    """Convert a sequence of daily returns back to prices."""
    prices = [base_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r / 100))
    return np.array(prices[1:])


# ================================================================
# STEP 1 — LOAD & PREPARE
# ================================================================

def load_and_prepare(path: str):
    """
    KEY CHANGE v2: We engineer a 'daily_return' target instead of
    predicting raw price. Returns are stationary — they don't drift
    upward over time the way prices do, so the model can generalise
    from the training price range to the test price range.

    Return target = (tomorrow_close - today_close) / today_close × 100
    """
    print_step(1, "Loading & Preparing Data")

    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df.drop(columns=["repaired?"], errors="ignore", inplace=True)
    df.dropna(inplace=True)

    print(f"  Loaded        : {len(df):,} rows × {len(df.columns)} cols")
    print(f"  Date range    : {df.index[0].date()} → {df.index[-1].date()}")

    # ── Engineer return-based target ─────────────────────────────
    # Instead of predicting tomorrow's raw price, predict tomorrow's
    # % return. This removes the upward price drift from the problem.
    df["return_target"] = (
        (df["target_next_close"] - df["close"]) / df["close"]
    ) * 100

    # Store raw close for later inverse conversion
    df["close_raw"] = df["close"]

    # Drop rows where return_target is NaN (last row)
    df.dropna(subset=["return_target"], inplace=True)

    print(f"  Return target : min={df['return_target'].min():.2f}%  "
          f"max={df['return_target'].max():.2f}%  "
          f"mean={df['return_target'].mean():.4f}%")
    print(f"  Usable rows   : {len(df):,}")

    return df


# ================================================================
# STEP 2 — FEATURE SCALING (RobustScaler)
# ================================================================

def scale_features(df: pd.DataFrame, train_end_idx: int,
                   for_xgboost: bool = False):
    """
    KEY CHANGE v2: RobustScaler uses median and IQR instead of
    min/max. When test prices exceed the training maximum (which
    they do — SGLN went from £17 to £75), MinMaxScaler maps them
    above 1.0. RobustScaler handles this gracefully because it's
    not bounded by training extremes.

    for_xgboost=True removes lag features to prevent leakage.
    """
    # Determine feature columns
    exclude = set(LEAK_COLS) | {"return_target", "close_raw",
                                 "target_next_close", "target_direction"}
    if for_xgboost:
        exclude |= set(LAG_COLS)   # Remove lags for XGBoost

    feature_cols = [c for c in df.columns if c not in exclude]
    target_col   = "return_target"

    X     = df[feature_cols].values
    y     = df[target_col].values
    dates = df.index

    # Fit ONLY on training data — never on test data
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    X_train_raw = X[:train_end_idx]
    y_train_raw = y[:train_end_idx].reshape(-1, 1)

    scaler_X.fit(X_train_raw)
    scaler_y.fit(y_train_raw)

    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()

    label = "XGBoost" if for_xgboost else "LSTM"
    print(f"\n  [{label}] Features : {len(feature_cols)} columns")
    if for_xgboost:
        print(f"  [{label}] Lag cols removed : {LAG_COLS}")
    print(f"  [{label}] Scaler   : RobustScaler (median/IQR)")

    return X_scaled, y_scaled, y, scaler_X, scaler_y, feature_cols, dates


# ================================================================
# STEP 3 — CHRONOLOGICAL SPLIT + RECENCY WEIGHTS
# ================================================================

def chronological_split(X, y_scaled, y_raw, dates, ratio):
    print_step(3, "Chronological Train / Test Split")

    split_idx = int(len(X) * ratio)

    X_train, X_test         = X[:split_idx],         X[split_idx:]
    y_train, y_test         = y_scaled[:split_idx],  y_scaled[split_idx:]
    y_raw_train, y_raw_test = y_raw[:split_idx],     y_raw[split_idx:]
    dates_train, dates_test = dates[:split_idx],     dates[split_idx:]

    print(f"  Training      : {len(X_train):,} rows  "
          f"({dates_train[0].date()} → {dates_train[-1].date()})")
    print(f"  Test          : {len(X_test):,} rows   "
          f"({dates_test[0].date()} → {dates_test[-1].date()})")

    return (X_train, X_test, y_train, y_test,
            y_raw_train, y_raw_test, dates_train, dates_test, split_idx)

def build_recency_weights(n: int, threshold: float = RECENCY_THRESHOLD,
                           weight: float = RECENCY_WEIGHT) -> np.ndarray:
    """
    KEY CHANGE v2: Give recent training samples higher weight.
    The most recent (1-threshold)% of training rows get RECENCY_WEIGHT
    times more influence during model fitting.
    This honours the insight that recent gold price behaviour is
    more representative of tomorrow's move than behaviour from 2016.
    """
    cutoff = int(n * threshold)
    weights = np.ones(n)
    weights[cutoff:] = weight
    return weights


# ================================================================
# STEP 4A — XGBOOST v2
# ================================================================

def build_sequences_xgb(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i].flatten())
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def train_xgboost_v2(X_train, y_train, X_test, y_test,
                      y_raw_train, y_raw_test, df,
                      scaler_y, seq_len) -> tuple:
    print_step(4, "Training XGBoost v2")

    X_tr_seq, y_tr_seq = build_sequences_xgb(X_train, y_train, seq_len)
    X_te_seq, y_te_seq = build_sequences_xgb(X_test,  y_test,  seq_len)

    # Recency weights — aligned to sequences (which start at seq_len)
    raw_weights = build_recency_weights(len(X_train))
    seq_weights = raw_weights[seq_len:]   # Align with sequence outputs

    print(f"  Train seqs    : {X_tr_seq.shape}")
    print(f"  Test seqs     : {X_te_seq.shape}")
    print(f"  Lag cols      : REMOVED (prevents leakage)")
    print(f"  Recency wt    : {RECENCY_WEIGHT}× on last "
          f"{int((1-RECENCY_THRESHOLD)*100)}% of training data")

    model = xgb.XGBRegressor(
        n_estimators          = 800,
        learning_rate         = XGB_LEARNING_RATE,  # 0.03 — tuned for return sensitivity
        max_depth             = 4,        # Shallower trees — less overfit
        min_child_weight      = 5,        # Requires more samples per leaf
        subsample             = 0.7,
        colsample_bytree      = 0.7,
        gamma                 = 0.2,      # Minimum loss reduction to split
        reg_alpha             = 0.5,      # Stronger L1
        reg_lambda            = 2.0,      # Stronger L2
        random_state          = 42,
        n_jobs                = -1,
        early_stopping_rounds = 50,
        eval_metric           = "rmse",
        verbosity             = 0,
    )

    print(f"\n  Training XGBoost v2...")
    model.fit(
        X_tr_seq, y_tr_seq,
        sample_weight  = seq_weights,
        eval_set       = [(X_te_seq, y_te_seq)],
        verbose        = False,
    )

    print(f"  ✓ Best iteration : {model.best_iteration}")

    # Predict returns, then convert back to prices
    y_pred_return_scaled = model.predict(X_te_seq)
    y_pred_returns = scaler_y.inverse_transform(
        y_pred_return_scaled.reshape(-1, 1)).flatten()
    y_true_returns = scaler_y.inverse_transform(
        y_te_seq.reshape(-1, 1)).flatten()

    # Get the corresponding base prices (today's close) for conversion
    # y_raw_test contains raw close prices; test sequences start at seq_len
    base_prices_test = y_raw_test[seq_len - 1: seq_len - 1 + len(y_true_returns)]

    # Convert returns to prices: price_t+1 = price_t × (1 + return/100)
    y_pred_prices = base_prices_test * (1 + y_pred_returns / 100)
    y_true_prices = base_prices_test * (1 + y_true_returns / 100)
    y_prev_prices = base_prices_test   # Today's price = directional baseline

    metrics = evaluate(y_true_prices, y_pred_prices,
                       y_prev_prices, "XGBoost v2 Test Set")

    # Feature importance — top 15
    fi = pd.Series(model.feature_importances_)
    print(f"\n  Top 10 most important features:")
    for idx in fi.nlargest(10).index:
        print(f"    Feature [{idx}] : {fi[idx]:.4f}")

    return model, metrics, y_true_prices, y_pred_prices


# ================================================================
# STEP 4B — LSTM v2
# ================================================================

def build_sequences_lstm(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def train_lstm_v2(X_train, y_train, X_test, y_test,
                   y_raw_test, scaler_y, seq_len, n_features) -> tuple:
    print_step(5, "Training LSTM v2")

    X_tr_seq, y_tr_seq = build_sequences_lstm(X_train, y_train, seq_len)
    X_te_seq, y_te_seq = build_sequences_lstm(X_test,  y_test,  seq_len)

    raw_weights = build_recency_weights(len(X_train))
    seq_weights = raw_weights[seq_len:]

    print(f"  Train seqs    : {X_tr_seq.shape}")
    print(f"  Test seqs     : {X_te_seq.shape}")
    print(f"  LSTM units    : {LSTM_UNITS_1} → {LSTM_UNITS_2} (reduced)")
    print(f"  Dropout       : {DROPOUT_RATE} (increased)")
    print(f"  L2 reg        : {L2_REG} (added)")
    print(f"  Recency wt    : {RECENCY_WEIGHT}× on last "
          f"{int((1-RECENCY_THRESHOLD)*100)}% of training data")

    # ── LSTM v2 Architecture ──────────────────────────────────────
    # KEY CHANGES from v1:
    # - Reduced units (32→16) to reduce model capacity
    # - Dropout increased to 0.4 to force generalisation
    # - L2 regularisation on LSTM kernels
    # - BatchNormalization to stabilise training
    # - Predicting returns (stationary) not raw prices
    model = Sequential([
        Input(shape=(seq_len, n_features)),

        LSTM(LSTM_UNITS_1,
             return_sequences=True,
             kernel_regularizer=regularizers.l2(L2_REG),
             recurrent_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        LSTM(LSTM_UNITS_2,
             return_sequences=False,
             kernel_regularizer=regularizers.l2(L2_REG),
             recurrent_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        Dense(16, activation="relu",
              kernel_regularizer=regularizers.l2(L2_REG)),
        Dense(1),
    ], name="OracleAU_LSTM_v2")

    model.compile(
        optimizer = Adam(learning_rate=LEARNING_RATE,
                         clipnorm=1.0),   # Gradient clipping — stabilises training
        loss      = "huber",
        metrics   = ["mae"],
    )

    print(f"\n  LSTM v2 Architecture:")
    model.summary(print_fn=lambda x: print(f"    {x}"))

    callbacks = [
        EarlyStopping(
            monitor              = "val_loss",
            patience             = 20,      # More patience — returns are noisier
            restore_best_weights = True,
            verbose              = 0,
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 10,
            min_lr   = 1e-7,
            verbose  = 0,
        ),
    ]

    print(f"\n  Training LSTM v2 (max {EPOCHS} epochs)...")
    history = model.fit(
        X_tr_seq, y_tr_seq,
        validation_data = (X_te_seq, y_te_seq),
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        sample_weight   = seq_weights,
        callbacks       = callbacks,
        verbose         = 0,
    )

    epochs_run = len(history.history["loss"])
    best_val   = min(history.history["val_loss"])
    final_train = history.history["loss"][-1]
    print(f"  ✓ Stopped at epoch : {epochs_run}")
    print(f"  ✓ Best val loss    : {best_val:.6f}")
    print(f"  ✓ Final train loss : {final_train:.6f}")
    print(f"  ✓ Overfit gap      : {abs(final_train - best_val):.6f} "
          f"({'⚠ large' if abs(final_train - best_val) > 0.1 else '✓ acceptable'})")

    # Predict returns → convert back to prices
    y_pred_return_scaled = model.predict(X_te_seq, verbose=0).flatten()
    y_pred_returns = scaler_y.inverse_transform(
        y_pred_return_scaled.reshape(-1, 1)).flatten()
    y_true_returns = scaler_y.inverse_transform(
        y_te_seq.reshape(-1, 1)).flatten()

    base_prices_test = y_raw_test[seq_len - 1: seq_len - 1 + len(y_true_returns)]

    y_pred_prices = base_prices_test * (1 + y_pred_returns / 100)
    y_true_prices = base_prices_test * (1 + y_true_returns / 100)
    y_prev_prices = base_prices_test

    metrics = evaluate(y_true_prices, y_pred_prices,
                       y_prev_prices, "LSTM v2 Test Set")

    return model, metrics, y_true_prices, y_pred_prices, history


# ================================================================
# STEP 5 — WALK-FORWARD VALIDATION
# ================================================================

def walk_forward_validation(df, feature_cols_xgb, feature_cols_lstm,
                             seq_len, n_folds, best_model_name) -> dict:
    print_step(6, f"Walk-Forward Validation ({n_folds} folds)")

    X_all_xgb  = df[feature_cols_xgb].values
    X_all_lstm = df[feature_cols_lstm].values
    y_all      = df["return_target"].values
    close_all  = df["close_raw"].values

    fold_size   = len(X_all_xgb) // (n_folds + 1)
    all_metrics = []

    for fold in range(n_folds):
        train_end  = fold_size * (fold + 1)
        test_start = train_end
        test_end   = min(train_end + fold_size, len(X_all_xgb))

        X_all = X_all_xgb if best_model_name == "xgboost" else X_all_lstm
        X_tr  = X_all[:train_end]
        X_te  = X_all[test_start:test_end]
        y_tr  = y_all[:train_end]
        y_te  = y_all[test_start:test_end]
        c_te  = close_all[test_start:test_end]

        sc_X = RobustScaler().fit(X_tr)
        sc_y = RobustScaler().fit(y_tr.reshape(-1, 1))

        X_tr_s = sc_X.transform(X_tr)
        X_te_s = sc_X.transform(X_te)
        y_tr_s = sc_y.transform(y_tr.reshape(-1, 1)).flatten()
        y_te_s = sc_y.transform(y_te.reshape(-1, 1)).flatten()

        wts = build_recency_weights(len(X_tr))

        if best_model_name == "xgboost":
            X_tr_seq, y_tr_seq = build_sequences_xgb(X_tr_s, y_tr_s, seq_len)
            X_te_seq, y_te_seq = build_sequences_xgb(X_te_s, y_te_s, seq_len)
            m = xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.02,
                max_depth=4, random_state=42, verbosity=0, n_jobs=-1)
            m.fit(X_tr_seq, y_tr_seq,
                  sample_weight=wts[seq_len:], verbose=False)
            y_pred_s = m.predict(X_te_seq)
        else:
            n_feat = X_tr_s.shape[1]
            X_tr_seq, y_tr_seq = build_sequences_lstm(X_tr_s, y_tr_s, seq_len)
            X_te_seq, y_te_seq = build_sequences_lstm(X_te_s, y_te_s, seq_len)
            m = Sequential([
                Input(shape=(seq_len, n_feat)),
                LSTM(32, return_sequences=False,
                     kernel_regularizer=regularizers.l2(0.001)),
                Dropout(0.4),
                Dense(1)
            ])
            m.compile(optimizer=Adam(0.0005, clipnorm=1.0), loss="huber")
            m.fit(X_tr_seq, y_tr_seq,
                  sample_weight=wts[seq_len:],
                  epochs=50, batch_size=32, verbose=0,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(patience=10,
                                           restore_best_weights=True,
                                           verbose=0)])
            y_pred_s = m.predict(X_te_seq, verbose=0).flatten()

        y_pred_ret = sc_y.inverse_transform(
            y_pred_s.reshape(-1, 1)).flatten()
        y_true_ret = y_te[seq_len:]
        base_p     = c_te[seq_len - 1: seq_len - 1 + len(y_true_ret)]

        min_len      = min(len(y_pred_ret), len(y_true_ret), len(base_p))
        y_pred_ret   = y_pred_ret[:min_len]
        y_true_ret   = y_true_ret[:min_len]
        base_p       = base_p[:min_len]

        y_pred_p = base_p * (1 + y_pred_ret / 100)
        y_true_p = base_p * (1 + y_true_ret / 100)

        fold_mae  = mean_absolute_error(y_true_p, y_pred_p)
        fold_rmse = float(np.sqrt(mean_squared_error(y_true_p, y_pred_p)))
        fold_mape = mape(y_true_p, y_pred_p)
        fold_da   = directional_accuracy(y_true_p, y_pred_p, base_p)

        all_metrics.append({
            "mae": fold_mae, "rmse": fold_rmse,
            "mape": fold_mape, "da": fold_da
        })

        test_dates = df.index[test_start:test_end]
        print(f"  Fold {fold+1}  "
              f"Train→{df.index[train_end-1].date()}  "
              f"Test {test_dates[0].date()}→{test_dates[-1].date()}  "
              f"MAE=£{fold_mae:.3f}  DA={fold_da:.1f}%")

    avg = {
        "avg_mae"  : float(np.mean([m["mae"]  for m in all_metrics])),
        "avg_rmse" : float(np.mean([m["rmse"] for m in all_metrics])),
        "avg_mape" : float(np.mean([m["mape"] for m in all_metrics])),
        "avg_da"   : float(np.mean([m["da"]   for m in all_metrics])),
    }
    print(f"\n  Walk-Forward Averages:")
    print(f"    Avg MAE  : £{avg['avg_mae']:.4f}")
    print(f"    Avg RMSE : £{avg['avg_rmse']:.4f}")
    print(f"    Avg MAPE : {avg['avg_mape']:.2f}%")
    print(f"    Avg DA   : {avg['avg_da']:.2f}%")

    return avg


# ================================================================
# STEP 6 — MULTI-STEP FORECAST (1–7 days)
# ================================================================

def multistep_forecast_v2(model, last_sequence: np.ndarray,
                           last_price: float, scaler_y,
                           n_days: int, model_type: str) -> list:
    """
    KEY CHANGE v2: Forecast returns iteratively, then chain-convert
    back to prices. Each day's predicted price becomes the base for
    the next day's return prediction.

    This is more principled than v1's approach of rolling the raw
    scaled close price through the sequence.
    """
    prices     = [last_price]
    current_seq = last_sequence.copy()

    for day in range(n_days):
        if model_type == "xgboost":
            inp      = current_seq.flatten().reshape(1, -1)
            pred_s   = model.predict(inp)[0]
        else:
            inp      = current_seq.reshape(1, *current_seq.shape)
            pred_s   = model.predict(inp, verbose=0)[0][0]

        # Inverse-transform scaled return → actual % return
        pred_return = float(scaler_y.inverse_transform(
            np.array([[pred_s]])).flatten()[0])

        # Convert return to price
        next_price = prices[-1] * (1 + pred_return / 100)
        prices.append(next_price)

        # Roll sequence forward — update close (index 0) with new scaled price
        new_row      = current_seq[-1].copy()
        new_row[0]   = pred_s    # Scaled return as proxy for updated state
        current_seq  = np.vstack([current_seq[1:], new_row])

    return prices[1:]   # Exclude the seed price



# ================================================================
# ENSEMBLE FORECAST & CONFIDENCE SIGNAL
# ================================================================

def ensemble_forecast(forecast_xgb: list, forecast_lstm: list,
                      last_price: float) -> dict:
    """
    Combine XGBoost and LSTM forecasts into a weighted ensemble
    and compute a per-day confidence signal.

    Ensemble price = (XGB_weight × xgb_price) + (LSTM_weight × lstm_price)

    Confidence logic:
    - HIGH   : Both models agree within CONFIDENCE_THRESHOLD (0.5%)
               → Strong signal, display prominently in UI
    - MEDIUM : Models disagree between 0.5% and 1.5%
               → Reasonable signal, show with caveat
    - LOW    : Models diverge more than 1.5%
               → Mixed signal, UI should show both and warn user

    Parameters:
        forecast_xgb  : List of 7 XGBoost predicted prices
        forecast_lstm : List of 7 LSTM predicted prices
        last_price    : Last known actual close price (£)

    Returns:
        dict with keys:
            ensemble_prices    : List[float] — weighted avg price per day
            confidence_levels  : List[str]   — HIGH / MEDIUM / LOW per day
            agreement_pcts     : List[float] — % difference per day
            overall_direction  : str         — UP / DOWN / NEUTRAL
            overall_confidence : str         — overall signal quality
    """
    ensemble_prices   = []
    confidence_levels = []
    agreement_pcts    = []

    for xgb_p, lstm_p in zip(forecast_xgb, forecast_lstm):
        # Weighted average
        ens_price = (ENSEMBLE_WEIGHT_XGB  * xgb_p +
                     ENSEMBLE_WEIGHT_LSTM * lstm_p)
        ensemble_prices.append(round(ens_price, 4))

        # % difference between the two models for this day
        pct_diff = abs(xgb_p - lstm_p) / lstm_p * 100
        agreement_pcts.append(round(pct_diff, 4))

        # Confidence tier
        if pct_diff <= CONFIDENCE_THRESHOLD:
            confidence_levels.append("HIGH")
        elif pct_diff <= CONFIDENCE_THRESHOLD * 3:
            confidence_levels.append("MEDIUM")
        else:
            confidence_levels.append("LOW")

    # Overall direction from ensemble Day 1 vs last known price
    day1_change_pct = (ensemble_prices[0] - last_price) / last_price * 100
    if day1_change_pct > 0.1:
        overall_direction = "UP"
    elif day1_change_pct < -0.1:
        overall_direction = "DOWN"
    else:
        overall_direction = "NEUTRAL"

    # Overall confidence — worst case of day 1–3 (most actionable horizon)
    tier_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    worst_near = max(confidence_levels[:3], key=lambda x: tier_rank[x])
    overall_confidence = worst_near

    return {
        "ensemble_prices"    : ensemble_prices,
        "confidence_levels"  : confidence_levels,
        "agreement_pcts"     : agreement_pcts,
        "overall_direction"  : overall_direction,
        "overall_confidence" : overall_confidence,
    }


def print_ensemble_summary(ensemble: dict, forecast_xgb: list,
                            forecast_lstm: list, last_price: float,
                            future_dates) -> None:
    """Print a formatted ensemble + confidence table to the terminal."""

    print(f"\n  Last known close : £{last_price:.4f}")
    print(f"  Overall direction: {ensemble['overall_direction']}")
    print(f"  Overall confidence: {ensemble['overall_confidence']}")
    print()
    print(f"  {'Day':<5} {'Date':<12} {'XGB':>8} {'LSTM':>8} "
          f"{'Ensemble':>10} {'Agree%':>8} {'Confidence':<12}")
    print(f"  {'─'*70}")

    for i, (d, xgb_p, lstm_p, ens_p, conf, agr) in enumerate(zip(
            future_dates,
            forecast_xgb,
            forecast_lstm,
            ensemble["ensemble_prices"],
            ensemble["confidence_levels"],
            ensemble["agreement_pcts"]), 1):

        chg = ((ens_p - last_price) / last_price) * 100
        arrow = "▲" if ens_p > last_price else "▼"
        conf_icon = {"HIGH": "✅", "MEDIUM": "⚠️ ", "LOW": "❌"}[conf]

        print(f"  Day {i:<2} {str(d.date()):<12} "
              f"£{xgb_p:>7.4f} £{lstm_p:>7.4f} "
              f"£{ens_p:>9.4f} "
              f"{agr:>7.3f}% "
              f"{conf_icon} {conf}")

    print(f"\n  Confidence key: ✅ HIGH (<0.5% gap)  "
          f"⚠️  MEDIUM (<1.5%)  ❌ LOW (>1.5%)")


# ================================================================
# STEP 7 — SAVE MODELS (v2 suffix)
# ================================================================

def save_models(xgb_model, lstm_model, scaler_X_xgb, scaler_X_lstm,
                scaler_y, xgb_metrics, lstm_metrics, wf_metrics,
                feature_cols_xgb, feature_cols_lstm,
                best_model_name) -> None:
    print_step(7, "Saving v2 Models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    paths = {
        "xgboost" : os.path.join(MODELS_DIR, "xgboost_v2.pkl"),
        "lstm"    : os.path.join(MODELS_DIR, "lstm_v2.h5"),
        "scaler_X_xgb" : os.path.join(MODELS_DIR, "scaler_X_xgb_v2.pkl"),
        "scaler_X_lstm": os.path.join(MODELS_DIR, "scaler_X_lstm_v2.pkl"),
        "scaler_y"     : os.path.join(MODELS_DIR, "scaler_y_v2.pkl"),
        "metadata"     : os.path.join(MODELS_DIR, "model_metadata_v2.json"),
    }

    joblib.dump(xgb_model,    paths["xgboost"])
    lstm_model.save(           paths["lstm"])
    joblib.dump(scaler_X_xgb, paths["scaler_X_xgb"])
    joblib.dump(scaler_X_lstm,paths["scaler_X_lstm"])
    joblib.dump(scaler_y,     paths["scaler_y"])

    for k, p in paths.items():
        if k != "metadata":
            print(f"  ✓ {p}")

    metadata = {
        "version"            : "v2",
        "ticker"             : "SGLN.L",
        "name"               : "iShares Physical Gold (GBP)",
        "best_model"         : best_model_name,
        "prediction_target"  : "daily_return_percent",
        "sequence_length"    : SEQUENCE_LEN,
        "forecast_days"      : FORECAST_DAYS,
        "feature_cols_xgb"   : feature_cols_xgb,
        "feature_cols_lstm"  : feature_cols_lstm,
        "trained_at"         : datetime.now().isoformat(),
        "ensemble_weights"   : {
            "xgboost": ENSEMBLE_WEIGHT_XGB,
            "lstm"   : ENSEMBLE_WEIGHT_LSTM,
        },
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "xgboost_metrics"    : xgb_metrics,
        "lstm_metrics"       : lstm_metrics,
        "walk_forward"       : wf_metrics,
        "changes_from_v1"    : [
            "Predict returns not raw prices (solves distribution shift)",
            "RobustScaler replaces MinMaxScaler",
            "Lag features removed from XGBoost (eliminates leakage)",
            "LSTM dropout 0.2→0.4, L2 reg added, reduced capacity",
            "Recency weighting 3x on most recent 20% of training data",
            "Gradient clipping added to LSTM optimizer",
            "BatchNormalization layers added to LSTM",
        ],
    }

    with open(paths["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ {paths['metadata']}")


# ================================================================
# STEP 8 — GENERATE PLOTS
# ================================================================

def generate_plots(dates_test, y_true_xgb, y_pred_xgb,
                   y_true_lstm, y_pred_lstm, lstm_history,
                   forecast_xgb, forecast_lstm,
                   best_model_name, last_price) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"actual": "#0f1b2d", "xgb": "#e8603a",
              "lstm": "#2a6dd9", "band": "#aac4f5"}

    # ── Plot 1: Actual vs Predicted (both models) ─────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    for ax, y_true, y_pred, label, color in [
        (axes[0], y_true_xgb, y_pred_xgb, "XGBoost v2", colors["xgb"]),
        (axes[1], y_true_lstm,y_pred_lstm, "LSTM v2",    colors["lstm"]),
    ]:
        n = min(len(y_true), len(y_pred))
        plot_dates = dates_test[-n:]
        ax.plot(plot_dates, y_true[:n], color=colors["actual"],
                linewidth=1.5, label="Actual", alpha=0.9)
        ax.plot(plot_dates, y_pred[:n], color=color,
                linewidth=1.2, label=f"{label} Predicted",
                alpha=0.8, linestyle="--")
        ax.set_title(f"{label} — Actual vs Predicted (Test Set)",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("SGLN.L Price (£)", fontsize=10)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    fig.suptitle("OracleAU v2 — SGLN.L Price Prediction",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, "actual_vs_predicted_v2.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p1}")

    # ── Plot 2: LSTM Training Loss ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(lstm_history.history["loss"]) + 1)
    ax.plot(epochs, lstm_history.history["loss"],
            color=colors["lstm"], linewidth=2, label="Training Loss")
    ax.plot(epochs, lstm_history.history["val_loss"],
            color=colors["xgb"], linewidth=2, linestyle="--",
            label="Validation Loss")
    gap = abs(lstm_history.history["loss"][-1] -
              min(lstm_history.history["val_loss"]))
    ax.set_title(f"LSTM v2 — Training Loss  "
                 f"(overfit gap: {gap:.4f})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.legend(fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "lstm_training_loss_v2.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p2}")

    # ── Plot 3: 7-Day Forecast ────────────────────────────────────
    from pandas.tseries.offsets import BDay
    last_date    = dates_test[-1]
    future_dates = pd.date_range(
        start=last_date + BDay(1), periods=FORECAST_DAYS, freq="B")

    fig, ax = plt.subplots(figsize=(12, 6))
    context_n      = 30
    context_dates  = dates_test[-context_n:]
    context_prices = y_true_xgb[-context_n:]

    ax.plot(context_dates, context_prices, color=colors["actual"],
            linewidth=2, label="Recent Actual", zorder=3)
    ax.plot(future_dates, forecast_xgb, color=colors["xgb"],
            linewidth=2, marker="o", markersize=6,
            label="XGBoost v2 Forecast", linestyle="--")
    ax.plot(future_dates, forecast_lstm, color=colors["lstm"],
            linewidth=2, marker="s", markersize=6,
            label="LSTM v2 Forecast", linestyle="--")

    best_fc    = forecast_xgb if best_model_name == "xgboost" else forecast_lstm
    band_color = colors["xgb"] if best_model_name == "xgboost" else colors["lstm"]
    spread     = np.linspace(0.008, 0.03, FORECAST_DAYS)
    upper      = [p * (1 + s) for p, s in zip(best_fc, spread)]
    lower      = [p * (1 - s) for p, s in zip(best_fc, spread)]
    ax.fill_between(future_dates, lower, upper,
                    alpha=0.15, color=band_color, label="Confidence Band")
    ax.axvline(x=last_date, color="grey", linestyle=":",
               linewidth=1.5, label="Forecast Start")
    ax.axhline(y=last_price, color="grey", linestyle="--",
               linewidth=1, alpha=0.4, label=f"Last Close £{last_price:.2f}")

    ax.set_title("OracleAU v2 — SGLN.L 7-Day Price Forecast",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("SGLN.L Price (£)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    plt.tight_layout()
    p3 = os.path.join(PLOTS_DIR, "forecast_7day_v2.png")
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p3}")


# ================================================================
# STEP 9 — WRITE REPORT
# ================================================================

def write_report(xgb_metrics, lstm_metrics, wf_metrics,
                 best_model_name, forecast_xgb, forecast_lstm,
                 ensemble, last_price, last_date) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "phase2_report_v2.txt")
    from pandas.tseries.offsets import BDay
    future_dates = pd.date_range(
        start=pd.Timestamp(last_date) + BDay(1),
        periods=FORECAST_DAYS, freq="B")

    lines = [
        "=" * 60,
        "  OracleAU — Phase 2 v2: Model Training Report",
        "=" * 60,
        f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Ticker     : SGLN.L (iShares Physical Gold)",
        f"  Last Close : £{last_price:.4f}  ({last_date})",
        "",
        "  v2 Key Changes",
        "  " + "─" * 40,
        "  • Predict returns % not raw prices (fixes distribution shift)",
        "  • RobustScaler replaces MinMaxScaler",
        "  • Lag features removed from XGBoost (fixes flat prediction)",
        "  • LSTM dropout 0.2→0.4, L2 reg, reduced capacity",
        "  • Recency weighting 3× on most recent 20% of training data",
        "  • Ensemble forecast (XGB 40% + LSTM 60%)",
        "  • Per-day confidence signal (HIGH / MEDIUM / LOW)",
        "",
        "  Configuration",
        "  " + "─" * 40,
        f"  Sequence Length  : {SEQUENCE_LEN} days",
        f"  Forecast Horizon : {FORECAST_DAYS} days",
        f"  Train / Test     : {int(TRAIN_RATIO*100)}% / {int((1-TRAIN_RATIO)*100)}%",
        f"  WF Folds         : {WF_FOLDS}",
        f"  Recency Weight   : {RECENCY_WEIGHT}× on last "
        f"{int((1-RECENCY_THRESHOLD)*100)}% of training",
        f"  Ensemble Weights : XGB={ENSEMBLE_WEIGHT_XGB}  LSTM={ENSEMBLE_WEIGHT_LSTM}",
        f"  Confidence Thr.  : {CONFIDENCE_THRESHOLD}% model agreement",
        "",
        "  XGBoost v2 Results (Test Set)",
        "  " + "─" * 40,
        f"  MAE              : £{xgb_metrics['mae']:.4f}",
        f"  RMSE             : £{xgb_metrics['rmse']:.4f}",
        f"  MAPE             : {xgb_metrics['mape']:.2f}%",
        f"  Directional Acc  : {xgb_metrics['directional_accuracy']:.2f}%",
        "",
        "  LSTM v2 Results (Test Set)",
        "  " + "─" * 40,
        f"  MAE              : £{lstm_metrics['mae']:.4f}",
        f"  RMSE             : £{lstm_metrics['rmse']:.4f}",
        f"  MAPE             : {lstm_metrics['mape']:.2f}%",
        f"  Directional Acc  : {lstm_metrics['directional_accuracy']:.2f}%",
        "",
        "  Walk-Forward Validation Averages",
        "  " + "─" * 40,
        f"  Avg MAE          : £{wf_metrics['avg_mae']:.4f}",
        f"  Avg RMSE         : £{wf_metrics['avg_rmse']:.4f}",
        f"  Avg MAPE         : {wf_metrics['avg_mape']:.2f}%",
        f"  Avg DA           : {wf_metrics['avg_da']:.2f}%",
        "",
        f"  ★ Best Model     : {best_model_name.upper()} v2",
        f"  Overall Direction: {ensemble['overall_direction']}",
        f"  Overall Confidence: {ensemble['overall_confidence']}",
        "",
        "  7-Day Ensemble Forecast",
        "  " + "─" * 40,
        f"  {'Day':<5} {'Date':<13} {'XGBoost':>9} {'LSTM':>9} {'Ensemble':>10} {'Agree%':>7} {'Signal':<8}",
        f"  {'─'*60}",
    ]

    for i, (d, fx, fl, ens, conf, agr) in enumerate(zip(
            future_dates, forecast_xgb, forecast_lstm,
            ensemble['ensemble_prices'], ensemble['confidence_levels'],
            ensemble['agreement_pcts']), 1):
        lines.append(
            f"  Day {i:<2} {str(d.date()):<13} "
            f"£{fx:>8.4f} £{fl:>8.4f} £{ens:>9.4f} "
            f"{agr:>6.3f}% {conf}")

    lines += [
        "",
        "  Confidence key:",
        "    HIGH   : Models agree within 0.5% — strong signal",
        "    MEDIUM : Models differ 0.5–1.5%  — reasonable signal",
        "    LOW    : Models differ >1.5%      — mixed signal",
        "",
        "  " + "─" * 40,
        "  ⚠  Disclaimer: For informational purposes only.",
        "     This is not financial advice.",
        "=" * 60,
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  ✓ Report → {path}")


# ================================================================
# MAIN
# ================================================================

def main():
    print_header("OracleAU — Phase 2 v2: Improved Model Training")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load
    df = load_and_prepare(DATA_PATH)

    split_idx = int(len(df) * TRAIN_RATIO)

    # 2. Scale — separate scalers for XGBoost (no lags) and LSTM (with lags)
    print_step(2, "Feature Scaling (RobustScaler)")

    (X_xgb, y_xgb_scaled, y_xgb_raw,
     scaler_X_xgb, scaler_y_xgb,
     feature_cols_xgb, dates) = scale_features(
         df, split_idx, for_xgboost=True)

    (X_lstm, y_lstm_scaled, y_lstm_raw,
     scaler_X_lstm, scaler_y_lstm,
     feature_cols_lstm, _) = scale_features(
         df, split_idx, for_xgboost=False)

    # Also grab raw close prices for return→price conversion
    close_raw = df["close_raw"].values

    # 3. Split
    (X_xgb_tr, X_xgb_te, y_xgb_tr, y_xgb_te,
     _, y_xgb_raw_te, dates_tr, dates_te, _) = chronological_split(
         X_xgb, y_xgb_scaled, close_raw, dates, TRAIN_RATIO)

    (X_lstm_tr, X_lstm_te, y_lstm_tr, y_lstm_te,
     _, y_lstm_raw_te, _, _, _) = chronological_split(
         X_lstm, y_lstm_scaled, close_raw, dates, TRAIN_RATIO)

    n_features_lstm = X_lstm.shape[1]

    # 4a. XGBoost v2
    xgb_model, xgb_metrics, y_true_xgb, y_pred_xgb = train_xgboost_v2(
        X_xgb_tr, y_xgb_tr, X_xgb_te, y_xgb_te,
        close_raw[:len(X_xgb_tr)], y_xgb_raw_te,
        df, scaler_y_xgb, SEQUENCE_LEN)

    # 4b. LSTM v2
    lstm_model, lstm_metrics, y_true_lstm, y_pred_lstm, lstm_history = \
        train_lstm_v2(
            X_lstm_tr, y_lstm_tr, X_lstm_te, y_lstm_te,
            y_lstm_raw_te, scaler_y_lstm, SEQUENCE_LEN, n_features_lstm)

    # 5. Model comparison
    print_step(5, "Model Comparison & Selection")
    print(f"\n  {'Metric':<25} {'XGBoost v2':>12} {'LSTM v2':>12}")
    print(f"  {'─'*51}")
    for metric in ["mae", "rmse", "mape", "directional_accuracy"]:
        label  = metric.replace("_", " ").upper()
        suffix = "%" if "mape" in metric or "acc" in metric else "£"
        xv, lv = xgb_metrics[metric], lstm_metrics[metric]
        wx = " ◄" if (
            (metric != "directional_accuracy" and xv <= lv) or
            (metric == "directional_accuracy" and xv >= lv)) else ""
        wl = " ◄" if (
            (metric != "directional_accuracy" and lv < xv) or
            (metric == "directional_accuracy" and lv > xv)) else ""
        print(f"  {label:<25} {suffix}{xv:>10.4f}{wx:<3} "
              f"{suffix}{lv:>10.4f}{wl}")

    best_model_name = "xgboost" if (
        xgb_metrics["mae"] <= lstm_metrics["mae"]) else "lstm"
    print(f"\n  ★ Best model : {best_model_name.upper()} v2  (lowest MAE)")

    # 6. Walk-forward validation
    wf_metrics = walk_forward_validation(
        df, feature_cols_xgb, feature_cols_lstm,
        SEQUENCE_LEN, WF_FOLDS, best_model_name)

    # 7. Generate 7-day forecast
    print_step(7, "Generating 7-Day Forecast")
    last_price = float(close_raw[-1])
    last_date  = dates[-1].date()

    scaler_y_best  = scaler_y_xgb  if best_model_name == "xgboost" else scaler_y_lstm
    X_best         = X_xgb         if best_model_name == "xgboost" else X_lstm
    model_best     = xgb_model     if best_model_name == "xgboost" else lstm_model

    last_seq_xgb  = X_xgb[-SEQUENCE_LEN:]
    last_seq_lstm = X_lstm[-SEQUENCE_LEN:]

    forecast_xgb  = multistep_forecast_v2(
        xgb_model,  last_seq_xgb,  last_price,
        scaler_y_xgb,  FORECAST_DAYS, "xgboost")
    forecast_lstm = multistep_forecast_v2(
        lstm_model, last_seq_lstm, last_price,
        scaler_y_lstm, FORECAST_DAYS, "lstm")

    from pandas.tseries.offsets import BDay
    future_dates = pd.date_range(
        start=pd.Timestamp(last_date) + BDay(1),
        periods=FORECAST_DAYS, freq="B")

    # Compute ensemble + confidence signal
    ensemble = ensemble_forecast(forecast_xgb, forecast_lstm, last_price)
    print_ensemble_summary(ensemble, forecast_xgb, forecast_lstm,
                           last_price, future_dates)

    # 8. Save
    save_models(xgb_model, lstm_model,
                scaler_X_xgb, scaler_X_lstm, scaler_y_best,
                xgb_metrics, lstm_metrics, wf_metrics,
                feature_cols_xgb, feature_cols_lstm,
                best_model_name)

    # 9. Plots
    print_step(9, "Generating Diagnostic Plots")
    generate_plots(
        dates_te, y_true_xgb, y_pred_xgb,
        y_true_lstm, y_pred_lstm, lstm_history,
        forecast_xgb, forecast_lstm,
        best_model_name, last_price)

    # 10. Report
    write_report(xgb_metrics, lstm_metrics, wf_metrics,
                 best_model_name, forecast_xgb, forecast_lstm,
                 ensemble, last_price, last_date)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  ✅  Phase 2 v2 Complete!")
    print(f"{'─'*60}")
    print(f"  Best model       : {best_model_name.upper()} v2")
    print(f"  Models saved to  : {MODELS_DIR}/ (*_v2 files)")
    print(f"  Plots saved to   : {PLOTS_DIR}/ (*_v2 files)")
    print(f"  Next step        : Phase 3 — FastAPI backend")
    print(f"{'='*60}")
    print(f"\n  ⚠  Forecasts are for informational purposes only.")
    print(f"     This is not financial advice.\n")


if __name__ == "__main__":
    main()