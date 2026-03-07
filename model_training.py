"""
================================================================
OracleAU — Phase 2: Model Training
================================================================
Script:      model_training.py
Description: Loads the feature-engineered SGLN data from Phase 1,
             trains two models (XGBoost + LSTM), evaluates both
             using walk-forward validation, and saves the best
             model for use in Phase 3 (FastAPI backend).

Models:
    - XGBoost Regressor  : Gradient boosting on tabular features
    - LSTM               : Deep learning on 45-day sequences

Evaluation Metrics:
    - MAE   : Mean Absolute Error (£)
    - RMSE  : Root Mean Squared Error (£)
    - MAPE  : Mean Absolute Percentage Error (%)
    - DA    : Directional Accuracy (%)

Usage:
    python model_training.py

Output:
    models/xgboost_model.pkl   — Trained XGBoost model
    models/lstm_model.h5       — Trained LSTM model
    models/scaler.pkl          — Fitted MinMaxScaler
    models/model_metadata.json — Metrics + best model name
    reports/phase2_report.txt  — Full training report

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
matplotlib.use("Agg")   # Non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# TensorFlow / Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Suppress TF verbose logs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")


# ================================================================
# CONFIGURATION
# ================================================================

# Paths
DATA_PATH     = "data/SGLN_features.csv"
MODELS_DIR    = "models"
REPORTS_DIR   = "reports"
PLOTS_DIR     = "reports/plots"

# Data
DROP_COLS = [
    "repaired?",        # yfinance metadata — not a feature
    "target_direction", # classification target — not used for regression
]

# The column we are predicting
TARGET_COL    = "target_next_close"

# Columns that are targets / future data — must NOT be in X
LEAK_COLS     = ["target_next_close", "target_direction"]

# Train / test split
TRAIN_RATIO   = 0.80     # 80% training, 20% testing (chronological)

# LSTM
SEQUENCE_LEN  = 45       # Look back 45 trading days (agreed)
FORECAST_DAYS = 7        # Predict up to 7 days ahead
LSTM_UNITS    = 64       # Neurons per LSTM layer
DROPOUT_RATE  = 0.2      # Dropout for regularisation
EPOCHS        = 100      # Max training epochs (early stopping will kick in)
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

# Walk-Forward Validation
WF_FOLDS      = 5        # Number of walk-forward folds


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
    """Mean Absolute Percentage Error — avoids division by zero."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                          y_prev: np.ndarray) -> float:
    """
    What % of the time did the model correctly predict
    whether tomorrow's price is higher or lower than today's?
    y_prev = today's actual close price (the baseline to compare against).
    """
    actual_dir    = (y_true > y_prev).astype(int)
    predicted_dir = (y_pred > y_prev).astype(int)
    return float(np.mean(actual_dir == predicted_dir) * 100)

def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_prev: np.ndarray, label: str) -> dict:
    """Compute and print all four evaluation metrics."""
    mae_val  = mean_absolute_error(y_true, y_pred)
    rmse_val = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape_val = mape(y_true, y_pred)
    da_val   = directional_accuracy(y_true, y_pred, y_prev)

    print(f"\n  {label} Results:")
    print(f"    MAE               : £{mae_val:.4f}")
    print(f"    RMSE              : £{rmse_val:.4f}")
    print(f"    MAPE              : {mape_val:.2f}%")
    print(f"    Directional Acc   : {da_val:.2f}%")

    return {"mae": mae_val, "rmse": rmse_val,
            "mape": mape_val, "directional_accuracy": da_val}


# ================================================================
# STEP 1 — LOAD & PREPARE DATA
# ================================================================

def load_and_prepare(path: str):
    """
    Load the features CSV, drop unwanted columns, handle NaNs,
    and return a clean DataFrame ready for splitting.
    """
    print_step(1, "Loading & Preparing Data")

    df = pd.read_csv(path, index_col="date", parse_dates=True)
    print(f"  Loaded        : {len(df):,} rows × {len(df.columns)} columns")

    # Drop metadata and classification target columns
    df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

    # Drop all rows with ANY NaN — removes warmup period (first ~199 rows)
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    print(f"  Dropped NaNs  : {dropped} warmup rows removed")
    print(f"  Usable rows   : {len(df):,}  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print(f"  Feature cols  : {len(df.columns) - 1}  (excl. target)")

    return df


# ================================================================
# STEP 2 — FEATURE SCALING
# ================================================================

def scale_features(df: pd.DataFrame, train_end_idx: int):
    """
    Fit MinMaxScaler on training data only, then transform
    the full dataset. Returns scaled arrays and the scaler.

    IMPORTANT: Scaler is fitted ONLY on train rows to prevent
    data leakage — test rows must be unseen during fitting.
    """
    print_step(2, "Feature Scaling")

    feature_cols = [c for c in df.columns if c not in LEAK_COLS]
    target_col   = TARGET_COL

    X = df[feature_cols].values
    y = df[target_col].values
    dates = df.index

    # Fit scaler only on training portion
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_train_raw = X[:train_end_idx]
    y_train_raw = y[:train_end_idx].reshape(-1, 1)

    scaler_X.fit(X_train_raw)
    scaler_y.fit(y_train_raw)

    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()

    print(f"  Feature scaler fitted on {train_end_idx:,} training rows")
    print(f"  Feature range  : 0.0 → 1.0 (MinMax)")
    print(f"  Features (X)   : {X_scaled.shape[1]} columns")
    print(f"  Target (y)     : '{target_col}' (next day close £)")

    return X_scaled, y_scaled, y, scaler_X, scaler_y, feature_cols, dates


# ================================================================
# STEP 3 — TRAIN / TEST SPLIT (Chronological)
# ================================================================

def chronological_split(X: np.ndarray, y_scaled: np.ndarray,
                         y_raw: np.ndarray, dates, ratio: float):
    """
    Split data in chronological order — never shuffle time series.
    Returns train and test sets with their corresponding dates.
    """
    print_step(3, "Chronological Train / Test Split")

    split_idx = int(len(X) * ratio)

    X_train, X_test       = X[:split_idx],         X[split_idx:]
    y_train, y_test       = y_scaled[:split_idx],  y_scaled[split_idx:]
    y_raw_train, y_raw_test = y_raw[:split_idx],   y_raw[split_idx:]
    dates_train, dates_test = dates[:split_idx],   dates[split_idx:]

    train_start = dates_train[0].date()
    train_end   = dates_train[-1].date()
    test_start  = dates_test[0].date()
    test_end    = dates_test[-1].date()

    print(f"  Training set  : {len(X_train):,} rows  ({train_start} → {train_end})")
    print(f"  Test set      : {len(X_test):,} rows   ({test_start} → {test_end})")
    print(f"  Split ratio   : {ratio*100:.0f}% / {(1-ratio)*100:.0f}%")

    return (X_train, X_test, y_train, y_test,
            y_raw_train, y_raw_test, dates_train, dates_test, split_idx)


# ================================================================
# STEP 4A — XGBOOST MODEL
# ================================================================

def build_sequences_xgb(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    For XGBoost, flatten the last seq_len rows into a single wide
    feature vector per sample. This lets XGBoost "see" sequential
    context without needing a recurrent architecture.
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        # Flatten the window: seq_len × n_features → 1D vector
        Xs.append(X[i - seq_len:i].flatten())
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  y_raw_test: np.ndarray, y_raw_train: np.ndarray,
                  scaler_y: MinMaxScaler, seq_len: int) -> tuple:
    """
    Build sequences, train XGBoost, evaluate on test set,
    and return the trained model + metrics.
    """
    print_step(4, "Training XGBoost Model")

    # Build windowed sequences
    X_tr_seq, y_tr_seq = build_sequences_xgb(X_train, y_train, seq_len)
    X_te_seq, y_te_seq = build_sequences_xgb(X_test,  y_test,  seq_len)
    print(f"  Train sequences : {X_tr_seq.shape}")
    print(f"  Test  sequences : {X_te_seq.shape}")

    # XGBoost model definition
    model = xgb.XGBRegressor(
        n_estimators      = 500,
        learning_rate     = 0.05,
        max_depth         = 6,
        min_child_weight  = 2,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        gamma             = 0.1,
        reg_alpha         = 0.1,    # L1 regularisation
        reg_lambda        = 1.0,    # L2 regularisation
        random_state      = 42,
        n_jobs            = -1,     # Use all CPU cores
        early_stopping_rounds = 30,
        eval_metric       = "rmse",
        verbosity         = 0,
    )

    print(f"\n  Training XGBoost (500 estimators, early stopping @ 30)...")
    model.fit(
        X_tr_seq, y_tr_seq,
        eval_set       = [(X_te_seq, y_te_seq)],
        verbose        = False,
    )

    best_iter = model.best_iteration
    print(f"  ✓ Best iteration  : {best_iter}")

    # Predict and inverse-transform back to £
    y_pred_scaled = model.predict(X_te_seq)
    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(
        y_te_seq.reshape(-1, 1)).flatten()

    # Previous day prices for directional accuracy
    # y_raw_test[seq_len-1:-1] = today's actual close for each test sample
    y_prev = y_raw_test[seq_len - 1: seq_len - 1 + len(y_true)]

    metrics = evaluate(y_true, y_pred, y_prev, "XGBoost Test Set")

    return model, metrics, y_true, y_pred


# ================================================================
# STEP 4B — LSTM MODEL
# ================================================================

def build_sequences_lstm(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    For LSTM, keep the 3D shape: (samples, timesteps, features).
    Each sample is a matrix of the last seq_len days.
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])    # shape: (seq_len, n_features)
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray, y_test: np.ndarray,
               y_raw_test: np.ndarray, scaler_y: MinMaxScaler,
               seq_len: int, n_features: int) -> tuple:
    """
    Build 3D sequences, define and train the LSTM architecture,
    evaluate on test set, and return trained model + metrics.
    """
    print_step(5, "Training LSTM Model")

    # Build 3D sequences
    X_tr_seq, y_tr_seq = build_sequences_lstm(X_train, y_train, seq_len)
    X_te_seq, y_te_seq = build_sequences_lstm(X_test,  y_test,  seq_len)
    print(f"  Train sequences : {X_tr_seq.shape}  (samples × days × features)")
    print(f"  Test  sequences : {X_te_seq.shape}")

    # ── LSTM Architecture ─────────────────────────────────────────
    # Two stacked LSTM layers with dropout for regularisation.
    # Final Dense layer outputs a single price prediction.
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(LSTM_UNITS, return_sequences=True),   # First LSTM layer
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS // 2, return_sequences=False),  # Second LSTM layer
        Dropout(DROPOUT_RATE),
        Dense(32, activation="relu"),
        Dense(1),                                   # Single output: next close
    ], name="OracleAU_LSTM")

    model.compile(
        optimizer = Adam(learning_rate=LEARNING_RATE),
        loss      = "huber",    # Huber loss: less sensitive to outliers than MSE
        metrics   = ["mae"],
    )

    print(f"\n  LSTM Architecture:")
    model.summary(print_fn=lambda x: print(f"    {x}"))

    # ── Callbacks ─────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor   = "val_loss",
            patience  = 15,           # Stop if no improvement for 15 epochs
            restore_best_weights = True,
            verbose   = 0,
        ),
        ReduceLROnPlateau(
            monitor   = "val_loss",
            factor    = 0.5,          # Halve the learning rate when plateauing
            patience  = 8,
            min_lr    = 1e-6,
            verbose   = 0,
        ),
    ]

    print(f"\n  Training LSTM (max {EPOCHS} epochs, early stopping @ patience=15)...")
    history = model.fit(
        X_tr_seq, y_tr_seq,
        validation_data = (X_te_seq, y_te_seq),
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        callbacks       = callbacks,
        verbose         = 0,
    )

    epochs_run = len(history.history["loss"])
    best_val   = min(history.history["val_loss"])
    print(f"  ✓ Stopped at epoch : {epochs_run}")
    print(f"  ✓ Best val loss    : {best_val:.6f}")

    # Predict and inverse-transform
    y_pred_scaled = model.predict(X_te_seq, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(
        y_te_seq.reshape(-1, 1)).flatten()

    y_prev = y_raw_test[seq_len - 1: seq_len - 1 + len(y_true)]

    metrics = evaluate(y_true, y_pred, y_prev, "LSTM Test Set")

    return model, metrics, y_true, y_pred, history


# ================================================================
# STEP 5 — WALK-FORWARD VALIDATION
# ================================================================

def walk_forward_validation(df: pd.DataFrame, scaler_X: MinMaxScaler,
                             scaler_y: MinMaxScaler, feature_cols: list,
                             seq_len: int, n_folds: int,
                             best_model_name: str) -> dict:
    """
    Evaluate model robustness across 5 chronological folds.
    Each fold trains on all data up to that point and tests
    on the next unseen period — simulating real-world use.
    """
    print_step(6, f"Walk-Forward Validation ({n_folds} folds)")

    X_all = df[feature_cols].values
    y_all = df[TARGET_COL].values

    fold_size  = len(X_all) // (n_folds + 1)
    all_metrics = []

    for fold in range(n_folds):
        train_end  = fold_size * (fold + 1)
        test_start = train_end
        test_end   = train_end + fold_size

        if test_end > len(X_all):
            break

        X_tr = X_all[:train_end]
        X_te = X_all[test_start:test_end]
        y_tr = y_all[:train_end]
        y_te = y_all[test_start:test_end]

        # Fit fresh scaler on this fold's training data
        sc_X = MinMaxScaler().fit(X_tr)
        sc_y = MinMaxScaler().fit(y_tr.reshape(-1, 1))

        X_tr_s = sc_X.transform(X_tr)
        X_te_s = sc_X.transform(X_te)
        y_tr_s = sc_y.transform(y_tr.reshape(-1, 1)).flatten()
        y_te_s = sc_y.transform(y_te.reshape(-1, 1)).flatten()

        if best_model_name == "xgboost":
            X_tr_seq, y_tr_seq = build_sequences_xgb(X_tr_s, y_tr_s, seq_len)
            X_te_seq, y_te_seq = build_sequences_xgb(X_te_s, y_te_s, seq_len)

            m = xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.05,
                max_depth=6, random_state=42,
                verbosity=0, n_jobs=-1
            )
            m.fit(X_tr_seq, y_tr_seq, verbose=False)
            y_pred_s = m.predict(X_te_seq)

        else:  # lstm
            n_feat = X_tr_s.shape[1]
            X_tr_seq, y_tr_seq = build_sequences_lstm(X_tr_s, y_tr_s, seq_len)
            X_te_seq, y_te_seq = build_sequences_lstm(X_te_s, y_te_s, seq_len)

            m = Sequential([
                Input(shape=(seq_len, n_feat)),
                LSTM(32, return_sequences=False),
                Dense(1)
            ])
            m.compile(optimizer="adam", loss="mse")
            m.fit(X_tr_seq, y_tr_seq,
                  epochs=30, batch_size=32, verbose=0,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(patience=5,
                                           restore_best_weights=True,
                                           verbose=0)])
            y_pred_s = m.predict(X_te_seq, verbose=0).flatten()

        # Inverse transform predictions back to £
        y_pred = sc_y.inverse_transform(
            y_pred_s.reshape(-1, 1)).flatten()
        y_true_fold = y_te[seq_len:]

        # Align lengths
        min_len = min(len(y_pred), len(y_true_fold))
        y_pred  = y_pred[:min_len]
        y_true_fold = y_true_fold[:min_len]
        y_prev_fold = y_tr[-1:].repeat(min_len)

        fold_mae  = mean_absolute_error(y_true_fold, y_pred)
        fold_rmse = float(np.sqrt(mean_squared_error(y_true_fold, y_pred)))
        fold_mape = mape(y_true_fold, y_pred)
        fold_da   = directional_accuracy(y_true_fold, y_pred, y_prev_fold)

        all_metrics.append({
            "mae": fold_mae, "rmse": fold_rmse,
            "mape": fold_mape, "da": fold_da
        })

        train_dates = df.index[:train_end]
        test_dates  = df.index[test_start:test_end]
        print(f"  Fold {fold+1}  "
              f"Train→{train_dates[-1].date()}  "
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

def multistep_forecast(model, last_sequence: np.ndarray,
                        scaler_y: MinMaxScaler, n_days: int,
                        model_type: str, n_features: int) -> list:
    """
    Generate a 1–7 day forecast by iteratively feeding each
    prediction back as input for the next step.

    This is called 'recursive multi-step forecasting':
    Day 1 prediction becomes part of the input for Day 2, etc.
    Each step adds uncertainty, so confidence naturally widens.
    """
    predictions = []
    current_seq = last_sequence.copy()   # shape: (seq_len, n_features)

    for day in range(n_days):
        if model_type == "xgboost":
            inp   = current_seq.flatten().reshape(1, -1)
            pred_s = model.predict(inp)[0]
        else:
            inp   = current_seq.reshape(1, current_seq.shape[0],
                                         current_seq.shape[1])
            pred_s = model.predict(inp, verbose=0)[0][0]

        # Inverse transform to get actual £ price
        pred_price = float(scaler_y.inverse_transform(
            np.array([[pred_s]])).flatten()[0])
        predictions.append(pred_price)

        # Roll the sequence forward: drop oldest day, append new prediction
        # We update only the 'close' position (index 0) with the prediction
        new_row = current_seq[-1].copy()
        new_row[0] = pred_s   # Update close price with scaled prediction
        current_seq = np.vstack([current_seq[1:], new_row])

    return predictions


# ================================================================
# STEP 7 — SAVE MODELS & METADATA
# ================================================================

def save_models(xgb_model, lstm_model, scaler_X: MinMaxScaler,
                scaler_y: MinMaxScaler, xgb_metrics: dict,
                lstm_metrics: dict, wf_metrics: dict,
                feature_cols: list, best_model_name: str) -> None:
    """
    Save both models, scalers, and a JSON metadata file
    that Phase 3 (FastAPI) will read on startup.
    """
    print_step(7, "Saving Models & Metadata")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save XGBoost
    xgb_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    joblib.dump(xgb_model, xgb_path)
    print(f"  ✓ XGBoost saved   → {xgb_path}")

    # Save LSTM
    lstm_path = os.path.join(MODELS_DIR, "lstm_model.h5")
    lstm_model.save(lstm_path)
    print(f"  ✓ LSTM saved      → {lstm_path}")

    # Save scalers
    scaler_X_path = os.path.join(MODELS_DIR, "scaler_X.pkl")
    scaler_y_path = os.path.join(MODELS_DIR, "scaler_y.pkl")
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    print(f"  ✓ Scaler X saved  → {scaler_X_path}")
    print(f"  ✓ Scaler y saved  → {scaler_y_path}")

    # Save metadata JSON — read by Phase 3 API
    metadata = {
        "ticker"          : "SGLN.L",
        "name"            : "iShares Physical Gold (GBP)",
        "best_model"      : best_model_name,
        "sequence_length" : SEQUENCE_LEN,
        "forecast_days"   : FORECAST_DAYS,
        "feature_cols"    : feature_cols,
        "trained_at"      : datetime.now().isoformat(),
        "xgboost_metrics" : xgb_metrics,
        "lstm_metrics"    : lstm_metrics,
        "walk_forward"    : wf_metrics,
    }

    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved  → {meta_path}")


# ================================================================
# STEP 8 — GENERATE PLOTS
# ================================================================

def generate_plots(dates_test, y_true_xgb: np.ndarray,
                   y_pred_xgb: np.ndarray, y_true_lstm: np.ndarray,
                   y_pred_lstm: np.ndarray, lstm_history,
                   forecast_xgb: list, forecast_lstm: list,
                   best_model_name: str) -> None:
    """
    Generate and save three diagnostic plots:
    1. Actual vs Predicted (both models)
    2. LSTM training loss curve
    3. 7-day forecast from both models
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"actual": "#0f1b2d", "xgb": "#e8603a", "lstm": "#2a6dd9"}

    # ── Plot 1: Actual vs Predicted ───────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    for ax, y_true, y_pred, label, color in [
        (axes[0], y_true_xgb, y_pred_xgb, "XGBoost", colors["xgb"]),
        (axes[1], y_true_lstm, y_pred_lstm, "LSTM",   colors["lstm"]),
    ]:
        n = min(len(y_true), len(y_pred))
        plot_dates = dates_test[-n:]
        ax.plot(plot_dates, y_true[:n],  color=colors["actual"],
                linewidth=1.5, label="Actual",    alpha=0.9)
        ax.plot(plot_dates, y_pred[:n],  color=color,
                linewidth=1.2, label=f"{label} Predicted", alpha=0.8,
                linestyle="--")
        ax.set_title(f"{label} — Actual vs Predicted (Test Set)",
                     fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("SGLN.L Price (£)", fontsize=10)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    fig.suptitle("OracleAU — SGLN.L Price Prediction", fontsize=15,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, "actual_vs_predicted.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Plot saved → {p1}")

    # ── Plot 2: LSTM Training Loss ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(lstm_history.history["loss"]) + 1)
    ax.plot(epochs, lstm_history.history["loss"],
            color=colors["lstm"], linewidth=2, label="Training Loss")
    ax.plot(epochs, lstm_history.history["val_loss"],
            color=colors["xgb"], linewidth=2, label="Validation Loss",
            linestyle="--")
    ax.set_title("LSTM — Training vs Validation Loss", fontsize=13,
                 fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Huber Loss", fontsize=10)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "lstm_training_loss.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Plot saved → {p2}")

    # ── Plot 3: 7-Day Forecast ────────────────────────────────────
    from pandas.tseries.offsets import BDay
    last_date   = dates_test[-1]
    future_dates = pd.date_range(
        start=last_date + BDay(1), periods=FORECAST_DAYS, freq="B")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Show last 30 days of actuals as context
    context_n = 30
    context_dates = dates_test[-context_n:]
    context_prices = y_true_xgb[-context_n:]
    ax.plot(context_dates, context_prices, color=colors["actual"],
            linewidth=2, label="Recent Actual", zorder=3)

    # Plot forecasts
    ax.plot(future_dates, forecast_xgb,  color=colors["xgb"],
            linewidth=2, marker="o", markersize=6,
            label="XGBoost Forecast", linestyle="--")
    ax.plot(future_dates, forecast_lstm, color=colors["lstm"],
            linewidth=2, marker="s", markersize=6,
            label="LSTM Forecast",    linestyle="--")

    # Confidence band on the best model
    best_fc = forecast_xgb if best_model_name == "xgboost" else forecast_lstm
    band_color = colors["xgb"] if best_model_name == "xgboost" else colors["lstm"]
    spread = np.linspace(0.005, 0.025, FORECAST_DAYS)  # widens over time
    upper = [p * (1 + s) for p, s in zip(best_fc, spread)]
    lower = [p * (1 - s) for p, s in zip(best_fc, spread)]
    ax.fill_between(future_dates, lower, upper,
                    alpha=0.15, color=band_color, label="Confidence Band")

    # Vertical line at forecast start
    ax.axvline(x=last_date, color="grey", linestyle=":", linewidth=1.5,
               label="Forecast Start")

    ax.set_title("OracleAU — SGLN.L 7-Day Price Forecast",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("SGLN.L Price (£)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    plt.tight_layout()
    p3 = os.path.join(PLOTS_DIR, "forecast_7day.png")
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Plot saved → {p3}")


# ================================================================
# STEP 9 — WRITE PHASE 2 REPORT
# ================================================================

def write_report(xgb_metrics: dict, lstm_metrics: dict,
                 wf_metrics: dict, best_model_name: str,
                 forecast_xgb: list, forecast_lstm: list,
                 last_price: float, last_date) -> None:
    """Save a human-readable Phase 2 summary report."""

    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "phase2_report.txt")

    lines = [
        "=" * 60,
        "  OracleAU — Phase 2: Model Training Report",
        "=" * 60,
        f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Ticker     : SGLN.L (iShares Physical Gold)",
        f"  Last Close : £{last_price:.4f}  ({last_date})",
        "",
        "  Configuration",
        "  " + "─" * 40,
        f"  Sequence Length  : {SEQUENCE_LEN} days",
        f"  Forecast Horizon : {FORECAST_DAYS} days",
        f"  Train / Test     : {int(TRAIN_RATIO*100)}% / {int((1-TRAIN_RATIO)*100)}%",
        f"  WF Folds         : {WF_FOLDS}",
        "",
        "  XGBoost Results (Test Set)",
        "  " + "─" * 40,
        f"  MAE              : £{xgb_metrics['mae']:.4f}",
        f"  RMSE             : £{xgb_metrics['rmse']:.4f}",
        f"  MAPE             : {xgb_metrics['mape']:.2f}%",
        f"  Directional Acc  : {xgb_metrics['directional_accuracy']:.2f}%",
        "",
        "  LSTM Results (Test Set)",
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
        f"  ★ Best Model     : {best_model_name.upper()}",
        "",
        "  7-Day Price Forecast",
        "  " + "─" * 40,
    ]

    from pandas.tseries.offsets import BDay
    future_dates = pd.date_range(
        start=pd.Timestamp(last_date) + BDay(1),
        periods=FORECAST_DAYS, freq="B")

    lines.append(f"  {'Day':<6} {'Date':<14} {'XGBoost':>10} {'LSTM':>10}")
    lines.append(f"  {'─'*44}")
    for i, (d, fx, fl) in enumerate(
            zip(future_dates, forecast_xgb, forecast_lstm), 1):
        marker = " ◄ BEST" if (
            (best_model_name == "xgboost" and i == 1) or
            (best_model_name == "lstm" and i == 1)
        ) else ""
        lines.append(
            f"  Day {i:<2}  {str(d.date()):<14} "
            f"£{fx:>8.4f}  £{fl:>8.4f}{marker}")

    lines += [
        "",
        "  " + "─" * 40,
        "  ⚠  Disclaimer: For informational purposes only.",
        "     This is not financial advice.",
        "=" * 60,
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n  ✓ Report saved → {path}")


# ================================================================
# MAIN
# ================================================================

def main():
    print_header("OracleAU — Phase 2: Model Training")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load data
    df = load_and_prepare(DATA_PATH)

    # 2. Scale
    split_idx = int(len(df) * TRAIN_RATIO)
    feature_cols = [c for c in df.columns if c not in LEAK_COLS]
    (X_scaled, y_scaled, y_raw,
     scaler_X, scaler_y,
     feature_cols, dates) = scale_features(df, split_idx)

    # 3. Split
    (X_train, X_test,
     y_train, y_test,
     y_raw_train, y_raw_test,
     dates_train, dates_test,
     split_idx) = chronological_split(
         X_scaled, y_scaled, y_raw, dates, TRAIN_RATIO)

    n_features = X_scaled.shape[1]

    # 4a. XGBoost
    xgb_model, xgb_metrics, y_true_xgb, y_pred_xgb = train_xgboost(
        X_train, y_train, X_test, y_test,
        y_raw_test, y_raw_train, scaler_y, SEQUENCE_LEN)

    # 4b. LSTM
    lstm_model, lstm_metrics, y_true_lstm, y_pred_lstm, lstm_history = train_lstm(
        X_train, y_train, X_test, y_test,
        y_raw_test, scaler_y, SEQUENCE_LEN, n_features)

    # 5. Select best model
    print_step(5, "Model Comparison & Selection")
    print(f"\n  {'Metric':<25} {'XGBoost':>10} {'LSTM':>10}")
    print(f"  {'─'*47}")
    for metric in ["mae", "rmse", "mape", "directional_accuracy"]:
        label = metric.replace("_", " ").upper()
        suffix_xgb = "%" if "mape" in metric or "acc" in metric else "£"
        suffix_lst = suffix_xgb
        xv = xgb_metrics[metric]
        lv = lstm_metrics[metric]
        winner_x = " ◄" if (
            (metric != "directional_accuracy" and xv < lv) or
            (metric == "directional_accuracy" and xv > lv)
        ) else ""
        winner_l = " ◄" if (
            (metric != "directional_accuracy" and lv < xv) or
            (metric == "directional_accuracy" and lv > xv)
        ) else ""
        print(f"  {label:<25} {suffix_xgb}{xv:>9.4f}{winner_x:<3} "
              f"{suffix_lst}{lv:>9.4f}{winner_l}")

    # Decide winner: lower MAE wins (most interpretable metric)
    best_model_name = "xgboost" if (
        xgb_metrics["mae"] <= lstm_metrics["mae"]) else "lstm"
    print(f"\n  ★ Best model selected : {best_model_name.upper()}")
    print(f"    (based on lowest MAE on test set)")

    # 6. Walk-forward validation on best model
    wf_metrics = walk_forward_validation(
        df, scaler_X, scaler_y, feature_cols,
        SEQUENCE_LEN, WF_FOLDS, best_model_name)

    # 7. Generate 7-day forecast from the last known sequence
    print_step(7, "Generating 7-Day Forecast")
    last_seq = X_scaled[-SEQUENCE_LEN:]   # Most recent 45 days

    forecast_xgb  = multistep_forecast(
        xgb_model,  last_seq, scaler_y,
        FORECAST_DAYS, "xgboost", n_features)
    forecast_lstm = multistep_forecast(
        lstm_model, last_seq, scaler_y,
        FORECAST_DAYS, "lstm", n_features)

    last_price = float(y_raw[-1])
    last_date  = dates[-1].date()

    print(f"\n  Last known close : £{last_price:.4f}  ({last_date})")
    print(f"\n  {'Day':<6} {'XGBoost':>10} {'LSTM':>10}")
    print(f"  {'─'*28}")
    from pandas.tseries.offsets import BDay
    future_dates = pd.date_range(
        start=pd.Timestamp(last_date) + BDay(1),
        periods=FORECAST_DAYS, freq="B")
    for i, (d, fx, fl) in enumerate(
            zip(future_dates, forecast_xgb, forecast_lstm), 1):
        print(f"  Day {i}  ({d.date()})  £{fx:.4f}  £{fl:.4f}")

    # 8. Save models
    save_models(xgb_model, lstm_model, scaler_X, scaler_y,
                xgb_metrics, lstm_metrics, wf_metrics,
                feature_cols, best_model_name)

    # 9. Generate plots
    print_step(9, "Generating Diagnostic Plots")
    generate_plots(
        dates_test, y_true_xgb, y_pred_xgb,
        y_true_lstm, y_pred_lstm, lstm_history,
        forecast_xgb, forecast_lstm, best_model_name)

    # 10. Write report
    write_report(
        xgb_metrics, lstm_metrics, wf_metrics,
        best_model_name, forecast_xgb, forecast_lstm,
        last_price, last_date)

    # ── Final Summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅  Phase 2 Complete!")
    print(f"{'─'*60}")
    print(f"  Best model       : {best_model_name.upper()}")
    print(f"  Saved to         : {MODELS_DIR}/")
    print(f"  Plots saved to   : {PLOTS_DIR}/")
    print(f"  Report saved to  : {REPORTS_DIR}/phase2_report.txt")
    print(f"  Next step        : Phase 3 — FastAPI backend")
    print(f"{'='*60}\n")
    print(f"  ⚠  Disclaimer: Forecasts are for informational")
    print(f"     purposes only. Not financial advice.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()