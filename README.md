# OracleAU

**7-day price forecasting for SGLN.L** (iShares Physical Gold ETF, LSE) using an XGBoost v2 + LSTM v2 ensemble model.

Live demo → `https://<your-username>.github.io/OracleAU`

---

## What It Does

- Fetches daily OHLCV data for SGLN.L from Stooq (free, no API key)
- Engineers 30+ technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, volatility)
- Predicts 7-day price returns using an ensemble of XGBoost and LSTM
- Serves forecasts, live price, indicators, and history via a FastAPI REST API
- Displays everything in a dark-themed trading dashboard (GitHub Pages frontend)

---

## Architecture

```
GitHub Pages (index.html)
        ↓  fetch()
Render API  (api.py / FastAPI)
        ↓  HTTP
Stooq CSV   (sgln.uk daily OHLCV)
        +
models/     (xgboost_v2.pkl + lstm_v2.h5 + scalers)
```

---

## Project Structure

```
OracleAU/
├── index.html              # Frontend — GitHub Pages
├── api.py                  # Backend  — FastAPI (deployed to Render)
├── data_pipeline.py        # Phase 1  — data fetch + feature engineering
├── model_training_v2.py    # Phase 2  — model training
├── requirements.txt
├── render.yaml             # Render deployment config
│
├── models/
│   ├── xgboost_v2.pkl
│   ├── lstm_v2.h5
│   ├── scaler_X_xgb_v2.pkl
│   ├── scaler_X_lstm_v2.pkl
│   ├── scaler_y_v2.pkl
│   └── model_metadata_v2.json
│
└── data/                   # gitignored — regenerate with data_pipeline.py
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | API status + model info |
| `GET /api/price` | Live SGLN price, OHLCV, 52-week range |
| `GET /api/forecast` | 7-day ensemble forecast + confidence |
| `GET /api/history?period=3m` | Historical OHLCV (1w/1m/3m/6m/1y/5y) |
| `GET /api/indicators` | RSI, MACD, Bollinger Bands, trend signals |

Interactive docs: `https://oracleau.onrender.com/docs`

---

## Model Performance

| Metric | XGBoost v2 | LSTM v2 | Ensemble |
|---|---|---|---|
| Test MAE | £0.50 | £0.51 | — |
| Test MAPE | 0.96% | 0.98% | — |
| Test DA | 53.4% | 45.4% | — |
| **Live DA (5 days)** | 60% | 60% | **80%** |
| **Live MAE** | £0.87 | £0.86 | **£0.85** |

Ensemble weights: XGBoost 40% · LSTM 60%

---

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch fresh data
python data_pipeline.py

# 3. Train models (or use pre-trained from models/)
python model_training_v2.py

# 4. Start API
uvicorn api:app --reload --port 8000

# 5. Open frontend
open index.html
```

---

## Retraining

Models are trained on historical data and frozen at inference time. The API always fetches the latest price data from Stooq but uses fixed model weights until you retrain.

To retrain with new data:
```bash
python data_pipeline.py       # Fetch latest data
python model_training_v2.py   # Retrain both models
git add models/
git commit -m "retrain: <date>"
git push                       # Render auto-redeploys
```

---

## Disclaimer

OracleAU is an experimental machine learning project for educational purposes only. Forecasts do not constitute financial advice. Past model performance does not guarantee future accuracy.

---

*Built with FastAPI · XGBoost · TensorFlow · Chart.js · Stooq*