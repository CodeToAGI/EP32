# EP32 — Time Series Forecasting with LSTM + TFT Intro

**CodeToAGI Deep Learning Series — Episode 32**  
Module 7 finale: apply RNNs, Seq2Seq and Attention ideas to real time-series forecasting.

## What you build
- Sliding-window dataset from any univariate series
- 2-layer LSTM + direct multi-step head (horizon = 24 for energy, 7 for BTC challenge)
- Correct chronological split + StandardScaler fitted **only on train**
- Full evaluation: MAE, RMSE, MAPE
- Comparison against naïve baseline

## Quick start (Challenge)

```bash
pip install torch pandas matplotlib yfinance scikit-learn
python ep32_ts_forecast.py
The script:

Downloads daily BTC-USD from Yahoo Finance
Builds a 30-day look-back → 7-day forecast LSTM
Trains for 50 epochs
Prints MAPE vs naïve model
Saves a forecast plot (btc_7day_forecast.png)

Key rules (never break these)

Never shuffle time-series data
Fit the scaler only on the training split
Split the raw series first, then create windows inside each split
Always inverse-transform before computing MAE / RMSE / MAPE

Results on the energy dataset (shown in the video)



EP33 — The Transformer Architecture

“Attention Is All You Need” from the ground up.
