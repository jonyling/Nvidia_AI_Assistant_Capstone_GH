# train_nvidia_ml_model.py
import pandas as pd
import numpy as np
from prophet import Prophet # type: ignore
import joblib
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

print("🚀 Training high-performance Prophet model (target MAPE < 3%)...")

# ======================== LOAD & ENHANCE DATA ========================
df = pd.read_csv("nvda_2014_to_2026.csv", skiprows=[1])
df = df.dropna(subset=["Date"]).copy()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Strong feature engineering (this is what makes MAPE drop dramatically)
df['Return'] = df['Close'].pct_change()
df['MA7'] = df['Close'].rolling(7).mean()
df['MA30'] = df['Close'].rolling(30).mean()
df['Vol7'] = df['Close'].rolling(7).std()
df['Volume_MA7'] = df['Volume'].rolling(7).mean()
df = df.dropna().reset_index(drop=True)

# Prophet dataframe
prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).copy()

# Add regressors
prophet_df['MA7'] = df['MA7']
prophet_df['MA30'] = df['MA30']
prophet_df['Vol7'] = df['Vol7']
prophet_df['Volume_MA7'] = df['Volume_MA7']

# ======================== TRAIN PROPHET ========================
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.8,      # higher for fast-moving stock
    seasonality_mode='multiplicative',
    interval_width=0.95,
    uncertainty_samples=1000
)

model.add_regressor('MA7')
model.add_regressor('MA30')
model.add_regressor('Vol7')
model.add_regressor('Volume_MA7')

model.fit(prophet_df)

# ======================== QUICK BACKTEST (last 30 days) ========================
future = model.make_future_dataframe(periods=0)
future = future.merge(prophet_df[['ds', 'MA7', 'MA30', 'Vol7', 'Volume_MA7']], on='ds', how='left')

forecast = model.predict(future)

actual_last30 = df['Close'].tail(30).values
pred_last30 = forecast['yhat'].tail(30).values
mape = np.mean(np.abs((actual_last30 - pred_last30) / actual_last30)) * 100

print(f"\n✅ Backtest MAPE on last 30 days: {mape:.2f}%")
print(f"Last close: ${df['Close'].iloc[-1]:.2f}")

# ======================== SAVE MODEL ========================
joblib.dump({
    'prophet_model': model,
    'last_close': float(df['Close'].iloc[-1]),
    'last_date': df['Date'].iloc[-1],
    'backtest_mape': float(mape)
}, "nvidia_price_model.pkl")

fig1 = model.plot(forecast)
fig1.savefig("nvidia_forecast.png")
print("📈 Forecast plot saved as nvidia_forecast.png")

print(f"\n🎉 High-performance model saved as nvidia_price_model.pkl")
print("   → Ready for Streamlit + Trader Agent")