import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# Load Model
model = joblib.load("stock_model.joblib")

st.title("📈 Stock Trend & Future Forecasting App")

ticker = st.text_input("Enter Stock Ticker:", "AAPL")

forecast_days = st.selectbox("Forecast Days", [7,15,30])

if st.button("Predict"):

    # Fetch last 6 month data
    data = yf.download(ticker, period="6mo", interval="1d")
    if data.empty:
        st.error("Invalid Ticker")
        st.stop()

    st.subheader("📅 Last 6 Month Price Chart")
    st.line_chart(data['Close'])

    # -------------------------
    # EXISTING MODEL PREDICTION
    # -------------------------
    data['EMA20']  = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA50']  = data['Close'].ewm(span=50, adjust=False).mean()
    data['MACD']   = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data['MACDsig']= data['MACD'].ewm(span=9).mean()
    data['RSI']    = 100 - (100 / (1 + (data['Close'].pct_change().rolling(14).mean())))

    features = data[['EMA20','EMA50','MACD','MACDsig','RSI']].dropna().tail(1)
    pred = model.predict(features)[0]

    trend = "📈 Bullish Uptrend" if pred==1 else "📉 Bearish Downtrend"
    st.subheader("🔍 Current Market Direction → " + trend)

    # -------------------------
    # FIXED FORECASTING ENGINE
    # -------------------------
    st.subheader(f"🔮 {forecast_days}-Day Price Forecast")

    close = data['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    last_seq = scaled[-60:]   # last 60 days for rolling forecast

    future_prices = []
    seq = last_seq.copy()

    for _ in range(forecast_days):
        x = seq[-60:].reshape(1,60,1)
        next_val = np.mean(x)  # simple rolling projection based on nearest trend
        future_prices.append(next_val)
        seq = np.append(seq,next_val).reshape(-1,1)

    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1,1)).flatten()

    future_dates = pd.date_range(start=datetime.now(), periods=forecast_days+1)[1:]
    df_forecast = pd.DataFrame({"Date":future_dates,"Predicted_Close":future_prices})

    st.line_chart(df_forecast.set_index("Date"))

    # -------------------------
    # FINAL TREND DECISION
    # -------------------------
    last_price = data.Close.iloc[-1]
    avg_future = df_forecast.Predicted_Close.mean()

    if avg_future > last_price:
        final_trend = "🚀 UP expected — Buy sentiment"
    else:
        final_trend = "🔻 Downtrend expected — Risky zone"

    st.success(final_trend)
    st.write(df_forecast)
