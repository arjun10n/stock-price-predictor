import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

st.title("📈 Stock Trend + Future Forecasting App")

ticker = st.text_input("Enter Stock Ticker:", "AAPL")
forecast_days = st.selectbox("Forecast Days", [7, 15, 30])

if st.button("Run Prediction"):

    # Load Price Data
    data = yf.download(ticker, period="6mo", interval="1d")
    if data.empty:
        st.error("Invalid Stock Ticker ❌")
        st.stop()

    # ---------------------- 🔥 SHOW PRICE HISTORY ----------------------
    st.subheader(f"📅 Price History ({ticker}) - Last 6 Months")
    st.line_chart(data['Close'])

    # ---------------------- Indicators (same logic as your model input) ----------------------
    data['EMA20']  = data['Close'].ewm(span=20).mean()
    data['EMA50']  = data['Close'].ewm(span=50).mean()
    data['MACD']   = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data['MACDsig']= data['MACD'].ewm(span=9).mean()
    data['RSI']    = 100 - (100 / (1 + data['Close'].pct_change().rolling(14).mean()))

    # Show indicator chart also
    st.subheader("📊 Technical Indicator Charts")
    st.write(data[['Close','EMA20','EMA50','RSI','MACD','MACDsig']])

    # ---------------------- ML Model Prediction ----------------------
    try:
        model = joblib.load("stock_model.joblib")
        features = data[['EMA20','EMA50','MACD','MACDsig','RSI']].dropna().tail(1)
        pred = model.predict(features)[0]
        st.success("Current Trend → 📈 UP" if pred==1 else "Current Trend → 📉 DOWN")

    except Exception as e:
        st.error(f"Model Error: {e}")
        st.stop()

    # ---------------------- FIXED FORECASTING SECTION 🛠 ----------------------
    st.subheader(f"🔮 {forecast_days}-Day Price Forecast")

    close = data['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    seq = scaled[-60:].copy()
    future_vals = []

    for _ in range(forecast_days):
        next_val = np.mean(seq[-60:])      # Rolling short-term momentum logic
        future_vals.append(next_val)
        seq = np.append(seq, next_val).reshape(-1,1)

    future_vals = scaler.inverse_transform(np.array(future_vals).reshape(-1,1)).flatten()

    future_dates = pd.date_range(datetime.now(), periods=forecast_days+1)[1:]
    forecast_df = pd.DataFrame({"Date":future_dates,"Predicted Price":future_vals})

    st.line_chart(forecast_df.set_index("Date"))
    st.write(forecast_df)

    # ---------------------- FINAL TREND FIX (Your Error Solved!) ✔ ----------------------
    last_price = float(data['Close'].iloc[-1])       # <-- FIX
    avg_future = float(forecast_df["Predicted Price"].mean())  # <-- FIX

    if avg_future > last_price:
        st.success(f"🚀 Expected Upside Ahead — BUY BIAS\n(Current: {last_price:.2f} → Avg Future: {avg_future:.2f})")
    else:
        st.error(f"🔻 Expected Down Trend — SELL / WAIT\n(Current: {last_price:.2f} → Avg Future: {avg_future:.2f})")
