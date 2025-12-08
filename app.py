# ===============================================================
# 📈 Stock Price Prediction + LSTM Future Forecast (Visually Clean 🌙)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta

st.title("📊 Stock Price Forecasting using LSTM (Next 30 Days)")
st.write("Upload any stock — see **history + future forecast 🔮**")

# ---------------------------------------------------------
# Select Stock Input
# ---------------------------------------------------------
ticker = st.text_input("Enter Stock Ticker (example: RELIANCE.NS, TCS.NS, AAPL)", "RELIANCE.NS")
forecast_days = st.slider("Forecast Future Days", 7, 60, 30)

# ---------------------------------------------------------
# Fetch Data
# ---------------------------------------------------------
@st.cache_data
def load_data(ticker):
    return yf.download(ticker, period="5y")

data = load_data(ticker)

if data is None or len(data)==0:
    st.error("❗ Invalid ticker or no data available")
    st.stop()

st.subheader("📜 Price History (5 Years)")
st.line_chart(data["Close"])

# ---------------------------------------------------------
# Prepare Data for LSTM
# ---------------------------------------------------------
df = data[['Close']]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df.astype(float))

# Generate sequences
X, y = [], []
lookback = 60
for i in range(lookback, len(scaled)):
    X.append(scaled[i-lookback:i, 0])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))   # LSTM format

# ---------------------------------------------------------
# Build LSTM Model
# ---------------------------------------------------------
with st.spinner("Training LSTM Model... takes 10–20 seconds ⚙"):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, batch_size=32, epochs=10, verbose=0)

# ---------------------------------------------------------
# Forecast Next X Days
# ---------------------------------------------------------
# prepare the input: last `lookback` values
input_seq = scaled[-lookback:].reshape(1, lookback, 1)
future_preds = []

for _ in range(forecast_days):
    pred = model.predict(input_seq, verbose=0)  # shape (1,1)
    # Convert pred to shape (1,1,1) so we can concatenate along axis=1
    pred_3d = pred.reshape(1, 1, 1)
    # Append the prediction and shift window
    input_seq = np.concatenate((input_seq[:, 1:, :], pred_3d), axis=1)
    future_preds.append(pred[0][0])

# inverse transform predictions
future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1)).flatten()

# Create future dates starting the next calendar day after last historical index
start_date = pd.to_datetime(data.index[-1]) + pd.Timedelta(days=1)
future_dates = pd.date_range(start=start_date, periods=forecast_days)
forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_preds})

# ---------------------------------------------------------
# 📊 Visual Forecast Graph (Beautiful Beginner Friendly)
# ---------------------------------------------------------
st.subheader("🔮 Future Forecast (LSTM)")
fig = go.Figure()

# Historical Price
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Historical Price", mode="lines"))

# Future Predictions
fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted Price"],
                         name="Forecast", mode="lines+markers"))

fig.update_layout(title=f"📈 {ticker}  — Next {forecast_days} Day Forecast",
                  xaxis_title="Date", yaxis_title="Price",
                  template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

# Trend conclusion
st.subheader("📌 Market Outlook")
last_price = float(data['Close'].iloc[-1])
future_mean = float(np.mean(future_preds))

if future_mean > last_price:
    st.success(f"📈 **UP Trend Expected** — Price likely rising 🚀\n(Current: {last_price:.2f} → Avg Future: {future_mean:.2f})")
else:
    st.error(f"📉 **Down Trend Expected** — Caution advised 🔻\n(Current: {last_price:.2f} → Avg Future: {future_mean:.2f})")

# Show table
st.write("🔍 Forecasted Prices:")
st.dataframe(forecast_df)
