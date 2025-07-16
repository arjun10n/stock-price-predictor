import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

st.title("📈 Stock Price Movement Predictor")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, INFY.BO):")

today = datetime.today()
default_start = today - timedelta(days=180)
start_date = st.date_input("From Date", default_start)
end_date = st.date_input("To Date", today)

if start_date > end_date:
    st.error("🚫 Start date must be before end date.")
elif ticker:
    data = yf.download(ticker, start=start_date, end=end_date)

    if not data.empty:
        # Calculate indicators
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()

        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))

        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # Fill missing values (no drop)
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # Require minimum data length
        if len(data) < 60:
            st.warning("⚠️ Not enough data for reliable analysis. Please select a longer date range.")
        else:
            # Prepare features correctly (make sure shape is (1,5))
            latest = data.iloc[-1]
            features = np.array([
                latest['MA10'],
                latest['MA50'],
                latest['RSI'],
                latest['MACD'],
                latest['Signal']
            ]).reshape(1, -1)  # This ensures 2D shape (1, 5)

            # Load model
            model_path = os.path.join(os.path.dirname(__file__), 'stock_model.joblib')
            if not os.path.exists(model_path):
                st.error(f"⚠️ Could not load model: File not found at {model_path}")
            else:
                try:
                    model = joblib.load(model_path)
                    prediction = model.predict(features)[0]
                    result = "📈 Stock is going UP" if prediction == 1 else "📉 Stock is going DOWN"

                    st.subheader("Prediction")
                    st.success(f"{result} (on {latest.name.date()})")

                except Exception as e:
                    st.error(f"⚠️ Could not use model: {e}")

        # Plot Closing Price
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            mode='lines+markers', name='Close Price',
            hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ))
        fig.update_layout(title=f"{ticker.upper()} Closing Price", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

        # Plot RSI
        st.subheader("RSI Indicator")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], line=dict(color='purple'), name='RSI'))
        fig_rsi.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=70, y1=70,
                          line=dict(color="red", dash="dot"))
        fig_rsi.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=30, y1=30,
                          line=dict(color="green", dash="dot"))
        fig_rsi.update_layout(template='plotly_white', height=300)
        st.plotly_chart(fig_rsi)

    else:
        st.warning("❗ No data found for the given ticker and date range.")
else:
    st.info("ℹ️ Please enter a stock ticker and select date range to get started.")
