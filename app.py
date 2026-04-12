import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ------------------ TITLE ------------------
st.title("📈 Stock Price Movement Predictor")

# ------------------ PASSWORD ------------------
PASSWORD = "heysexyladies"
pwd = st.text_input("🔒 Enter access password:", type="password")

if pwd != PASSWORD:
    st.warning("Access denied! Please enter a valid password.")
    st.stop()

# ------------------ INPUT ------------------
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, INFY.BO):")

today = datetime.today()
default_start = today - timedelta(days=180)

start_date = st.date_input("From Date", default_start)
end_date = st.date_input("To Date", today)

# ------------------ VALIDATION ------------------
if start_date > end_date:
    st.error("🚫 Start date must be before end date.")

elif ticker:
    data = yf.download(ticker, start=start_date, end=end_date)

    if not data.empty:

        # ------------------ INDICATORS ------------------
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA50'] = data['Close'].rolling(50).mean()

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

        # ------------------ FIXED (PANDAS 2.0+) ------------------
        data.ffill(inplace=True)
        data.bfill(inplace=True)

        # ------------------ CHECK DATA ------------------
        if len(data) < 60:
            st.warning("⚠️ Not enough data. Select a longer date range.")
        else:
            latest = data.iloc[-1]

            features = np.array([
                latest['MA10'],
                latest['MA50'],
                latest['RSI'],
                latest['MACD'],
                latest['Signal']
            ]).reshape(1, -1)

            # Safety check
            if np.isnan(features).any():
                st.error("⚠️ Invalid data. Try a longer range.")
            else:
                # ------------------ LOAD MODEL ------------------
                model_path = os.path.join(os.path.dirname(__file__), 'stock_model.joblib')

                if not os.path.exists(model_path):
                    st.error("⚠️ Model file not found.")
                else:
                    try:
                        model = joblib.load(model_path)
                        prediction = model.predict(features)[0]

                        result = "📈 Stock is going UP" if prediction == 1 else "📉 Stock is going DOWN"

                        st.subheader("Prediction")
                        st.success(f"{result} (on {latest.name.date()})")

                    except Exception as e:
                        st.error(f"⚠️ Model error: {e}")

        # ------------------ PRICE PLOT ------------------
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines+markers',
            name='Close Price'
        ))

        fig.update_layout(
            title=f"{ticker.upper()} Closing Price",
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(fig)

        # ------------------ RSI PLOT ------------------
        st.subheader("RSI Indicator")

        fig_rsi = go.Figure()

        fig_rsi.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            line=dict(color='purple'),
            name='RSI'
        ))

        fig_rsi.add_shape(
            type="line",
            x0=data.index[0], x1=data.index[-1],
            y0=70, y1=70,
            line=dict(color="red", dash="dot")
        )

        fig_rsi.add_shape(
            type="line",
            x0=data.index[0], x1=data.index[-1],
            y0=30, y1=30,
            line=dict(color="green", dash="dot")
        )

        fig_rsi.update_layout(template='plotly_white', height=300)

        st.plotly_chart(fig_rsi)

    else:
        st.warning("❗ No data found.")

else:
    st.info("ℹ️ Enter a stock ticker to begin.")