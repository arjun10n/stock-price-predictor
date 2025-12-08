import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from prophet import Prophet     # NEW - forecasting model

st.title("📈 Stock Price Movement Predictor + Future Forecasting")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, INFY.BO):")

# User can choose how much history to load
days = st.selectbox("History Range", [30, 60, 90, 120, 180])
today = datetime.today()
start_date = today - timedelta(days=days)

# Forecast length options
forecast_days = st.selectbox("Forecast Future", [7, 15, 30])

if ticker:
    data = yf.download(ticker, start=start_date, end=today)

    if not data.empty:

        # ---------------- INDICATORS (for ML UP/DOWN prediction) --------------------
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

        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # ---------------- CURRENT PRICE MOVEMENT PREDICTION --------------------
        if len(data) >= 60:
            latest = data.iloc[-1]
            features = np.array([
                latest['MA10'],
                latest['MA50'],
                latest['RSI'],
                latest['MACD'],
                latest['Signal']
            ]).reshape(1, -1)

            model_path = os.path.join(os.path.dirname(__file__), 'stock_model.joblib')
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                prediction = model.predict(features)[0]
                result = "📈 UP Trend Expected" if prediction == 1 else "📉 DOWN Trend Expected"

                st.subheader("Trend Direction (Machine Learning)")
                st.success(result)
            else:
                st.warning("⚠ ML Model not found — only forecast available.")
        else:
            st.warning("⚠ Need more historical data to predict UP/DOWN trend")

        # ---------------- PRICE FORECASTING USING PROPHET --------------------
        st.subheader(f"🔮 {forecast_days}-Day Future Price Forecast")

        df = data[['Close']].reset_index()
        df.columns = ['ds', 'y']  # Prophet format requirement

        model_prophet = Prophet()
        model_prophet.fit(df)

        future = model_prophet.make_future_dataframe(periods=forecast_days)
        forecast = model_prophet.predict(future)

        # Plot forecast
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=df['ds'], y=df['y'],
            mode='lines', name="Historical Price"
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines', name="Forecast Price", line=dict(color="orange")
        ))
        fig_forecast.update_layout(
            title=f"{ticker.upper()} - Next {forecast_days} Days Forecast",
            xaxis_title="Date", yaxis_title="Price"
        )

        st.plotly_chart(fig_forecast)

        # ---------------- INDICATOR CHARTS --------------------
        st.subheader("📊 Indicators Used for Prediction")

        # Price Chart
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'],
                                       mode='lines', name='Close Price'))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['MA10'],
                                       mode='lines', name='MA10'))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['MA50'],
                                       mode='lines', name='MA50'))
        st.plotly_chart(fig_price)

        # RSI Chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
        fig_rsi.update_layout(title="RSI Indicator", height=300)
        st.plotly_chart(fig_rsi)

    else:
        st.error("❗ No data found — Try another stock symbol.")
