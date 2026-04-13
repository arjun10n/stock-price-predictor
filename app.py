import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# UI
st.set_page_config(page_title="Hybrid Stock AI", layout="wide")
st.title("🌿📈 Hybrid Stock Predictor (ML + LSTM)")

# PASSWORD
PASSWORD = "1234"
pwd = st.text_input("Enter Password", type="password")

if pwd != PASSWORD:
    st.warning("Enter correct password")
else:

    # LOAD MODELS
    model = joblib.load("stock_model.joblib")

    try:
        feature_names = joblib.load("features.joblib")
    except:
        feature_names = ['MA10','MA50','RSI','MACD','Signal']

    lstm_model = load_model("lstm_model.h5")
    scaler = joblib.load("scaler.joblib")

    # INPUT
    ticker = st.text_input("Enter Stock Ticker (AAPL, INFY.BO):")

    today = datetime.today()
    start_date = st.date_input("From Date", today - timedelta(days=365))
    end_date = st.date_input("To Date", today)

    def compute_rsi(series):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    if ticker:

        data = yf.download(ticker, start=start_date, end=end_date)

        if not data.empty:

            # FEATURES
            data['MA10'] = data['Close'].rolling(10).mean()
            data['MA50'] = data['Close'].rolling(50).mean()
            data['RSI'] = compute_rsi(data['Close'])

            data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
            data['Signal'] = data['MACD'].ewm(span=9).mean()

            data['Volatility'] = data['Close'].pct_change().rolling(10).std()

            data['BB_MID'] = data['Close'].rolling(20).mean()
            data['BB_STD'] = data['Close'].rolling(20).std()
            data['BB_UP'] = data['BB_MID'] + 2 * data['BB_STD']
            data['BB_LOW'] = data['BB_MID'] - 2 * data['BB_STD']

            data['Returns'] = data['Close'].pct_change()

            data.ffill(inplace=True)
            data.bfill(inplace=True)

            # ---------------- PREDICTIONS ----------------

            st.subheader("🤖 AI Predictions")

            col1, col2 = st.columns(2)

            # 🔹 Logistic Regression
            with col1:
                if len(data) > 10:
                    latest = data.iloc[-1]
                    features = latest[feature_names].values.reshape(1, -1)

                    if not np.isnan(features).any():
                        prediction = model.predict(features)[0]
                        prob = model.predict_proba(features)[0][1]

                        st.success(f"ML Prediction: {'UP 📈' if prediction==1 else 'DOWN 📉'}")
                        st.write(f"Confidence: {prob*100:.2f}%")

            # 🔹 LSTM
            with col2:
                if len(data) > 60:
                    close_prices = data['Close'].values.reshape(-1,1)
                    scaled = scaler.transform(close_prices)

                    sequence = scaled[-60:]
                    sequence = sequence.reshape(1,60,1)

                    lstm_pred = lstm_model.predict(sequence)
                    price = scaler.inverse_transform(lstm_pred)[0][0]

                    st.success(f"LSTM Price: {price:.2f}")
                else:
                    st.warning("Need 60+ days for LSTM")

            # ---------------- GRAPHS ----------------

            st.subheader("📊 Market Analysis")

            st.plotly_chart(go.Figure(data=[go.Candlestick(
                x=data.index, open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close']
            )]), use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close']))
            fig.add_trace(go.Scatter(x=data.index, y=data['MA10']))
            fig.add_trace(go.Scatter(x=data.index, y=data['MA50']))
            st.plotly_chart(fig, use_container_width=True)

            st.plotly_chart(go.Figure(data=[go.Scatter(x=data.index, y=data['RSI'])]), use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD']))
            fig.add_trace(go.Scatter(x=data.index, y=data['Signal']))
            st.plotly_chart(fig, use_container_width=True)

            st.plotly_chart(go.Figure(data=[go.Bar(x=data.index, y=data['Volume'])]), use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close']))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_UP']))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_LOW']))
            st.plotly_chart(fig, use_container_width=True)

            st.plotly_chart(go.Figure(data=[go.Scatter(x=data.index, y=data['Returns'])]), use_container_width=True)

            st.plotly_chart(go.Figure(data=[go.Scatter(x=data.index, y=data['Volatility'])]), use_container_width=True)

            st.plotly_chart(go.Figure(data=[go.Histogram(x=data['Close'])]), use_container_width=True)

            # Kaggle dataset
            st.subheader("📊 Historical Dataset")
            try:
                df_k = pd.read_csv("data/AAPL.csv")
                df_k['Date'] = pd.to_datetime(df_k['Date'])
                st.plotly_chart(go.Figure(data=[go.Scatter(x=df_k['Date'], y=df_k['Close'])]))
            except:
                st.warning("Kaggle dataset not found")

        else:
            st.error("No data found")
