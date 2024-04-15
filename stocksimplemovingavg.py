import streamlit as st
from datetime import date

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Close'], label='Close Price')
    plt.title('Stock Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    st.pyplot()

plot_raw_data()

# Simple Moving Average Forecast
def simple_moving_average_forecast(data, window):
    forecast = data['Close'].rolling(window=window).mean().iloc[-1]
    return forecast

forecast_window = st.slider('Select window size for Simple Moving Average (days):', 5, 100, 30)
forecast = simple_moving_average_forecast(data, forecast_window)

st.subheader('Forecast data')
st.write(f'Forecasted Close Price using Simple Moving Average: {forecast}')

