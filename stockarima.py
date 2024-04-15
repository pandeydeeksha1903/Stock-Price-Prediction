import streamlit as st
from datetime import date

import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache
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
    plt.figure(figsize=(10,6))
    plt.plot(data['Date'], data['Open'], label='stock_open')
    plt.plot(data['Date'], data['Close'], label='stock_close')
    plt.title('Time Series data with Rangeslider')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

plot_raw_data()

# Forecast with ARIMA
df_train = data[['Date','Close']]
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_train.set_index('Date', inplace=True)

model = ARIMA(df_train, order=(5,1,0))
results = model.fit()

forecast = results.forecast(steps=period)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast)

plt.figure(figsize=(10,6))
plt.plot(df_train.index, df_train, label='Actual')
forecast_dates = pd.date_range(start=df_train.index[-1], periods=period+1)[1:]
forecast_series = pd.Series(forecast, index=forecast_dates)
plt.plot(forecast_dates, forecast, label='Forecast', color='red')
plt.title('Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Get the last forecasted closing price
last_forecasted_closing_price = forecast_series.iloc[-1]

# Display the forecasted closing price at the end of the forecast graph
plt.text(forecast_dates[-1], last_forecasted_closing_price, f'Forecasted Close Price using ARIMA: {last_forecasted_closing_price:.5f}', ha='right', fontsize=10, color='blue')
st.pyplot(plt)
