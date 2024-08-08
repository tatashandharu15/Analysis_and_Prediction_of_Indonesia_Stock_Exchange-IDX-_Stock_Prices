import streamlit as st 
from streamlit_option_menu import option_menu
import util
import yfinance as yf
import plotly.graph_objs as go
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
import pandas as pd
import numpy as np
import base64
from sklearn.metrics import r2_score, mean_absolute_error
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import datetime

def get_ticker_data(ticker_symbol, data_period, data_interval):
    ticker_data = yf.download(tickers=ticker_symbol, period=data_period, interval=data_interval)

    if len(ticker_data) == 0:
        st.write("Could not find the ticker data. Modify the ticker symbol or reduce the Period value.")
    else:
        ticker_data.index = ticker_data.index.strftime("%d-%m-%Y %H:%M")
    return ticker_data

def plot_candle_chart(ticker_data):
    candle_fig = go.Figure()
    candle_fig.add_trace(
        go.Candlestick(x=ticker_data.index,
                       open=ticker_data['Open'],
                       close=ticker_data['Close'],
                       low=ticker_data['Low'],
                       high=ticker_data['High']
                       )
    )
    candle_fig.update_layout(
        height=800
    )
    st.write(candle_fig)

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # encode CSV to bytes
    href = f'<a href="data:file/csv;base64,{b64}" download="DataSaham.csv">Download Data Saham</a>'
    return href

# ---------------------------- TEKNIKAL FUNCTION AWAL ----------------------------
def get_ema(ticker_data, period):
    ema = ticker_data['Close'].ewm(span=period).mean()
    column_name = 'ema_' + str(period)
    ticker_data[column_name] = ema
    return ticker_data

def get_ticker_data(ticker_symbol, data_period, data_interval):
    ticker_data = yf.download(tickers=ticker_symbol, period=data_period, interval=data_interval)
    if len(ticker_data) == 0:
        st.write('Could not find the ticker data. Modify ticker symbol or reduce the Period value.')
    else:
        #Format the x-axis to skip dates with missing values
        ticker_data.index = ticker_data.index.strftime("%d-%m-%Y %H:%M")
    return ticker_data

def get_candle_chart(ticker_data):
    candle_fig = go.Figure()
    candle_fig.add_trace(
        go.Candlestick(x=ticker_data.index,
        open=ticker_data['Open'],
        close=ticker_data['Close'],
        low=ticker_data['Low'],
        high=ticker_data['High'],
        name='Market Data'
        )
    )
    candle_fig.update_layout(
        height=800,
    )
    return candle_fig

def add_ema_trace(candle_fig, timestamp, ema, trace_name, color):
    candle_fig.add_trace(
        go.Scatter(
            x = timestamp,
            y = ema,
            name = trace_name,
            line = dict(color=color)
        )
    )
    return candle_fig

def add_trades_trace(candle_fig, ticker_data):
    candle_fig.add_trace(
        go.Scatter(
            x = ticker_data.index,
            y = ticker_data['Trade Price'],
            name = 'Trade Triggers',
            marker_color = ticker_data['Trade Color'],
            mode = 'markers'
        )
    )
    return candle_fig

def create_ema_trade_list(ticker_data, ema1_col_name, ema2_col_name):
    ticker_data['ema_diff'] = ticker_data[ema1_col_name] - ticker_data[ema2_col_name]
    prev_state = 'unknown'
    trades = []
    for i in range(len(ticker_data)):
        if ticker_data['ema_diff'][i] >= 0:
            state = 'positive'
        else:
            state = 'negative'
        if prev_state != 'unknown':
            if state == 'positive' and prev_state == 'negative':
                 try:
                     trade = str(ticker_data.index[i+1]) + ',' + str(ticker_data['Open'][i+1]) + ',cyan,buy'
                     trade = trade.split(',')
                     trades.append(trade) 
                 except:
                    continue
            elif state == 'negative' and prev_state == 'positive':
                 try:
                    trade = str(ticker_data.index[i+1]) + ',' + str(ticker_data['Open'][i+1]) + ',magenta,sell'
                    trade = trade.split(',')
                    trades.append(trade) 
                 except:
                    continue
        prev_state = state
    return trades

def join_trades_to_ticker_data(trades, ticker_data):
    trades_df = pd.DataFrame(trades, columns=['Time','Trade Price','Trade Color','Trade Type']).set_index('Time')
    trades_df['Trade Price'] = trades_df['Trade Price'].astype(float).round(4)
    ticker_data = pd.concat([ticker_data, trades_df], axis=1, join='outer')
    return ticker_data

def simulate_ema_cross_trading(trades):
    results = []
    buy_price = 0
    for trade in trades:
        if trade[3] == 'buy':
            buy_price = float(trade[1])
        elif buy_price != 0:
            sell_price = float(trade[1])
            results.append(sell_price - buy_price)
    return results

def get_sim_summary(sim_results, share_amount, initial_capital):
    accumulative_account_value = []
    np_sim_results = np.array(sim_results)
    win_rate = (np_sim_results > 0).sum() / len(sim_results) * 100
    sim_results_df = pd.DataFrame(sim_results, columns=['Change'])  
    sim_fig = go.Figure()
    sim_fig.add_trace(
        go.Scatter(
            x = sim_results_df.index,
            y = sim_results_df['Change']
        )
    )
    accumulative_account_value.append(initial_capital)
    total = initial_capital
    for item in sim_results:
        total = total + item*share_amount
        accumulative_account_value.append(total)
    accumulative_account_value_df = pd.DataFrame(accumulative_account_value, columns=['Acc Value']) 
    accumulative_fig = go.Figure()
    accumulative_fig.add_trace(
        go.Scatter(
            x = accumulative_account_value_df.index,
            y = accumulative_account_value_df['Acc Value']
        )
    )
    return win_rate, sim_results_df, sim_fig, accumulative_fig
# ---------------------------- TEKNIKAL FUNCTION AKHIR ----------------------------

@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

def model_engine_lstm_manual_input(num, lstm_units, lstm_epochs, test_size, option, duration, start_date, end_date):
    option = option.upper()
    data = download_data(option, start_date, end_date)
    scaler = StandardScaler()

    df = data[['Close']]
    df['preds'] = data.Close.shift(-num)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df.preds.values
    y = y[:-num]

    # Manually input the test size
    train_size = 1 - test_size
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, train_size=train_size, random_state=7)

    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    x_forecast = np.reshape(x_forecast, (x_forecast.shape[0], 1, x_forecast.shape[1]))

    lstm_units_key = 'lstm_units_' + str(lstm_units)
    lstm_epochs_key = 'lstm_epochs_' + str(lstm_epochs)

    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(1, x_train.shape[2])))
    model.add(LSTM(units=lstm_units))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    st.text("Training LSTM model...")
    for epoch in range(1, lstm_epochs + 1):
        model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=0)
        preds = model.predict(x_test)
        mse = np.mean(np.square(preds - y_test))
        st.text(f'Epoch {epoch}/{lstm_epochs} - Mean Squared Error: {mse}')

    st.text("Training completed.")

    preds = model.predict(x_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    st.text(f'R2 Score: {r2} \nMean Absolute Error: {mae}')

# Definisikan variabel global untuk menyimpan nilai input sebelumnya
previous_values = {
    'lstm_units': 100,
    'lstm_epochs': 100,
    'test_size': 0.2,
    'option': "BBRI.JK",
    'duration': 5000,
    'start_date': datetime.date.today() - datetime.timedelta(days=3000),
    'end_date': datetime.date.today()
}

def model_engine_prediksi(num):
    lstm_units = previous_values['lstm_units']
    lstm_epochs = previous_values['lstm_epochs']
    test_size = previous_values['test_size']
    option = previous_values['option']
    duration = previous_values['duration']
    start_date = previous_values['start_date']
    end_date = previous_values['end_date']

    option = option.upper()
    data = download_data(option, start_date, end_date)
    scaler = StandardScaler()

    df = data[['Close']]
    df['preds'] = data.Close.shift(-num)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df.preds.values
    y = y[:-num]

    # Manually input the test size
    train_size = 1 - test_size
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, train_size=train_size, random_state=7)

    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    x_forecast = np.reshape(x_forecast, (x_forecast.shape[0], 1, x_forecast.shape[1]))

    lstm_units_key = 'lstm_units_' + str(lstm_units)
    lstm_epochs_key = 'lstm_epochs_' + str(lstm_epochs)

    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(1, x_train.shape[2])))
    model.add(LSTM(units=lstm_units))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    st.text("Please Waiting for Prediction")
    for epoch in range(1, lstm_epochs + 1):
        model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=0)
        preds = model.predict(x_test)
        mse = np.mean(np.square(preds - y_test))

    st.text("Prediction completed.")

    preds = model.predict(x_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    # st.text(f'R2 Score: {r2} \nMean Absolute Error: {mae}')

    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i[0]}')
        day += 1

