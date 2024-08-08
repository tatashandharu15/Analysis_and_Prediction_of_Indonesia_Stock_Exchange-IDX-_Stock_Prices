# Reference : https://stock-market-prediction-2piq85jecgi.streamlit.app/

import streamlit as st
from streamlit_option_menu import option_menu
import util
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas_datareader as web
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Function to scale the data
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    return scaler, scaled_data

# Function to prepare training data
def prepare_training_data(data, training_data_len=0):
    train_data = data[0:training_data_len,:]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train

# Function to create and train LSTM model
def create_train_model(x_train, y_train, epochs, progress_text):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    for epoch in range(epochs):
        model.fit(x_train, y_train, batch_size=1, verbose=0)  # Train for one epoch 
        progress_text.write(f"Training epoch {epoch+1}/{epochs}")

    return model

# Function to create and train LSTM model
def create_train_model_predict(x_train, y_train, epochs, progress_text):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    st.write("Process predicted close price, please waiting...")
        
    # Train the model
    for epoch in range(epochs):
        model.fit(x_train, y_train, batch_size=1, verbose=0)  # Train for one epoch 
        

    return model

# Function to prepare test data and make predictions
def prepare_and_predict(model, data, scaler, training_data_len, days_to_predict):
    test_data = data[training_data_len-60: , : ]
    x_test = []

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i,0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions[-days_to_predict:]

# Function to fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    df = yf.Ticker(ticker)
    df = df.history(start=start_date, end=end_date)
    return df

# Function to evaluate the model
def evaluate_model(model, x_train, y_train):
    # Get predictions
    train_predictions = model.predict(x_train)
    train_predictions = scaler.inverse_transform(train_predictions)

    # Inverse transform y_train
    y_train_inverse = scaler.inverse_transform(y_train.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_train_inverse, train_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train_inverse, train_predictions)

    return mse, rmse, mae
    
with st.sidebar:
    selected = option_menu("Main Menu", ["Bursa Efek Indonesia", "Training Model", "Prediksi Harga Saham", "Analisis Teknikal"], 
        icons=['activity', 'gear', 'book'], menu_icon="cast", default_index=0)
    selected

if selected == 'Bursa Efek Indonesia':
    # st.title("Bursa Efek Indonesia :")
    # ticker_symbol = st.text_input(
    #     "Please enter the stock symbol", 'BBRI.JK'
    # )
    ticker_symbol = st.selectbox(
        "Please enter the stock symbol",
        ("BBRI.JK", "BBCA.JK", "BBTN.JK", "BMRI.JK", "BRIS.JK", "ARTO.JK", "TLKM.JK", "GOTO.JK", "RAJA.JK", "ADRO.JK", "UNVR.JK", "MSKY.JK", "ASII.JK", "PTBA.JK", "PGEO.JK", "ANTM.JK", "ALII.JK", "NICE.JK", "CUAN.JK", "SURI.JK")
    )
    st.write("You selected:", ticker_symbol)

    data_period = st.text_input("Period", "10d")
    data_interval = st.radio("Interval", ['15m', '30m', '1h', '1d', '5d'])

    st.header(ticker_symbol)

    ticker_data = util.get_ticker_data(ticker_symbol, data_period, data_interval)
    ticker_data 

    util.plot_candle_chart(ticker_data)

    st.markdown(util.get_table_download_link(ticker_data), unsafe_allow_html=True)

elif selected == 'Training Model':
    st.title("Training Model (LSTM) :")
    ticker = st.selectbox(
        "Please enter the stock symbol",
        ("BBRI.JK", "BBCA.JK", "BBTN.JK", "BMRI.JK", "BRIS.JK", "ARTO.JK", "TLKM.JK", "GOTO.JK", "RAJA.JK", "ADRO.JK", "UNVR.JK", "MSKY.JK", "ASII.JK", "PTBA.JK", "PGEO.JK", "ANTM.JK", "ALII.JK", "NICE.JK", "CUAN.JK", "SURI.JK")
    )
    start_date = st.date_input('Start Date', pd.to_datetime('2018-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2024-03-01'))
    epochs = st.number_input('Enter the number of LSTM epochs', value=5, key='lstm_epochs')
    days_to_predict = 5

    training_data_len = 500

    if st.button('Train Model'):
        # Fetch historical data
        df = fetch_stock_data(ticker, start_date, end_date)
        data = df.filter(['Close']).values

        # Scale the data
        scaler, scaled_data = scale_data(data)

        # Prepare training data
        x_train, y_train = prepare_training_data(scaled_data, training_data_len)

        # Display progress
        progress_text = st.empty()

        # Create and train LSTM model
        model = create_train_model(x_train, y_train, epochs, progress_text)

        # Get predictions
        predictions = prepare_and_predict(model, scaled_data, scaler, training_data_len, days_to_predict)

        # Display predicted prices
        st.header('Evaluation Training LSTM')
        mse, rmse, mae = evaluate_model(model, x_train, y_train)
        
        # Print evaluation results
        st.write(f'Mean Squared Error (MSE): {mse}')
        st.write(f'Root Mean Squared Error (RMSE): {rmse}')
        st.write(f'Mean Absolute Error (MAE): {mae}')

elif selected == 'Prediksi Harga Saham':
    st.title("Prediksi Harga Saham :")
    ticker = st.selectbox(
        "Please enter the stock symbol",
        ("BBRI.JK", "BBCA.JK", "BBTN.JK", "BMRI.JK", "BRIS.JK", "ARTO.JK", "TLKM.JK", "GOTO.JK", "RAJA.JK", "ADRO.JK", "UNVR.JK", "MSKY.JK", "ASII.JK", "PTBA.JK", "PGEO.JK", "ANTM.JK", "ALII.JK", "NICE.JK", "CUAN.JK", "SURI.JK")
    )
    start_date = st.date_input('Start Date', pd.to_datetime('2018-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2024-03-01'))
    epochs = st.number_input('Enter the number of LSTM epochs', value=5, key='lstm_epochs')
    days_to_predict = st.slider('Days to Predict', min_value=1, max_value=30, value=10)

    training_data_len = 500

    if st.button('Prediction'):
        # Fetch historical data
        df = fetch_stock_data(ticker, start_date, end_date)
        data = df.filter(['Close']).values

        # Scale the data
        scaler, scaled_data = scale_data(data)

        # Prepare training data
        x_train, y_train = prepare_training_data(scaled_data, training_data_len)

        # Display progress
        progress_text = st.empty()

        # Create and train LSTM model
        model = create_train_model_predict(x_train, y_train, epochs, progress_text)

        # Get predictions
        predictions = prepare_and_predict(model, scaled_data, scaler, training_data_len, days_to_predict)

        # Create dataframe for predictions
        pred_dates = [df.index[-1] + pd.Timedelta(days=i+1) for i in range(days_to_predict)]
        pred_df = pd.DataFrame({'Date': pred_dates, 'Prediction (Close)': [pred[0] for pred in predictions]})

        # Display predicted prices
        st.header('Predicted Prices')
        st.dataframe(pred_df)

elif selected == 'Analisis Teknikal':
    st.title("Analisis Teknikal")

    ticker_symbol = st.selectbox(
        "Please enter the stock symbol",
        ("BBRI.JK", "BBCA.JK", "BBTN.JK", "BMRI.JK", "BRIS.JK", "ARTO.JK", "TLKM.JK", "GOTO.JK", "RAJA.JK", "ADRO.JK", "UNVR.JK", "MSKY.JK", "ASII.JK", "PTBA.JK", "PGEO.JK", "ANTM.JK", "ALII.JK", "NICE.JK", "CUAN.JK", "SURI.JK")
    )
    st.write("You selected:", ticker_symbol)
    data_period = st.text_input('Period', '10d')
    data_interval = st.radio('Interval', ['15m','30m','1h','1d'])
    ema1 = st.text_input('EMA 1', 20)
    ema2 = st.text_input('EMA 2', 50)
    share_amount = st.text_input('Number of Shares', 1)
    initial_capital = st.text_input('Initial Capital (USD)', 10000)

    ticker_data = util.get_ticker_data(ticker_symbol, data_period, data_interval)

    if len(ticker_data) != 0:
        ticker_data = util.get_ema(ticker_data, int(ema1))
        ticker_data = util.get_ema(ticker_data, int(ema2))

        candle_fig = util.get_candle_chart(ticker_data)
        candle_fig = util.add_ema_trace(candle_fig, ticker_data.index, ticker_data['ema_' + ema1], 'EMA ' + ema1, "#ffeb3b")
        candle_fig = util.add_ema_trace(candle_fig, ticker_data.index, ticker_data['ema_' + ema2], 'EMA ' + ema2, "#2962ff")

        trades = util.create_ema_trade_list(ticker_data, 'ema_' + ema1, 'ema_' + ema2)
        ticker_data = util.join_trades_to_ticker_data(trades, ticker_data)
        candle_fig = util.add_trades_trace(candle_fig, ticker_data)

        simulation_results = util.simulate_ema_cross_trading(trades)
        win_rate, sim_results_df, sim_fig, accumulative_fig = util.get_sim_summary(simulation_results, int(share_amount), float(initial_capital))
        win_rate_str = str(win_rate.round(1))
        
        st.write("### Simulation Results - ", ticker_symbol)
        st.write("#### Win Rate: ", win_rate_str, "%")
        st.write(sim_results_df.describe())
        st.write("#### Share Price Change Per Trade")
        st.write(sim_fig)
        st.write("#### Account Value Change")
        st.write(accumulative_fig)
        st.write("#### Trade List")
        trades
        st.write(candle_fig)
