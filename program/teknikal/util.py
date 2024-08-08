import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import yfinance as yf
import pandas as pd
import numpy as np

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