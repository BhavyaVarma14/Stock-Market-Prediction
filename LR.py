# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import datetime as dt

# Define the function
def lr(data, taf):
    lt = []
    ls = []
    
    # Loop through each column
    for col in ['Open', 'High', 'Low', 'Close']:
        data['Target'] = data[col].shift(-taf)
        data.dropna(inplace=True)
        train_size = 0.7
        test_size = 0.15
        val_size = 0.15

        num_data = len(data)
        data['Date'] = pd.to_datetime(data['Date'])
        num_tr = int(num_data * train_size)
        num_val = int(num_data * val_size)
        num_ts = int(num_data * test_size)
        train_data = data[:num_tr]
        test_data = data[num_tr:num_tr + num_ts]
        val_data = data[num_tr + num_ts:num_tr + num_ts + num_val]
        k = val_data['Date']

        X_train = train_data.drop(columns=['Target'])
        y_train = train_data['Target']

        X_test = test_data.drop(columns=['Target'])
        y_test = test_data['Target']

        X_val = val_data.drop(columns=['Target'])
        y_val = val_data['Target']
        ls = y_val

        X_test['Year'] = X_test['Date'].dt.year
        X_test['Month'] = X_test['Date'].dt.month
        X_test['Day'] = X_test['Date'].dt.day
        X_test.drop(columns=['Date'], inplace=True)

        X_val['Year'] = X_val['Date'].dt.year
        X_val['Month'] = X_val['Date'].dt.month
        X_val['Day'] = X_val['Date'].dt.day
        X_val.drop(columns=['Date'], inplace=True)

        X_train['Year'] = X_train['Date'].dt.year
        X_train['Month'] = X_train['Date'].dt.month
        X_train['Day'] = X_train['Date'].dt.day
        X_train.drop(columns=['Date'], inplace=True)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # ans=r2_score(y_test,y_pred)

        y_pval = model.predict(X_val)
        y_pval = y_pval[:len(y_pval) - 1]
        # ans1=r2_score(y_val,y_pval)

    # Calculate moving averages
    k = val_data['Date']
    ma100 = val_data['Close'].rolling(100).mean()
    ma200 = val_data['Close'].rolling(200).mean()

    # Create line traces with correct x-axis values
    line_trace1 = go.Scatter(
        x=k,  # Use the index of the DataFrame containing the moving averages
        y=ma100,  # Access the 'Close' column of the DataFrame
        mode='lines',
        name='MA100'
    )
    line_trace2 = go.Scatter(
        x=k,  # Use the index of the DataFrame containing the moving averages
        y=ma200,  # Access the 'Close' column of the DataFrame
        mode='lines',
        name='MA200'
    )

    # Create the figure
    fig = go.Figure(data=[go.Candlestick(
        x=val_data.index,  # Assuming val_data.index contains the date index
        open=val_data['Open'],
        high=val_data['High'],
        close=val_data['Close'],
        low=val_data['Low'],
        increasing_line_color='green',
        decreasing_line_color='red'
    ), line_trace1, line_trace2])

    # Update layout
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(autosize=False, width=800, height=500)

    # Show the figure
    st.plotly_chart(fig)
