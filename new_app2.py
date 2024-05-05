# Import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
# from new_LR import LRR
# from new_GB import gbn
# from new_RF import rfn
from LR import lr
from GB import gb
from RF import rf
from accu import accuracy
from LSTM import lstm
from TLSTM import tlstm
import rsi_try as rt


# setting the side bar to collapsed taa k footer jo ha wo sahi dikhay
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")


# Title

st.write("<h1 style=' color:#cc0cf7; font-weight:bold; text-align:center;'>Stock Market Prediction</h1>",unsafe_allow_html=True)

# st.write("<h3 style=' font-weight:bold; text-align:center;'>This app is created to forecast the stock market price of the selected company.</h3>",unsafe_allow_html=True)



# Take input from the user of the app about the start and end date

# Sidebar
st.write("<h4 style=' font-weight:bold; ;'>Select the parameters from below</h4>",unsafe_allow_html=True)


start_date = st.date_input('Start date', date(2014, 1, 1))
end_date = st.date_input('End date', date(2022, 12, 31))
# Add ticker symbol list
# Load the CSV file containing company names and ticker symbols
@st.cache_resource
def load_data():
    data = pd.read_csv("company_tickers.csv")
    return data

companies_data = load_data()

# Display full company names in the sidebar select box
selected_company = st.selectbox('Select the company', companies_data['Company'])

# Fetch ticker symbol corresponding to the selected full company name
ticker_symbol = companies_data.loc[companies_data['Company'] == selected_company, 'Ticker'].iloc[0]

# Fetch data from user inputs using yfinance library
data = yf.download(ticker_symbol, start=start_date, end=end_date)
d1=data.copy()
print(data.head())
custom_css = """
<style>
[data-testid="stMarkdownContainer"]{
color:#b7ebdf;
}
[data-testid="stImage"]{
    width:60%;
    height:20%;
}



</style>
"""
if selected_company != "":
    
    st.markdown(custom_css, unsafe_allow_html=True)
    # Check if data is empty or contains missing values
    if data.empty or data.isnull().values.any():
        st.write("Please select a stock and a model for forecasting.*")
    else:
        # Add Date as a column to the dataframe
        data.insert(0, "Date", data.index, True)
        d1=data.copy()
        data.reset_index(drop=True, inplace=True)
        st.write('Data from', start_date, 'to', end_date)
        # Plot the data
        #st.header('Data Visualization')
        #st.subheader('Plot of the data')
        s_d_g=st.button("Show Data Graph")
        if(s_d_g==True or s_d_g==False):
            ticker = yf.Ticker(ticker_symbol)
            live_data = ticker.history(period="1d", interval="1m")
            k=live_data['Close']
            ma100=k.rolling(5).mean()
            ma200=k.rolling(20).mean()

            # Create line traces with correct x-axis values
            line_trace1 = go.Scatter(
            x=live_data.index,  # Use the index of the DataFrame containing the moving averages
            y=ma100,  # Access the 'Close' column of the DataFrame
            mode='lines',
            name='MA10'
            )
            line_trace2 = go.Scatter(
            x=live_data.index,  # Use the index of the DataFrame containing the moving averages
            y=ma200,  # Access the 'Close' column of the DataFrame
            mode='lines',
            name='MA20'
            )
            st.write('NOTE:If MA10 cross MA20 in up direction its BUY call')
            st.write('If MA10 cross MA20 in down direction its SELL call')
            #RSE
            fig = go.Figure(data=[go.Candlestick(
            x=live_data.index,  # Assuming data.index contains the date index
            open=live_data['Open'],
            high=live_data['High'],
            close=live_data['Close'],
            low=live_data['Low'],
            increasing_line_color='green',
            decreasing_line_color='red'
            ), line_trace1, line_trace2])
            fig.update_layout(title=f'Live Candlestick Chart - {selected_company}', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig)
            st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")
            sd= data.drop(columns=['Adj Close'])
            kl=d1['Close']
            ma100=kl.rolling(50).mean()
            ma200=kl.rolling(200).mean()
            line_trace3 = go.Scatter(
            x=d1.index,  # Use the index of the DataFrame containing the moving averages
            y=ma100,  # Access the 'Close' column of the DataFrame
            mode='lines',
            name='MA50'
            )
            line_trace4 = go.Scatter(
            x=d1.index,  # Use the index of the DataFrame containing the moving averages
            y=ma200,  # Access the 'Close' column of the DataFrame
            mode='lines',
            name='MA200'
            )

# Assuming you have already calculated RSI values and generated signals
            rsi_values = rt.calculate_rsi(d1['Close'])
            d1['RSI'] = rsi_values
            d1['Signal'] = rt.generate_signals(rsi_values)

            # Create traces for RSI and moving averages
            rsi_trace = go.Scatter(x=d1.index, y=d1['RSI'], mode='lines', name='RSI')
            line_trace3 = go.Scatter(x=d1.index, y=ma100, mode='lines', name='MA100')
            line_trace4 = go.Scatter(x=d1.index, y=ma200, mode='lines', name='MA200')

            # Create buy and sell annotations
            buy_annotations = []
            sell_annotations = []
            for i in range(len(d1)):
                if d1['Signal'][i] == 'BUY':
                    buy_annotations.append(dict(x=d1.index[i], y=d1['Close'][i], text='BUY', showarrow=True, arrowhead=1, arrowcolor='green', ax=0, ay=-40))
                elif d1['Signal'][i] == 'SELL':
                    sell_annotations.append(dict(x=d1.index[i], y=d1['Close'][i], text='SELL', showarrow=True, arrowhead=1, arrowcolor='red', ax=0, ay=40))

            # Create the candlestick trace
            candlestick_trace = go.Candlestick(x=d1.index, open=d1['Open'], high=d1['High'], close=d1['Close'], low=d1['Low'],
                                                increasing_line_color='green', decreasing_line_color='red', name='Candlestick')

            # Create the figure
            fig = go.Figure(data=[candlestick_trace, line_trace3, line_trace4])

            # Add buy and sell annotations
            for annotation in buy_annotations:
                fig.add_annotation(annotation)
            for annotation in sell_annotations:
                fig.add_annotation(annotation)

            # Update layout
            fig.update_layout(
                title='Closing Price and RSI',
                xaxis=dict(title='Date'),
                yaxis=dict(title='Price'),
                yaxis2=dict(title='RSI', overlaying='y', side='right'),
            )

            # Show the figure
            st.plotly_chart(fig)

            #volume
            

            # Sample candlestick and volume data (replace with your own data)
            candlestick_data = pd.DataFrame({
             'Date': d1['Date'],
            'Open': d1['Open'],
            'High': d1['High'],
            'Low': d1['Close'],
            'Close': d1['Low']
            })

            volume_data = pd.DataFrame({
             'Date': d1['Date'],
             'Volume':d1['Volume']  # Sample volume data
            })

            # Example threshold for identifying volume spikes
            volume_threshold =4*(volume_data['Volume'].mean())
            volume_spike_mask = volume_data['Volume'] > volume_threshold
            candlestick_data['Volume_Spike'] = volume_spike_mask
            # Plot candlestick chart with volume bars
            fig = go.Figure(data=[
            go.Candlestick(x=candlestick_data['Date'],
                   open=candlestick_data['Open'],
                   high=candlestick_data['High'],
                   low=candlestick_data['Low'],
                   close=candlestick_data['Close'],
                   name='Candlestick'),
            go.Bar(x=volume_data['Date'], y=volume_data['Volume'], name='Volume')
            ])

            # Highlight volume spikes
            volume_spike_dates = candlestick_data[candlestick_data['Volume_Spike']]['Date']
            fig.add_trace(go.Scatter(x=volume_spike_dates, y=candlestick_data.loc[candlestick_data['Volume_Spike'], 'High'],
                         mode='markers',
                         marker=dict(size=10, color='red'),
                         name='Volume Spike'))

            # Update layout
            #fig.update_layout(title='Candlestick Chart with Volume Spike Indicator',
                  #xaxis_title='Date',
                 # yaxis_title='Price',
                  #yaxis2_title='Volume',
                  #yaxis2=dict(anchor='x', overlaying='y', side='right'))

            st.plotly_chart(fig)

            
        
        
        # Add a select box to choose the column for forecasting
        
        
        column = st.selectbox('Select the column to be used for forecasting', data.columns[1:])

        # Subsetting the data
        data = data[['Date', column]]
        s_d_p=st.button("Predict price")
        t=2
        if(s_d_p==True):
            #mo=[tlstm(d1,column),lstm(d1,column)]
            #lt_model = st.selectbox('Select the column to be used for forecasting', data.columns[1:])
            #accurate_model,value=accuracy(d1,column,t)
            #s_t_p=st.button("Show time wise")
            #s_t_d=st.button("Show day wise")
            #if(s_t_p==True):
            #st.write("Day wise")
            #tlstm(d1,column) 
            #st.write('NOTE:If MA10 cross MA20 in up direction its BUY call')
            #st.write('If MA10 cross MA20 in down direction its SELL call')
            #lstm(d1,column)
            # SARIMA Model
        # User input for SARIMA parameters
        #p = st.slider('Select the value of p', 0, 5, 2)
        #d = st.slider('Select the value of d', 0, 5, 1)
        #q = st.slider('Select the value of q', 0, 5, 2)
        #seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 12)

            model = sm.tsa.statespace.SARIMAX(data[column], order=(2, 2, 3), seasonal_order=(2, 2, 3, 4))
            model = model.fit()
        # Print model summary
        #st.header('Model Summary')
        #st.write(model.summary())
        #st.write("---")

        # Forecasting using SARIMA
            st.write("<p style='color:green; font-size: 50px; font-weight: bold;'>Forecasting the data with SARIMA</p>",
                 unsafe_allow_html=True)

            forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)
        # Predict the future values
            predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period)
            predictions1 = model.get_prediction(start=len(data), end=len(data) + forecast_period)
        #accuacy=mean_squared_error(data[:,1:2],predictions1)
            #st.write(data)
            predictions = predictions.predicted_mean
        # Add index to the predictions
            predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
            predictions = pd.DataFrame(predictions)
            predictions.insert(0, "Date", predictions.index, True)
            predictions.reset_index(drop=True, inplace=True)
            st.write("Predictions", predictions)
            #st.write("Actual Data", data)
            st.write("---")

        # Plot the data
            fig = go.Figure()
        # Add actual data to the plot
            fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
        # Add predicted data to the plot
            fig.add_trace(
            go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted',
                       line=dict(color='red')))
        # Set the title and axis labels
            fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
        # Display the plot
            st.plotly_chart(fig)
            #if(s_t_d==True):
            #lstm(d1,column)
            #st.write("Day wise")
            # if accurate_model=='Gradient':
            #     gb(d1,column,t)
            # elif accurate_model=='Linear':
            #     lr(d1,column,t)
            # elif accurate_model=='Random':
            #     rf(d1,column,t)
            #st.write(value*100)
        # s_d_p1=st.button("show other")  
        # if(s_d_p1==True):
        #     lr(d1,t)
        #     st.write('note:if MA10 cross MA20 in up direction its BUY call')
        #     st.write('note:if MA10 cross MA20 in down direction its SELL call')
        #     # rf(d1,column,t)




