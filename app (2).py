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
from LR import lr
from GB import gb
from RF import rf



# setting the side bar to collapsed taa k footer jo ha wo sahi dikhay
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")


# Title

st.write("<h1 style=' color:#cc0cf7; font-weight:bold; text-align:center;'>Stock Market Prediction</h1>",unsafe_allow_html=True)

#st.write("<h3 style=' font-weight:bold; color:#01CF73; text-align:center;'>This app is created to forecast the stock market price of the selected company.</h3>",unsafe_allow_html=True)



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
# d1=data.copy()
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
        data.reset_index(drop=True, inplace=True)
        st.write('Data from', start_date, 'to', end_date)
        col1,col2=st.columns(2)
        with col1:
            s_d=st.button("Show Data")
        d1=data.copy()

        # Plot the data
        #st.header('Data Visualization')
        #st.subheader('Plot of the data')
        with col2:
            s_d_g=st.button("Show Data Graph")
        if(s_d_g==True and s_d==False):
            st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")
            sd= data.drop(columns=['Volume', 'Adj Close'])
            fig = px.line(data, x='Date', y=sd.columns, title='Data graph', width=1000, height=600)
            st.plotly_chart(fig)
        elif(s_d_g==False and s_d==True):
            st.write(data)
        
        # Add a select box to choose the column for forecasting
        
        
        # column = st.selectbox('Select the column to be used for forecasting', data.columns[1:])

        # Subsetting the data
        # data = d1[['Date', column]]
        #s_d_p=st.button("Predict price")
        
        fig=go.Figure(data=[go.Candlestick(x=d1.index,open=d1['Open'],high=d1['High'],close=d1['Close'],
        low=d1['Low'],
        increasing_line_color='purple',
        decreasing_line_color='orange'
        )])
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(autosize=False,width=700,height=500)
        st.plotly_chart(fig)
        st.write("HELOOOOOO")
       





