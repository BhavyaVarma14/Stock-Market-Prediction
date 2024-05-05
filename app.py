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
app_name = 'Stock Market Forecasting App'
st.title(app_name)
st.subheader('This app is created to forecast the stock market price of the selected company.')
# Add an image from an online resource
st.image("https://akm-img-a-in.tosshub.com/sites/visualstory/wp/2023/12/stock.jpg?size=*:900")

# Take input from the user of the app about the start and end date

# Sidebar

st.header('Select the parameters from below')

start_date = st.date_input('Start date', date(2020, 1, 1))
end_date = st.date_input('End date', date(2020, 12, 31))
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
        fig = px.line(data, x='Date', y=data.columns, title='Data graph', width=1000, height=600)
        st.plotly_chart(fig)
    elif(s_d_g==False and s_d==True):
        st.write(data)
    elif(s_d_g==True and s_d==True):
        with col2:
            st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")
            fig = px.line(data, x='Date', y=data.columns, title='Data graph', width=1000, height=600)
            st.plotly_chart(fig)
        with col1:
            st.write(data)
    # Add a select box to choose the column for forecasting
    column = st.selectbox('Select the column to be used for forecasting', data.columns[1:])

    # Subsetting the data
    data = data[['Date', column]]
    st.write("Selected Data")
    st.write(data)

    # ADF test to check stationarity
    #st.header('Is data Stationary?')
    #st.write(adfuller(data[column])[1] < 0.05)

    # Decompose the data
    st.header('Decomposition of the data')
    decomposition = seasonal_decompose(data[column], model='additive', period=12)
    #st.write(decomposition.plot())
    # Make same plot in Plotly
    st.write("## Plotting the decomposition in Plotly")
    d_g=st.button("Show Data Trend and other info")
    if(d_g==True):
        st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title='Trend', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
        st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality', width=1000, height=400,
        labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='green'))
        st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title='Residuals', width=1000, height=400,
        labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red', line_dash='dot'))

    # Model selection
    models = ['SARIMA', 'LSTM', 'Prophet','LinearRegression','GradientBoost','RandomForest']
    selected_model = st.selectbox('Select the model for forecasting', models)

    if selected_model == 'SARIMA':
        # SARIMA Model
        # User input for SARIMA parameters
        p = st.slider('Select the value of p', 0, 5, 2)
        d = st.slider('Select the value of d', 0, 5, 1)
        q = st.slider('Select the value of q', 0, 5, 2)
        seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 12)

        model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
        model = model.fit()
        # Print model summary
        st.header('Model Summary')
        #st.write(model.summary())
        st.write("---")

        # Forecasting using SARIMA
        st.write("<p style='color:green; font-size: 50px; font-weight: bold;'>Forecasting the data with SARIMA</p>",
                 unsafe_allow_html=True)

        forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)
        # Predict the future values
        predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period)
        predictions1 = model.get_prediction(start=len(data), end=len(data) + forecast_period)
        #accuacy=mean_squared_error(data[:,1:2],predictions1)
        st.write(data)
        predictions = predictions.predicted_mean
        # Add index to the predictions
        predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
        predictions = pd.DataFrame(predictions)
        predictions.insert(0, "Date", predictions.index, True)
        predictions.reset_index(drop=True, inplace=True)
        st.write("Predictions", predictions)
        st.write("Actual Data", data)
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
        st.write("Model selected:", selected_model)
    elif selected_model == 'LinearRegression':
        colnm=['Open','Close','High','Low']
        timeaf=[1,2,3,4,5,6,7]
        cv=st.selectbox("select column",colnm)
        taf=st.selectbox("day",timeaf)
        print(data.columns)
        print(cv)
        print(taf)
        
        lr(d1,cv,taf)
    elif selected_model == 'GradientBoost':
        colnm=['Open','Close','High','Low']
        timeaf=[1,2,3,4,5,6,7]
        cv=st.selectbox("select column",colnm)
        taf=st.selectbox("day",timeaf) 
        gb(d1,cv,taf)
    elif selected_model == 'RandomForest':
        colnm=['Open','Close','High','Low']
        timeaf=[1,2,3,4,5,6,7]
        cv=st.selectbox("select column",colnm)
        taf=st.selectbox("day",timeaf) 
        rf(d1,cv,taf)
    elif selected_model == 'Random Forest':
        # Random Forest Model
        st.header('Random Forest Regression')

        # Splitting data into training and testing sets
        train_size = int(len(data) * 0.8)
        train_data, test_data = data[:train_size], data[train_size:]

        # Feature engineering
        train_X, train_y = train_data['Date'], train_data[column]
        test_X, test_y = test_data['Date'], test_data[column]

        # Initialize and fit the Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
        rf_model.fit(train_X.values.reshape(-1, 1), train_y.values)

        # Predict the future values
        predictions = rf_model.predict(test_X.values.reshape(-1, 1))

        # Calculate mean squared error
        mse = mean_squared_error(test_y, predictions)
        rmse = np.sqrt(mse)

        st.write(f"Root Mean Squared Error (RMSE): {rmse}")

        # Combine training and testing data for plotting
        combined_data = pd.concat([train_data, test_data])

        # Plot the data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined_data["Date"], y=combined_data[column], mode='lines', name='Actual',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test_data["Date"], y=predictions, mode='lines', name='Predicted',
                                 line=dict(color='red')))
        fig.update_layout(title='Actual vs Predicted (Random Forest)', xaxis_title='Date', yaxis_title='Price',
                          width=1000, height=400)
        st.plotly_chart(fig)
        st.write("Model selected:", selected_model)

    elif selected_model == 'LSTM':
        # LSTM Model
        st.header('Long Short-Term Memory (LSTM)')

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[column].values.reshape(-1, 1))

        # Split the data into training and testing sets
        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

        # Create sequences for LSTM model
        def create_sequences(dataset, seq_length):
            X, y = [], []
            for i in range(len(dataset) - seq_length):
                X.append(dataset[i:i + seq_length, 0])
                y.append(dataset[i + seq_length, 0])
            return np.array(X), np.array(y)

        seq_length = st.slider('Select the sequence length', 1, 30, 10)

        train_X, train_y = create_sequences(train_data, seq_length)
        test_X, test_y = create_sequences(test_data, seq_length)

        train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
        test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

        # Build the LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(train_X.shape[1], 1)))
        lstm_model.add(LSTM(units=50))
        lstm_model.add(Dense(units=1))

        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(train_X, train_y, epochs=10, batch_size=16)

        # Predict the future values
        train_predictions = lstm_model.predict(train_X)
        test_predictions = lstm_model.predict(test_X)
        train_predictions = scaler.inverse_transform(train_predictions)
        test_predictions = scaler.inverse_transform(test_predictions)

        # Calculate mean squared error
        train_mse = mean_squared_error(train_data[seq_length:], train_predictions)
        train_rmse = np.sqrt(train_mse)
        test_mse = mean_squared_error(test_data[seq_length:], test_predictions)
        test_rmse = np.sqrt(test_mse)

        st.write(f"Train RMSE: {train_rmse}")
        st.write(f"Test RMSE: {test_rmse}")

        # Combine training and testing data for plotting
        train_dates = data['Date'][:train_size + seq_length]
        test_dates = data['Date'][train_size + seq_length:]
        combined_dates = pd.concat([train_dates, test_dates])
        combined_predictions = np.concatenate([train_predictions, test_predictions])

        # Plot the data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined_dates, y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test_dates, y=combined_predictions, mode='lines', name='Predicted',
                             line=dict(color='red')))
        fig.update_layout(title='Actual vs Predicted (LSTM)', xaxis_title='Date', yaxis_title='Price',
                      width=1000, height=400)
        st.plotly_chart(fig)
        st.write("Model selected:", selected_model)

    elif selected_model == 'Prophet':
        # Prophet Model
        st.header('Facebook Prophet')

        # Prepare the data for Prophet
        prophet_data = data[['Date', column]]
        prophet_data = prophet_data.rename(columns={'Date': 'ds', column: 'y'})

        # Create and fit the Prophet model
        prophet_model = Prophet()
        prophet_model.fit(prophet_data)

        # Forecast the future values
        future = prophet_model.make_future_dataframe(periods=365)
        forecast = prophet_model.predict(future)

        # Plot the forecast
        fig = prophet_model.plot(forecast)
        plt.title('Forecast with Facebook Prophet')
        plt.xlabel('Date')
        plt.ylabel('Price')
        st.pyplot(fig)
        st.write("Model selected:", selected_model)
    elif selected_model.isempty():
        st.write("Please select a model for forecasting.")
        





