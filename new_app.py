# Import libraries
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

start_date = st.date_input('Start date', date(2020, 1, 1))
end_date = st.date_input('End date', date(2020, 12, 31))
# Add ticker symbol list
# Load the CSV file containing company names and ticker symbols
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

# Check if data is empty or contains missing values
    # Add Date as a column to the dataframe
    data.insert(0, "Date", data.index, True)
    data.reset_index(drop=True, inplace=True)
    d1=data.copy()

    # Plot the data
    #st.header('Data Visualization')
    #st.subheader('Plot of the data')

    # Subsetting the data
    data = data[['Date', column]]
    # ADF test to check stationarity
    #st.header('Is data Stationary?')
    #st.write(adfuller(data[column])[1] < 0.05)


    # Model selection
    
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
        





