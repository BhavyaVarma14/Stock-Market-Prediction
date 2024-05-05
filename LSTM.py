import streamlit as st
def lstm(data,column):
    
    opn = data[[column]]
    devraj=data.iloc[-10:-3,1]
    import matplotlib.pyplot as plt
    ds = opn.values
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    normalizer = MinMaxScaler(feature_range=(0,1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))
    train_size = int(len(ds_scaled)*0.70)
    test_size = len(ds_scaled) - train_size
    ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]
    def create_ds(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)
    time_stamp = 10
    X_train, y_train = create_ds(ds_train,time_stamp)
    X_test, y_test = create_ds(ds_test,time_stamp)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    from keras.models import Sequential
    from keras.layers import Dense, LSTM,Dropout
    model = Sequential()
# First LSTM layer with Dropout regularisation
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(Dropout(0.3))

    model.add(LSTM(units=80, return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=30))
    model.add(Dropout(0.3))

    model.add(Dense(units=1))


    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=64)
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = normalizer.inverse_transform(train_predict)
    test_predict = normalizer.inverse_transform(test_predict)
    test = np.vstack((train_predict,test_predict))
    nd=np.array([[devraj[0]],[devraj[1]],[devraj[2]],[devraj[3]],[devraj[4]],[devraj[5]],[devraj[6]]])
    ypp=[]
    x_p=nd
    x_p_reshaped = np.reshape(x_p, (1, x_p.shape[0], x_p.shape[1]))
    for i in range(7):
        y_p=model.predict(x_p_reshaped)
        x_p_reshaped=x_p_reshaped[:,1:,:]
        new=np.array([[y_p]])
        ypp.append(y_p)
        x_p_reshaped=np.concatenate((x_p_reshaped,new.reshape(1,1,1)),axis=1)
    ypp=np.concatenate(ypp,axis=0)
    ypp2=normalizer.inverse_transform(ypp)
    # df=data.reset_index()
    import pandas as pd
    # df['Date'] = pd.to_datetime(df['Date'])
    # from datetime import timedelta
    # a=df.iloc[-1,0].date()
    # dtt=[]
    # for i in range(7):
    #     dtt.append(a)
    #     a += timedelta(days=1)
    import matplotlib.pyplot as plt
    import pandas as pd

# Example data (replace with your own data)
# dates = ['2024-03-01', '2024-03-02', '2024-03-03', '2024-03-04', '2024-03-05']
# predicted_prices = [100, 102, 105, 103, 107]

# Convert dates to datetime objects

    import plotly.graph_objects as go
    # Plotting
    #plt.figure(figsize=(10, 6))
    #fig=plt.plot( ypp2, marker='o', linestyle='-')
    fig=go.Figure(ypp2)
    # Formatting
    #plt.title('Stock Predicted Prices')
    #plt.xlabel('Date')
    #plt.ylabel('Predicted Price')
    #plt.grid(True)

    # Rotate x-axis labels for better readability
    #plt.xticks(rotation=45)

    # Display the plot
    st.plotly_chart(fig)
    #st.pyplot(fig)
    import numpy as np
    from sklearn.metrics import mean_squared_error

    # Example arrays (replace with your actual arrays)
    predicted_values = ypp2
    actual_values = nd

    # Calculate mean squared error
    mse = mean_squared_error(actual_values, predicted_values)

    # Calculate root mean squared error (RMSE)
    rmse = np.sqrt(mse)

    print("Root Mean Squared Error (RMSE):", rmse)


    # print("Mean Squared Error (MSE):", mse)
    st.write(rmse)


    # fut_inp = ds_test[277:]
    # fut_inp = fut_inp.reshape(1,-1)
    # tmp_inp = list(fut_inp)
    # tmp_inp = tmp_inp[0].tolist()
    # lst_output=[]
    # n_steps=100
    # i=0
    # while(i<30):
        
    #     if(len(tmp_inp)>100):
    #         fut_inp = np.array(tmp_inp[1:])
    #         fut_inp=fut_inp.reshape(1,-1)
    #         fut_inp = fut_inp.reshape((1, n_steps, 1))
    #         yhat = model.predict(fut_inp, verbose=0)
    #         tmp_inp.extend(yhat[0].tolist())
    #         tmp_inp = tmp_inp[1:]
    #         lst_output.extend(yhat.tolist())
    #         i=i+1
    #     else:
    #         fut_inp = fut_inp.reshape((1, n_steps,1))
    #         yhat = model.predict(fut_inp, verbose=0)
    #         tmp_inp.extend(yhat[0].tolist())
    #         lst_output.extend(yhat.tolist())
    #         i=i+1
    # plot_new=np.arange(1,101)
    # plot_pred=np.arange(101,21)
    # ds_new = ds_scaled.tolist()
    # ds_new.extend(lst_output)
    # data['Open'][1206:]
    # import plotly.graph_objects as go
    # import pandas as pd

    # # Assuming data contains the original data and lst_output contains the predicted closing values
    # # Concatenate original closing values with predicted values
    # all_close_values = list(data['Close']) + [item[0] for item in lst_output]
    # k=data['Close']
    # ma100=k.rolling(100).mean()
    # ma200=k.rolling(200).mean()

    # # Create line traces with correct x-axis values
    # line_trace1 = go.Scatter(
    #     x=data.index,  # Use the index of the DataFrame containing the moving averages
    #     y=ma100,  # Access the 'Close' column of the DataFrame
    #     mode='lines',
    #     name='MA100'
    # )
    # line_trace2 = go.Scatter(
    #     x=data.index,  # Use the index of the DataFrame containing the moving averages
    #     y=ma200,  # Access the 'Close' column of the DataFrame
    #     mode='lines',
    #     name='MA200'
    # )

    # # Create the figure
    # fig = go.Figure(data=[go.Candlestick(
    #     x=data.index,  # Assuming data.index contains the date index
    #     open=data['Open'],
    #     high=data['High'],
    #     close=all_close_values,
    #     low=data['Low'],
    #     increasing_line_color='green',
    #     decreasing_line_color='red'
    # ), line_trace1, line_trace2])

    # # Update layout
    # fig.update_layout(xaxis_rangeslider_visible=False)
    # fig.update_layout(autosize=False, width=700, height=500)

    # # Show the figure
    # st.plotly_chart(fig)

