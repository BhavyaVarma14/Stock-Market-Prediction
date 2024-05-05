import streamlit as st
def tlstm(data,column):
    data=data[:86]
    opn = data[[column]]
    import matplotlib.pyplot as plt
    ds = opn.values
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    normalizer = MinMaxScaler(feature_range=(0,1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))
    train_size = int(len(ds_scaled)*0.60)
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
    from keras.layers import Dense, LSTM
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1,activation='linear'))
    model.summary()
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = normalizer.inverse_transform(train_predict)
    test_predict = normalizer.inverse_transform(test_predict)
    test = np.vstack((train_predict,test_predict))
    fut_inp = ds_test[10:]
    fut_inp = fut_inp.reshape(1,-1)
    tmp_inp = list(fut_inp)
    tmp_inp = tmp_inp[0].tolist()
    lst_output=[]
    n_steps=len(tmp_inp)-1
    i=0
    while(i<10):
        
        if(len(tmp_inp)>10):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    plot_new=np.arange(1,11)
    plot_pred=np.arange(11,21)
    ds_new = ds_scaled.tolist()
    ds_new.extend(lst_output)
    data['Open'][1206:]
    import plotly.graph_objects as go
    import pandas as pd

    # Assuming data contains the original data and lst_output contains the predicted closing values
    # Concatenate original closing values with predicted values
    all_close_values = list(data['Close']) + [item[0] for item in lst_output]
    k=data['Close']
    ma100=k.rolling(5).mean()
    ma200=k.rolling(20).mean()

    # Create line traces with correct x-axis values
    line_trace1 = go.Scatter(
        x=data.index,  # Use the index of the DataFrame containing the moving averages
        y=ma100,  # Access the 'Close' column of the DataFrame
        mode='lines',
        name='MA10'
    )
    line_trace2 = go.Scatter(
        x=data.index,  # Use the index of the DataFrame containing the moving averages
        y=ma200,  # Access the 'Close' column of the DataFrame
        mode='lines',
        name='MA20'
    )

    # Create the figure
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,  # Assuming data.index contains the date index
        open=data['Open'],
        high=data['High'],
        close=all_close_values,
        low=data['Low'],
        increasing_line_color='green',
        decreasing_line_color='red'
    ), line_trace1, line_trace2])

    # Update layout
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(autosize=False, width=800, height=500)

    # Show the figure
    st.plotly_chart(fig)
