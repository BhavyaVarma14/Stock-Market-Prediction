def rfn(data,cv,taf):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import yfinance as yf
    import streamlit as st
    import plotly.graph_objects as go
    import datetime as dt
    from sklearn.ensemble import RandomForestRegressor
    data['Target']=data[cv].shift(-taf)
    data.dropna(inplace=True)
    print(data.head())
    train_size=0.7
    test_size=0.15
    val_size=0.15

    num_data=len(data)
    data['Date']=pd.to_datetime(data['Date'])
    num_tr=int(num_data*train_size)
    num_val=int(num_data*val_size)
    num_ts=int(num_data*test_size)
    train_data=data[:num_tr]
    test_data=data[num_tr:num_tr+num_ts]
    val_data=data[num_tr+num_ts:num_tr+num_ts+num_val]
    
    X_train=train_data.drop(columns=['Target'])
    y_train=train_data['Target']

    X_test=test_data.drop(columns=['Target'])
    y_test=test_data['Target']

    X_val=val_data.drop(columns=['Target'])
    y_val=val_data['Target']
    X_test['Year']=X_test['Date'].dt.year
    X_test['Month']=X_test['Date'].dt.month
    X_test['day']=X_test['Date'].dt.day
    xd=X_test['Date']
    X_test.drop(columns=['Date'],inplace=True)

    X_val['Year']=X_val['Date'].dt.year
    X_val['Month']=X_val['Date'].dt.month
    X_val['day']=X_val['Date'].dt.day
    xvd=X_val['Date']
    X_val.drop(columns=['Date'],inplace=True)

    X_train['Year']=X_train['Date'].dt.year
    X_train['Month']=X_train['Date'].dt.month
    X_train['day']=X_train['Date'].dt.day
    xtd=X_train['Date']

    X_train.drop(columns=['Date'],inplace=True)
    
    model1=RandomForestRegressor(random_state=1)
    model1.fit(X_train,y_train)
    
    y_pred=model1.predict(X_test)
    ans=r2_score(y_test,y_pred)
    y_pval=model1.predict(X_val)
    ans1=r2_score(y_val,y_pval)

    
    import plotly.express as px
    df=pd.DataFrame()

    df['Date']=val_data['Date']
    df['test']=y_val
    df['pred']=y_pval

    return ans1
