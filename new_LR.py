  
def lrn(data,cv,taf):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import yfinance as yf
    #import streamlit as st
    import plotly.graph_objects as go
    import datetime as dt
        # ticker_symbol = "HINDPETRO.NS"  
        # data = yf.download(ticker_symbol, start="2018-01-01", end="2022-12-31")
        # data.reset_index(inplace=True)

        # data['Date']=pd.to_datetime(data['Date'])
        # data.sort_values(by='Date',inplace=True)

        # data['Target']=data['Close'].shift(-2)#colnm='open/close/high/low' timd=-1,-2,-3,-4
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

    model=LinearRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    ans=r2_score(y_test,y_pred)

    y_pval=model.predict(X_val)

    ans1=r2_score(y_val,y_pval)

    import plotly.express as px
    df=pd.DataFrame()

    df['Date']=val_data['Date']
    df['test']=y_val
    df['pred']=y_pval

    return ans1

#     def pltg(self):
#         combined_data = pd.concat([train_data, test_data])
#         combined_data1 = pd.concat([train_data, val_data])
#         # k.show()
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=combined_data["Date"], y=combined_data[cv], mode='lines', name='Actual',line=dict(color='blue')))
#         fig.add_trace(go.Scatter(x=test_data["Date"], y=y_pred, mode='lines', name='Predicted',line=dict(color='red')))
#         fig.update_layout(title='Actual vs Predicted (LR)', xaxis_title='Date', yaxis_title='Price',width=1000, height=400)
#         st.plotly_chart(fig)
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=combined_data1["Date"], y=combined_data1[cv], mode='lines', name='Actual',line=dict(color='blue')))
#         fig.add_trace(go.Scatter(x=val_data["Date"], y=y_pval, mode='lines', name='Predicted',line=dict(color='red')))
#         fig.update_layout(title='Actual vs Predicted(Validation) (LR)', xaxis_title='Date', yaxis_title='Price',width=1000, height=400)
#         st.plotly_chart(fig)

# # """
#  combined_data = pd.concat([train_data, test_data])
#     combined_data1 = pd.concat([train_data, val_data])
#     # k.show()
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=combined_data["Date"], y=combined_data[cv], mode='lines', name='Actual',line=dict(color='blue')))
#     fig.add_trace(go.Scatter(x=test_data["Date"], y=y_pred, mode='lines', name='Predicted',line=dict(color='red')))
#     fig.update_layout(title='Actual vs Predicted (LR)', xaxis_title='Date', yaxis_title='Price',width=1000, height=400)
#     st.plotly_chart(fig)
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=combined_data1["Date"], y=combined_data1[cv], mode='lines', name='Actual',line=dict(color='blue')))
#     fig.add_trace(go.Scatter(x=val_data["Date"], y=y_pval, mode='lines', name='Predicted',line=dict(color='red')))
#     fig.update_layout(title='Actual vs Predicted(Validation) (LR)', xaxis_title='Date', yaxis_title='Price',width=1000, height=400)
#     st.plotly_chart(fig)

# """