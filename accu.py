def accuracy(data,cv,t):
    import pandas as pd
    from new_GB import gbn
    from new_LR import lrn
    from new_RF import rfn
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




    # grad = gbn(data,cv,t)
    linear = lrn(data,cv,t)
    # randforest = rfn(data,cv,t)
    # st.write(grad*100)
    # st.write(randforest*100)
    
    lst = {'Linear':linear}
    x = max(lst)
    y=lst[x]
    return x,y

