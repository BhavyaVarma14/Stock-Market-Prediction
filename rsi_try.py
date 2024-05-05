import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Fetch historical price data for TSLA from Yahoo Finance

# Calculate RSI
def calculate_rsi(prices, period=30):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Generate Trading Signals
def generate_signals(rsi_values):
    signals = []
    for rsi in rsi_values:
        if rsi > 70:
            signals.append('SELL')
        elif rsi < 30:
            signals.append('BUY')
        else:
            signals.append('HOLD')
    return signals

