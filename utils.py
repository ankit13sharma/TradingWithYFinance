import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def stock_data(symbol,time_period,time_interval):
    stock = pd.DataFrame(yf.download(symbol,period = time_period,interval = time_interval))
    stock.index = pd.to_datetime(stock.index, format="%Y%m%d").to_period('D')
    if len(stock.index) != len(stock.index.unique()):
        stock = stock.groupby(stock.index).mean().reset_index() 
    return stock


def all_returns(data: pd.DataFrame, frequency):
    if frequency == "annual":
        factor = 12
    elif frequency == "monthly":
        factor = 252
    returns = data.pct_change(fill_method='bfill')
    returns = returns.dropna()
    returns_prod = (1+returns).prod(axis = 0)
    # daily_returns = returns_prod**(1/len(returns)) -1
    period_returns = returns_prod**(1/len(returns)) -1
    annual_returns = returns_prod**(factor/len(returns)) -1
    annual_vol = returns.std()*(factor**0.5)
    return returns,period_returns,annual_returns, annual_vol

def sharp_ratio(annual_returns, annual_vol, riskfree_rate):
    return (annual_returns - riskfree_rate)/annual_vol

def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()   
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})

def get_thresholds_mean_k(arr:pd.DataFrame,scale):
    kt = pd.DataFrame(index = arr.index)
    kt_mean = pd.DataFrame(index = arr.index)
    for column in arr.columns:
        y = arr[column]/scale
        kt[column] = y /(y.shift(1)*(1-y.shift(1))) 
        # kt[column] = kt[column].replace(np.inf,np.nan).mean(skipna=True)
        kt_mean[column] = [kt[column][:i].replace(np.inf,np.nan).dropna().mean(skipna=True) for i in range(len(arr.index))]
    return kt,kt_mean