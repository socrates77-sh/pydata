from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt

now = datetime.now()
delta = datetime(2017, 11, 6) - datetime(1977, 2, 7)

start = datetime(2011, 1, 7)
start + timedelta(12)

stamp = datetime(2011, 1, 3)

value = '2011-01-03'
datetime.strptime(value, '%Y-%m-%d')

datestrs = ['7/6/2011', '8/6/2011']
[datetime.strptime(x, '%m/%d/%Y') for x in datestrs]

parse('2011-01-03')

parse('Jan 31, 1977 10:45 PM')

parse('6/12/2011', dayfirst=True)

pd.to_datetime(datestrs)
idx = pd.to_datetime(datestrs + [None])

dates = [datetime(2011, 1, 2), datetime(2011, 1, 5),
         datetime(2011, 1, 7), datetime(2011, 1, 8),
         datetime(2011, 1, 10), datetime(2011, 1, 12)]

ts = Series(np.random.randn(6), index=dates)

longer_ts = Series(np.random.randn(1000),
                   index=pd.date_range('1/1/2000', periods=1000))

close_px_all = pd.read_csv('stock_px.csv', parse_dates=True, index_col=0)
close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
close_px = close_px.resample('B', fill_method='ffill')

close_px['AAPL'].plot()

close_px.loc['2009'].plot()
close_px['AAPL'].loc['2011-1': '2011-3'].plot()

appl_q = close_px['AAPL'].resample('Q-DEC', fill_method='ffill')
appl_q.loc['2009':].plot()

close_px['AAPL'].plot()
# pd.rolling_mean(close_px['AAPL'], 250).plot()
close_px['AAPL'].rolling(window=250, center=False).mean().plot()

appl_std250 = close_px['AAPL'].rolling(
    window=250, center=False, min_periods=10).std()
appl_std250.plot()

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                         sharey=True, figsize=(12, 7))

appl_px = close_px['AAPL']['2005':'2009']
ma60 = pd.rolling_mean(appl_px, 60, min_periods=50)
ewma60 = pd.ewma(appl_px, span=60)

appl_px.plot(style='k-', ax=axes[0])
ma60.plot(style='k--', ax=axes[0])
appl_px.plot(style='k-', ax=axes[1])
ewma60.plot(style='k--', ax=axes[1])
axes[0].set_title('Simple MA')
axes[1].set_title('Exponentially-weighted MA')

spx_px = close_px_all['SPX']
spx_rets = spx_px / spx_px.shift(1) - 1
returns = close_px.pct_change()
corr = pd.rolling_corr(returns['AAPL'], spx_rets, 125, min_periods=100)
corr.plot()

corr = pd.rolling_corr(returns, spx_rets, 125, min_periods=100)
corr.plot()