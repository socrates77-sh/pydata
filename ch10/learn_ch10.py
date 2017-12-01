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

#347