from datetime import datetime
from datetime import timedelta, time
from dateutil.parser import parse
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web

ts1 = Series(np.random.randn(3),
             index=pd.date_range('2012-6-13', periods=3, freq='W-WED'))

ts1.resample('B')
ts1.resample('B', fill_method='ffill')

gdp = Series([1.78, 1.94, 2.08, 2.01, 2.15, 2.31, 2.46],
             index=pd.period_range('1984Q2', periods=7, freq='Q-SEP'))
infl = Series([0.024, 0.045, 0.037, 0.04],
              index=pd.period_range('1982', periods=4, freq='A-DEC'))
infl_q = infl.asfreq('Q-SEP', how='end')
infl_q.reindex(gdp.index, method='ffill')

rng = pd.date_range('2012-6-1 9:30', '2012-6-1 15:59', freq='T')
rng = rng.append([rng + pd.offsets.BDay(i) for i in range(1, 4)])
ts = Series(np.arange(len(rng), dtype=float), index=rng)

indexer = np.sort(np.random.permutation(len(ts))[700:])
irr_ts = ts.copy()
irr_ts[indexer] = np.nan

selection = pd.date_range('2012-6-1 10:00', periods=4, freq='B')

data1 = DataFrame(np.ones((6, 3), dtype=float),
                  columns=['a', 'b', 'c'],
                  index=pd.date_range('2012-6-12', periods=6))
data2 = DataFrame(np.ones((6, 3), dtype=float) * 2,
                  columns=['a', 'b', 'c'],
                  index=pd.date_range('2012-6-13', periods=6))
spliced = pd.concat([data1.loc[:'2012-6-14'], data2.loc['2012-6-15':]])

data2 = DataFrame(np.ones((6, 4), dtype=float) * 2,
                  columns=['a', 'b', 'c', 'd'],
                  index=pd.date_range('2012-6-13', periods=6))
spliced = pd.concat([data1.loc[:'2012-6-14'], data2.loc['2012-6-15':]])
spliced_filled = spliced.combine_first(data2)
spliced.update(data2, overwrite=False)

close_px = pd.read_csv('stock_px.csv', parse_dates=True, index_col=0)
volume = pd.read_csv('volume.csv', parse_dates=True, index_col=0)
prices = close_px.loc['2011-09-05':'2011-09-14', ['AAPL', 'JNJ', 'SPX', 'XOM']]
volume = volume.loc['2011-09-05':'2011-09-12', ['AAPL', 'JNJ', 'XOM']]
vwap = (prices * volume).sum() / volume.sum()

s1 = Series(range(3), index=['a', 'b', 'c'])
s2 = Series(range(4), index=['d', 'b', 'c', 'e'])
s3 = Series(range(3), index=['f', 'a', 'c'])
DataFrame({'one': s1, 'two': s2, 'three': s3})
DataFrame({'one': s1, 'two': s2, 'three': s3}, index=list('face'))

ts1 = Series(np.random.randn(3),
             index=pd.date_range('2012-6-13', periods=3, freq='W-WED'))
ts1.resample('B')
ts1.resample('B', fill_method='ffill')

dates = pd.DatetimeIndex(['2012-6-12', '2012-6-17', '2012-6-18',
                          '2012-6-21', '2012-6-22', '2012-6-29'])
ts2 = Series(np.random.randn(6), index=dates)

gdp = Series([1.78, 1.94, 2.08, 2.01, 2.15, 2.31, 2.46],
             index=pd.period_range('1984Q2', periods=7, freq='Q-SEP'))
infl = Series([0.025, 0.045, 0.037, 0.04],
              index=pd.period_range('1982', periods=4, freq='A-DEC'))
infl_q = infl.asfreq('Q-SEP', how='end')

rng = pd.date_range('2012-6-1 9:30', '2012-6-1 15:59', freq='T')
rng = rng.append([rng + pd.offsets.BDay(i) for i in range(1, 4)])
ts = Series(np.arange(len(rng), dtype=float), index=rng)
ts[time(10, 0)]

indexer = np.sort(np.random.permutation(len(ts))[700:])
irr_ts = ts.copy()
irr_ts[indexer] = np.nan
irr_ts['2012-6-1 9:50':'2012-6-1 10:00']

data1 = DataFrame(np.ones((6, 3), dtype=float),
                  columns=['a', 'b', 'c'],
                  index=pd.date_range('6/12/2012', periods=6))
data2 = DataFrame(np.ones((6, 3), dtype=float) * 2,
                  columns=['a', 'b', 'c'],
                  index=pd.date_range('6/13/2012', periods=6))
spliced = pd.concat([data1.ix[:'2012-06-14'], data2.ix['2012-06-15':]])

data2 = DataFrame(np.ones((6, 4), dtype=float) * 2,
                  columns=['a', 'b', 'c', 'd'],
                  index=pd.date_range('6/13/2012', periods=6))
spliced = pd.concat([data1.ix[:'2012-06-14'], data2.ix['2012-06-15':]])
spliced_filled = spliced.combine_first(data2)
spliced.update(data2, overwrite=False)

cp_spliced = spliced.copy()
cp_spliced[['a', 'c']] = data1[['a', 'c']]

price = web.get_data_yahoo('AAPL', '2011-01-01')['Adj Close']

returns = price.pct_change()
ret_index = (1 + returns).cumprod()
