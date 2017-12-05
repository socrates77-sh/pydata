from datetime import datetime
from datetime import timedelta, time
from dateutil.parser import parse
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt

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

#365
