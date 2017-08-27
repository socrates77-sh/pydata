import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from pandas_datareader import data
from numpy import nan as NA

obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)
obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')

frame = DataFrame(np.arange(9).reshape((3, 3)), index=[
                  'a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)
# frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill', columns=states)
frame.reindex(index=['a', 'b', 'c', 'd'],  columns=states)
frame.ix[['a', 'b', 'c', 'd'],  states]

obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')
obj.drop(['d', 'c'])

data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data.drop(['Colorado', 'Ohio'])
data.drop('two', axis=1)
data.drop(['two', 'four'], axis=1)

obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj['b']
obj[1]
obj[2:4]
obj[['b', 'a', 'd']]
obj[[1, 3]]
obj[obj < 2]
obj['b':'c']
obj['b':'c'] = 5
obj

data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data
data['two']
data[['three', 'one']]
data[:2]
data[data['three'] > 5]
data < 5
data[data < 5] = 0
data
data.ix['Colorado', ['two', 'three']]
data.ix[['Colorado', 'Utah'], [3, 0, 1]]
data.ix[2]
data.ix[:'Utah', 'two']
data.ix[data.three > 5, :3]

s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'b', 'c', 'd'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1
s2
s1 + s2
df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list(
    'bcd'), index=['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list(
    'bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1
df2
df1 + df2

df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df1.add(df2, fill_value=0)
df1.reindex(columns=df2.columns, fill_value=0)

arr = np.arange(12.).reshape((3, 4))
arr - arr[0]

frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list(
    'bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.ix[0]
frame - series
series2 = Series(range(3), index=['b', 'e', 'f'])
frame + series2
series3 = frame['d']
frame.sub(series3, axis=0)

frame = DataFrame(np.random.randn(4, 3), columns=list(
    'bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
np.abs(frame)

obj = Series(np.arange(4.), index=['d', 'a', 'b', 'c'])
obj.sort_index()
frame = DataFrame(np.arange(8).reshape((2, 4)), columns=list(
    'dabc'), index=['three', 'one'])
frame.sort_index()
frame.sort_index(axis=1)
frame.sort_index(axis=1, ascending=False)
obj = Series([4, 7, -3, 2])
obj.sort_values()
obj = Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()
frame = DataFrame({'b': [4, 7, -3, -2], 'a': [0, 1, 0, 1]})
frame.sort_index(by='b')
frame.sort_values(by=['a', 'b'])
obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()
obj.rank(method='first')
obj.rank(ascending=False, method='max')
frame = DataFrame(
    {'b': [4.3, 7, -3, -2], 'a': [0, 1, 0, 1], 'c': [-2, 5, 8, -2.5]})
frame.rank(axis=1)
obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj.index.is_unique()
obj['a']

df = DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan],
                [0.75, -1.3]], index=['a', 'b', 'c', 'd'],
               columns=['one', 'two'])
df.sum()
df.sum(axis=1)
df.mean(axis=1, skipna=False)
df.idxmax()
df.cumsum()
df.describe()
obj = Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()

all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']:
    all_data[ticker] = data.get_data_yahoo(ticker, '1/1/2000', '1/1/2010')
price = DataFrame({tic: data['Adj Close'] for tic, data in all_data.items()})
volume = DataFrame({tic: data['Volume'] for tic, data in all_data.items()})
returns = price.pct_change()
returns.tail()
returns.MSFT.corr(returns.IBM)
returns.MSFT.cov(returns.IBM)

obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
uniques = obj.unique()
obj.value_counts()
pd.value_counts(obj.values, sort=False)
mask = obj.isin(['b', 'c'])
obj[mask]
data = DataFrame({'Qu1': [1, 3, 4, 3, 4],
                  'Qu2': [2, 3, 1, 2, 3],
                  'Qu3': [1, 5, 2, 4, 4]})
data.apply(pd.value_counts).fillna(0)

string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data.isnull()

data = Series([1, NA, 3.5, NA, 7])
data.dropna()
data[data.notnull()]
data = DataFrame([[1, 6.5, 3], [1, NA, NA], [NA, NA, NA], [NA, 6.5, 3]])
data.dropna()
data.dropna(how='all')
data[4] = NA
data.dropna(axis=1, how='all')
df = DataFrame(np.random.randn(7, 3))
df.ix[:4, 1] = NA
df.ix[:2, 2] = NA
df.dropna(thresh=3)
df.fillna(0)
df.fillna({1: 0.5})

data = Series(np.random.randn(10),
              index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],
                     [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])

frame = DataFrame(np.arange(12).reshape((4, 3)),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=[['Ohio', 'Ohio', 'Colorado'],
                           ['Green', 'Red', 'Green']])
