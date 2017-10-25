import numpy as np
import pandas as pd
from pandas import Series, DataFrame

df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                'key2': ['one', 'two', 'one', 'two', 'one'],
                'data1': np.random.randn(5),
                'data2': np.random.randn(5)})

grouped = df['data1'].groupby(df['key1'])
grouped.mean()

means = df['data1'].groupby([df['key1'], df['key2']]).mean()
means.unstack()

states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
df['data1'].groupby([states, years]).mean()
df.groupby('key1').mean()
df.groupby(['key1', 'key2']).mean()
df.groupby(['key1', 'key2']).size()

for name, group in df.groupby('key1'):
    print(name)
    print(group)

for (k1, k2), group in df.groupby(['key1', 'key2']):
    print(k1, k2)
    print(group)

pieces = dict(list(df.groupby('key1')))

people = DataFrame(np.random.randn(5, 5),
                   columns=['a', 'b', 'c', 'd', 'e'],
                   index=['Joe', 'Steve', 'Wes', 'Jim', 'Traivs'])
people.ix[2:3, ['b', 'c']] = np.nan

mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
           'd': 'blue', 'e': 'red', 'f': 'orange'}
by_column = people.groupby(mapping, axis=1)
by_column.sum()

map_series = Series(mapping)
people.groupby(map_series, axis=1).count()

people.groupby(len).sum()

#271
