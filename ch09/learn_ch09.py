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

grouped = df.groupby('key1')
grouped['data1'].quantile(0.9)


def peak_to_peak(arr):
    return arr.max() - arr.min()

grouped.agg(peak_to_peak)
grouped.describe()

tips = pd.read_csv('tips.csv')
tips['tip_pct'] = tips['tip'] / tips['total_bill']
grouped = tips.groupby(['sex', 'smoker'])
grouped_pct = grouped['tip_pct']
grouped_pct.agg('mean')
grouped_pct.agg(['mean', 'std', peak_to_peak])
grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])
functions = ['count', 'mean', 'max']
result = grouped['tip_pct', 'total_bill'].agg(functions)
grouped.agg({'tip': np.max, 'size': 'sum'})
grouped.agg({'tip_pct': ['min', 'max', 'mean', 'std'], 'size': 'sum'})

k1_means = df.groupby('key1').mean().add_prefix('mean_')
pd.merge(df, k1_means, left_on='key1', right_index=True)

key = ['one', 'two', 'one', 'two', 'one']
people.groupby(key).mean()
people.groupby(key).transform(np.mean)


def demean(arr):
    return arr - arr.mean()

demeaned = people.groupby(key).transform(demean)
demeaned.groupby(key).mean()


def top(df, n=5, column='tip_pct'):
    return df.sort_values(by=column)[-n:]

top(tips, n=6)

tips.groupby('smoker').apply(top)
tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill')
result = tips.groupby('smoker')['tip_pct'].describe()
result.unstack('smker')


def f = lambda x: x.describe()
grouped.apply(f)

tips.groupby('smoker', group_keys=False).apply(top)

frame = DataFrame({'data1': np.random.randn(1000),
                   'data2': np.random.randn(1000)})
factor = pd.cut(frame.data1, 4)


def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}

grouped = frame.data2.groupby(factor)
grouped.apply(get_stats).unstack()

grouping = pd.qcut(frame.data1, 10, labels=False)
grouped = frame.data2.groupby(grouping)
grouped.apply(get_stats).unstack()

s = Series(np.random.randn(6))
s[::2] = np.nan
s.fillna(s.mean())

states = ['Ohio', 'New York', 'Vermont', 'Floria',
          'Oregon', 'Nevada', 'Clifornia', 'Idaho']
group_key = ['East'] * 4 + ['West'] * 4
data = Series(np.random.randn(8), index=states)
data[['Vermont', 'Nevada', 'Idaho']] = np.nan
data.groupby(group_key).mean()


def fill_mean = lambda g: g.fillna(g.mean())
data.groupby(group_key).apply(fill_mean)

fill_values = {'East': 0.5, 'West': -1}


def fill_func = lambda g: g.fillna(fill_func[g.name])
data.groupby(group_key).apply(fill_func)

suits = ['H', 'S', 'C', 'D']
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ['A'] + list(range(2, 11)) + ['J', 'Q', 'K']
cards = []
for suit in suits:
    cards.extend(str(num) + suit for num in base_names)
deck = Series(card_val, index=cards)


def draw(deck, n=5):
    return deck.take(np.random.permutation(len(deck))[:n])

draw(deck)

#296