import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

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

df = DataFrame({'category': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                'data': np.random.randn(8),
                'weights': np.random.rand(8)})
grouped = df.groupby('category')
get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
grouped.apply(get_wavg)

close_px = pd.read_csv('stock_px.csv', parse_dates=True, index_col=0)
rets = close_px.pct_change().dropna()
spx_corr = lambda x: x.corrwith(x['SPX'])
by_year = rets.groupby(lambda x: x.year)
by_year.apply(spx_corr)
by_year.apply(lambda g: g['AAPL'].corr(g['MSFT']))

tips.pivot_table(index=['sex', 'smoke'])
tips.pivot_table(['tip_pct', 'size'], index=['sex', 'day'], columns='smoker')
tips.pivot_table(['tip_pct', 'size'], index=['sex', 'day'], columns='smoker',
                 margins=True)
tips.pivot_table(['tip_pct', 'size'], index=['sex', 'day'], columns='smoker',
                 margins=True, aggfunc=len)
tips.pivot_table('size', index=['time', 'sex', 'smoker'], columns='day',
                 aggfunc=sum, fill_value=0)
pd.crosstab([tips['time'], tips['day']], tips['smoker'], margins=True)

fec = pd.read_csv('P00000001-ALL.csv')
unique_cands = fec.cand_nm.unique()

parties = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           Roemer, Charles E. 'Buddy' III: 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}

fec.cand_nm[123456:123461]
fec.cand_nm[123456:123461].map(parties)
fec['party'] = fec.cand_nm.map(parties)
fec['party'].value_counts()
(fec.contb_receipt_amt > 0).value_counts()
fec = fec[fec.contb_receipt_amt > 0]
fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]

fec.contbr_occupation.value_counts()[:10]

occ_mapping = {
    'INFORMATION REQUESTED PER BEST EFFORTS': 'NOT PROVIDED',
    'INFORMATION REQUESTED': 'NOT PROVIDED',
    'INFORMATION REQUESTED (BEST EFFORTS)': 'NOT PROVIDED',
    'C.E.O.': 'CEO'
}

f = lambda x: occ_mapping.get(x, x)
fec.contbr_occupation = fec.contbr_occupation.map(f)


emp_mapping = {
    'INFORMATION REQUESTED PER BEST EFFORTS': 'NOT PROVIDED',
    'INFORMATION REQUESTED': 'NOT PROVIDED',
    'SELF': 'SELF-EMPLOYED',
    'SELF EMPLOYED': 'SELF-EMPLOYED',
}

f = lambda x: emp_mapping.get(x, x)
fec.contbr_employer = fec.contbr_employer.map(f)

by_occupation = fec.pivot_table('contb_receipt_amt',
                                index='contbr_occupation',
                                columns='party',
                                aggfunc='sum')
over_2mm = by_occupation[by_occupation.sum(1) > 2000000]

over_2mm.plot(kind='barh')


def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)['contb_receipt_amt'].sum()
    return totals.sort_values(ascending=False)[:n]

grouped = fec_mrbo.groupby('cand_nm')
grouped.apply(get_top_amounts, 'contbr_occupation', n=7)
grouped.apply(get_top_amounts, 'contbr_employer', n=10)

bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
labels = pd.cut(fec_mrbo.contb_receipt_amt, bins)
grouped = fec_mrbo.groupby(['cand_nm', labels])
grouped.size().unstack()

bucket_sums = grouped.contb_receipt_amt.sum().unstack(0)
normed_sums = bucket_sums.div(bucket_sums.sum(axis=1), axis=0)

normed_sums[:-2].plot(kind='barh', stacked=True)
plt.show()

grouped = fec_mrbo.groupby(['cand_nm', 'contbr_st'])
totals = grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
totals = totals[totals.sum(1) > 100000]
percent = totals.div(totals.sum(1), axis=0)

