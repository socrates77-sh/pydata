import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def add_prop(group):
    births = group.births
    group['prop'] = births / births.sum()
    return group


def get_top1000(group):
    return group.sort_values(by='births', ascending=False)[:1000]


def get_quantile_count(group, q=0.5):
    group = group.sort_values(by='prop', ascending=False)
    return group.prop.cumsum().searchsorted(q) + 1

years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']

for year in years:
    path = 'ch02/names/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)

names = pd.concat(pieces, ignore_index=True)
# print(names)
total_births = names.pivot_table(
    'births', index='year', columns='sex', aggfunc=sum)

# print(total_births.tail())
# total_births.plot(title='Total births by sex and year')
# plt.show()

names = names.groupby(['year', 'sex']).apply(add_prop)
# print(names)
# print(np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1))

grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)
# print(top1000)

boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']

total_births = top1000.pivot_table(
    'births', index='year', columns='name', aggfunc=sum)

# print(total_births)

subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
# print(subset)

# subset.plot(subplots=True, figsize=(12, 10), grid=False,
#             title='Number of births per year')
# plt.show()

table = top1000.pivot_table('prop', index='year', columns='sex', aggfunc=sum)
# print(table)
# table.plot(title='Sum of table100.prop by year and sex',
#            yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))
# plt.show()

# df = boys[boys.year == 1900]
# in1900 = df.sort_values(by='prop', ascending=False).prop.cumsum()
# print(in1900.searchsorted(0.5) + 1)

# diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
# diversity = diversity.unstack('sex')
# # print(diversity)
# diversity.plot(title='Number of popular names in top 50%')
# plt.show()

get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
# print(last_letter)
last_letters.name = 'last_letter'

table = names.pivot_table('births', index='last_letters', columns=[
                          'sex', 'year'], aggfunc=sum)
subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
print(subtable.head())
