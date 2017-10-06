import numpy as np
import pandas as pd
from pandas import Series, DataFrame

df1 = DataFrame(
    {'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df2 = DataFrame({'key': ['a', 'b', 'd'], 'data2': range(3)})
pd.merge(df1, df2)
pd.merge(df1, df2, on='key')

df3 = DataFrame(
    {'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df4 = DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(3)})
pd.merge(df3, df4, left_on='lkey', right_on='rkey')

pd.merge(df1, df2, how='outer')

df1 = DataFrame(
    {'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'], 'data2': range(5)})
pd.merge(df1, df2, on='key', how='left')

left = DataFrame({'key1': ['foo', 'foo', 'bar'],
                  'key2': ['one', 'two', 'one'],
                  'lval': [1, 2, 3]})
right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                   'key2': ['one', 'one', 'one', 'two'],
                   'rval': [4, 5, 6, 7]})
pd.merge(left, right, on=['key1', 'key2'], how='outer')

pd.merge(left, right, on='key1')
pd.merge(left, right, on='key1', suffixes=('_left', '_right'))

left1 = DataFrame(
    {'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
pd.merge(left1, right1, left_on='key', right_index=True)
pd.merge(left1, right1, left_on='key', right_index=True, how='outer')

lefth = DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                   'key2': [2000, 2001, 2002, 2001, 2002],
                   'data': np.arange(5.)})
righth = DataFrame(np.arange(12).reshape((6, 2)),
                   index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
                          [2001, 2000, 2000, 2000, 2001, 2002]],
                   columns=['event1', 'event2'])
pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True)
pd.merge(lefth, righth, left_on=['key1', 'key2'],
         right_index=True, how='outer')

left2 = DataFrame([[1, 2], [3, 4], [5, 6]],
                  index=['a', 'c', 'e'], columns=['Ohio', 'Nevada'])
right2 = DataFrame([[7, 8], [9, 10], [11, 12], [13, 14]],
                   index=['b', 'c', 'd', 'e'], columns=['Missouri', 'Alabama'])
pd.merge(left2, right2, how='outer', left_index=True, right_index=True)
left2.join(right2, how='outer')
left1.join(right1, on='key')

another = DataFrame([[7, 8], [9, 10], [11, 12], [16, 17]],
                    index=['a', 'c', 'e', 'f'], columns=['New York', 'Oregon'])
left2.join([right2, another])
left2.join([right2, another], how='outer')

arr = np.arange(12).reshape((3, 4))
np.concatenate([arr, arr], axis=1)

s1 = Series([0, 1], index=['a', 'b'])
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])
pd.concat([s1, s2, s3])
pd.concat([s1, s2, s3], axis=1)

s4 = pd.concat([s1 * 5, s3])
pd.concat([s1, s4], axis=1)
pd.concat([s1, s4], axis=1, join='inner')
pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']])
pd.concat([s1, s2, s3], keys=['one', 'two', 'three'])
pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])

df1 = DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
                columns=['one', 'two'])
df2 = DataFrame(np.arange(4).reshape(2, 2), index=['a',  'c'],
                columns=['three', 'four'])
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])
pd.concat({'level1': df1, 'level2': df2}, axis=1)
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'],
          names=['upper', 'lower'])

df1 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
pd.concat([df1, df2], ignore_index=True)

a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
           index=['f', 'e', 'd', 'c', 'b', 'a'])
b = Series(np.arange(len(a), dtype=np.float64),
           index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] = np.nan

df1 = DataFrame({'a': [1, np.nan, 5, np.nan],
                 'b': [np.nan, 2, np.nan, 6],
                 'c': range(2, 18, 4)})
df2 = DataFrame({'a': [5, 4, np.nan, 3, 7],
                 'b': [np.nan, 3, 4, 6, 8]})
df1.combine_first(df2)

data = DataFrame(np.arange(6).reshape((2, 3)),
                 index=pd.Index(['Ohio', 'Colorado'], name='state'),
                 columns=pd.Index(['one', 'two', 'three'], name='number'))
result = data.stack()
result.unstack()

s1 = Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])

df = DataFrame({'left': result, 'right': right + 5},
               columns=pd.Index(['left', 'right'], name='side'))

data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                  'k2': [1, 1, 2, 3, 3, 4, 4]})
data.duplicated()
data.drop_duplicates()
data['v'] = range(7)
data.drop_duplicates(['k1'])
data.drop_duplicates(['k1', 'k2'])

data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami',
                           'corned beef', 'Bacon', 'pastrami', 'honey ham', 'nova lox'],
                  'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
meat_to_animal = {
    'bacon': 'pig',
    'pulled pork': 'pig',
    'pastrami': 'cow',
    'corned beef': 'cow',
    'honey ham': 'pig',
    'nova lox': 'salmon'
}
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
data['food'].map(lambda x: meat_to_animal[x.lower()])

data = Series([1, -999, 2, -999, -1000, 3])
data.replace(-999, np.nan)
data.replace([-999, -1000], np.nan)

data = DataFrame(np.arange(12).reshape((3, 4)),
                 index=['Ohio', 'Colorado', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data.index.map(str.upper)
data.index = data.index.map(str.upper)
data.rename(index=str.title, columns=str.upper)
data.rename(index={'OHIO': 'INDIANA'},
            columns={'three': 'peekaboo'})

ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bin)

data = np.random.randn(1000)
cats = pd.qcut(data, 4)
pd.value_counts(cats)

np.random.seed(12345)
data = DataFrame(np.random.randn(1000, 4))
data.describe()

col = data[3]
col[np.abs(col) > 3]

data[(np.abs(data) > 3).any(1)]
data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()

df = DataFrame(np.arange(5 * 4).reshape(5, 4))
sampler = np.random.permutation(5)

val = 'a,b,  guido'
val.split(',')
pieces = [x.strip() for x in val.split(',')]
'::'.join(pieces)

import re

text = 'foo  bar\t baz  \tqux'

re.split('\s+', text)
regex = re.compile('\s+')
regex.split(text)
regex.findall(text)

text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
regex = re.compile(pattern, flags=re.IGNORECASE)
regex.findall(text)

m = regex.search(text)
text[m.start():m.end()]

regex.match(text)
s = regex.sub('REDACTED', text)

pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)
m = regex.match('wesm@bright.net')
m.groups()
regex.findall(text)
print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))

regex = re.compile(r"""
(?P<username>[A-Z0-9._%+-]+)
@
(?P<domain>[A-Z0-9.-]+)
\.
(?P<suffix>[A-Z]{2,4})
""", flags=re.IGNORECASE | re.VERBOSE)
m = regex.match('wesm@bright.net')
m.groupdict()

data = {'Dave': 'dave@google.com',
        'Steve': 'steve@gmail.com',
        'Rob': 'rob@gmail.com',
        'Ryan': 'ryan@yahoo.com',
        'Wes': np.nan}
data = Series(data)
data.isnull()
data.str.contains('gmail')
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
data.str.findall(pattern, flags=re.IGNORECASE)
matches = data.str.match(pattern, flags=re.IGNORECASE)

import json
db = json.load(open('foods-2011-10-03.json'))
len(db)

nutrients = DataFrame(db[0]['nutrients'])
info_keys = ['description', 'group', 'id',  'manufacturer']
info = DataFrame(db, columns=info_keys)
pd.value_counts(info.group)

nutrients = []
for rec in db:
    fnuts = DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)
nutrients = pd.concat(nutrients, ignore_index=True)
nutrients.duplicated().sum()
nutrients = nutrients.drop_duplicates()

col_mapping = {'description': 'food',
               'group': 'fgroup'}
info = info.rename(columns=col_mapping, copy=False)
col_mapping = {'description': 'nutrient',
               'group': 'nutgroup'}
nutrients = nutrients.rename(columns=col_mapping, copy=False)
ndata = pd.merge(nutrients, info, on='id', how='outer')
