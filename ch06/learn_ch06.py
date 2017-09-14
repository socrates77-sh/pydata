import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import sys
import csv
import json


result = pd.read_csv('ex6.csv')
result = pd.read_csv('ex6.csv', nrows=5)
chunker = pd.read_csv('ex6.csv', chunksize=1000)
tot = Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)

data = pd.read_csv('ex5.csv')
data.to_csv('out.csv')
data.to_csv(sys.stdout, sep='|')
data.to_csv(sys.stdout, na_rep='NULL')
data.to_csv(sys.stdout, index=False, header=False)
data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])

dates = pd.date_range('1/1/2000', periods=7)
ts = Series(np.arange(7), index=dates)
ts.to_csv('tseries.csv')

Series.from_csv('tseries.csv', parse_dates=True)

f = open('ex7.csv')
reader = csv.reader(f)
for line in reader:
    print(line)

lines = list(csv.reader(open('ex7.csv')))
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}

obj = """
{
    "name": "Wes",
    "places_lived": ["United States", "Spain", "Germany"],
    "pet": null,
    "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
                 {"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""
result = json.loads(obj)
asjson = json.dumps(result)

siblings = DataFrame(result['siblings'], columns=['name', 'age'])

frame = pd.read_csv('ex1.csv')