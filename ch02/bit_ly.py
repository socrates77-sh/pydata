
import json
from collections import defaultdict
from collections import Counter
from pandas import DataFrame, Series
import pandas as pd
import numpy as np


def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]


def get_counts2(sequence):
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    return counts


def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

path = 'ch02/usagov_bitly_data2012-03-16-1331923249.txt'
# a = open(path).readline()
# print(a)

records = [json.loads(line) for line in open(path)]
# print(open(path))
# print(records[0])
# print(records[0]['tz'])

time_zones = [rec['tz'] for rec in records if 'tz' in rec]
# print(time_zones[:10])

# counts = get_counts2(time_zones)
# # print(counts)

# print(top_counts(counts))

counts = Counter(time_zones)
# print(counts)
# print(counts.most_common(10))

frame = DataFrame(records)
# print(frame['tz'][:10])
tz_counts = frame['tz'].value_counts()
# print(tz_counts[:10])

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
print(tz_counts[:10])

tz_counts[:10].plot(kind='barh', rot=0)
