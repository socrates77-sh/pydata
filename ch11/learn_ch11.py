from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt

ts1 = Series(np.random.randn(3),
             index=pd.date_range('2012-6-13', periods=3, freq='W-WED'))

ts1.resample('B')
ts1.resample('B', fill_method='ffill')

#359
