import tensorflow as tf
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

path = Path('datasets/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals_20250324.csv')

# W for weekdays | A for Saturdays | and U for Sundays or Hollidays
df = pd.read_csv(path, parse_dates=['service_date'])
df.columns = ["date", "day_type", "bus", "rail", "total"]

# Sets the date as the index instead of an integer
df = df.sort_values('date').set_index('date')
df = df.drop('total', axis=1) # Remove it since its Bus + train
df = df.drop_duplicates() # It has two duplicates, and sice the df is ordered by the date its going to remove duplicated dates

diff_7 = df[["bus", "rail"]].diff(7)["2019-03":"2019-05"]
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
df.plot(ax=axs[0], legend=False, marker=".") # original time series
df.shift(7).plot(ax=axs[0], grid=True, legend=False, linestyle=":") 
diff_7.plot(ax=axs[1], grid=True, marker=".") # 7-day difference time
plt.show()