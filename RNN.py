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
# plt.show()

from statsmodels.tsa.arima.model import ARIMA
origin, today = "2019-01-01", "2019-05-31"
rail_series = df.loc[origin:today]["rail"].asfreq("D")
model = ARIMA(rail_series, order=(1, 0, 0), seasonal_order=(0, 1, 1, 7)) # Order -> p=1; d=0; q=0 (p->How many past values; d->How many diffs; q->moving average from previous data)
model = model.fit()                                                      # Seasonal Order -> P=0; D=1; Q=1; s=7. Sazonality hyperparameters
y_pred = model.forecast() 
print(y_pred)
# The prediction is slightly worse than naive forecasting (repeating the previous week data)

origin, start_date, end_date = "2019-01-01", "2019-03-01", "2019-05-31"

time_period = pd.date_range(start_date, end_date)
rail_series = df.loc[origin:end_date]["rail"].asfreq("D")

y_preds = []

for today in time_period.shift(-1):
    model = ARIMA(rail_series[origin:today], order=(1, 0, 0), seasonal_order=(0, 1, 1, 7))

    model = model.fit() 
    y_pred = model.forecast().iloc[0]
    y_preds.append(y_pred)

y_preds = pd.Series(y_preds, index=time_period)
mae = (y_preds - rail_series[time_period]).abs().mean()
print(mae)

# The mean absolute error is way batter than naive forecasting, beating by aproximatelly 10k