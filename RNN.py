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

# Generates a sequence of data and then transforms it into blocks called windows (4 numbers 1 label and so on)
dataset = tf.data.Dataset.range(6).window(4, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window_dataset: window_dataset.batch(4))

for window_tensor in dataset:
    print(f"{window_tensor}")

# Function that recieves a dataset and transforms it into windows blocks 
def to_windows(dataset, length):
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    return dataset.flat_map(lambda window_ds: window_ds.batch(length))

# Splits the dataset on 3 parts and divide them by a million (ensuring that the data keeps in the 0 - 1 range)
rail_train = df["rail"]["2016-01":"2018-12"] / 1e6
rail_valid = df["rail"]["2019-01":"2019-05"] / 1e6
rail_test = df["rail"]["2019-06":] / 1e6

seq_length = 56

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    rail_train.to_numpy(),
    targets=rail_train[seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    rail_valid.to_numpy(),
    targets=rail_valid[seq_length:],
    sequence_length=seq_length,
    batch_size=32
)


# Using a simple regression model to predict the number of passengers given a date
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[seq_length])
])

et_callback = tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_mae', restore_best_weights=True)
opt = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)

# model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=['mae'])
# history = model.fit(train_ds, validation_data=valid_ds, epochs=500, callbacks=[et_callback])

# Outputs a MAE of 37k, which is better then naive forecasting but is worse than SARIMA


# Using actual RNNs

rnn_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])
])

# rnn_model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=['mae'])
# rnn_model.fit(train_ds, validation_data=valid_ds, epochs=500, callbacks=[et_callback])

# Runs badly because it only has one neuron (cant keep up with this memory size) 
# and uses a tanh activation function, which returns values from -1 to 1, while
# the outputs would be at 1 - 1.4

better_rnn = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 1]),
    tf.keras.layers.Dense(1)
])
better_rnn.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=['mae'])
better_rnn.fit(train_ds, validation_data=valid_ds, epochs=500, callbacks=[et_callback])