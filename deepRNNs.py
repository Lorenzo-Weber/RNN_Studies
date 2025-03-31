import tensorflow as tf
from tensorflow import keras
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

deep_model = keras.Sequential([
    keras.layers.SimpleRNN(32, input_shape=[None, 1], return_sequences=True),
    keras.layers.SimpleRNN(32, return_sequences=True),
    keras.layers.SimpleRNN(32),
    keras.layers.Dense(1)
])

et_callback = tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_mae', restore_best_weights=True)
opt = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)

deep_model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=['mae'])
deep_model.fit(train_ds, validation_data=valid_ds, epochs=500, callbacks=[et_callback])