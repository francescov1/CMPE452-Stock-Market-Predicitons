from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

# add a new column with the data for the next day for each column in the original file 
def Preprocess(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % ( j +1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % ( j +1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % ( j +1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# load dataset and set values to float
dataset = read_csv('5yr S&P data.csv')
dataset.drop(dataset.columns[[0]], axis=1, inplace=True)
values = dataset.values
values = values.astype('float32')

# normalize the input features to be between 0 and 1
normalized = MinMaxScaler(feature_range=(0, 1))
normed = normalized.fit_transform(values)
# append collumns with values for the next day
processed_data = Preprocess(normed, 1, 1)
# drop columns we don't want to predict (we only want to keep the column with the S&P
# values for the next day)
processed_data.drop(processed_data.columns[[6, 7, 8, 9]], axis=1, inplace=True)
print(processed_data.head())

# split data into training and testing sets (80 - 20 split)
values = processed_data.values
test_num = 1007
training = values[:test_num, :]
testing = values[test_num:, :]

# get input and output arrays
train_input, train_output = training[:, :-1], training[:, -1]
np.delete(train_input, [0], axis=0)
test_input, test_output = testing[:, :-1], testing[:, -1]
np.delete(test_input, [0], axis=0)
# reshape to pass into the neural network
train_input = train_input.reshape((train_input.shape[0], 1, train_input.shape[1]))
test_input = test_input.reshape((test_input.shape[0], 1, test_input.shape[1]))

# build lstm neural network [4-50-100-1] nodes per layer
model = Sequential()
# input (layer 1) and layer 2
model.add(LSTM(
    input_shape=(train_input.shape[1], train_input.shape[2]),
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

# layer 3
model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

# layer 4
model.add(Dense(
    output_dim=1))
# using a linear activation function for output
model.add(Activation("linear"))

model.compile(loss='mae', optimizer='adam')


# fit network (epochs can be changed but 65 was found to work best through testing)
fit = model.fit(train_input, train_output, epochs=65, batch_size=72, validation_data=(test_input, test_output), verbose=2,
                    shuffle=False)

# run test data through the trained network for validation
prediction = model.predict(test_input)
test_input = test_input.reshape((test_input.shape[0], test_input.shape[2]))

# plot the epoch losses
plt.plot(fit.history['loss'], label='train')
plt.plot(fit.history['val_loss'], label='test')
plt.legend()
plt.show()

# invert normalization
inv_prediction = concatenate((prediction, test_input[:, 1:]), axis=1)
inv_prediction = normalized.inverse_transform(inv_prediction)
inv_prediction = inv_prediction[:, 0]
test_output = test_output.reshape((len(test_output), 1))
inv_output = concatenate((test_output, test_input[:, 1:]), axis=1)
inv_output = normalized.inverse_transform(inv_output)
inv_output = inv_output[:, 0]

# plot network prediction against the actual S&P values
plt.plot(inv_output, label='S&P Value')
plt.plot(inv_prediction, label='Predicted S&P Value')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_output, inv_prediction))
print('RMSE: %.3f' % rmse)
