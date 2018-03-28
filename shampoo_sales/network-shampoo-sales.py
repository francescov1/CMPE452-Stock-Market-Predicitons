# tutorial: https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/

import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# load dataset
def parser(x):
    return pd.datetime.strptime('190'+x, '%Y-%m')

# convert time series to supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop row with NaN vals
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i-interval]
        diff.append(value)
    return pd.Series(diff)


# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw vals
    raw_values = series.values

    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)

    # rescale vals to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)

    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values

    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test


# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, n_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    # design network
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(keras.layers.Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # fit network
    for i in range(n_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model

def update_lstm(model, train, n_lag, n_batch, n_epoch):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    # fit network
    for i in range(n_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model

def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))

    # make forecast
    forecast = model.predict(X, batch_size=n_batch)

    # convert to array
    return [x for x in forecast[0, :]]

# make a persistence forecast
def persistence(last_ob, n_seq):
    return [last_ob for i in range(n_seq)]

# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):

        X, y = test[i, 0:n_lag], test[i, n_lag:]

        # make forecast
        # presistant forecast (simple, take last value and persist, use to get baseline rmse)
        #forecast = persistence(X[-1], n_seq)

        # LSTM forecast
        forecast = forecast_lstm(model, X, n_batch)

        # store forecast
        forecasts.append(forecast)

        # add new data to retrain
        row = train.shape[0]
        col = train.shape[1]
        train = np.append(train, test[i])
        train = train.reshape(row + 1, col)

        model = update_lstm(model, train, n_lag, n_batch, n_epoch=5)

    return forecasts


# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)

    # propagate difference forecast using inverted first val
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted

# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = np.array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))

        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]

        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)

        # store
        inverted.append(inv_diff)
    return inverted


# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):

    for i in range(n_seq):
        deviations = []
        mis = 0

        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]

        for j, currentPrice in enumerate(actual):

            dev = np.abs(currentPrice - predicted[j]) / currentPrice
            deviations.append(dev)

            if j == (len(actual)-1):
                break

            predictedDirection = np.sign(predicted[j+1] - currentPrice)
            actualDirection = np.sign(actual[j+1] - currentPrice)

            mis += np.abs(actualDirection-predictedDirection)/2

        error = mis/(len(actual)-1)
        rmse = (mean_squared_error(actual, predicted))**0.5
        avgDev = np.mean(deviations)

        print('\nt+%d' % (i+1))
        print('RMSE: %f' % rmse)
        print('Error in direction: %f' % error)
        print('Avg deviation of price: %f' % avgDev)


# plot forecasts in context of original dataset
# also connect persisted forecast to actual persisted value in original dataset
def plot_forecasts(series, forecasts, n_test):
    # plot entire dataset in blue
    plt.plot(series.values)

    # plot forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        plt.plot(xaxis, yaxis, color='red')
    plt.show()


# save trained network
def save_network(model, n_neurons, n_epochs):
    # serialize model to JSON
    model_json = model.to_json()

    filename = "model_%dneu_%depoch" % (n_neurons, n_epochs)

    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename + ".h5")
    print("Saved model to disk")


# load previously trained network
def load_network(n_neurons, n_epochs):

    filename = "model_%dneu_%depoch" % (n_neurons, n_epochs)

    # load json and create model
    json_file = open(filename + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename + ".h5")
    loaded_model.compile(loss='mean_squared_error', optimizer='adam')

    print("Loaded model from disk")
    return loaded_model


# load dataset
series = pd.read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# configure
n_lag = 1
n_seq = 3
n_test = 10
n_epochs = 1000
n_batch = 1
n_neurons = 1

# specifies if network should retrain with new data after making each prediction
updateLSTM = False

# specifies if network should be trained or loaded from last training
load_model = False

# prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)


if load_model:
    model = load_network()
else:
    # fit network
    model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)

# make forecast
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)

# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2)

# evaluate and plot forecasts
print('\n\nParameters:\nNeurons =', n_neurons, '\nEpochs =', n_epochs)
evaluate_forecasts(actual, forecasts, n_lag, n_seq)

plot_forecasts(series, forecasts, n_test+2)

if not load_model:
    save_model = input('Do you want to save this network? This will overwrite previously saved network\n(y/n) ')

    if save_model.lower() == 'y':
        # save network
        save_network(model, n_neurons, n_epochs)