# tutorial: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


# load dataset
def parser(x):
    return pd.datetime.strptime('190'+x, '%Y-%m')


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# # create a differenced series (convert dataset to be time independant, getting changes in each step rather than value)
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = keras.models.Sequential()

    # stateful decides whether to clear LSTM layer state between batches or not
    model.add(keras.layers.LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # verbose suppresses some debug info, shuffle shuffles samples in an epoch
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0,shuffle=False)
        # reset states (since it isnt doing it automatically)
        model.reset_states()
    return model


# make a one-step forecast
def forcast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))

    # predict takes a 3d np array input (in this case its 1 val, observation at previosu step)
    # since we are providing a single input, it will output a 2d np array with 1 val
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]



# load data
series = pd.read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test
train, test = supervised_values[0:-12], supervised_values[-12:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# repeat, get average
repeats = 1
error_scores = list()
for r in range(repeats):
    # fit model
    # Consider trying 1500 epochs and 1 neuron, performance may be better
    lstm_model = fit_lstm(train_scaled, 1, 3000, 4)

    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)

    # walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forcast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forcast_lstm(lstm_model, 1, X)
        #invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
        # store forcast
        predictions.append(yhat)
        expected = raw_values[len(train)+i+1]
        print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

    # report performance
    rmse = (mean_squared_error(raw_values[-12:], predictions))**0.5
    print('Test RMSE: %.3f' % rmse)
    error_scores.append(rmse)

    # line plot of observed vs predicted
    plt.plot(raw_values[-12:])
    plt.plot(predictions)
    plt.show()

# summarize results
results = pd.DataFrame()
results['rmse'] = error_scores
print(results.describe())
results.boxplot()
plt.show()
