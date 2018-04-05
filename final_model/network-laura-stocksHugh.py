# tutorial: https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/

import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, confusion_matrix
from matplotlib import pyplot as plt
import qexpy as q



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
def prepare_data(series, n_lag, n_seq):

    # rescale vals to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(series)
    scaled_values = scaled_values.reshape(series.shape[0], series.shape[1])

    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    indexes = [7+i*8 for i in range(n_lag)]
    extra = [(7+(n_lag-1)*8)+i for i in range(1,8)]
    for x in extra:
        indexes.append(x)
    supervised.drop(supervised.columns[indexes], axis=1,inplace=True) #for n_lag=5: drop(7,15,23,31,39,40,41,42,43,44,45,46
    supervised_values = supervised.values
    
    return scaler, supervised_values


# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, n_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, :-1], train[:, -1]
    X = X.reshape(X.shape[0],n_lag,7)
    
    # design network
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences = True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(100, input_shape=(X.shape[1], X.shape[2])))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')

    print("Training progress:")
    # fit network
    for i in range(n_epoch):
        if np.mod(i, n_epoch/10) == 0 and i != 0:
            print('%d%% complete' % (i*100/n_epoch))
        model.fit(X, y, epochs=1, verbose=0, shuffle=False)
        model.reset_states()
    print('100% complete')

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
    # reshape input pattern to [samples, timesteps, features
    X = X.reshape(1,X.shape[0],X.shape[1],)
    # make forecast
    forecast = model.predict(X)

    # convert to array
    return [x for x in forecast[0, :]]

# make a persistence forecast
def persistence(last_ob, n_seq):
    return [last_ob for i in range(n_seq)]

# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq, updateLSTM=False):
    forecasts = list()
    print('\nMaking predictions')
    X, Y = test[:, :-1], train[:, -1]
    X = X.reshape(X.shape[0],n_lag,7)
    for i in range(len(test)):
        
        x = X[i,:,:]
        # make forecast
        # presistant forecast (simple, take last value and persist, use to get baseline rmse)
        #forecast = persistence(X[-1], n_seq)

        # LSTM forecast
        forecast = forecast_lstm(model, x, n_batch)

        # store forecast
        forecasts.append(forecast)

        # add new data to retrain
        if updateLSTM:
            if np.mod(i, len(test) / 10) == 0 and i != 0:
                print('%d%% complete' % (i * 100 / len(test)))

            row = train.shape[0]
            col = train.shape[1]
            train = np.append(train, test[i])
            train = train.reshape(row + 1, col)
            model = update_lstm(model, train, n_lag, n_batch, n_epoch=5)

    print('100%% complete')

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
        temp = [-1,-1,-1,-1,-1,-1,-1,forecasts[i][0]]
        forecast = np.array(temp)
        forecast = forecast.reshape(1, len(forecast))

        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, 7]

        # invert differencing
#        index = len(series) - n_test + i - 1
#        last_ob = series.values[index]
#        inv_diff = inverse_difference(last_ob, inv_scale)

        # store
        inverted.append(inv_scale)
    return inverted


# evaluate the RMSE for each forecast time step
def evaluate_forecasts(actual, forecasts, n_lag, n_seq):

        predictedDirections = np.sign(forecasts)
        actualDirections = np.sign(actual)

        tn, fp, fn, tp = confusion_matrix(actualDirections, predictedDirections, [-1, 1]).ravel()
        accuracy = (tp+tn) / (tp+tn+fp+fn)
        rmse = (mean_squared_error(actual, forecasts))**0.5
        print('\nt')
        print('RMSE: %f' % rmse)
        print('Accuracy of direction: %f' % accuracy)


def evaluate_forecast(forecasts, actual):
    correct = np.zeros(len(forecasts))
    for i in range(len(forecasts)):
        if ((forecasts[i]-actual[i])>0.0 and actual[i]<actual[i+1]) or ((forecasts[i]-actual[i])<0.0 and actual[i]>actual[i+1]):
            correct[i] = 1
    accuracy = correct.sum()/correct.shape[0]
    print('Accuracy: %f' % accuracy)
    

# plot forecasts in context of original dataset
# also connect persisted forecast to actual persisted value in original dataset
'''def plot_forecasts(series, forecasts, n_test):
    # plot entire dataset in blue
    X = [i for i in range(len(series))]
    r = [i for i in range(len(forecasts))]
    N=len(series)
    series1=series
    forecasts1=forecasts
    xaxis = [i + N - n_test -1for i in r]
    yaxis = [series[i + N - n_test-1] + forecasts[i] for i in r]
    
    X = q.MeasurementArray(X, error = 0,units = '')
    series = q.MeasurementArray(series, error = 0,units = '')
    xy1 = q.XYDataSet(X,series)
    xaxis = q.MeasurementArray(xaxis, error = 0,units = '')
    yaxis = q.MeasurementArray(yaxis, error = 0,units = '')
    xy2 = q.XYDataSet(xaxis,yaxis)

    q.plot_engine = 'mpl'
    figure = q.MakePlot(xy1)
    figure.add_dataset(xy2)
    figure.show()
    a = np.array([series1[i + N - n_test-1] + forecasts1[i] for i in r])
    b = series1[(N - n_test-2):]
    evaluate_forecast(a,b)
    
    # plot entire dataset in blue
    plt.plot(series1)

    # plot forecasts in red
    off_s = (series1.shape[0]) - n_test - 1
    off_e = off_s + n_test + 1
    xaxis = [x for x in range(off_s, off_e)]
    yaxis = [series1[xaxis]] + forecasts
    plt.plot(xaxis, yaxis, color='red')
    plt.show()'''
    
def plot_forecasts(series, forecasts, n_test):
    # plot entire dataset in blue
    plt.plot(series)

    # plot forecasts in red
    off_s = (series.shape[0]) - n_test - 1
    off_e = off_s + n_test + 1
    xaxis = [x for x in range(off_s, off_e)]
    yaxis = [series[xaxis]] + forecasts
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

def split_dataset(data, n_test):
    train = data[:-n_test,:]
    test = data[-n_test:,:]
    return train, test


# load dataset
series = pd.read_csv('5yr-SP-data.csv')
series = series.drop(['Date'], 1)

# use SP data
sp_price = series['S&P Open']
sp_vol = series['S&P Volume']
sp_20day = series['S&P 20 day moving avrg']
sp_macd = series['S&P MACD']
sp_macde = series['S&P MACD EMA']
sp_div = series['S&P divergence']
vix = series['VIX Open']

# get daily SP price differential
sp_diff = difference(sp_price.values, 1)

# configure parameters
n_lag = 20
n_seq = 1
n_test = int(len(sp_price) * 0.25)
n_epochs = 100
n_batch = 1
n_neurons = 1

# specifies if network should retrain with new data after making each prediction
updateLSTM = False

# specifies if network should be trained or loaded from last training
load_model = False

# drop the last row of each varibale so they are all of length(sp_diff)
sp_price = sp_price.values[:-1]
sp_vol = sp_vol.values[:-1]
sp_20day = sp_20day.values[:-1]
sp_macd = sp_macd.values[:-1]
sp_macde = sp_macde.values[:-1]
sp_div = sp_div.values[:-1]
vix = vix.values[:-1]
sp_diff = sp_diff.values[:] #do nothing but pointing out it's still a variable
data = np.ndarray([len(sp_price),8])
data[:,0] = sp_price[:]
data[:,1] = sp_vol[:]
data[:,2] = sp_20day[:]
data[:,3] = sp_macd[:]
data[:,4] = sp_macde[:]
data[:,5] = sp_div[:]
data[:,6] = vix[:]
data[:,7] = sp_diff[:]

# prepare data
scaler, dataset = prepare_data(data, n_lag, n_seq)
train, test = split_dataset(dataset, n_test)


if load_model:
    model = load_network(n_neurons, n_epochs)
else:
    # fit network
    model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)

# make forecast
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq, updateLSTM)

# inverse transform forecasts and test
forecasts = inverse_transform(data, forecasts, scaler,  n_test+n_seq-1)
actual = [[row[7]] for row in test]
actual = inverse_transform(data, actual, scaler,  n_test+n_seq-1)

# evaluate and plot forecasts
print('\n\nParameters:\nNeurons =', n_neurons, '\nEpochs =', n_epochs)
#evaluate_forecasts(actual, forecasts, n_lag, n_seq)

plot_forecasts(sp_price, forecasts,  n_test+n_seq-1)

evaluate_forecasts(sp_diff[-n_test:],forecasts,n_lag,n_seq)
if not load_model:
    save_model = input('Do you want to save this network? This will overwrite any previously saved network with the same parameters\n(y/n) ')

    if save_model.lower() == 'y':
        # save network
        save_network(model, n_neurons, n_epochs)