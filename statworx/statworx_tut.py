import numpy as np
import pandas as pd
import tensorflow
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('data_stocks.csv')

# drop date var
data = data.drop(['DATE'], 1)

n = data.shape[0]
p = data.shape[1]

# make np array
data = data.values

#plt.plot(data['SP500'])
#plt.show()

train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# scale
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# build x and y
x_train = data_train[:, 1:]
y_train = data_train[:, 0]
x_test = data_test[:, 1:]
y_test = data_test[:, 0]
