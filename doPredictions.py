from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

from prepareData import *

# fix random seed for reproducibility
np.random.seed(7)

# splits data into 67% train data and 33% test data
def splitTestAndTrainingData(dataset: np.array):
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print("split the dataset into ", len(train), " train data and ", len(test), "test data")
    return train, test

# convert an array of values into a dataset matrix, by adding a column with the original data shifted by loo_back steps
# against the original dataset
# @look_back: steps to shift the array against itself
# @array_index: index of the target column in the input array
# @return dataX: original target array; dataY dataX shifted by look_back steps
def createDatasetWithLookback(dataset: np.array, look_back=1, array_index = 0):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), array_index]
        dataX.append(a)
        dataY.append(dataset[i + look_back, array_index])
    return np.array(dataX), np.array(dataY)

# this function takes the input, shifts it by look_back timesteps and trains a model based on these time-shifted values
def doPredictionWithLookback(dataset: np.array, look_back = 1, _epochs=1):
    # fix random seed for reproducibility
    np.random.seed(7)

    # since LSTMs are better suited for value ranges between 0 and 1 normalize the input to this value range
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = dataset[:, 1].reshape(-1, 1)
    dataset = scaler.fit_transform(dataset)

    train, test = splitTestAndTrainingData(dataset)

    trainX, trainY = createDatasetWithLookback(train, look_back, 0)
    testX, testY = createDatasetWithLookback(test, look_back, 0)

    # reshape input to be [samples, time steps, features] TODO: understand
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(trainX, trainY, epochs=_epochs, batch_size=1, verbose=2)
    for i in range(_epochs):
        model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2, shuffle=False)
        model.reset_states()

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions (turn back the normalization)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # draw the results
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()


def multivariateForecast(data: pd.DataFrame):
    # scale the input values to value rage of 0 to 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    data = series_to_supervised(data_scaled, 1, 1)

    # drop columns we don't want to predict
    data.drop(data.columns[[9, 8, 7, 6]], axis=1, inplace=True)

    # split in training and test data
    values = data.values # convert DataFrame to np array
    n_train_hours = round(data.shape[0] * 0.67)
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)


def doPredictionsForAlfonsPechStrasse(data: pd.DataFrame):

    # group data by day
    # groupedData = data.groupby(['Tag der Woche', 'Wochennummer']).sum()
    # groupedData = groupedData.to_numpy()
    # doPredictionWithLookback(groupedData, 4, 10000)

    # predict with influence factors
    multivariateForecast(data)
    print("debug")


