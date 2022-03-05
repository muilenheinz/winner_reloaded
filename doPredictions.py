from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
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

    # reshape input to be [samples, time steps, features]
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


# given the input data calculates a model which makes one prediction out of one timestep back
def multivariateForecastOneBackOneForward(data: pd.DataFrame):
    # scale the input values to value rage of 0 to 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    data_supervised = series_to_supervised(data_scaled, 1, 1)

    # drop columns we don't want to predict
    data_supervised.drop(data_supervised.columns[[9, 8, 7, 6]], axis=1, inplace=True)

    # split in training and test data
    values = data_supervised.values # convert DataFrame to np array
    n_train_hours = round(data_supervised.shape[0] * 0.67)
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape input to be 3D [samples, timesteps (per sample), features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='huber', optimizer='adam')

    # fit network
    history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    pyplot.plot(history.history['accuracy'], label='train2')
    pyplot.plot(history.history['val_accuracy'], label='test2')
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

# given a dataset predicts the next timestep based on the n_hours previous timesteps
# @dataset with target and influencing factors
# @n_stepsIntoPast: based on how many past observations shall the prediction be done
# @predict_index: which index in the data is the target size?
def multivariateForecastNBackMForward(_data: pd.DataFrame, n_stepsIntoPast, n_stepsIntoFuture=1, predict_index=0):
    n_features = _data.shape[1]  # count number of influencing factors
    data = np.array(_data, dtype=float)

    # frame as supervised learning
    reframed = series_to_supervised(data, n_stepsIntoPast, n_stepsIntoFuture)

    # scale the input values to value rage of 0 to 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(reframed)

    # split into train and test sets
    n_train_hours = round(reframed.shape[0] * 0.67)
    print(n_train_hours)
    train = data_scaled[:n_train_hours, :]
    test = data_scaled[n_train_hours:, :]

    # split into input and outputs
    n_obs = n_stepsIntoPast * n_features
    n_predictions = n_features * n_stepsIntoFuture
    train_X, train_y = train[:, :n_obs], train[:, -n_predictions:]
    test_X, test_y = test[:, :n_obs], test[:, -n_predictions:]

    # for day in range(10):
    #     start = day * 1440
    #     end = start + 1440
    #     pyplot.plot(test_X[start:end, 0], label='test_X Tag' + str(day))
    #     pyplot.plot(data[n_train_hours+start:n_train_hours+start+1440, 0], label='Originaldaten Tag ' + str(day))
    #
    #     pyplot.plot(train_X[start:end, 0], label='train_X Tag '+ str(day))
    #     pyplot.plot(data[start:end, 0], label='Originaldaten Tag' + str(day))
    #
    #     pyplot.legend()
    #     pyplot.show()

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_stepsIntoPast, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_stepsIntoPast, n_features))

    # design network
    print("creating model...")
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(n_stepsIntoFuture * n_features))
    model.compile(loss='mae', optimizer='adam')
    # mse -> overfitting

    # fit network
    print("train the model...")
    history = model.fit(
                    train_X,
                    train_y,
                    epochs=100,
                    batch_size=128,
                    validation_data=(test_X, test_y),
                    verbose=2,
                    shuffle=False
                )

    # plot history
    pyplot.plot(history.history['loss'], label='train loss')
    pyplot.plot(history.history['val_loss'], label='test loss')
    plt.legend(loc='upper right')
    pyplot.show()

    # make a prediction
    yhat = model.predict(test_X)

    # invert scaling for forecast
    # the scaler needs an input with the same shape as the input shape, so concat the predictions with test_x to
    # achieve that
    test_X = test_X.reshape((test_X.shape[0], n_stepsIntoPast * n_features))
    inv_yhat = np.concatenate((test_X, yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)

    # invert scaling for actual data
    inv_y = np.concatenate((test_X, test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)

    plotNForwardMBackwardsResults(inv_yhat, inv_y, n_stepsIntoFuture, test_X, test_y, n_features)

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

def plotNForwardMBackwardsResults(inv_yhat, inv_y, n_stepsIntoFuture, test_X, test_Y, n_features):

    # get the target data out of predictions for the target factor
    target_factor_index = 0

    # get every target_factor_count's column out of the predictions
    onlyTargetValuePredictions = np.zeros((inv_yhat.shape[0], n_stepsIntoFuture))
    onlyTargetValueTestData = np.zeros((inv_yhat.shape[0], n_stepsIntoFuture))

    for i in range(n_stepsIntoFuture):
        targetColIndex = test_X.shape[1] + i * n_features + target_factor_index
        targetCol = inv_yhat[:, targetColIndex]
        onlyTargetValuePredictions[:, i] = targetCol

        targetCol = inv_y[:, targetColIndex]
        onlyTargetValueTestData[:, i] = targetCol

    # plot the complete prediction series for the first three days as predicted at midnight from
    # the previous 24 hours
    loop_top = int(inv_yhat.shape[0] / n_stepsIntoFuture)
    for i in range(loop_top):
        # prepend None values to the day to display, so it is shifted against the previously printed day
        prependNoneData = np.zeros((n_stepsIntoFuture * i))
        prependNoneData[:] = None
        printOneDayTestData = np.concatenate((prependNoneData, onlyTargetValueTestData[i * n_stepsIntoFuture, :]))
        printOneDayPredictionData = np.concatenate((prependNoneData, onlyTargetValuePredictions[i * n_stepsIntoFuture, :]))

        pyplot.plot(printOneDayTestData, label='Originaldaten Tag' + str(i))
        pyplot.plot(printOneDayPredictionData, label='Vorhersagen Tag' + str(i))

    # pyplot.legend()
    pyplot.show()

def plotResults(originaldata, predictions):
    pyplot.plot(originaldata, label='Originaldaten')
    pyplot.plot(predictions, label='Vorhersagen')
    pyplot.legend()
    pyplot.show()

def getHourFromTimestamp(datarow):
    datetimeValue = datetime.fromtimestamp(int(datarow["Zeitstempel"]) / 1000)
    hourVal = int(datetimeValue.strftime('%H'))
    datarow["Stunde"] = hourVal
    return datarow

def getTimeCosFromTimestamp(datarow):
    seconds_in_day = 24*60*60
    datarow["Zeit (cos)"] = np.cos(2*np.pi*datarow["Zeitstempel"]/seconds_in_day)
    return datarow

def doPredictionsForAlfonsPechStrasse(data: pd.DataFrame):
    print("-- calc predictions for Alfons-Pech-Strasse --")
    # group data by day
    # groupedData = data.groupby(['Wochennummer', 'Tag der Woche']).sum()
    # groupedData = groupedData.to_numpy()
    # doPredictionWithLookback(groupedData, 4, 10000)

    # group data by hour
    data = data.apply(getHourFromTimestamp, axis=1)
    data = data.astype("float")
    groupedData = data.groupby(['Wochennummer', 'Tag der Woche', 'Stunde']).sum()
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Messwert", 0.3)
    multivariateForecastNBackMForward(onlyRelevantFactors, 48, 24)
    print("debug")


