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
import time
from keras.callbacks import EarlyStopping

from prepareData import *
targetFilePath = ""

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
def multivariateForecastNBackMForward(
        _data: pd.DataFrame,
        n_stepsIntoPast,
        n_stepsIntoFuture=1,
        predict_index=0,
        lstm_units=256,
        batch_size=32,
        share_traindata=0.8,
        _dropout=0.2,
        _modelNumber=0,
        _epochs=50,
        _run=0,
        _lossFunction="mae"
):
    n_features = _data.shape[1]  # count number of influencing factors
    data = np.array(_data, dtype=float)

    # frame as supervised learning
    reframed = series_to_supervised(data, n_stepsIntoPast, n_stepsIntoFuture)

    # scale the input values to value rage of 0 to 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(reframed)

    # split into train and test sets
    n_train_hours = round(reframed.shape[0] * share_traindata)
    train = data_scaled[:n_train_hours, :]
    test = data_scaled[n_train_hours:, :]

    # split into input and outputs
    n_obs = n_stepsIntoPast * n_features
    n_predictions = n_features * n_stepsIntoFuture
    train_X, train_y = train[:, :n_obs], train[:, -n_predictions:]
    test_X, test_y = test[:, :n_obs], test[:, -n_predictions:]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_stepsIntoPast, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_stepsIntoPast, n_features))

    # design network
    print("creating model...")
    model = Sequential()
    model.add(LSTM(lstm_units, dropout=_dropout, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(lstm_units, dropout=_dropout))
    model.add(Dense(n_stepsIntoFuture * n_features))
    model.compile(loss=_lossFunction, optimizer='adam')

    # earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

    # fit network
    print("train the model...")
    history = model.fit(
                    train_X,
                    train_y,
                    epochs=_epochs,
                    batch_size=batch_size,
                    validation_data=(test_X, test_y),
                    verbose=2,
                    shuffle=False,
                    # callbacks=[earlyStopping]
                )

    # plot history
    pyplot.plot(history.history['loss'], label='train loss')
    pyplot.plot(history.history['val_loss'], label='test loss')
    plt.legend(loc='upper right')
    if _run == 0:
        plt.savefig(targetFilePath + str(_modelNumber) + '_loss.jpg', bbox_inches='tight', dpi=150)
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

    # filter only the target cols (energy production for aps) out of the forecast
    onlyTargetValuePredictions = np.zeros((inv_yhat.shape[0], n_stepsIntoFuture))
    onlyTargetValueTestData = np.zeros((inv_yhat.shape[0], n_stepsIntoFuture))

    for i in range(n_stepsIntoFuture):
        targetColIndex = test_X.shape[1] + i * n_features + predict_index
        targetCol = inv_yhat[:, targetColIndex]
        onlyTargetValuePredictions[:, i] = targetCol

        targetCol = inv_y[:, targetColIndex]
        onlyTargetValueTestData[:, i] = targetCol

    plotNForwardMBackwardsResults(inv_yhat, n_stepsIntoFuture, onlyTargetValuePredictions, onlyTargetValueTestData, _run, _modelNumber)

    rmse = sqrt(mean_squared_error(onlyTargetValueTestData, onlyTargetValuePredictions))
    print("results for config lstm_units=", str(lstm_units), ", steps_forward = ", n_stepsIntoFuture, ", steps_backward=", n_stepsIntoPast, " batch_size=",  batch_size)
    print('RMSE of test data: %.3f' % rmse)

    return rmse


def plotNForwardMBackwardsResults(inv_yhat, n_stepsIntoFuture, onlyTargetValuePredictions, onlyTargetValueTestData, _run, _modelNumber):
    global targetFilePath

    # plot the complete prediction series for the first three days as predicted at midnight from
    # the previous 24 hours
    loop_top = int(inv_yhat.shape[0] / n_stepsIntoFuture)
    plottableTestData = np.zeros((1))
    plottablePredictionData = np.zeros((1))
    for i in range(loop_top):
        # plot the forecasts as sepearted values
        # prepend None values to the day to display, so it is shifted against the previously printed day
        prependNoneData = np.zeros((n_stepsIntoFuture * i))
        prependNoneData[:] = None
        printOneDayTestData = np.concatenate((prependNoneData, onlyTargetValueTestData[i * n_stepsIntoFuture, :]))
        printOneDayPredictionData = np.concatenate((prependNoneData, onlyTargetValuePredictions[i * n_stepsIntoFuture, :]))

        pyplot.plot(printOneDayTestData, label='Originaldaten Tag' + str(i))
        pyplot.plot(printOneDayPredictionData, label='Vorhersagen Tag' + str(i))

        # prepare to plot those forecasts in one line
        plottableTestData = np.concatenate((plottableTestData, onlyTargetValueTestData[i * n_stepsIntoFuture, :]))
        plottablePredictionData = np.concatenate((plottablePredictionData, onlyTargetValuePredictions[i * n_stepsIntoFuture, :]))

    if loop_top < 3:
        pyplot.legend()

    # store the plot on the first run as an image
    if _run == 0:
        plt.savefig(targetFilePath + str(_modelNumber) + '_predictions_split.jpg', bbox_inches='tight', dpi=150)

    pyplot.show()

    # plot all lines as one in one chart
    pyplot.plot(plottableTestData, label='Originaldaten')
    pyplot.plot(plottablePredictionData, label='Vorhersagen')
    pyplot.legend()
    if _run == 0:
        plt.savefig(targetFilePath + str(_modelNumber) + '_predictions.jpg', bbox_inches='tight', dpi=150)
    pyplot.show()

    pyplot.plot(plottableTestData[:1440], label='Originaldaten')
    pyplot.plot(plottablePredictionData[:1440], label='Vorhersagen')
    pyplot.legend()
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

def checkModuleParameters(
        _data: pd.DataFrame,
        n_stepsIntoPast,
        n_stepsIntoFuture=1,
        predict_index=0,
        lstm_units=256,
        batch_size=32,
        share_traindata=0.8,
        _dropout=0.2,
        _modelNumber=0,
        _epochs = 50,
        _lossFunction="mae"):
    global targetFilePath
    rmse = np.array([0.0, 0.0, 0.0])
    executionTime = np.array([0.0, 0.0, 0.0])

    # store the results in csv file
    for i in range(3):
        startTime = time.time()
        rmse[i] = multivariateForecastNBackMForward(
            _data,
            n_stepsIntoPast,
            n_stepsIntoFuture,
            predict_index,
            lstm_units,
            batch_size,
            share_traindata,
            _dropout,
            _modelNumber,
            _epochs,
            i,
            _lossFunction
        )
        executionTime[i] = (time.time() - startTime)

    averageRMSE = np.sum(rmse) / 3
    averageExecutionTime = np.sum(executionTime) / 3

    file_path = targetFilePath + 'regressionModels.csv'
    df = pd.read_csv(file_path, sep=";")
    newEntry = {
        "Nummer": _modelNumber,
        "LSTM-Layer": "2",
        "LSTM Units": lstm_units,
        "Epochen": _epochs,
        "steps_into_past": n_stepsIntoPast,
        "steps_into_future": n_stepsIntoFuture,
        "Trainingdata share": share_traindata,
        "batch_size": batch_size,
        "dropout": _dropout,
        "Trainingsdauer": averageExecutionTime,
        "Test-RMSE Run 1": rmse[0],
        "Run 2": rmse[1],
        "Run 3": rmse[2],
        "Durchschnitt": averageRMSE,
        "lossFunction": _lossFunction
    }
    df = df.append(newEntry, ignore_index=True)
    df.to_csv(file_path, sep=";", index=False)


def doPredictionsForAlfonsPechStrasse(data: pd.DataFrame):
    print("-- calc predictions for Alfons-Pech-Strasse --")
    global targetFilePath
    targetFilePath = "../results/aps_regression_60minutes/"

    # 60-Minute forecast on "plain" (ungrouped) data
    onlyRelevantFactors = filterDataBasedOnKendallRanks(data, "Messwert", 0.3)
    # optimize steps into past
    checkModuleParameters(onlyRelevantFactors, 60, 60, 0, 100, 128, 0.67, 0.2, 1, 100, "mae")
    checkModuleParameters(onlyRelevantFactors, 120, 60, 0, 100, 128, 0.67, 0.2, 1, 100, "mae")
    checkModuleParameters(onlyRelevantFactors, 1440, 60, 0, 100, 128, 0.67, 0.2, 1, 100, "mae")

    # group data by day
    # groupedData = data.groupby(['Wochennummer', 'Tag der Woche']).sum()
    # groupedData = groupedData.to_numpy()
    # doPredictionWithLookback(groupedData, 4, 10000)

    # group data by hour
    # data = data.apply(getHourFromTimestamp, axis=1)
    # data = data.astype("float")
    # groupedData = data.groupby(['Wochennummer', 'Tag der Woche', 'Stunde']).sum()
    # onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Messwert", 0.3)
    #
    # checkModuleParameters(onlyRelevantFactors, 20, 20, 0, 100, 128, 0.67, 0.2, 2, 10, "mae")
    # multivariateForecastNBackMForward(onlyRelevantFactors, 96, 24, 0, 1024, 128, 0.8, 0.2)
    # multivariateForecastNBackMForward(onlyRelevantFactors, 96, 24, 0, 1024, 128, 0.67, 0.2)


