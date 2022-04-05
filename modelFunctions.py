from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import scipy
import time
from keras.callbacks import EarlyStopping
from scipy.optimize import curve_fit
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
import numpy as np
from lmfit.models import StepModel, LinearModel
import matplotlib.pyplot as plt
from prepareData import *
import matplotlib.dates as mdates

# fix random seed for reproducibility
np.random.seed(7)


# given a dataset predicts the next timestep based on the n_hours previous timesteps
# @dataset with target and influencing factors
# @n_stepsIntoPast: based on how many past observations shall the prediction be done
# @predict_index: which index in the data is the target size?
# @saveWeights: False when not required, path to folder to store weights in when required
def multivariateForecastNBackMForward(
        targetFilePath,
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
        _lossFunction="mae",
        _saveWeights=False,
        _plotLegendItems=False,
        _plotDateFormat="%d.%m"
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

    earlyStopping = EarlyStopping(monitor='val_loss', patience=75, mode='min', verbose=1)

    # fit network
    print("train the model...")
    history = model.fit(
                    train_X,
                    train_y,
                    epochs=_epochs,
                    batch_size=batch_size,
                    validation_data=(test_X, test_y),
                    verbose=1,
                    shuffle=False,
                    callbacks=[earlyStopping]
                )

    # plot history
    pyplot.figure(targetFilePath + str(_modelNumber) + "losses")
    pyplot.plot(history.history['loss'], label='train loss')
    pyplot.plot(history.history['val_loss'], label='test loss')
    plt.legend(loc='upper right')
    if _run == 0:
        plt.savefig(targetFilePath + str(_modelNumber) + '_loss.jpg', bbox_inches='tight', dpi=150)

        # save model weights
        if _saveWeights != False:
            model.save(_saveWeights)

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

    if _run == 0:
        plotNForwardMBackwardsResults(inv_yhat, n_stepsIntoFuture, onlyTargetValuePredictions, onlyTargetValueTestData, _run, _modelNumber, targetFilePath, _plotLegendItems, _plotDateFormat)

    rmse = sqrt(mean_squared_error(onlyTargetValueTestData, onlyTargetValuePredictions))
    epochs = len(history.history['loss'])
    print("results for config lstm_units=", str(lstm_units), ", steps_forward = ", n_stepsIntoFuture, ", steps_backward=", n_stepsIntoPast, " batch_size=",  batch_size)
    print('RMSE of test data: %.3f' % rmse)

    return [rmse, epochs]


def plotNForwardMBackwardsResults(inv_yhat, n_stepsIntoFuture, onlyTargetValuePredictions, onlyTargetValueTestData, _run, _modelNumber, targetFilePath, _plotLegendItems, _plotDateFormat):

    # plot the complete prediction series for the first three days as predicted at midnight from
    # the previous 24 hours
    loop_top = int(inv_yhat.shape[0] / n_stepsIntoFuture)
    plottableTestData = np.zeros((0))
    plottablePredictionData = np.zeros((0))
    # pyplot.figure(targetFilePath + str(_modelNumber) + "forecasts")
    for i in range(loop_top):
        # plot the forecasts as sepearted values
        # prepend None values to the day to display, so it is shifted against the previously printed day
        prependNoneData = np.zeros((n_stepsIntoFuture * i))
        prependNoneData[:] = None

        # prepare to plot forecasts in one line
        plottableTestData = np.concatenate((plottableTestData, onlyTargetValueTestData[i * n_stepsIntoFuture, :]))
        plottablePredictionData = np.concatenate((plottablePredictionData, onlyTargetValuePredictions[i * n_stepsIntoFuture, :]))

    pyplot.figure(targetFilePath + str(_modelNumber) + "forecasts")

    # plot all lines as one in one chart
    if type(_plotLegendItems) != bool:
        # for final outputs
        pyplot.plot(_plotLegendItems[:len(plottableTestData)].values, plottableTestData, label='Originaldaten')
        pyplot.plot(_plotLegendItems[:len(plottableTestData)].values, plottablePredictionData, label='Vorhersagen', alpha=0.7)
        pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter(_plotDateFormat))
        pyplot.legend()
        plt.savefig(targetFilePath + str(_modelNumber) + '_predictions.jpg', bbox_inches='tight', dpi=150)
        pyplot.clf()

        # plot first n forecast
        multiplicationFactor = 3
        if n_stepsIntoFuture == 4 or n_stepsIntoFuture == 60:
            multiplicationFactor = 24
        threeTimesPredictionSpan = min(n_stepsIntoFuture * multiplicationFactor, len(plottableTestData))

        beginIndex = 0
        if n_stepsIntoFuture == 60:
            beginIndex = 60 * 14
            threeTimesPredictionSpan += beginIndex - 2 * 60

        pyplot.plot(_plotLegendItems[beginIndex:threeTimesPredictionSpan].values, plottableTestData[beginIndex:threeTimesPredictionSpan], label='Originaldaten')
        pyplot.plot(_plotLegendItems[beginIndex:threeTimesPredictionSpan].values, plottablePredictionData[beginIndex:threeTimesPredictionSpan], label='Vorhersagen', alpha=0.7)
        pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter(_plotDateFormat))
        pyplot.legend()
        plt.savefig(targetFilePath + str(_modelNumber) + '_predictions_3times.jpg', bbox_inches='tight', dpi=150)
        pyplot.show()
    else:
        pyplot.plot(plottableTestData,  label='Originaldaten')
        pyplot.plot(plottablePredictionData, label='Vorhersagen', alpha=0.7)
        pyplot.legend()
        plt.savefig(targetFilePath + str(_modelNumber) + '_predictions.jpg', bbox_inches='tight', dpi=150)
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
        targetFilePath,
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
        _lossFunction="mae",
        _saveWeights=False,
        _plotLegendItems=False,
        _plotDateFormat="%d.%m"
):

    rmse = np.array([0.0, 0.0, 0.0])
    epochs = np.array([0.0, 0.0, 0.0])
    executionTime = np.array([0.0, 0.0, 0.0])

    # execute model 3 times
    for i in range(3):
        startTime = time.time()
        result = multivariateForecastNBackMForward(
            targetFilePath,
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
            _lossFunction,
            _saveWeights,
            _plotLegendItems,
            _plotDateFormat
        )
        rmse[i] = result[0]
        epochs[i] = result[1]
        executionTime[i] = (time.time() - startTime)

    averageRMSE = np.sum(rmse) / 3
    averageExecutionTime = np.sum(executionTime) / 3

    # store the results in csv file
    file_path = targetFilePath + 'regressionModels.csv'
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path, sep=";")
    else:
        df = pd.DataFrame()

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
        "lossFunction": _lossFunction,
        "epochs": epochs
    }
    df = df.append(newEntry, ignore_index=True)
    df.to_csv(file_path, sep=";", index=False)



def function(x, a, b, c, d):
    return a*x**3 + b * x ** 2 + c * x + d

def approximateFunctionToData(_data: pd.DataFrame):
    data = _data.copy()
    data = data[["Messwert", "Zeitstempel"]]
    x = pd.Series(list(range(1441)))
    y = data.loc[:1440, "Messwert"]

    np.random.seed(6)

    popt, cov = scipy.optimize.curve_fit(function, x, y, maxfev=20000)
    a, b, c, d = popt

    x_new_value = np.arange(min(x), max(x), 5)
    y_new_value = function(x_new_value, a, b, c, d)

    plt.scatter(x,y,color="green")
    plt.plot(x_new_value,y_new_value,color="red")
    plt.xlabel('X')
    plt.ylabel('Y')
    print("Estimated value of a : "+ str(a))
    print("Estimated value of b : " + str(b))
    plt.show()






