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
targetFilePath = ""

# fix random seed for reproducibility
np.random.seed(7)


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
                    verbose=1,
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
    pyplot.figure(0)
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
    pyplot.figure(1)

    # plot all lines as one in one chart
    pyplot.plot(plottableTestData, label='Originaldaten')
    pyplot.plot(plottablePredictionData, label='Vorhersagen')
    pyplot.legend()
    if _run == 0:
        plt.savefig(targetFilePath + str(_modelNumber) + '_predictions.jpg', bbox_inches='tight', dpi=150)
    pyplot.show()

    pyplot.figure(2)
    pyplot.plot(plottableTestData[:96], label='Originaldaten')
    pyplot.plot(plottablePredictionData[:96], label='Vorhersagen')
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

    # execute model 3 times
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
        "lossFunction": _lossFunction
    }
    df = df.append(newEntry, ignore_index=True)
    df.to_csv(file_path, sep=";", index=False)


def determineOptimalParametersForAlfonsPechStrasse(data: pd.DataFrame):
    print("-- calc predictions for Alfons-Pech-Strasse --")
    data = data.astype("float")

    # 60-Minute forecast on "plain" (ungrouped) data
    onlyRelevantFactors = filterDataBasedOnKendallRanks(data, "Messwert", 0.3)
    approximateFunctionToData(data)
    determineOptimalParametersForModel(onlyRelevantFactors, "../results/aps_regression_60minutes/", [30, 60, 120, 180], 60)

    # forecast for next 24 hours on hourly basis
    data = data.apply(getHourFromTimestamp, axis=1)
    groupedData = data.groupby(['Wochennummer', 'Tag der Woche', 'Stunde']).sum()
    groupedData = groupedData / 60
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Messwert", 0.3)
    determineOptimalParametersForModel(onlyRelevantFactors, "../results/aps_regression_24hours/", [12, 48, 96], 24)

    # forecast for complete days
    groupedData = data.groupby(['Wochennummer', 'Tag der Woche']).sum()
    groupedData = groupedData / 1440
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Messwert", 0.3)
    determineOptimalParametersForModel(onlyRelevantFactors, "../results/aps_regression_1day/", [7, 14, 21], 7)

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

def determineOptimalParametersForTanzendeSiedlung(data: pd.DataFrame):
    print("-- calc predictions for tanzende Siedlung --")
    data = data.astype("float")

    # forecast on "plain" (ungrouped) data for the net 4 quarter hours
    onlyRelevantFactors = filterDataBasedOnKendallRanks(data, "Netzeinspeisung", 0.3)
    determineOptimalParametersForModel(onlyRelevantFactors, "../results/ts_regression_feedin_60minutes/", [4, 8, 12], 4, 1)

    # forecast for next 24 hours on hourly basis
    data = data.apply(getHourFromTimestamp, axis=1)
    groupedData = data.groupby(['Wochennummer', 'Tag der Woche', 'Stunde']).sum()
    groupedData = groupedData / 4
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Netzeinspeisung", 0.3)
    determineOptimalParametersForModel(onlyRelevantFactors, "../results/ts_regression_feedin_24hours/", [24, 48, 72], 24, 1)

    # forecast for complete days
    groupedData = data.groupby(['Wochennummer', 'Tag der Woche']).sum()
    groupedData = groupedData / 96
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Netzeinspeisung", 0.3)
    determineOptimalParametersForModel(onlyRelevantFactors, "../results/ts_regression_feedin_1day/", [7, 14, 21], 7, 1)

    print("###################### Tanzende Siedlung: Nezeinspeisung durch, berechne Gesamtverbrauch ##################")

    # forecast on "plain" (ungrouped) data for the net 4 quarter hours
    onlyRelevantFactors = filterDataBasedOnKendallRanks(data, "Gesamtverbrauch", 0.3)
    determineOptimalParametersForModel(onlyRelevantFactors, "../results/ts_regression_usage_60minutes/", [4, 8, 12], 4, 1)

    # forecast for next 24 hours on hourly basis
    data = data.apply(getHourFromTimestamp, axis=1)
    groupedData = data.groupby(['Wochennummer', 'Tag der Woche', 'Stunde']).sum()
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Gesamtverbrauch", 0.3)
    determineOptimalParametersForModel(onlyRelevantFactors, "../results/ts_regression_usage_24hours/", [24, 48, 72], 24, 1)

    # forecast for complete days
    groupedData = data.groupby(['Wochennummer', 'Tag der Woche']).sum()
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Gesamtverbrauch", 0.3)
    determineOptimalParametersForModel(onlyRelevantFactors, "../results/ts_regression_usage_1day/", [7, 14, 21], 7, 1)


# test all "plausible" values for the given factors
def determineOptimalParametersForModel(onlyRelevantFactors, targetFile, stepsIntoPast, stepsIntoFuture, predictIndex=0, trainDataShare=0.67):
    global targetFilePath

    targetFilePath = targetFile

    # test number of units
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.2, 1, 1, "mae")
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 256, 128, 0.8, 0.2, 2, 100, "mae")
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 512, 128, 0.8, 0.2, 3, 100, "mae")
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 1024, 128, 0.8, 0.2, 4, 100, "mae")

    # test batch_size
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 32, 0.8, 0.2, 5, 100, "mae")
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 64, 0.8, 0.2, 6, 100, "mae")
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.2, 7, 100, "mae")

    # test dropout
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0, 9, 100, "mae")
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.1, 10, 100, "mae")

    # test optimization function
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.1, 11, 100, "mse")
    cosine_loss_fn = tf.keras.losses.CosineSimilarity(axis=1)
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.1, 12, 100, cosine_loss_fn)
    huber_loss = tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.1, 9, 13, 100, huber_loss)
    meanAbsolutePercentageError = tf.keras.losses.MeanAbsolutePercentageError(reduction="auto", name="mean_absolute_percentage_error")
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.1, 14, 100, meanAbsolutePercentageError)
    meanSquaredLogarithmicError = tf.keras.losses.MeanSquaredLogarithmicError(reduction="auto", name="mean_squared_logarithmic_error")
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.1, 15, 100, meanSquaredLogarithmicError)
    logCosh = tf.keras.losses.LogCosh(reduction="auto", name="log_cosh")
    checkModuleParameters(onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.1, 16, 100, logCosh)

    # test steps into past
    index = 0
    for i in stepsIntoPast:
        checkModuleParameters(onlyRelevantFactors, i, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.2, (17 + index), 100, "mae")
        index += 1



