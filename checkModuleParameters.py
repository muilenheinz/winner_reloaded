import pandas as pd
import tensorflow as tf
import sys
import re

from modelFunctions import *

# test all "plausible" values for the given factors
def determineOptimalParametersForModel(onlyRelevantFactors, _targetFilePath, stepsIntoPast, stepsIntoFuture, predictIndex=0):
    # test number of units
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.2, 1, 2, "mae")
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 256, 128, 0.8, 0.2, 2, 1000, "mae")
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 512, 128, 0.8, 0.2, 3, 1000, "mae")
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 1024, 128, 0.8, 0.2, 4, 1000, "mae")

    # test batch_size
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 32, 0.8, 0.2, 5, 1000, "mae")
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 64, 0.8, 0.2, 6, 1000, "mae")
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.2, 7, 1000, "mae")

    # test dropout
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0, 9, 1000, "mae")
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.1, 10, 1000, "mae")

    # test optimization function
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.1, 11, 1000, "mse")
    cosine_loss_fn = tf.keras.losses.CosineSimilarity(axis=1)
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.1, 12, 1000, cosine_loss_fn)

    huber_loss = tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.1, 13, 1000, huber_loss)
    meanAbsolutePercentageError = tf.keras.losses.MeanAbsolutePercentageError(reduction="auto", name="mean_absolute_percentage_error")
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.1, 14, 1000, meanAbsolutePercentageError)
    meanSquaredLogarithmicError = tf.keras.losses.MeanSquaredLogarithmicError(reduction="auto", name="mean_squared_logarithmic_error")
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.1, 15, 1000, meanSquaredLogarithmicError)
    logCosh = tf.keras.losses.LogCosh(reduction="auto", name="log_cosh")
    checkModuleParameters(_targetFilePath, onlyRelevantFactors, stepsIntoFuture, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.1, 16, 1000, logCosh)

    # test steps into past
    index = 0
    for i in stepsIntoPast:
        checkModuleParameters(_targetFilePath, onlyRelevantFactors, i, stepsIntoFuture, predictIndex, 100, 128, 0.8, 0.2, (17 + index), 100, "mae")
        index += 1

def determineOptimalParametersForAlfonsPechStrasse(data: pd.DataFrame, _calc60MinuteModel = True, _calc24HourModel = True, _calc7DaysModel = True):
    print("-- calc predictions for Alfons-Pech-Strasse --")
    data = data.astype("float")

    # 60-Minute forecast on "plain" (ungrouped) data
    if _calc60MinuteModel:
        onlyRelevantFactors = filterDataBasedOnKendallRanks(data, "Messwert", 0.3)
        determineOptimalParametersForModel(onlyRelevantFactors, "../results/aps_regression_60minutes/aps_60min_", [30, 60, 120, 180], 60, 0)

    # forecast for next 24 hours on hourly basis
    if _calc24HourModel:
        data = data.apply(getHourFromTimestamp, axis=1)
        groupedData = data.groupby(['Wochennummer', 'Tag der Woche', 'Stunde']).sum()
        groupedData = groupedData / 60
        onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Messwert", 0.3)
        determineOptimalParametersForModel(onlyRelevantFactors, "../results/aps_regression_24hours/aps_24h_", [12, 48, 96], 24)

    # forecast for complete days
    if _calc7DaysModel:
        groupedData = data.groupby(['Wochennummer', 'Tag der Woche']).sum()
        groupedData = groupedData / 1440
        onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Messwert", 0.3)
        determineOptimalParametersForModel(onlyRelevantFactors, "../results/aps_regression_7days/aps_1day_", [7, 14, 21], 7)

def determineOptimalParametersForTanzendeSiedlung(
        data: pd.DataFrame,
        _calc60minutesFeedIn=True,
        _calc24hoursFeedIn=True,
        _calc7daysFeedIn=True,
        _calc60minutesUsage=True,
        _calc24hoursUsage=True,
        _calc7daysUsage=True,
):
    print("-- calc predictions for tanzende Siedlung --")
    data = data.astype("float")

    # forecast on "plain" (ungrouped) data for the net 4 quarter hours
    if _calc60minutesFeedIn:
        onlyRelevantFactors = filterDataBasedOnKendallRanks(data, "Netzeinspeisung", 0.3)
        determineOptimalParametersForModel(onlyRelevantFactors, "../results/ts_regression_feedin_60minutes/ts_feedIn_60min_", [4, 8, 12], 4, 1)

    # forecast for next 24 hours on hourly basis
    if _calc24hoursFeedIn:
        data = data.apply(getHourFromTimestamp, axis=1)
        groupedData = data.groupby(['Wochennummer', 'Tag der Woche', 'Stunde']).sum()
        onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Netzeinspeisung", 0.3)
        determineOptimalParametersForModel(onlyRelevantFactors, "../results/ts_regression_feedin_24hours/ts_feedIn_24h_", [24, 48, 72], 24, 1)

    # forecast for complete days
    if _calc7daysFeedIn:
        groupedData = data.groupby(['Wochennummer', 'Tag der Woche']).sum()
        onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Netzeinspeisung", 0.3)
        determineOptimalParametersForModel(onlyRelevantFactors, "../results/ts_regression_feedin_7days/ts_feedIn_7days_", [7, 14, 21], 7, 1)

    print("###################### Tanzende Siedlung: Nezeinspeisung durch, berechne Gesamtverbrauch ##################")

    # forecast on "plain" (ungrouped) data for the next 4 quarter hours
    if _calc60minutesUsage:
        onlyRelevantFactors = filterDataBasedOnKendallRanks(data, "Gesamtverbrauch", 0.3)
        determineOptimalParametersForModel(onlyRelevantFactors, "../results/ts_regression_usage_60minutes/ts_usage_60min_", [4, 8, 12], 4, 7)

    # forecast for next 24 hours on hourly basis
    if _calc24hoursUsage:
        data = data.apply(getHourFromTimestamp, axis=1)
        groupedData = data.groupby(['Wochennummer', 'Tag der Woche', 'Stunde']).sum()
        onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Gesamtverbrauch", 0.3)
        determineOptimalParametersForModel(onlyRelevantFactors, "../results/ts_regression_usage_24hours/ts_usage_24h_", [24, 48, 72], 24, 7)

    # forecast for complete days
    if _calc7daysUsage:
        groupedData = data.groupby(['Wochennummer', 'Tag der Woche']).sum()
        onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Gesamtverbrauch", 0.3)
        determineOptimalParametersForModel(onlyRelevantFactors, "../results/ts_regression_usage_7days/ts_usage_7days_", [7, 14, 21], 7, 7)

# read parameters from command line and execute the requested analysis based on that
def triggerAnalysisExecution():

    # check parameters for alfons-pech-strasse
    # convert input string to boolean array
    apsParameters = sys.argv[1]
    params = re.findall('.', apsParameters)
    params = [int(num) == 1 for num in params]

    # execute aps analysis
    apsData = prepareAlfonsPechStrasseData()
    determineOptimalParametersForAlfonsPechStrasse(apsData, params[0], params[1], params[2])

    # check parameters for tanzende Siedlung
    # convert input string to boolean array
    tasParameters = sys.argv[2]
    params = re.findall('.', tasParameters)
    params = [int(num) == 1 for num in params]

    tanzendeSiedlungData = prepareTanzendeSiedlungData()
    determineOptimalParametersForTanzendeSiedlung(tanzendeSiedlungData, params[0], params[1], params[2], params[3], params[4], params[5])


triggerAnalysisExecution()