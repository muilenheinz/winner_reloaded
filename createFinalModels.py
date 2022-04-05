import pandas as pd

from prepareData import *
from modelFunctions import *

runs = 1

# define optimization functions
huber_loss = tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")
meanAbsolutePercentageError = tf.keras.losses.MeanAbsolutePercentageError(reduction="auto", name="mean_absolute_percentage_error")
cosine_similarity = tf.keras.losses.CosineSimilarity(axis=1)
logCosh = tf.keras.losses.LogCosh(reduction="auto", name="log_cosh")
meanSquaredLogarithmicError = tf.keras.losses.MeanSquaredLogarithmicError(reduction="auto", name="mean_squared_logarithmic_error")

def calcModelsForAlfonsPechStrasse():
    # prepare data
    apsData = prepareAlfonsPechStrasseData()
    apsData = apsData.astype("float")
    apsData = apsData.apply(getHourFromTimestamp, axis=1)

    # 60 minute forecast
    onlyRelevantFactors = filterDataBasedOnKendallRanks(apsData, "Messwert", 0.3)
    url = "../results/finalModels/aps_60min"
    multivariateForecastNBackMForward(
        url, onlyRelevantFactors,
        60, 60, 0, 100, 128, 0.8, 0.2, 0, runs, 0, "mae", url + "_weights"
    )

    # # 24 hours forecast
    groupedData = apsData.groupby(['Wochennummer', 'Tag der Woche', 'Stunde']).sum()
    groupedData = groupedData / 60
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Messwert", 0.3)
    url = "../results/finalModels/aps_24h"
    multivariateForecastNBackMForward(
        url, onlyRelevantFactors,
        24, 24, 0, 100, 128, 0.8, 0.2, 0, runs, 0, huber_loss, url + "_weights"
    )

    # 7 days forecast
    groupedData = apsData.groupby(['Wochennummer', 'Tag der Woche']).sum()
    groupedData = groupedData / 1440
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData, "Messwert", 0.3)
    url = "../results/finalModels/aps_7d"
    multivariateForecastNBackMForward(
        url, onlyRelevantFactors,
        7, 7, 0, 100, 128, 0.8, 0.2, 0, runs, 0, "mse", url + "_weights"
    )

def convertTimestampToDate(input):
    return datetime.utcfromtimestamp(int(input))


def calcModelsForTanzendeSiedlung():
    # prepare data
    tasData = prepareTanzendeSiedlungData()
    tasData = tasData.astype("float")
    tasData = tasData.apply(getHourFromTimestamp, axis=1)
    tasData = tasData.sort_values(by="Zeitstempel")
    tasData["Zeitstempel"] = tasData["Zeitstempel"] / 1000

    # get forecast timestamps plain (for 60 minute model)
    n_train_hours = round(tasData.shape[0] * 0.8)
    timestamps = tasData.agg({"Zeitstempel": [convertTimestampToDate]})[n_train_hours:]

    # get forecast data 24 hours
    groupedData24Hours = tasData.groupby(['Wochennummer', 'Tag der Woche', 'Stunde']).sum()
    countGroupMembers = tasData.groupby(['Wochennummer', 'Tag der Woche', 'Stunde']).count()["Zeitstempel"]
    groupedData24Hours = groupedData24Hours.divide(countGroupMembers, axis=0)
    groupedData24Hours = groupedData24Hours.sort_values(by="Zeitstempel")

    # get timestamp vales for the output chart scale for 24 hours data
    n_train_hours = round(groupedData24Hours.shape[0] * 0.8)
    timestamps24Hours = groupedData24Hours.agg({"Zeitstempel": [convertTimestampToDate]})[n_train_hours:]
    # timestamps24Hours = timestamps24Hours["Zeitstempel"][n_train_hours:]

    # get forecast data 7 days
    countGroupMembers = tasData.groupby(['Wochennummer', 'Tag der Woche']).count()["Stunde"]
    groupedData7Days = tasData.groupby(['Wochennummer', 'Tag der Woche']).sum()
    groupedData7Days["Zeitstempel"] = groupedData7Days["Zeitstempel"].divide(countGroupMembers, axis=0)  # calc average timestamp
    groupedData7Days = groupedData7Days.sort_values(by="Zeitstempel")

    # get timestamp vales for the output chart scale for 7 days data
    n_train_hours = round(groupedData7Days.shape[0] * 0.8)
    timestamps7Days = groupedData7Days.agg({"Zeitstempel": [convertTimestampToDate]})[n_train_hours:]
    # timestamps7Days = groupedData7Days["Zeitstempel"][n_train_hours:]

    baseURL = "../results/finalModels/"

    # forecast 60 min feedIn
    onlyRelevantFactors = filterDataBasedOnKendallRanks(tasData, "Netzeinspeisung", 0.3)
    checkModuleParameters(
        baseURL + "tas_60min_feedIn", onlyRelevantFactors, 4, 4, 1, 100, 128, 0.8, 0.2, 0, 3, "mae",
        baseURL + "tas_60min_feedIn_weights", timestamps, "%d.%m \n %H:%M"
    )

    # forecast 24 hour feedIn
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData24Hours, "Netzeinspeisung", 0.3)
    checkModuleParameters(
        baseURL + "tas_24h_feedin", onlyRelevantFactors, 24, 24, 1, 100, 128, 0.8, 0.2, 1, 3, "mae",
        baseURL + "tas_24h_feedin_weights", timestamps, "%d.%m \n %H:%M"
    )

    # forecast 7 days feedIn
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData7Days, "Netzeinspeisung", 0.3)
    checkModuleParameters(
        baseURL + "tas_7d_feedin", onlyRelevantFactors, 7, 7, 1, 100, 128, 0.8, 0.2, 1, 3, cosine_similarity,
        baseURL + "tas_7d_feedin_weights", timestamps, "%d.%m"
    )

    # forecast 60 minute usage
    onlyRelevantFactors = filterDataBasedOnKendallRanks(tasData, "Gesamtverbrauch", 0.3)
    checkModuleParameters(
        baseURL + "tas_60min_usage", onlyRelevantFactors, 60, 60, 7, 100, 128, 0.8, 0.2, 1, 3, meanSquaredLogarithmicError,
        baseURL + "tas_60min_usage_weights", timestamps, "%d.%m \n %H:%M"
    )

    # forecast 24 hour usage
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData24Hours, "Gesamtverbrauch", 0.3)
    checkModuleParameters(
        baseURL + "tas_24h_usage", onlyRelevantFactors, 24, 24, 7, 100, 128, 0.8, 0.2, 1, 3, meanAbsolutePercentageError,
        baseURL + "tas_24h_usage_weights", timestamps, "%d.%m \n %H:%M"
    )

    # forecast 7 days usage
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData7Days, "Gesamtverbrauch", 0.3)
    checkModuleParameters(
        baseURL + "tas_7d_usage", onlyRelevantFactors, 7, 7, 7, 100, 128, 0.8, 0.2, 1, 3, logCosh,
        baseURL + "tas_7d_usage_weights", timestamps, "%d.%m \n %H:%M"
    )


# calcModelsForAlfonsPechStrasse()
calcModelsForTanzendeSiedlung()