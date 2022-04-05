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
    apsData["Zeitstempel"] = apsData["Zeitstempel"] / 1000

    # get forecast timestamps plain (for 60 minute model)
    n_train_hours = round(apsData.shape[0] * 0.8)
    timestamps = apsData.agg({"Zeitstempel": [convertTimestampToDate]})[n_train_hours:]

    # get data for 24 hours model
    groupedData24Hours = apsData.groupby(['Wochennummer', 'Tag der Woche', 'Stunde']).sum()
    countGroupMembers = apsData.groupby(['Wochennummer', 'Tag der Woche', 'Stunde']).count()
    groupedData24Hours = groupedData24Hours.divide(countGroupMembers, axis=0)
    groupedData24Hours = groupedData24Hours.sort_values(by="Zeitstempel")

    # get timestamps for 24 hours model forecasts
    n_train_hours = round(groupedData24Hours.shape[0] * 0.8)
    timestamps24Hours = groupedData24Hours.agg({"Zeitstempel": [convertTimestampToDate]})[n_train_hours:]

    # get 7 days data
    groupedData7Days = apsData.groupby(['Wochennummer', 'Tag der Woche']).sum()
    countGroupMembers = apsData.groupby(['Wochennummer', 'Tag der Woche']).count()["Stunde"]
    groupedData7Days= groupedData7Days.divide(countGroupMembers, axis=0)  # calc average timestamp
    groupedData7Days = groupedData7Days.sort_values(by="Zeitstempel")

    # get timestamps for 7 days model forecasts
    n_train_hours = round(groupedData7Days.shape[0] * 0.8)
    timestamps7Days = groupedData7Days.agg({"Zeitstempel": [convertTimestampToDate]})[n_train_hours:]

    baseURL = "../results/finalModels/"

    # 60 minute forecast
    onlyRelevantFactors = filterDataBasedOnKendallRanks(apsData, "Messwert", 0.3)
    checkModuleParameters(
        baseURL + "aps_60min", onlyRelevantFactors, 60, 60, 0, 100, 128, 0.8, 0.2, 0, 1000, "mae",
        baseURL + "aps_60min_weights", timestamps, "%d.%m \n %H:%M"
    )

    # 24 hours forecast
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData24Hours, "Messwert", 0.3)
    checkModuleParameters(
        baseURL + "aps_24h", onlyRelevantFactors, 24, 24, 0, 100, 128, 0.8, 0.2, 1, 1000, huber_loss,
        baseURL + "aps_24h_weights", timestamps24Hours, "%d.%m \n %H:%M"
    )

    # 7 days forecast
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData7Days, "Messwert", 0.3)
    # use 67% as train data to have at least one prediction to visualize
    checkModuleParameters(
        baseURL + "aps_7d", onlyRelevantFactors, 7, 7, 0, 100, 128, 0.67, 0.2, 2, 1000, "mse",
        baseURL + "aps_7d_weights", timestamps7Days, "%a \n %d.%m"
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
    groupedData24Hours["Zeitstempel"] = groupedData24Hours["Zeitstempel"].divide(countGroupMembers, axis=0)
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

    baseURL = "../results/finalModels/"

    # forecast 60 min feedIn
    onlyRelevantFactors = filterDataBasedOnKendallRanks(tasData, "Netzeinspeisung", 0.3)
    checkModuleParameters(
        baseURL + "tas_60min_feedIn", onlyRelevantFactors, 4, 4, 1, 100, 128, 0.8, 0.2, 0, 1000, "mae",
        baseURL + "tas_60min_feedIn_weights", timestamps, "%d.%m \n %H:%M"
    )

    # forecast 24 hour feedIn
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData24Hours, "Netzeinspeisung", 0.3)
    checkModuleParameters(
        baseURL + "tas_24h_feedin", onlyRelevantFactors, 24, 24, 1, 100, 128, 0.8, 0.2, 1, 1000, meanAbsolutePercentageError,
        baseURL + "tas_24h_feedin_weights", timestamps24Hours, "%d.%m \n %H:%M"
    )

    # forecast 7 days feedIn
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData7Days, "Netzeinspeisung", 0.3)
    checkModuleParameters(
        baseURL + "tas_7d_feedin", onlyRelevantFactors, 7, 7, 1, 1024, 128, 0.8, 0, 1, 1000, cosine_similarity,
        baseURL + "tas_7d_feedin_weights", timestamps7Days, "%a \n %d.%m"
    )

    # forecast 60 minute usage
    onlyRelevantFactors = filterDataBasedOnKendallRanks(tasData, "Gesamtverbrauch", 0.3)
    checkModuleParameters(
        baseURL + "tas_60min_usage", onlyRelevantFactors, 4, 4, 7, 100, 128, 0.8, 0.1, 1, 1000, meanSquaredLogarithmicError,
        baseURL + "tas_60min_usage_weights", timestamps, "%d.%m \n %H:%M"
    )

    # forecast 24 hour usage
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData24Hours, "Gesamtverbrauch", 0.3)
    checkModuleParameters(
        baseURL + "tas_24h_usage", onlyRelevantFactors, 24, 24, 7, 100, 128, 0.8, 0.2, 1, 1000, huber_loss,
        baseURL + "tas_24h_usage_weights", timestamps24Hours, "%d.%m \n %H:%M"
    )

    # forecast 7 days usage
    onlyRelevantFactors = filterDataBasedOnKendallRanks(groupedData7Days, "Gesamtverbrauch", 0.3)
    checkModuleParameters(
        baseURL + "tas_7d_usage", onlyRelevantFactors, 14, 7, 7, 100, 128, 0.8, 0.2, 1, 1000, logCosh,
        baseURL + "tas_7d_usage_weights", timestamps7Days, "%a \n %d.%m"
    )


calcModelsForAlfonsPechStrasse()
calcModelsForTanzendeSiedlung()