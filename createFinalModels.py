from prepareData import *
from modelFunctions import *

runs = 2

# define optimization functions
huber_loss = tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")

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
    url = "../results/finalModels/aps_24h"
    multivariateForecastNBackMForward(
        url, onlyRelevantFactors,
        7, 7, 0, 100, 128, 0.8, 0.2, 0, runs, 0, "mse", url + "_weights"
    )

calcModelsForAlfonsPechStrasse()