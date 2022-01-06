from prepareData import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import kendalltau

def plotTimeSeries(dataY, dataX=None, _label="scatter", _type="scatter"):
    # if no xAxis is given mock the axis
    if dataX is None:
        dataX = np.array(list(np.arange(1, dataY.size+1)))
    else:
        if _type == "scatter":
            # append date range as info to the legend
            beginDatetimeValue = datetime.fromtimestamp(dataX[0] / 1000)
            endDatetimeValue = datetime.fromtimestamp(dataX[dataX.size - 1] / 1000)
            printableBegin = beginDatetimeValue.strftime("%d/%m/%Y")
            printableEnd = endDatetimeValue.strftime("%d/%m/%Y")
            _label = _label + " of " + printableBegin + "-" + printableEnd

    if _type == "scatter":
        plt.scatter(dataX, dataY, marker=".", label=_label)
    elif _type == "bar":
        plt.bar(dataX, dataY, label=_label)
        plt.xticks(rotation=45, ha="right")

    plt.legend(loc='upper right')
    plt.show()

# calcs the autocorrelation of the values of dataY,
# dataX contains the corresponding timestamps used later for plotting
def checkAutocorrelation(dataY: np.array, dataX: np.array,  _chartLabel=None):
    result = np.correlate(dataY, dataY, mode='full')
    result = result[result.size//2:]
    normalizedResult = result / float(result.max())

    plotTimeSeries(normalizedResult, dataX, "normalized Autocorrelation")

# for the 1D-np.array dataY calcs the deviation of a value from its predecessor
def calcDifferenceSeries(dataY: np.array, dataX: np.array):
    result = np.diff(dataY)
    plotTimeSeries(result, dataX[:-1], "differential data")

# calculate the kendalCoefficient, comparing the production datat to every other column in the given np.array
def doCorrelationAnalysis(data: np.array):
    productionData = data[:,0]
    kendallCoefficients = np.zeros(shape=data.shape[1] - 1)

    # iterate all available candidate factors
    for i in range(1, data.shape[1]):
        candidateData = data[:,i]
        corr, _ = kendalltau(productionData, candidateData)
        kendallCoefficients[i - 1] = corr

    colNames = np.array((
        "time",
        "dayOfWeek",
        "isWeekend",
        "weekNumber",
        "isHoliday (Feiertag)",
        "isSchoolHoliday",
        "diffuse Himmelstrahlung 10min (DS_10)",
        "globalstrahlung joule (GS_10)",
        "sonnenscheindauer (SD_10)",
        "Langwellige Strahlung (LS_10)",
        "Niederschlagsdauer 10min (RWS_DAU_10)",
        "Summe der Niederschlagsh. der vorangeg.10Min (RWS_10)",
        "Niederschlagsindikator  10min (RWS_IND_10)"
     ))

    # plot coefficients as barchart
    plotTimeSeries(kendallCoefficients, colNames, "Kendall Rank correlations", "bar")

def executeAnalysis():
    # get sample data
    useData = dataWithWeatherInformation # from the prepareData script
    # the first three days
    threeDaysDataX = np.array(useData[:4320,0], dtype=float)
    threeDaysDataY = np.array(useData[:4320,1], dtype=float)
    # all available data
    allDaysDataX = np.array(useData[:,0], dtype=float)
    allDaysDataY = np.array(useData[:,1], dtype=float)

    # draw the plain production data
    # plotTimeSeries(threeDaysDataY, threeDaysDataX, "production data")
    # plotTimeSeries(allDaysDataY, allDaysDataX, "production data")
    #
    # # check the autocorrelations
    # checkAutocorrelation(threeDaysDataY, threeDaysDataX)
    # checkAutocorrelation(allDaysDataY, allDaysDataX)
    #
    # # check the difference values
    # calcDifferenceSeries(threeDaysDataY, threeDaysDataX)
    # calcDifferenceSeries(allDaysDataY, allDaysDataX)

    doCorrelationAnalysis(finalData) # from prepareData.py

executeAnalysis()
print("debug checkFeasibility")