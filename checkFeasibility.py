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
            _label = _label + " (" + printableBegin + "-" + printableEnd + ")"

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

def executeFeasibilityAnalysisalfonsPechStr(_data: np.array):
    # the first three days
    threeDaysTimestampData = np.array(_data[:4320,0], dtype=float)
    threeDaysMeasuremntData = np.array(_data[:4320,1], dtype=float)
    # all available data
    allDaysTimestampData = np.array(_data[:,0], dtype=float)
    allDaysMeasurementData = np.array(_data[:,1], dtype=float)

    # draw the plain production data
    plotTimeSeries(threeDaysMeasuremntData, threeDaysTimestampData, "production data")
    plotTimeSeries(allDaysMeasurementData, allDaysTimestampData, "production data")

    # check the autocorrelations
    checkAutocorrelation(threeDaysMeasuremntData, threeDaysTimestampData)
    checkAutocorrelation(allDaysMeasurementData, allDaysTimestampData)

    # check the difference values
    calcDifferenceSeries(threeDaysMeasuremntData, threeDaysTimestampData)
    calcDifferenceSeries(allDaysMeasurementData, allDaysTimestampData)

    dataWithoutTimestamp = deleteColFromNpArray(_data, 0)
    doCorrelationAnalysis(dataWithoutTimestamp) # from prepareData.py

def executeFeasibilityAnalysistanzendeSiedlung(_data: np.array):
    # the first three days
    threeDayData = {}
    threeDayData["timestamp"] = np.array(_data[:96,0], dtype=float)
    threeDayData["networkObtainanceQuarter"] = np.array(_data[:96,1], dtype=float)
    threeDayData["networkFeedInQuarter"] = np.array(_data[:96,2], dtype=float)
    threeDayData["PVConsumption"] = np.array(_data[:96,3], dtype=float)
    threeDayData["PVFeedIn"] = np.array(_data[:96,4], dtype=float)

    # plot time series of the plain data
    plotTimeSeries(threeDayData["networkObtainanceQuarter"], threeDayData["timestamp"], "Netzbezug durch das gesamete Quartier")
    plotTimeSeries(threeDayData["networkFeedInQuarter"], threeDayData["timestamp"], "Netzeinspeisung durch das Quartier")
    plotTimeSeries(threeDayData["PVConsumption"], threeDayData["timestamp"], "Bezug durch PV-Analge")
    plotTimeSeries(threeDayData["PVFeedIn"], threeDayData["timestamp"], "Einspeisung der PV-Anlage ins Quartier")

    # all available data
    # allDaysTimestampData = np.array(_data[:,0], dtype=float)
    # allDaysMeasurementData = np.array(_data[:,1], dtype=float)

    # # draw the plain production data
    # plotTimeSeries(allDaysMeasurementData, allDaysTimestampData, "production data")
    #
    # # check the autocorrelations
    # checkAutocorrelation(threeDaysMeasuremntData, threeDaysTimestampData)
    # checkAutocorrelation(allDaysMeasurementData, allDaysTimestampData)
    #
    # # check the difference values
    # calcDifferenceSeries(threeDaysMeasuremntData, threeDaysTimestampData)
    # calcDifferenceSeries(allDaysMeasurementData, allDaysTimestampData)
    #
    # dataWithoutTimestamp = deleteColFromNpArray(_data, 0)
    # doCorrelationAnalysis(dataWithoutTimestamp) # from prepareData.py
